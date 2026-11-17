"""Low-level protobuf wire-format parsing.

This module knows nothing about schemas; it only splits a byte buffer into
(field number, wire type, payload) records exactly as they appear on the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Union

MAX_FIELD_NUMBER = (1 << 29) - 1
# Field numbers reserved for the protobuf implementation; a .proto may not
# declare them, though they can appear on the wire.
RESERVED_RANGE = range(19000, 20000)


class WireType(IntEnum):
    VARINT = 0
    I64 = 1
    LEN = 2
    SGROUP = 3
    EGROUP = 4
    I32 = 5


class DecodeError(ValueError):
    """Raised when a buffer is not a well-formed protobuf message."""


@dataclass
class RawField:
    """One field record as it appears on the wire.

    ``value`` is an ``int`` for VARINT/I32/I64, ``bytes`` for LEN, and a list of
    nested :class:`RawField` for SGROUP (groups are parsed eagerly because the
    wire format leaves no other choice).
    """

    number: int
    wire_type: WireType
    value: Union[int, bytes, List["RawField"]]
    offset: int  # offset of the tag within the buffer that was parsed
    end: int  # offset just past this field
    data_offset: int = 0  # offset of the payload (after tag / length prefix)
    extra: dict = field(default_factory=dict)


def read_varint(buf: bytes, pos: int, end: int | None = None) -> Tuple[int, int]:
    """Read a base-128 varint starting at ``pos``; return (value, new_pos)."""
    if end is None:
        end = len(buf)
    result = 0
    shift = 0
    start = pos
    while True:
        if pos >= end:
            raise DecodeError(f"truncated varint at offset {start}")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            break
        shift += 7
        if shift >= 70:
            raise DecodeError(f"varint too long at offset {start}")
    if result >= 1 << 64:
        raise DecodeError(f"varint overflows 64 bits at offset {start}")
    return result, pos


def encode_varint(value: int) -> bytes:
    """Encode an int64 or uint64 value as a varint."""
    if not -(1 << 63) <= value < (1 << 64):
        raise ValueError(f"{value} is outside the int64/uint64 range")
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def parse_message(
    buf: bytes,
    start: int = 0,
    end: int | None = None,
    max_field_number: int = MAX_FIELD_NUMBER,
) -> List[RawField]:
    """Strictly parse ``buf[start:end]`` as a sequence of protobuf fields.

    Raises :class:`DecodeError` if the bytes are not a complete, well-formed
    message (bad wire type, field number 0, truncated payload, unbalanced
    groups, ...).
    """
    if end is None:
        end = len(buf)
    fields, pos = _parse(buf, start, end, None, max_field_number, 0)
    if pos != end:  # pragma: no cover - _parse always consumes to end or raises
        raise DecodeError("trailing data")
    return fields


def _parse(buf, pos, end, group_number, max_field_number, depth):
    if depth > 100:
        raise DecodeError("nesting too deep")
    fields: List[RawField] = []
    while pos < end:
        tag_offset = pos
        tag, pos = read_varint(buf, pos, end)
        number = tag >> 3
        wt = tag & 7
        if number == 0:
            raise DecodeError(f"field number 0 at offset {tag_offset}")
        if number > max_field_number:
            raise DecodeError(f"field number {number} too large at offset {tag_offset}")
        if wt > 5:
            raise DecodeError(f"invalid wire type {wt} at offset {tag_offset}")
        wire_type = WireType(wt)
        data_offset = pos
        if wire_type == WireType.VARINT:
            value, pos = read_varint(buf, pos, end)
        elif wire_type == WireType.I64:
            if pos + 8 > end:
                raise DecodeError(f"truncated fixed64 at offset {pos}")
            value = int.from_bytes(buf[pos : pos + 8], "little")
            pos += 8
        elif wire_type == WireType.I32:
            if pos + 4 > end:
                raise DecodeError(f"truncated fixed32 at offset {pos}")
            value = int.from_bytes(buf[pos : pos + 4], "little")
            pos += 4
        elif wire_type == WireType.LEN:
            length, pos = read_varint(buf, pos, end)
            data_offset = pos
            if pos + length > end:
                raise DecodeError(
                    f"length {length} at offset {tag_offset} runs past end of buffer"
                )
            value = bytes(buf[pos : pos + length])
            pos += length
        elif wire_type == WireType.SGROUP:
            value, pos = _parse(buf, pos, end, number, max_field_number, depth + 1)
        else:  # EGROUP
            if group_number != number:
                raise DecodeError(f"unexpected end-group for field {number} at offset {tag_offset}")
            return fields, pos
        fields.append(RawField(number, wire_type, value, tag_offset, pos, data_offset))
    if group_number is not None:
        raise DecodeError(f"unterminated group {group_number}")
    return fields, pos


def parse_packed_varints(data: bytes) -> List[int]:
    values = []
    pos = 0
    while pos < len(data):
        v, pos = read_varint(data, pos)
        values.append(v)
    return values


def parse_packed_fixed(data: bytes, width: int) -> List[int]:
    if len(data) % width:
        raise DecodeError(f"length {len(data)} is not a multiple of {width}")
    return [
        int.from_bytes(data[i : i + width], "little") for i in range(0, len(data), width)
    ]
