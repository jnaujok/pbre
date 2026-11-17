"""Heuristic decoding of raw wire records into an annotated tree.

Every length-delimited payload is ambiguous on the wire: it could be a string,
raw bytes, a nested message or a packed repeated scalar. This module records
*which* interpretations are possible for each payload so that schema inference
can pick one consistently across every occurrence of a field.
"""

from __future__ import annotations

import struct
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .wire import (
    MAX_FIELD_NUMBER,
    DecodeError,
    RawField,
    WireType,
    parse_message,
    parse_packed_fixed,
    parse_packed_varints,
)


@dataclass
class Node:
    number: int
    wire_type: WireType
    offset: int
    end: int
    # VARINT / I32 / I64: the unsigned integer value.  LEN: the payload bytes.
    value: object = None
    # LEN only ---------------------------------------------------------------
    text: Optional[str] = None  # set if payload is valid UTF-8
    printable: bool = False  # text consists of printable characters only
    children: Optional[List["Node"]] = None  # set if payload parses as a message
    packed_varints: Optional[List[int]] = None
    # SGROUP: children holds the group's fields

    @property
    def is_len(self) -> bool:
        return self.wire_type == WireType.LEN

    @property
    def is_group(self) -> bool:
        return self.wire_type == WireType.SGROUP


@dataclass
class DecodeOptions:
    max_field_number: int = MAX_FIELD_NUMBER
    max_depth: int = 64
    # Fraction of characters that must be printable for a payload to be
    # considered "text" rather than a candidate nested message.
    min_printable_ratio: float = 1.0


def is_printable_text(s: str) -> bool:
    for ch in s:
        if ch in "\t\n\r":
            continue
        cat = unicodedata.category(ch)
        if cat[0] == "C":  # control, format, unassigned, private use, surrogate
            return False
    return True


def decode(buf: bytes, options: DecodeOptions | None = None) -> List[Node]:
    """Decode a complete message. Raises DecodeError if ``buf`` is not one."""
    options = options or DecodeOptions()
    # The top level is known to be a message, so only the protocol's own limit
    # applies; options.max_field_number only guards speculative nested parses.
    raw = parse_message(buf)
    return _convert(raw, options, 0, 0)


def _convert(raw: List[RawField], options: DecodeOptions, depth: int, base: int) -> List[Node]:
    nodes = []
    for rf in raw:
        node = Node(rf.number, rf.wire_type, base + rf.offset, base + rf.end)
        if rf.wire_type == WireType.LEN:
            data: bytes = rf.value  # type: ignore[assignment]
            node.value = data
            _analyse_len(node, data, options, depth, base + rf.data_offset)
        elif rf.wire_type == WireType.SGROUP:
            node.children = _convert(rf.value, options, depth + 1, base)  # type: ignore[arg-type]
        else:
            node.value = rf.value
        nodes.append(node)
    return nodes


def _analyse_len(node: Node, data: bytes, options: DecodeOptions, depth: int, base: int) -> None:
    try:
        node.text = data.decode("utf-8")
        node.printable = is_printable_text(node.text)
    except UnicodeDecodeError:
        pass
    if data and depth < options.max_depth:
        try:
            raw = parse_message(data, max_field_number=options.max_field_number)
            node.children = _convert(raw, options, depth + 1, base)
        except DecodeError:
            node.children = None
    elif not data:
        node.children = []
    try:
        node.packed_varints = parse_packed_varints(data)
    except DecodeError:
        node.packed_varints = None


# --------------------------------------------------------------------------
# Scalar reinterpretation helpers (used by schema inference and rendering)
# --------------------------------------------------------------------------

def to_signed(value: int, bits: int) -> int:
    if value >= 1 << (bits - 1):
        return value - (1 << bits)
    return value


def zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def as_float(value: int) -> float:
    return struct.unpack("<f", value.to_bytes(4, "little"))[0]


def as_double(value: int) -> float:
    return struct.unpack("<d", value.to_bytes(8, "little"))[0]


def plausible_float(x: float) -> bool:
    """Is this a number a human would plausibly have stored as a float?"""
    if x != x or x in (float("inf"), float("-inf")):
        return False
    if x == 0.0:
        return True
    return 1e-7 <= abs(x) <= 1e12


def unpack_values(node: Node, packed_kind: str) -> List[int]:
    """Re-read a LEN payload as packed varints / fixed32 / fixed64."""
    data: bytes = node.value  # type: ignore[assignment]
    if packed_kind == "varint":
        return parse_packed_varints(data)
    if packed_kind == "fixed32":
        return parse_packed_fixed(data, 4)
    if packed_kind == "fixed64":
        return parse_packed_fixed(data, 8)
    raise ValueError(packed_kind)


def children_by_number(nodes: List[Node]) -> Dict[int, List[Node]]:
    out: Dict[int, List[Node]] = {}
    for n in nodes:
        out.setdefault(n.number, []).append(n)
    return out
