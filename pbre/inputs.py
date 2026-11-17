"""Reading blobs from files in various encodings and framings."""

from __future__ import annotations

import base64
import binascii
import re
from typing import List

from .wire import DecodeError, read_varint

_HEX_RE = re.compile(rb"^(?:0x)?[0-9a-fA-F\s:,]+$")
_HEX_STRIP = re.compile(rb"0x|[\s:,]")


def decode_text_encoding(data: bytes, fmt: str) -> bytes:
    """Convert ``data`` from ``fmt`` (raw, hex, base64, auto) to raw bytes."""
    if fmt == "raw":
        return data
    if fmt == "hex":
        return _from_hex(data)
    if fmt == "base64":
        return _from_base64(data)
    if fmt == "auto":
        stripped = data.strip()
        if stripped and _HEX_RE.match(stripped):
            try:
                return _from_hex(stripped)
            except ValueError:
                pass
        return data
    raise ValueError(f"unknown input format {fmt!r}")


def _from_hex(data: bytes) -> bytes:
    cleaned = _HEX_STRIP.sub(b"", data)
    try:
        return binascii.unhexlify(cleaned)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid hex input: {e}") from None


def _from_base64(data: bytes) -> bytes:
    cleaned = re.sub(rb"\s", b"", data)
    cleaned += b"=" * (-len(cleaned) % 4)
    try:
        if b"-" in cleaned or b"_" in cleaned:
            return base64.urlsafe_b64decode(cleaned)
        return base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid base64 input: {e}") from None


def split_frames(data: bytes, framing: str) -> List[bytes]:
    """Split a stream into individual messages.

    * ``none``      - the whole buffer is one message
    * ``delimited`` - each message is prefixed with its varint length
      (``writeDelimitedTo`` / ``parseDelimitedFrom``)
    * ``grpc``      - each message has the 5-byte gRPC prefix
      (1 byte compressed flag, 4 byte big-endian length)
    """
    if framing == "none":
        return [data]
    out = []
    pos = 0
    if framing == "delimited":
        while pos < len(data):
            length, pos = read_varint(data, pos)
            if pos + length > len(data):
                raise DecodeError(f"delimited frame at offset {pos} runs past end of input")
            out.append(data[pos : pos + length])
            pos += length
        return out
    if framing == "grpc":
        while pos < len(data):
            if pos + 5 > len(data):
                raise DecodeError(f"truncated gRPC frame header at offset {pos}")
            flag = data[pos]
            length = int.from_bytes(data[pos + 1 : pos + 5], "big")
            pos += 5
            if flag:
                raise DecodeError("compressed gRPC frames are not supported; decompress first")
            if pos + length > len(data):
                raise DecodeError(f"gRPC frame at offset {pos} runs past end of input")
            out.append(data[pos : pos + length])
            pos += length
        return out
    raise ValueError(f"unknown framing {framing!r}")
