import pytest

from pbre.wire import (
    DecodeError,
    WireType,
    encode_varint,
    parse_message,
    parse_packed_fixed,
    parse_packed_varints,
    read_varint,
)


def test_varint_roundtrip():
    for v in [0, 1, 127, 128, 300, 2**32, 2**63, 2**64 - 1]:
        enc = encode_varint(v)
        assert read_varint(enc, 0) == (v, len(enc))


def test_negative_varint_is_ten_bytes():
    assert len(encode_varint(-1)) == 10
    assert read_varint(encode_varint(-1), 0)[0] == 2**64 - 1


def test_truncated_varint():
    with pytest.raises(DecodeError):
        read_varint(b"\x80\x80", 0)


def test_parse_basic():
    fields = parse_message(bytes.fromhex("089601120774657374696e67"))
    assert [(f.number, f.wire_type) for f in fields] == [(1, WireType.VARINT), (2, WireType.LEN)]
    assert fields[0].value == 150
    assert fields[1].value == b"testing"


def test_parse_fixed():
    fields = parse_message(bytes.fromhex("0d01000000110200000000000000"))
    assert fields[0].wire_type == WireType.I32 and fields[0].value == 1
    assert fields[1].wire_type == WireType.I64 and fields[1].value == 2


def test_groups():
    # field 1 start group, inner field 2 varint 5, end group 1
    fields = parse_message(bytes([0x0B, 0x10, 0x05, 0x0C]))
    assert fields[0].wire_type == WireType.SGROUP
    assert fields[0].value[0].number == 2


@pytest.mark.parametrize(
    "blob",
    [
        b"\x00",  # field 0
        b"\x0f",  # wire type 7
        b"\x0a\x05abc",  # truncated length
        b"\x0b\x10\x05",  # unterminated group
        b"\x0c",  # stray end group
        b"\x0d\x00\x00",  # truncated fixed32
    ],
)
def test_invalid(blob):
    with pytest.raises(DecodeError):
        parse_message(blob)


def test_packed_helpers():
    assert parse_packed_varints(bytes.fromhex("03 8e02 9ea705".replace(" ", ""))) == [3, 270, 86942]
    assert parse_packed_fixed(b"\x01\x00\x00\x00\x02\x00\x00\x00", 4) == [1, 2]
    with pytest.raises(DecodeError):
        parse_packed_fixed(b"\x01\x02\x03", 4)
