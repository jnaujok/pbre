"""Regression tests for issues found in code review."""

import pytest

from helpers import assert_roundtrip, message_class
from pbre import DecodeOptions, analyze, decode, generate_proto, parse_hint, render_text, to_json
from pbre.wire import encode_varint


@pytest.mark.parametrize("value", [-(1 << 64) - 1, -(1 << 63) - 1, 1 << 64])
def test_encode_varint_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        encode_varint(value)


def test_encode_varint_range_edges():
    assert len(encode_varint(-(1 << 63))) == 10
    assert len(encode_varint((1 << 64) - 1)) == 10


def test_max_field_number_only_limits_nested_guesses():
    # field 11 = 1, field 1 = b"\x58\x01" (which would be a message with field 11)
    blob = bytes([0x58, 0x01, 0x0A, 0x02, 0x58, 0x01])
    nodes = decode(blob, DecodeOptions(max_field_number=10))
    assert [n.number for n in nodes] == [11, 1]
    assert nodes[1].children is None  # nested interpretation was rejected
    assert decode(blob)[1].children is not None


def test_reserved_field_numbers_are_omitted():
    blob = b"\x08\x07" + encode_varint(19001 << 3) + b"\x05"
    decoded, schema = analyze(blob)
    proto = generate_proto(schema)
    assert "19001 =" not in proto
    assert "field 19001 omitted" in proto
    cls = message_class(proto, "Message")
    msg = cls()
    msg.ParseFromString(blob)  # compiles, and the reserved field is kept as unknown
    assert msg.field_1 == 7
    assert msg.SerializeToString() == blob


def test_enum_hint_rejected():
    with pytest.raises(ValueError):
        parse_hint("1=enum")


def test_mixed_packed_unpacked_is_flagged_and_not_byte_identical():
    blob = bytes.fromhex("08070a020102")
    _, schema = analyze(blob)
    f = schema.fields[1]
    assert any("packed and unpacked" in n for n in f.notes)
    cls = message_class(generate_proto(schema), "Message")
    msg = cls()
    msg.ParseFromString(blob)
    assert list(msg.field_1) == [7, 1, 2]
    assert msg.SerializeToString() != blob


@pytest.mark.parametrize(
    "hex_blob, hint, expected",
    [
        ("08ffffffff0f", "1=int32", "-1"),
        ("08ffffffffffffffffff01", "1=int32", "-1"),
        ("08ffffffff0f", "1=uint32", "4294967295"),
        ("08ffffffff0f", "1=sint32", "-2147483648"),
        ("08feffffffffffffffff01", "1=sint32", "2147483647"),
        ("08ffffffffffffffffff01", "1=int64", "-1"),
    ],
)
def test_32_bit_types_use_low_32_bits(hex_blob, hint, expected):
    blob = bytes.fromhex(hex_blob)
    decoded, schema = analyze(blob, hints=dict([parse_hint(hint)]))
    assert render_text(decoded[0], schema).endswith(f"= {expected}")
    assert str(to_json(decoded[0], schema)[0]["value"]) == expected
    # and the runtime agrees
    cls = message_class(generate_proto(schema), "Message")
    msg = cls()
    msg.ParseFromString(blob)
    assert str(msg.field_1) == expected


def test_reserved_field_omission_respects_no_comments():
    blob = b"\x08\x07" + encode_varint(19001 << 3) + b"\x05"
    _, schema = analyze(blob)
    proto = generate_proto(schema, comments=False)
    assert "//" not in proto
    assert "19001" not in proto
    assert "field_1 = 1;" in proto
