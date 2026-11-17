"""End-to-end tests: serialize real messages with the protobuf runtime, run
pbre over the bytes, and check both the inferred types and that the generated
.proto compiles and reproduces the original bytes exactly."""

import pytest

from helpers import assert_roundtrip, message_class
from pbre import analyze, generate_proto, parse_hint, render_text, to_json

ORIGINAL = """
syntax = "proto3";
message Person {
  string name = 1;
  int32 id = 2;
  string email = 3;
  enum PhoneType { MOBILE = 0; HOME = 1; WORK = 2; }
  message PhoneNumber {
    string number = 1;
    PhoneType type = 2;
  }
  repeated PhoneNumber phones = 4;
  double balance = 5;
  float ratio = 6;
  int64 big = 7;
  int32 negative = 8;
  repeated int32 scores = 9;
  bytes blob = 10;
  fixed32 crc = 11;
  sfixed64 offset = 12;
  Address address = 13;
}
message Address {
  string street = 1;
  string city = 2;
  repeated string tags = 3;
}
"""


@pytest.fixture(scope="module")
def Person():
    return message_class(ORIGINAL, "Person")


def make_person(Person, **overrides):
    p = Person(
        name="Ada Lovelace",
        id=1815,
        email="ada@example.com",
        balance=1234.5,
        ratio=0.25,
        big=2**40 + 7,
        negative=-42,
        scores=[10, 200, 3000],
        blob=bytes([0x00, 0xFF, 0x10, 0x80, 0x07]),
        crc=0xDEADBEEF,
        offset=-1000,
    )
    p.phones.add(number="555-0100", type=1)
    p.phones.add(number="555-0199", type=2)
    p.address.street = "12 St James's Square"
    p.address.city = "London"
    p.address.tags.extend(["home", "historic"])
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def test_types_are_inferred(Person):
    blob = make_person(Person).SerializeToString()
    _, schema = analyze(blob, "Person")
    f = schema.fields
    assert f[1].type == "string"
    assert f[2].type == "int32"
    assert f[3].type == "string"
    assert f[4].type == "message" and f[4].repeated
    assert f[4].message.fields[1].type == "string"
    assert f[4].message.fields[2].type == "int32"
    assert f[5].type == "double"
    assert f[6].type == "float"
    assert f[7].type == "int64"
    assert f[8].type == "int32"
    # Packed scalars are indistinguishable from opaque bytes without more
    # evidence; pbre keeps them as bytes but says so.
    assert f[9].type == "bytes" and "could be packed varints" in f[9].notes
    assert f[10].type == "bytes"
    assert f[11].type == "fixed32"
    assert f[12].type == "sfixed64"
    assert f[13].type == "message"
    assert f[13].message.fields[3].type == "string" and f[13].message.fields[3].repeated


def test_generated_proto_roundtrips(Person):
    blob = make_person(Person).SerializeToString()
    _, schema = analyze(blob, "Person", hints={(9,): parse_hint("9=int32")[1]})
    proto = generate_proto(schema, package="recovered")
    msg = assert_roundtrip(proto, "Person", blob)
    assert msg.field_1 == "Ada Lovelace"
    assert msg.field_4[1].field_1 == "555-0199"
    assert msg.field_5 == 1234.5
    assert msg.field_8 == -42
    assert list(msg.field_9) == [10, 200, 3000]


def test_multiple_samples_merge(Person):
    a = make_person(Person)
    a.ClearField("phones")
    a.phones.add(number="1")  # only one phone in this sample -> not repeated here
    b = Person(id=3)  # most fields absent
    c = make_person(Person)
    blobs = [x.SerializeToString() for x in (a, b, c)]
    decoded, schema = analyze(blobs, "Person")
    assert schema.fields[4].repeated  # repeated because sample c has two
    assert schema.fields[2].present_in == 3
    proto = generate_proto(schema, sample_count=3)
    for blob in blobs:
        assert_roundtrip(proto, "Person", blob)


def test_unpacked_repeated_proto2():
    src = """
syntax = "proto2";
message M {
  repeated int32 a = 1;
  repeated int32 b = 2 [packed = true];
  optional group G = 3 { optional int32 x = 4; optional string y = 5; }
}
"""
    M = message_class(src, "M")
    m = M(a=[1, 2, 3], b=[4, 5])
    m.g.x = 9
    m.g.y = "hi"
    blob = m.SerializeToString()
    hints = dict([parse_hint("2=int32")])
    _, schema = analyze(blob, "M", hints=hints)
    assert schema.fields[1].packed is False
    assert schema.fields[2].packed is True
    assert schema.fields[3].type == "group"
    proto = generate_proto(schema)
    assert 'syntax = "proto2"' in proto
    assert_roundtrip(proto, "M", blob)
    # proto3 output must keep unpacked encoding too
    proto3_schema = analyze(blob[: blob.index(bytes([0x1B]))], "M", hints=hints)[1]
    proto3 = generate_proto(proto3_schema)
    assert "[packed = false]" in proto3
    assert_roundtrip(proto3, "M", blob[: blob.index(bytes([0x1B]))])


def test_hints_override():
    src = 'syntax = "proto3"; message M { sint64 delta = 1; bytes raw = 2; repeated uint32 ids = 3; }'
    M = message_class(src, "M")
    blob = M(delta=-5, raw=b"\x08\x01", ids=[1, 2]).SerializeToString()
    _, schema = analyze(blob)
    assert schema.fields[1].type == "int32"  # zigzag cannot be detected
    assert schema.fields[2].type == "message"  # b"\x08\x01" looks like a message
    assert schema.fields[3].type == "bytes"  # packed looks like bytes without hints
    hints = dict(parse_hint(h) for h in ["1=sint64:delta", "2=bytes", "3=uint32:ids"])
    _, schema = analyze(blob, hints=hints)
    assert schema.fields[1].type == "sint64" and schema.fields[1].name == "delta"
    assert schema.fields[2].type == "bytes"
    assert schema.fields[3].type == "uint32" and schema.fields[3].packed
    proto = generate_proto(schema)
    msg = assert_roundtrip(proto, "Message", blob)
    assert msg.delta == -5 and list(msg.ids) == [1, 2]


def test_mixed_packed_and_unpacked_varints():
    # field 1 appears once unpacked (value 7) and once packed ([1, 2])
    blob = bytes([0x08, 0x07, 0x0A, 0x02, 0x01, 0x02])
    _, schema = analyze(blob)
    f = schema.fields[1]
    assert f.type == "int32" and f.repeated


def test_render_and_json(Person):
    blob = make_person(Person).SerializeToString()
    decoded, schema = analyze(blob, "Person")
    text = render_text(decoded[0], schema)
    assert '1 field_1: string = "Ada Lovelace"' in text
    assert "9 field_9: bytes = 0ac801b817  (or packed varints: [10, 200, 3000])" in text
    assert "8 field_8: int32 = -42" in text
    assert "11 field_11: fixed32 = 3735928559" in text
    verbose = render_text(decoded[0], schema, verbose=True)
    assert "[@0 len]" in verbose
    data = to_json(decoded[0], schema)
    assert data[0]["value"] == "Ada Lovelace"
    phones = [e for e in data if e["field"] == 4]
    assert phones[1]["fields"][0]["value"] == "555-0199"


def test_string_not_mistaken_for_message():
    # "hello" happens to parse as a message (0x68 = field 13 varint) but is text
    blob = bytes([0x0A, 0x05]) + b"hello"
    _, schema = analyze(blob)
    assert schema.fields[1].type == "string"


def test_conflicting_wire_types_are_reported():
    blob = bytes([0x08, 0x01, 0x0D, 0x00, 0x00, 0x80, 0x3F, 0x08, 0x02])
    decoded, schema = analyze(blob)
    assert schema.fields[1].type == "int32"
    assert any("conflicting" in n for n in schema.fields[1].notes)
    text = render_text(decoded[0], schema)
    assert "1: <i32>" in text
