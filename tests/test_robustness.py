import random

from google.protobuf import descriptor_pb2

from helpers import assert_roundtrip
from pbre import DecodeError, analyze, generate_proto, render_text, to_json


def test_random_bytes_never_crash():
    rng = random.Random(1234)
    for _ in range(3000):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
        try:
            decoded, schema = analyze(blob)
        except DecodeError:
            continue
        render_text(decoded[0], schema, verbose=True)
        to_json(decoded[0], schema)
        generate_proto(schema)


def test_real_world_descriptor_roundtrip():
    # descriptor.proto's own FileDescriptorProto is a large, deeply nested,
    # real-world message (strings, enums, nested repeated messages, options).
    fdp = descriptor_pb2.FileDescriptorProto()
    descriptor_pb2.DESCRIPTOR.CopyToProto(fdp)
    blob = fdp.SerializeToString()
    decoded, schema = analyze(blob, "FileDescriptorProto")
    assert schema.fields[1].type == "string"  # name
    assert schema.fields[4].type == "message" and schema.fields[4].repeated  # message_type
    assert schema.fields[4].message.fields[2].type == "message"  # field
    proto = generate_proto(schema)
    msg = assert_roundtrip(proto, "FileDescriptorProto", blob)
    assert msg.field_1 == "google/protobuf/descriptor.proto"
