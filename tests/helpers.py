"""Test helpers: build real protobuf messages from .proto text and compile
pbre-generated .proto files with grpc_tools so they can be checked against
the original bytes."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import uuid

import pytest

protoc = pytest.importorskip("grpc_tools.protoc")
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory  # noqa: E402


def compile_proto(source: str):
    """Compile .proto ``source`` and return the generated Python module."""
    tmp = tempfile.mkdtemp(prefix="pbre-test-")
    stem = "m_" + uuid.uuid4().hex[:12]
    proto_path = os.path.join(tmp, stem + ".proto")
    # Every compiled file goes into the process-wide default descriptor pool,
    # so give each one a unique package to avoid duplicate-symbol errors.
    lines = [l for l in source.splitlines() if not l.strip().startswith("package ")]
    for i, l in enumerate(lines):
        if l.strip().startswith("syntax"):
            lines.insert(i + 1, f"package {stem};")
            break
    with open(proto_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    include = os.path.join(os.path.dirname(protoc.__file__), "_proto")
    rc = protoc.main(["protoc", f"-I{tmp}", f"-I{include}", f"--python_out={tmp}", proto_path])
    if rc != 0:
        raise AssertionError(f"protoc failed on:\n{source}")
    mod_path = os.path.join(tmp, stem + "_pb2.py")
    spec = importlib.util.spec_from_file_location(stem + "_pb2", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def message_class(source: str, name: str):
    module = compile_proto(source)
    return getattr(module, name)


def assert_roundtrip(proto_source: str, message_name: str, blob: bytes):
    """The generated schema must parse ``blob`` and re-serialize it identically."""
    cls = message_class(proto_source, message_name)
    msg = cls()
    msg.ParseFromString(blob)
    assert msg.SerializeToString() == blob, proto_source
    return msg
