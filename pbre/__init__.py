"""pbre - protobuf reverse engineering.

Decode binary protobuf blobs without a schema, show their contents in a
readable form and infer a .proto definition that describes them.
"""

from .decoder import DecodeOptions, Node, decode
from .protogen import generate_proto
from .render import render_text, to_json
from .scan import Candidate, ScanResult, find_start, scan
from .schema import FieldSchema, Hint, MessageSchema, infer_schema, parse_hint
from .wire import DecodeError

__version__ = "0.2.0"

__all__ = [
    "DecodeError",
    "DecodeOptions",
    "FieldSchema",
    "Hint",
    "MessageSchema",
    "Node",
    "Candidate",
    "ScanResult",
    "analyze",
    "decode",
    "find_start",
    "generate_proto",
    "infer_schema",
    "parse_hint",
    "render_text",
    "scan",
    "to_json",
]


def analyze(blobs, root_name="Message", hints=None, options=None):
    """Decode every blob and infer one schema covering all of them.

    Returns ``(decoded, schema)`` where ``decoded`` is a list of node lists,
    one per blob.
    """
    if isinstance(blobs, (bytes, bytearray)):
        blobs = [bytes(blobs)]
    decoded = [decode(b, options) for b in blobs]
    return decoded, infer_schema(decoded, root_name, hints)
