"""Infer a message schema from one or more decoded samples.

Inference is done per field, looking at *every* occurrence of that field in
*every* instance of the enclosing message, so that a field is only treated as a
nested message if all of its payloads parse as one, only as a string if all of
them are printable text, and so on.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .decoder import (
    Node,
    as_double,
    as_float,
    plausible_float,
    to_signed,
    unpack_values,
)
from .wire import DecodeError, WireType

VARINT_TYPES = {"int32", "int64", "uint32", "uint64", "sint32", "sint64", "bool", "enum"}
FIXED32_TYPES = {"fixed32", "sfixed32", "float"}
FIXED64_TYPES = {"fixed64", "sfixed64", "double"}
SCALAR_TYPES = VARINT_TYPES | FIXED32_TYPES | FIXED64_TYPES
LEN_TYPES = {"string", "bytes", "message"}
ALL_TYPES = SCALAR_TYPES | LEN_TYPES | {"group"}

Path = Tuple[int, ...]


def wire_class(type_name: str) -> str:
    if type_name in VARINT_TYPES:
        return "varint"
    if type_name in FIXED32_TYPES:
        return "fixed32"
    if type_name in FIXED64_TYPES:
        return "fixed64"
    if type_name == "group":
        return "group"
    return "len"


_WIRE_TO_CLASS = {
    WireType.VARINT: "varint",
    WireType.I32: "fixed32",
    WireType.I64: "fixed64",
    WireType.LEN: "len",
    WireType.SGROUP: "group",
}


@dataclass
class FieldSchema:
    number: int
    name: str
    type: str  # one of ALL_TYPES
    repeated: bool = False
    packed: Optional[bool] = None  # for repeated scalars: observed encoding
    message: Optional["MessageSchema"] = None  # for message / group
    notes: List[str] = field(default_factory=list)
    samples: List[str] = field(default_factory=list)
    occurrences: int = 0
    # instances of the parent that contained this field / total instances
    present_in: int = 0
    # a zero / empty value was explicitly present on the wire, so the field
    # needs explicit presence (proto3 "optional") to round-trip exactly
    default_seen: bool = False

    @property
    def is_scalar(self) -> bool:
        return self.type in SCALAR_TYPES


@dataclass
class MessageSchema:
    name: str
    path: Path
    fields: Dict[int, FieldSchema] = field(default_factory=dict)
    instances: int = 0
    is_group: bool = False

    def sorted_fields(self) -> List[FieldSchema]:
        return [self.fields[k] for k in sorted(self.fields)]

    def uses_groups(self) -> bool:
        for f in self.fields.values():
            if f.type == "group":
                return True
            if f.message is not None and f.message.uses_groups():
                return True
        return False


@dataclass
class Hint:
    type: str
    name: Optional[str] = None


def parse_hint(spec: str) -> Tuple[Path, Hint]:
    """Parse ``"3.1=string"`` / ``"2=message:User"`` / ``"4=int32:count"``."""
    if "=" not in spec:
        raise ValueError(f"hint {spec!r} must look like PATH=TYPE[:NAME]")
    path_s, type_s = spec.split("=", 1)
    try:
        path = tuple(int(p) for p in path_s.strip().split("."))
    except ValueError:
        raise ValueError(f"bad field path in hint {spec!r}") from None
    name = None
    if ":" in type_s:
        type_s, name = type_s.split(":", 1)
    type_s = type_s.strip()
    if type_s not in ALL_TYPES:
        raise ValueError(
            f"unknown type {type_s!r} in hint {spec!r}; expected one of {sorted(ALL_TYPES)}"
        )
    return path, Hint(type_s, name.strip() if name else None)


def infer_schema(
    samples: Sequence[List[Node]],
    root_name: str = "Message",
    hints: Optional[Dict[Path, Hint]] = None,
) -> MessageSchema:
    return _infer_message(list(samples), root_name, (), hints or {}, False)


def _infer_message(instances, name, path, hints, is_group) -> MessageSchema:
    msg = MessageSchema(name=name, path=path, instances=len(instances), is_group=is_group)
    # number -> list of (instance index, node)
    by_number: Dict[int, List[Tuple[int, Node]]] = {}
    for idx, nodes in enumerate(instances):
        for n in nodes:
            by_number.setdefault(n.number, []).append((idx, n))
    for number in sorted(by_number):
        occ = by_number[number]
        msg.fields[number] = _infer_field(number, occ, path + (number,), hints)
    return msg


def _infer_field(number, occ, path, hints) -> FieldSchema:
    per_instance = Counter(idx for idx, _ in occ)
    nodes = [n for _, n in occ]
    hint = hints.get(path)
    fs = FieldSchema(
        number=number,
        name=(hint.name if hint and hint.name else f"field_{number}"),
        type="bytes",
        repeated=any(c > 1 for c in per_instance.values()),
        occurrences=len(nodes),
        present_in=len(per_instance),
    )

    classes = Counter(_WIRE_TO_CLASS[n.wire_type] for n in nodes)
    scalar_classes = [c for c in classes if c in ("varint", "fixed32", "fixed64")]

    # A scalar field can legitimately appear both unpacked and packed (LEN).
    if (
        len(classes) == 2
        and "len" in classes
        and len(scalar_classes) == 1
        and (hint is None or wire_class(hint.type) == scalar_classes[0])
        and _all_unpack(nodes, scalar_classes[0])
    ):
        fs.repeated = True
        _infer_scalar(fs, nodes, scalar_classes[0], hint)
        fs.packed = classes["len"] >= classes[scalar_classes[0]]
        return fs

    if len(classes) > 1:
        chosen, _ = classes.most_common(1)[0]
        if hint is not None and wire_class(hint.type) in classes:
            chosen = wire_class(hint.type)
        if hint is not None and hint.type in SCALAR_TYPES and "len" in classes:
            chosen = "len" if _all_unpack([n for n in nodes if n.is_len], wire_class(hint.type)) else chosen
        fs.notes.append(
            "conflicting wire types seen ("
            + ", ".join(f"{k} x{v}" for k, v in sorted(classes.items()))
            + f"); using {chosen}"
        )
        nodes = [n for n in nodes if _WIRE_TO_CLASS[n.wire_type] == chosen]
        cls = chosen
    else:
        cls = next(iter(classes))

    if cls in ("varint", "fixed32", "fixed64"):
        _infer_scalar(fs, nodes, cls, hint)
        if fs.repeated:
            fs.packed = False
    elif cls == "group":
        fs.type = "group"
        group_name = hint.name if hint and hint.name else f"Group{number}"
        group_name = group_name[:1].upper() + group_name[1:]
        fs.name = group_name.lower()
        fs.message = _infer_message([n.children for n in nodes], group_name, path, hints, True)
    else:
        _infer_len(fs, nodes, path, hints, hint)
    return fs


def _all_unpack(nodes, cls) -> bool:
    for n in nodes:
        if n.is_len:
            try:
                unpack_values(n, cls)
            except DecodeError:
                return False
    return True


def _scalar_values(nodes, cls) -> List[int]:
    values: List[int] = []
    for n in nodes:
        if n.is_len:
            values.extend(unpack_values(n, cls))
        else:
            values.append(n.value)  # type: ignore[arg-type]
    return values


def _infer_scalar(fs: FieldSchema, nodes, cls, hint) -> None:
    values = _scalar_values(nodes, cls)
    fs.default_seen = any(not n.is_len and n.value == 0 for n in nodes)
    if hint is not None and wire_class(hint.type) == cls:
        fs.type = hint.type
    elif cls == "varint":
        fs.type = _guess_varint(values, fs)
    elif cls == "fixed32":
        fs.type = _guess_fixed(values, 32)
    else:
        fs.type = _guess_fixed(values, 64)
    seen = []
    for v in values:
        s = format_scalar(fs.type, v)
        if s not in seen:
            seen.append(s)
        if len(seen) >= 3:
            break
    fs.samples = seen


def _guess_varint(values: List[int], fs: FieldSchema) -> str:
    if not values:
        return "int32"
    if set(values) <= {0, 1} and len(values) > 1 and 0 in values:
        fs.notes.append("only 0/1 seen: may be a bool")
    elif set(values) <= {0, 1}:
        fs.notes.append("only 1 seen: may be a bool or enum")
    if any(v >= 1 << 63 for v in values):
        signed = [to_signed(v, 64) for v in values]
        if all(-(1 << 31) <= s < (1 << 31) for s in signed):
            return "int32"
        return "int64"
    if max(values) >= 1 << 31:
        return "int64"
    return "int32"


def _guess_fixed(values: List[int], bits: int) -> str:
    conv = as_float if bits == 32 else as_double
    if values and all(plausible_float(conv(v)) for v in values) and any(v for v in values):
        return "float" if bits == 32 else "double"
    if any(v >= 1 << (bits - 1) for v in values):
        signed = [to_signed(v, bits) for v in values]
        if all(abs(s) < 1 << (bits // 2) for s in signed):
            return f"sfixed{bits}"
    return f"fixed{bits}"


def _infer_len(fs: FieldSchema, nodes: List[Node], path, hints, hint) -> None:
    nonempty = [n for n in nodes if n.value]
    fs.default_seen = len(nonempty) < len(nodes)
    forced = hint.type if hint else None

    if forced in SCALAR_TYPES:
        cls = wire_class(forced)
        if _all_unpack(nodes, cls):
            fs.repeated = True
            fs.packed = True
            fs.type = forced
            _infer_scalar(fs, nodes, cls, hint)
            return
        fs.notes.append(f"hint {forced} ignored: payload is not packed {cls}")
        forced = None
    if forced == "message" and not all(n.children is not None for n in nodes):
        fs.notes.append("hint message ignored: not every payload parses as a message")
        forced = None
    if forced == "string" and not all(n.text is not None for n in nodes):
        fs.notes.append("hint string ignored: not every payload is valid UTF-8")
        forced = None

    if forced:
        kind = forced
    elif not nonempty:
        kind = "string"
        fs.notes.append("always empty: could equally be bytes or a message")
    elif all(n.printable for n in nonempty):
        kind = "string"
    elif all(n.children is not None for n in nonempty):
        kind = "message"
    else:
        kind = "bytes"
        if all(n.text is not None for n in nonempty):
            fs.notes.append("valid UTF-8 but contains control characters")
        elif all(n.packed_varints is not None for n in nonempty):
            fs.notes.append("could be packed varints")

    if kind == "group":  # a group hint cannot apply to a LEN field
        fs.notes.append("hint group ignored: field is length-delimited")
        kind = "bytes"
    fs.type = kind
    if kind == "message":
        type_name = hint.name if hint and hint.name else f"Field{fs.number}"
        if hint and hint.name:
            fs.name = _snake(hint.name)
        type_name = type_name[:1].upper() + type_name[1:]
        fs.message = _infer_message(
            [n.children or [] for n in nodes], type_name, path, hints, False
        )
    else:
        seen = []
        for n in nodes:
            s = format_len(kind, n.value)  # type: ignore[arg-type]
            if s not in seen:
                seen.append(s)
            if len(seen) >= 3:
                break
        fs.samples = seen


def _snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


# --------------------------------------------------------------------------
# Value formatting
# --------------------------------------------------------------------------

def format_scalar(type_name: str, v: int) -> str:
    if type_name in ("int32", "int64", "enum"):
        return str(to_signed(v, 64))
    if type_name == "uint32":
        return str(v & 0xFFFFFFFF)
    if type_name == "uint64" or type_name in ("fixed32", "fixed64"):
        return str(v)
    if type_name in ("sint32", "sint64"):
        return str((v >> 1) ^ -(v & 1))
    if type_name == "bool":
        return "true" if v else "false"
    if type_name == "sfixed32":
        return str(to_signed(v, 32))
    if type_name == "sfixed64":
        return str(to_signed(v, 64))
    if type_name == "float":
        return f"{as_float(v):.7g}"
    if type_name == "double":
        return repr(as_double(v))
    raise ValueError(type_name)


def quote_string(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def format_len(kind: str, data: bytes, limit: int = 48) -> str:
    if kind == "string":
        text = data.decode("utf-8", errors="replace")
        if limit and len(text) > limit:
            return quote_string(text[:limit]) + f"... ({len(text)} chars)"
        return quote_string(text)
    if limit and len(data) > limit:
        return f"{data[:limit].hex()}... ({len(data)} bytes)"
    return data.hex() if data else '""'
