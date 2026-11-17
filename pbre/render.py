"""Human-readable and JSON renderings of a decoded message."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .decoder import Node, as_double, as_float, to_signed, unpack_values, zigzag_decode
from .schema import (
    FieldSchema,
    MessageSchema,
    format_len,
    format_scalar,
    wire_class,
)
from .wire import WireType

_WIRE_NAMES = {
    WireType.VARINT: "varint",
    WireType.I64: "i64",
    WireType.LEN: "len",
    WireType.SGROUP: "group",
    WireType.I32: "i32",
}


def _matches(node: Node, f: Optional[FieldSchema]) -> bool:
    if f is None:
        return False
    cls = wire_class(f.type)
    if node.is_len:
        return cls == "len" or (f.is_scalar and _unpacks(node, cls))
    if node.is_group:
        return cls == "group"
    return cls == {WireType.VARINT: "varint", WireType.I32: "fixed32", WireType.I64: "fixed64"}[
        node.wire_type
    ]


def _unpacks(node: Node, cls: str) -> bool:
    try:
        unpack_values(node, cls)
        return True
    except Exception:
        return False


def _alternates(node: Node) -> str:
    v = node.value
    if node.wire_type == WireType.VARINT:
        parts = [f"uint={v}", f"sint={zigzag_decode(v)}", f"hex=0x{v:x}"]
        if v >= 1 << 63:
            parts.insert(1, f"int={to_signed(v, 64)}")
        return ", ".join(parts)
    if node.wire_type == WireType.I32:
        return f"fixed32={v}, sfixed32={to_signed(v, 32)}, float={as_float(v):.7g}"
    if node.wire_type == WireType.I64:
        return f"fixed64={v}, sfixed64={to_signed(v, 64)}, double={as_double(v)!r}"
    return ""


def render_text(
    nodes: List[Node],
    schema: MessageSchema,
    verbose: bool = False,
    indent: int = 0,
    bytes_limit: int = 48,
    lost: Optional[Set[int]] = None,
) -> str:
    """Render the tree. ``lost`` is a set of byte offsets whose values are
    unknown (from ``--repair``); fields containing any are flagged."""
    lines: List[str] = []
    _render(nodes, schema, verbose, indent, lines, bytes_limit, lost or set())
    return "\n".join(lines)


def _own_ranges(n: Node, as_message: bool):
    """Byte ranges that belong to this node itself, not to its children."""
    if not as_message or not n.children:
        return [(n.offset, n.end)]
    ranges = [(n.offset, n.children[0].offset)]
    for a, b in zip(n.children, n.children[1:]):
        if a.end < b.offset:
            ranges.append((a.end, b.offset))
    ranges.append((n.children[-1].end, n.end))
    return ranges


def lost_in(n: Node, lost: Set[int], as_message: bool) -> int:
    if not lost:
        return 0
    return sum(1 for a, b in _own_ranges(n, as_message) for i in range(a, b) if i in lost)


def _lost_note(count: int) -> str:
    return f"  [{count} lost byte{'s' if count != 1 else ''}: value uncertain]" if count else ""


def _render(nodes, schema, verbose, depth, lines, limit, lost):
    ind = "  " * depth
    for n in nodes:
        first = len(lines)
        _render_node(n, schema, verbose, depth, lines, limit, lost)
        f = schema.fields.get(n.number) if schema else None
        as_message = bool(n.children is not None and (n.is_group or (f is not None and f.type in ("message", "group"))))
        count = lost_in(n, lost, as_message)
        if count and first < len(lines):
            lines[first] += _lost_note(count)


def _render_node(n, schema, verbose, depth, lines, limit, lost):
    ind = "  " * depth
    f = schema.fields.get(n.number) if schema else None
    prefix = f"{ind}"
    if verbose:
        prefix += f"[@{n.offset} {_WIRE_NAMES[n.wire_type]}] "
    if not _matches(n, f):
        lines.append(prefix + _raw_line(n, verbose, limit))
        if n.is_group and n.children is not None:
            _render(n.children, None, verbose, depth + 1, lines, limit, lost)
            lines.append(f"{ind}}}")
        return
    assert f is not None
    label = f"{n.number} {f.name}"
    if f.type in ("message", "group"):
        children = n.children or []
        lines.append(f"{prefix}{label}: {f.message.name} {{")
        _render(children, f.message, verbose, depth + 1, lines, limit, lost)
        lines.append(f"{ind}}}")
    elif f.is_scalar:
        cls = wire_class(f.type)
        if n.is_len:
            vals = [format_scalar(f.type, v) for v in unpack_values(n, cls)]
            lines.append(f"{prefix}{label}: {f.type} = [{', '.join(vals)}]  (packed)")
        else:
            line = f"{prefix}{label}: {f.type} = {format_scalar(f.type, n.value)}"
            if verbose:
                line += f"  ({_alternates(n)})"
            lines.append(line)
    else:
        data: bytes = n.value
        line = f"{prefix}{label}: {f.type} = {format_len(f.type, data, 0 if verbose else limit)}"
        if verbose:
            line += f"  ({len(data)} bytes)"
        if f.type == "bytes" and data and n.packed_varints is not None and n.children is None:
            vals = n.packed_varints
            shown = ", ".join(str(v) for v in vals[:8]) + (", ..." if len(vals) > 8 else "")
            line += f"  (or packed varints: [{shown}])"
        lines.append(line)


def _raw_line(n: Node, verbose: bool, limit: int) -> str:
    """A node the schema does not describe (conflicting wire type)."""
    if n.is_group:
        return f"{n.number}: group {{"
    if n.is_len:
        return f"{n.number}: <len> = {format_len('bytes', n.value, 0 if verbose else limit)}"
    return f"{n.number}: <{_WIRE_NAMES[n.wire_type]}> = {n.value}  ({_alternates(n)})"


def to_json(
    nodes: List[Node], schema: Optional[MessageSchema], lost: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    lost = lost or set()
    out = []
    for n in nodes:
        f = schema.fields.get(n.number) if schema else None
        entry: Dict[str, Any] = {
            "field": n.number,
            "wire_type": _WIRE_NAMES[n.wire_type],
            "offset": n.offset,
            "length": n.end - n.offset,
        }
        if not _matches(n, f):
            entry["type"] = None
            if n.is_group:
                entry["fields"] = to_json(n.children or [], None, lost)
            elif n.is_len:
                entry["value"] = n.value.hex()
            else:
                entry["value"] = n.value
            _json_lost(entry, n, lost, n.is_group)
            out.append(entry)
            continue
        assert f is not None
        entry["name"] = f.name
        entry["type"] = f.message.name if f.type in ("message", "group") else f.type
        if f.type in ("message", "group"):
            entry["fields"] = to_json(n.children or [], f.message, lost)
        elif f.is_scalar:
            cls = wire_class(f.type)
            if n.is_len:
                entry["packed"] = True
                entry["value"] = [_json_scalar(f.type, v) for v in unpack_values(n, cls)]
            else:
                entry["value"] = _json_scalar(f.type, n.value)
        elif f.type == "string":
            entry["value"] = n.value.decode("utf-8", errors="replace")
        else:
            entry["value"] = n.value.hex()
        _json_lost(entry, n, lost, f.type in ("message", "group"))
        out.append(entry)
    return out


def _json_lost(entry: Dict[str, Any], n: Node, lost: Set[int], as_message: bool) -> None:
    count = lost_in(n, lost, as_message and n.children is not None)
    if count:
        entry["lost_bytes"] = count


def _json_scalar(type_name: str, v: int):
    s = format_scalar(type_name, v)
    if type_name == "bool":
        return bool(v)
    if type_name in ("float", "double"):
        d = as_float(v) if type_name == "float" else as_double(v)
        return d if d == d and d not in (float("inf"), float("-inf")) else s
    return int(s)


def preview(nodes: List[Node], schema: MessageSchema, max_fields: int = 4, width: int = 70) -> str:
    """A one-line summary of the first few top-level fields."""
    parts = []
    for n in nodes[:max_fields]:
        f = schema.fields.get(n.number)
        if not _matches(n, f):
            parts.append(f"{n.number}: <{_WIRE_NAMES[n.wire_type]}>")
            continue
        assert f is not None
        if f.type in ("message", "group"):
            k = len(n.children or [])
            parts.append(f"{n.number}: {{{k} field{'s' if k != 1 else ''}}}")
        elif f.is_scalar:
            if n.is_len:
                parts.append(f"{n.number}: [packed {f.type}]")
            else:
                parts.append(f"{n.number}: {format_scalar(f.type, n.value)}")
        else:
            parts.append(f"{n.number}: {format_len(f.type, n.value, 16)}")
    if len(nodes) > max_fields:
        parts.append("...")
    text = "  ".join(parts)
    if len(text) > width:
        text = text[: width - 3] + "..."
    return text
