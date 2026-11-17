"""Command line interface: ``pbre decode`` and ``pbre proto``."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List, Optional, Sequence

from . import __version__
from .decoder import DecodeOptions, decode
from .inputs import decode_text_encoding, split_frames
from .protogen import generate_proto
from .render import render_text, to_json
from .schema import infer_schema, parse_hint
from .wire import MAX_FIELD_NUMBER, DecodeError

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PACKAGE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "files",
        nargs="*",
        default=["-"],
        help="input files ('-' or nothing for stdin); several files are treated "
        "as samples of the same message type",
    )
    p.add_argument(
        "-f",
        "--format",
        choices=["auto", "raw", "hex", "base64"],
        default="auto",
        help="input encoding (default: auto = hex if the input looks like hex, else raw)",
    )
    framing = p.add_mutually_exclusive_group()
    framing.add_argument(
        "--delimited",
        action="store_true",
        help="input is a stream of varint-length-prefixed messages",
    )
    framing.add_argument(
        "--grpc", action="store_true", help="input is gRPC-framed (5-byte prefix per message)"
    )
    p.add_argument(
        "--skip", type=int, default=0, metavar="N", help="skip N leading bytes of each input"
    )
    p.add_argument(
        "-H",
        "--hint",
        action="append",
        default=[],
        metavar="PATH=TYPE[:NAME]",
        help="override inference for a field, e.g. 3=string, 2.1=sint64:delta, "
        "4=message:User, 5=int32 (on a length-delimited field: packed). Repeatable.",
    )
    p.add_argument(
        "-m",
        "--message-name",
        default="Message",
        help="name of the top-level message (default: Message)",
    )
    p.add_argument(
        "--max-field-number",
        type=int,
        default=MAX_FIELD_NUMBER,
        help="reject nested-message interpretations that use field numbers above this "
        "(lower it to reduce false positives on binary data)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbre",
        description="Protobuf reverse engineering: decode schemaless protobuf blobs "
        "and infer a .proto definition for them.",
    )
    parser.add_argument("--version", action="version", version=f"pbre {__version__}")
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("decode", help="show the fields in a blob (default command)")
    _add_common(d)
    d.add_argument("-v", "--verbose", action="store_true", help="show offsets, wire types and alternate interpretations")
    d.add_argument("--json", action="store_true", help="emit JSON instead of text")
    d.add_argument("--proto", metavar="PATH", help="also write the inferred .proto to PATH ('-' for stdout)")
    d.add_argument("--no-prompt", action="store_true", help="never offer to generate a .proto")
    _add_proto_opts(d)

    p = sub.add_parser("proto", help="infer and print a .proto definition")
    _add_common(p)
    p.add_argument("-o", "--output", metavar="PATH", default="-", help="write to PATH (default: stdout)")
    _add_proto_opts(p)
    return parser


def _add_proto_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--syntax", choices=["proto2", "proto3"], help="default: proto3 (proto2 if groups are present)")
    p.add_argument("--package", help="package name to put in the .proto")
    p.add_argument("--no-comments", action="store_true", help="omit sample values and notes from the .proto")


def _read_inputs(args) -> List[bytes]:
    blobs: List[bytes] = []
    for path in args.files:
        if path == "-":
            data = sys.stdin.buffer.read()
            name = "<stdin>"
        else:
            with open(path, "rb") as fh:
                data = fh.read()
            name = path
        try:
            data = decode_text_encoding(data, args.format)
        except ValueError as e:
            raise SystemExit(f"pbre: {name}: {e}")
        data = data[args.skip :]
        framing = "delimited" if args.delimited else "grpc" if args.grpc else "none"
        try:
            blobs.extend(split_frames(data, framing))
        except DecodeError as e:
            raise SystemExit(f"pbre: {name}: {e}")
    return blobs


def _analyze(args):
    if not _IDENT.match(args.message_name):
        raise SystemExit(f"pbre: invalid message name {args.message_name!r}")
    if getattr(args, "package", None) and not _PACKAGE.match(args.package):
        raise SystemExit(f"pbre: invalid package name {args.package!r}")
    hints = {}
    for spec in args.hint:
        try:
            path, hint = parse_hint(spec)
        except ValueError as e:
            raise SystemExit(f"pbre: {e}")
        if hint.name and not _IDENT.match(hint.name):
            raise SystemExit(f"pbre: invalid name in hint {spec!r}")
        hints[path] = hint
    blobs = _read_inputs(args)
    if not blobs:
        raise SystemExit("pbre: no input")
    options = DecodeOptions(max_field_number=args.max_field_number)
    decoded = []
    for i, blob in enumerate(blobs):
        try:
            decoded.append(decode(blob, options))
        except DecodeError as e:
            label = f"message {i + 1}" if len(blobs) > 1 else "input"
            raise SystemExit(
                f"pbre: {label} is not a valid protobuf message: {e}\n"
                "hint: try --format hex/base64, --skip N, --delimited or --grpc"
            )
    schema = infer_schema(decoded, args.message_name, hints)
    return decoded, schema


def _proto_text(args, schema, n_samples) -> str:
    try:
        return generate_proto(
            schema,
            syntax=args.syntax,
            package=args.package,
            sample_count=n_samples,
            comments=not args.no_comments,
        )
    except ValueError as e:
        raise SystemExit(f"pbre: {e}")


def _write(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {path}", file=sys.stderr)


def cmd_decode(args) -> int:
    decoded, schema = _analyze(args)
    if args.json:
        payload = [to_json(nodes, schema) for nodes in decoded]
        json.dump(payload[0] if len(payload) == 1 else payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        for i, nodes in enumerate(decoded):
            if len(decoded) > 1:
                print(f"--- message {i + 1} ---")
            print(f"{schema.name} {{")
            print(render_text(nodes, schema, verbose=args.verbose, indent=1))
            print("}")
    if args.proto:
        _write(args.proto, _proto_text(args, schema, len(decoded)))
    elif not args.no_prompt and not args.json and _interactive():
        _offer_proto(args, schema, len(decoded))
    return 0


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _offer_proto(args, schema, n_samples) -> None:
    try:
        answer = input("\nGenerate a .proto file for this data? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return
        default = f"{_snake(schema.name)}.proto"
        path = input(f"Output path [{default}] ('-' to print): ").strip() or default
    except EOFError:
        return
    text = _proto_text(args, schema, n_samples)
    if path == "-":
        print()
    _write(path, text)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def cmd_proto(args) -> int:
    decoded, schema = _analyze(args)
    _write(args.output, _proto_text(args, schema, len(decoded)))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `pbre FILE` is shorthand for `pbre decode FILE`.
    if not argv or argv[0] not in ("decode", "proto", "-h", "--help", "--version"):
        argv.insert(0, "decode")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "proto":
            return cmd_proto(args)
        return cmd_decode(args)
    except BrokenPipeError:
        return 0
    except FileNotFoundError as e:
        print(f"pbre: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
