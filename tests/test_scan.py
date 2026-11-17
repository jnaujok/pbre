"""Tests for finding a message inside a blob with extra bytes around it."""

import gzip
import json
import random

import pytest
from google.protobuf import descriptor_pb2

from pbre import decode
from pbre.cli import main
from pbre.scan import LOW_CONFIDENCE, find_start, scan
from pbre.wire import parse_message

SMALL = bytes.fromhex("089601120774657374696e671a03089601")


@pytest.fixture(scope="module")
def big():
    fdp = descriptor_pb2.FileDescriptorProto()
    descriptor_pb2.DESCRIPTOR.CopyToProto(fdp)
    return fdp.SerializeToString()


def test_clean_input_is_used_as_is(big):
    for msg in (SMALL, big):
        c = find_start(msg)
        assert (c.start, c.end, c.leading, c.trailing) == (0, len(msg), 0, 0)


def test_header_and_trailer(big):
    blob = b"PBRE\x01\x00\x10\x00\x00\x00\x37\x7f" + big + bytes.fromhex("deadbeef")
    c = find_start(blob)
    assert (c.start, c.end) == (12, 12 + len(big))
    assert c.confidence > 0.95


def test_fragment_starts_at_next_field_boundary(big):
    boundaries = [f.offset for f in parse_message(big)]
    frag = big[3:]  # cut into the first field
    c = find_start(frag)
    assert c.start == boundaries[1] - 3
    assert c.end == len(frag)


def test_every_candidate_span_decodes():
    rng = random.Random(5)
    for _ in range(2000):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 80)))
        for c in scan(blob).candidates:
            decode(blob[c.start : c.end])


def test_random_junk_prefixes(big):
    """Junk bytes can form valid fields, so the best guess can't always be
    right, but the true start should almost always be among the top 3."""
    rng = random.Random(11)
    exact = top3 = total = 0
    for msg in (SMALL, big):
        for length in range(1, 25):
            for _ in range(5):
                blob = bytes(rng.randrange(256) for _ in range(length)) + msg
                result = scan(blob, top=3)
                want = (length, len(blob))
                total += 1
                exact += (result.best.start, result.best.end) == want
                top3 += any((c.start, c.end) == want for c in result.candidates)
    assert exact / total > 0.7
    assert top3 / total > 0.95


def test_suffix_candidates_are_labelled():
    result = scan(SMALL)
    first, second = result.candidates[:2]
    assert (first.start, first.same_as) == (0, None)
    assert second.same_as == 1 and second.dropped_fields == 1


def test_random_data_has_low_confidence():
    rng = random.Random(3)
    blob = bytes(rng.randrange(256) for _ in range(20000))
    best = scan(blob).best
    assert best is not None and best.confidence < LOW_CONFIDENCE


def test_groups_are_followed():
    group = bytes([0x0B, 0x10, 0x05, 0x0C, 0x18, 0x01])  # 1: group {2: 5}, 3: 1
    c = find_start(b"\xff\xff" + group)
    assert (c.start, c.end) == (2, 8)


def test_empty_and_hopeless_input():
    assert scan(b"").candidates == []
    assert find_start(b"\xff") is None


@pytest.mark.parametrize(
    "blob, needle",
    [
        (gzip.compress(SMALL), "gzip"),
        (b"\x78\x9c" + b"\x00" * 8, "zlib"),
        (b"\x28\xb5\x2f\xfd" + b"\x00" * 8, "zstd"),
        (b"\x00" + len(SMALL).to_bytes(4, "big") + SMALL, "--grpc"),
        (bytes([len(SMALL)]) + SMALL, "--delimited"),
    ],
)
def test_hints(blob, needle):
    assert any(needle in h for h in scan(blob).hints)


# --- CLI ---------------------------------------------------------------


def run(capsys, *args):
    rc = main(list(args))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli_scan(tmp_path, capsys):
    p = tmp_path / "m.bin"
    p.write_bytes(b"HDR\x00\x07" + SMALL + b"\xff\xfe")
    rc, out, _ = run(capsys, "scan", str(p))
    assert rc == 0
    lines = out.splitlines()
    assert lines[2].split()[:6] == ["1", "5", "22", "5", "2", "3"]
    assert '2: "testing"' in lines[2]
    assert "pbre decode --find-start FILE" in out


def test_cli_scan_json_and_skip_offsets(tmp_path, capsys):
    p = tmp_path / "m.bin"
    p.write_bytes(b"\x00\x00\x00\x00" + b"\xff\xff" + SMALL)
    rc, out, _ = run(capsys, "scan", "--skip", "4", "--json", str(p))
    data = json.loads(out)
    best = data["candidates"][0]
    assert (best["start"], best["end"], best["leading"]) == (6, 6 + len(SMALL), 2)


def test_cli_decode_find_start(tmp_path, capsys):
    p = tmp_path / "m.bin"
    p.write_bytes(b"HDR\x00\x07" + SMALL + b"\xff\xfe")
    with pytest.raises(SystemExit) as e:
        main(["decode", str(p), "--no-prompt"])
    assert "--find-start" in str(e.value)
    rc, out, err = run(capsys, "decode", "--find-start", str(p), "--no-prompt")
    assert '2 field_2: string = "testing"' in out
    assert "using bytes 5..22 (skipped 5 leading, ignored 2 trailing)" in err


def test_cli_find_start_rank(tmp_path, capsys):
    p = tmp_path / "m.bin"
    p.write_bytes(SMALL)
    rc, out, err = run(capsys, "decode", "--candidate", "2", str(p), "--no-prompt")
    top_level = [l for l in out.splitlines() if l.startswith("  ") and not l.startswith("    ")]
    assert top_level[0] == '  2 field_2: string = "testing"'  # candidate 2 drops field 1
    assert "using bytes 3..17 (skipped 3 leading)" in err


def test_cli_proto_find_start(tmp_path, capsys):
    p = tmp_path / "m.bin"
    p.write_bytes(b"HDR\x00\x07" + SMALL)
    rc, out, _ = run(capsys, "proto", "--find-start", str(p))
    assert "string field_2 = 2;" in out


def test_cli_find_start_nothing_found(tmp_path, capsys):
    p = tmp_path / "m.bin"
    p.write_bytes(b"\xff")
    with pytest.raises(SystemExit) as e:
        main(["decode", "--find-start", str(p), "--no-prompt"])
    assert "found no plausible message" in str(e.value)


def test_cli_low_confidence_warning(tmp_path, capsys):
    rng = random.Random(3)
    p = tmp_path / "r.bin"
    p.write_bytes(bytes(rng.randrange(256) for _ in range(20000)))
    rc, out, _ = run(capsys, "scan", str(p))
    assert "low confidence" in out
