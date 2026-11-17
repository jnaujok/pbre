"""Tests for --repair: data mangled by a UTF-8 decode/encode round trip."""

import random

import pytest

from helpers import message_class
from pbre import decode
from pbre.cli import main
from pbre.repair import REPLACEMENT, count_replacements, repair
from pbre.scan import scan
from pbre.wire import parse_message

SRC = """
syntax = "proto3";
message Status { string code = 1; int32 level = 2; fixed32 crc = 3; }
message Reading { string sensor = 1; double value = 2; int64 ts = 3; sint32 delta = 4; }
message Body {
  Status status = 1;
  repeated Reading readings = 2;
  string host = 3;
  uint32 seq = 4;
  bytes token = 5;
}
message Envelope { Body body = 8; }
"""


@pytest.fixture(scope="module")
def Envelope():
    return message_class(SRC, "Envelope")


def mangle(data: bytes) -> bytes:
    return data.decode("utf-8", errors="replace").encode("utf-8")


def shape(buf, base=0):
    """Field layout (number, wire type, offset, end), recursing into payloads
    that parse as messages."""
    out = []
    for f in parse_message(buf):
        out.append((f.number, int(f.wire_type), base + f.offset, base + f.end))
    return out


def make(Envelope, readings=5):
    e = Envelope()
    b = e.body
    b.status.code = "U"
    b.status.level = 3
    b.status.crc = 0x1234
    for i in range(readings):
        b.readings.add(sensor=f"temp-{i}", value=20.5 + i * 0.25, ts=1763412345000 + i * 1000, delta=-i * 70)
    b.host = "gateway-01.local"
    b.seq = 40000
    b.token = bytes(range(0x90, 0x9C))
    return e.SerializeToString()


def test_only_the_length_byte_lost_is_recovered_exactly(Envelope):
    """Like the reported case: everything but the first length byte is ASCII
    or small, so the original comes back byte for byte."""
    e = Envelope()
    e.body.status.code = "U"
    e.body.host = "h" * 100
    for i in range(4):  # short fields whose lengths stay below 128 ...
        e.body.readings.add(sensor=f"sensor-{i}-" + "x" * 30)
    orig = e.SerializeToString()
    assert 256 <= len(orig) - 3 < 384  # ... while field 8's needs two bytes
    bad = mangle(orig)
    assert bad[:4] == b"\x42" + REPLACEMENT and count_replacements(bad) == 1
    r = repair(bad)
    assert r.data == orig
    (lost,) = r.lost
    assert lost.offset == 1 and lost.value == orig[1]
    assert lost.what.startswith(f"length of field 8 = {len(orig) - 3} bytes")


def test_structure_recovered_and_unknowns_flagged(Envelope):
    orig = make(Envelope)
    bad = mangle(orig)
    assert count_replacements(bad) > 10
    r = repair(bad)
    assert len(r.data) == len(orig)
    assert shape(r.data) == shape(orig)
    assert shape(r.data[3:]) == shape(orig[3:])  # the body too
    # every byte that differs from the original is reported as uncertain
    diff = {i for i in range(len(orig)) if orig[i] != r.data[i]}
    assert diff <= r.uncertain
    # the lost length of field 8 is recovered, as a tie-broken best guess
    assert r.data[1] == orig[1]
    assert r.lost[0].status == "guess" and "other lengths also parse" in r.lost[0].what


def test_strings_survive_intact(Envelope):
    orig = make(Envelope)
    r = repair(mangle(orig))
    for s in (b"gateway-01.local", b"temp-3", b"U"):
        assert s in r.data


def test_repaired_bytes_always_decode(Envelope):
    rng = random.Random(4)
    for n in range(1, 8):
        r = repair(mangle(make(Envelope, readings=n)))
        decode(r.data)
    # arbitrary mangled junk: repair either finds a structure that decodes or gives up
    for _ in range(300):
        junk = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))
        r = repair(mangle(junk))
        if r is not None:
            decode(r.data)


def test_no_structure_returns_none():
    assert repair(b"\x0a\x05" + REPLACEMENT) is None  # length 5, only 1 byte left


def test_scan_hint_for_mangled_data(Envelope):
    hints = scan(mangle(make(Envelope))).hints
    assert any("EF BF BD" in h and "--repair" in h for h in hints)


# --- CLI ---------------------------------------------------------------


def run(capsys, *args):
    rc = main(list(args))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli_decode_suggests_repair(tmp_path, Envelope):
    p = tmp_path / "bad.bin"
    p.write_bytes(mangle(make(Envelope)))
    with pytest.raises(SystemExit) as e:
        main(["decode", str(p), "--no-prompt"])
    assert "try --repair" in str(e.value)


def test_cli_decode_repair(tmp_path, capsys, Envelope):
    p = tmp_path / "bad.bin"
    p.write_bytes(mangle(make(Envelope)))
    rc, out, err = run(capsys, "decode", "--repair", str(p), "--no-prompt")
    assert rc == 0
    assert '3 field_3: string = "gateway-01.local"' in out
    assert "lost byte" in out and "value uncertain" in out
    assert "assuming each stands for one lost byte" in err
    assert "length of field 8 = " in err


def test_cli_repair_json_and_proto(tmp_path, capsys, Envelope):
    import json

    p = tmp_path / "bad.bin"
    p.write_bytes(mangle(make(Envelope)))
    rc, out, _ = run(capsys, "decode", "--repair", "--json", str(p))
    body = json.loads(out)[0]["fields"]
    assert any(e.get("lost_bytes") for e in body)
    rc, out, _ = run(capsys, "proto", "--repair", str(p))
    assert "Field8 field_8 = 8;" in out


def test_cli_repair_nothing_to_do(tmp_path, capsys):
    p = tmp_path / "ok.bin"
    p.write_bytes(bytes.fromhex("089601"))
    rc, out, err = run(capsys, "decode", "--repair", str(p), "--no-prompt")
    assert "nothing to repair" in err
    assert "field_1: int32 = 150" in out


def test_cli_repair_rejects_find_start(tmp_path):
    p = tmp_path / "ok.bin"
    p.write_bytes(bytes.fromhex("089601"))
    with pytest.raises(SystemExit) as e:
        main(["decode", "--repair", "--find-start", str(p), "--no-prompt"])
    assert "can't be combined" in str(e.value)


def test_search_is_bounded(monkeypatch):
    import pbre.repair as rep

    monkeypatch.setattr(rep, "_MAX_STATES", 5)
    with pytest.raises(rep.RepairTooHard):
        rep.repair(b"\x0a" + REPLACEMENT + b"\x01" + b"\x0a\x01\x41" * 50)
