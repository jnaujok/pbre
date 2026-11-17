"""Find where a protobuf message starts (and ends) inside a larger blob.

Protobuf has no sync markers: a parse only works if it starts exactly on a
field boundary. A real message must also end exactly at a field boundary, so
we work *backwards* from the end of the buffer:

* ``rec(k)`` parses the single field record whose tag starts at offset ``k``
  (or says there is none), which tells us where that record ends.
* Records are positional, so following ``end -> rec(end) -> ...`` from any
  offset gives one deterministic *chain*. ``stop[k]`` is where the chain from
  ``k`` breaks (``len(buf)`` if it runs cleanly to the end), and
  ``score[k]`` adds up how plausible each record on it looks.

Both are filled in one right-to-left pass, so the whole scan is linear in
the size of the input. Every offset is then a candidate start for the span
``[k, stop[k])``, which is guaranteed to parse as a message. Candidates are
ranked by how much plausible-looking data they explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .decoder import as_double, as_float, is_printable_text, plausible_float
from .wire import (
    MAX_FIELD_NUMBER,
    RESERVED_RANGE,
    DecodeError,
    WireType,
    encode_varint,
    parse_message,
)

# Below this confidence, a candidate is probably not a real message.
LOW_CONFIDENCE = 0.75

# How far to follow a chain when checking whether a group or a nested payload
# is well formed. Bounds the cost of a scan on pathological input.
_MAX_WALK = 4096
# Bonus, as a fraction of the input length, for a span that runs exactly to
# the end of the input (no trailing bytes).
_CLEAN_END_BONUS = 0.02

_EGROUP = "egroup"


@dataclass
class Candidate:
    start: int  # offset of the first byte of the message
    end: int  # offset just past the last byte of the message
    fields: int  # number of top-level fields in [start, end)
    score: float
    leading: int  # bytes before start
    trailing: int  # bytes after end
    # plausibility-weighted bytes covered by the span (score minus bonuses)
    explained: float = 0.0
    # set when this start lies on a higher-ranked candidate's chain
    same_as: Optional[int] = None  # rank (1-based) of that candidate
    dropped_fields: int = 0  # how many of its leading fields this one skips

    @property
    def clean_end(self) -> bool:
        return self.trailing == 0

    @property
    def confidence(self) -> float:
        """Plausibility per byte of the span, 0..1. Close to 1 for real
        messages; around 0.5 when the span is mostly opaque bytes fields,
        which is what random, compressed or encrypted data looks like."""
        span = self.end - self.start
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, self.explained / span))


@dataclass
class ScanResult:
    length: int
    candidates: List[Candidate]
    hints: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


def _read_varint(buf: bytes, pos: int, end: int) -> Optional[Tuple[int, int]]:
    result = 0
    shift = 0
    start = pos
    while pos < end:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            if result >= 1 << 64:
                return None
            return result, pos
        shift += 7
        if shift >= 70 or pos - start >= 10:
            return None
    return None


def _canonical(value: int, nbytes: int) -> bool:
    return len(encode_varint(value)) == nbytes


def _plausible_fixed(v: int, width: int) -> bool:
    """Real fixed-width fields mostly hold ordinary floats/doubles or
    smallish integers; random bytes rarely look like either."""
    bits = width * 8
    if v < 1 << (bits // 2) or v >= (1 << bits) - (1 << (bits // 2)):
        return True  # small unsigned or small negative signed value
    x = as_float(v) if width == 4 else as_double(v)
    return plausible_float(x)


def _field_number_score(number: int) -> float:
    if number in RESERVED_RANGE:
        return -3.0
    if number <= 64:
        return 0.0
    if number <= 1000:
        return -0.25
    if number <= 1 << 16:
        return -1.0
    return -3.0


class _Scanner:
    def __init__(self, buf: bytes, max_field_number: int):
        self.buf = buf
        self.n = len(buf)
        self.max_field_number = max_field_number
        n = self.n
        # Per-offset record info: end offset (or -1 if no record), kind, number.
        self.rec_end = [-1] * (n + 1)
        self.rec_kind: List[Optional[str]] = [None] * (n + 1)
        self.rec_num = [0] * (n + 1)
        self.rec_score = [0.0] * (n + 1)
        # Chain info.
        self.stop = list(range(n + 1))
        self.total = [0.0] * (n + 1)
        self.count = [0] * (n + 1)

    def run(self) -> None:
        n = self.n
        for k in range(n - 1, -1, -1):
            self._record(k)
            end = self.rec_end[k]
            if end > k and self.rec_kind[k] != _EGROUP:
                self.stop[k] = self.stop[end]
                self.total[k] = self.rec_score[k] + self.total[end]
                self.count[k] = 1 + self.count[end]

    def _walk_to(self, pos: int, limit: int, group: Optional[int]) -> Optional[int]:
        """Follow records from ``pos``. For a group, return the offset just
        past its matching end-group tag; otherwise return ``pos`` if the
        records land exactly on ``limit``. None if neither happens."""
        steps = 0
        while pos < limit and steps < _MAX_WALK:
            end = self.rec_end[pos]
            if end <= pos or end > limit:
                return None
            if self.rec_kind[pos] == _EGROUP:
                return end if group is not None and self.rec_num[pos] == group else None
            pos = end
            steps += 1
        if group is None and pos == limit:
            return pos
        return None

    def _record(self, k: int) -> None:
        buf, n = self.buf, self.n
        tag = _read_varint(buf, k, n)
        if tag is None:
            return
        value, pos = tag
        number, wt = value >> 3, value & 7
        if number == 0 or number > self.max_field_number or wt > 5:
            return
        # Plausibility multiplier for this record; its score is this times the
        # number of bytes it covers, so a start is ranked by how many
        # plausible-looking bytes it explains.
        score = 1.0 + _field_number_score(number)
        if not _canonical(value, pos - k):
            score -= 2.0
        if wt == WireType.VARINT:
            v = _read_varint(buf, pos, n)
            if v is None:
                return
            val, end = v
            if not _canonical(val, end - pos):
                score -= 2.0
        elif wt in (WireType.I64, WireType.I32):
            width = 8 if wt == WireType.I64 else 4
            end = pos + width
            if end > n:
                return
            if not _plausible_fixed(int.from_bytes(buf[pos:end], "little"), width):
                score -= 1.0
        elif wt == WireType.LEN:
            ln = _read_varint(buf, pos, n)
            if ln is None:
                return
            length, data = ln
            if not _canonical(length, data - pos):
                score -= 2.0
            end = data + length
            if end > n:
                return
            if length:
                payload = buf[data:end]
                plausible = self._walk_to(data, end, None) == end
                if not plausible:
                    try:
                        plausible = is_printable_text(payload.decode("utf-8"))
                    except UnicodeDecodeError:
                        pass
                if not plausible:
                    score -= 0.5  # opaque bytes: possible, but weak evidence
        elif wt == WireType.SGROUP:
            end = self._walk_to(pos, n, number)
            if end is None:
                return
            score -= 0.5  # groups are rare in modern data
        else:  # EGROUP: only valid as the terminator of a group
            self.rec_end[k] = pos
            self.rec_kind[k] = _EGROUP
            self.rec_num[k] = number
            return
        if end > n:
            return
        self.rec_end[k] = end
        self.rec_kind[k] = "field"
        self.rec_num[k] = number
        self.rec_score[k] = max(score, -1.0) * (end - k)


def _detect_hints(buf: bytes) -> List[str]:
    hints = []
    if buf[:2] == b"\x1f\x8b":
        hints.append("starts with 1f 8b: looks gzip-compressed; decompress it first")
    elif buf[:1] == b"\x78" and buf[1:2] in (b"\x01", b"\x5e", b"\x9c", b"\xda"):
        hints.append("starts with a zlib header; decompress it first")
    elif buf[:4] == b"\x28\xb5\x2f\xfd":
        hints.append("starts with 28 b5 2f fd: looks zstd-compressed; decompress it first")
    if len(buf) >= 5 and buf[0] in (0, 1) and int.from_bytes(buf[1:5], "big") == len(buf) - 5:
        hints.append("first 5 bytes look like a gRPC frame header: try --grpc")
    else:
        v = _read_varint(buf, 0, len(buf))
        if v is not None and v[0] > 0 and v[1] + v[0] == len(buf):
            hints.append(
                f"first {v[1]} byte(s) are a varint equal to the remaining length: try --delimited"
            )
    return hints


def scan(
    buf: bytes,
    top: int = 5,
    max_field_number: int = MAX_FIELD_NUMBER,
    min_fields: int = 1,
) -> ScanResult:
    """Rank the most plausible (start, end) spans of a message in ``buf``."""
    s = _Scanner(buf, max_field_number)
    s.run()
    n = len(buf)

    def value(k: int) -> float:
        return s.total[k] + (_CLEAN_END_BONUS * n if s.stop[k] == n else 0.0)

    starts = [k for k in range(n) if s.count[k] >= min_fields and s.total[k] > 0]
    starts.sort(key=lambda k: (-value(k), k))

    chosen: List[Candidate] = []
    # offset -> (rank, number of records from that candidate's start)
    on_chain: dict = {}
    for k in starts:
        if len(chosen) >= top:
            break
        same = on_chain.get(k)
        if same is not None and same[1] > 3:
            # Just a later field boundary of a better candidate; only worth
            # showing when it drops a few (possibly bogus) leading fields.
            continue
        cand = Candidate(
            start=k,
            end=s.stop[k],
            fields=s.count[k],
            score=round(value(k), 2),
            leading=k,
            trailing=n - s.stop[k],
            explained=s.total[k],
        )
        if same is not None:
            cand.same_as, cand.dropped_fields = same
        chosen.append(cand)
        rank = len(chosen)
        pos, steps = k, 0
        while pos < s.stop[k]:
            on_chain.setdefault(pos, (rank, steps))
            pos = s.rec_end[pos]
            steps += 1
    return ScanResult(length=n, candidates=chosen, hints=_detect_hints(buf))


def find_start(buf: bytes, max_field_number: int = MAX_FIELD_NUMBER) -> Optional[Candidate]:
    """Pick the span to decode.

    If the whole buffer already parses as a message it is used as is, since
    the input is then valid and there is nothing to find. Otherwise the
    best-ranked candidate from :func:`scan` is returned (None if there is
    none).
    """
    try:
        fields = parse_message(buf)
        return Candidate(0, len(buf), len(fields), 0.0, 0, 0, explained=float(len(buf)))
    except DecodeError:
        pass
    return scan(buf, top=1, max_field_number=max_field_number).best
