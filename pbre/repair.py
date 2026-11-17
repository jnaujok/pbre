"""Best-effort repair of protobuf data that was mangled by a UTF-8 text decode.

When binary data is decoded as UTF-8 with replacement (``res.text()``,
``new String(bytes, UTF_8)``, ``bytes.decode(errors="replace")`` ...) and then
encoded again, each byte that wasn't part of a valid UTF-8 sequence becomes
the three bytes ``EF BF BD`` (U+FFFD). Everything else survives exactly.

That leaves more structure than it seems:

* A lost byte was always ``0x80`` or higher (lower bytes are ASCII and always
  valid), so it had its varint *continuation bit* set. The length of every
  varint, and so where every field starts and ends, is still knowable; only
  7 bits of value per lost byte are gone.
* Lengths are constrained by the structure: a length-delimited payload has to
  end exactly where its parent does, or where the next field starts.

:func:`repair` treats each replacement as one unknown byte and searches for
the field structure that best explains the whole buffer, scoring candidates
the same way ``pbre scan`` does. Bytes the structure pins down (typically
lengths) are recovered exactly; the rest get a placeholder and are reported,
so their decoded values are marked as uncertain rather than trusted.

Limits: a replacement can occasionally stand for two or three bytes (a
truncated multi-byte sequence); pbre assumes one, and says so if no
consistent structure exists under that assumption.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .scan import _field_number_score, _plausible_fixed
from .decoder import is_printable_text
from .wire import MAX_FIELD_NUMBER, encode_varint

REPLACEMENT = b"\xef\xbf\xbd"
_PLACEHOLDER = 0x80  # continuation bit set, value bits unknown (shown as 0)
_MAX_LENGTH_OPTIONS = 128  # one unknown 7-bit group in a length
_PLAUSIBLE_MESSAGE = 0.75  # fraction of payload bytes a nested parse must explain
# Upper bound on (start, end) regions the search may evaluate. Lost bytes in
# lengths multiply the possibilities, so cap the work instead of hanging.
_MAX_STATES = 300_000


class RepairTooHard(Exception):
    """The search space is too large to explore."""


@dataclass
class LostByte:
    offset: int  # in the repaired buffer
    value: int
    status: str  # "exact", "guess" or "unknown"
    what: str  # human-readable role of the byte


@dataclass
class RepairResult:
    data: bytes  # the repaired buffer
    replacements: int  # number of EF BF BD sequences found
    lost: List[LostByte]
    notes: List[str] = field(default_factory=list)

    @property
    def uncertain(self) -> Set[int]:
        """Offsets whose value is a placeholder or a guess."""
        return {b.offset for b in self.lost if b.status != "exact"}


def count_replacements(buf: bytes) -> int:
    return buf.count(REPLACEMENT)


def _template(buf: bytes) -> List[Optional[int]]:
    """Bytes with each replacement collapsed to one unknown (None)."""
    out: List[Optional[int]] = []
    i, n = 0, len(buf)
    while i < n:
        if buf.startswith(REPLACEMENT, i):
            out.append(None)
            i += 3
        else:
            out.append(buf[i])
            i += 1
    return out


@dataclass
class _Sol:
    score: float
    assign: Dict[int, Tuple[int, str, str]]  # pos -> (byte, status, what)
    fields: int = 0


class _Repairer:
    def __init__(self, t: List[Optional[int]]):
        self.t = t
        self.n = len(t)
        self.memo: Dict[Tuple[int, int], Optional[_Sol]] = {}

    # -- varints with holes ------------------------------------------------

    def _varint(self, p: int, limit: int):
        """Read a varint whose bytes may be unknown. Unknown bytes always
        continue (they were >= 0x80). Returns (groups, end) where groups are
        7-bit values or None, or None if no varint fits before ``limit``."""
        groups: List[Optional[int]] = []
        while p < limit and len(groups) < 10:
            b = self.t[p]
            p += 1
            if b is None:
                groups.append(None)
                continue
            groups.append(b & 0x7F)
            if not b & 0x80:
                return groups, p
        return None

    @staticmethod
    def _value(groups) -> int:
        v = 0
        for i, g in enumerate(groups):
            v |= (g or 0) << (7 * i)
        return v

    # -- the search ----------------------------------------------------------

    def best(self, s: int, e: int) -> Optional[_Sol]:
        """Best parse of t[s:e] as a sequence of fields, or None."""
        key = (s, e)
        if key in self.memo:
            return self.memo[key]
        if len(self.memo) >= _MAX_STATES:
            raise RepairTooHard(f"more than {_MAX_STATES} partial structures to consider")
        self.memo[key] = None  # guards against cycles (there are none)
        if s == e:
            sol: Optional[_Sol] = _Sol(0.0, {})
        else:
            sol = None
            options = self._options(s, e)
            viable = []
            for end, score, assign, desc in options:
                rest = self.best(end, e)
                if rest is None:
                    continue
                viable.append((score + rest.score, end, score, assign, rest, desc))
            if viable:
                # Highest score wins. Exact ties are common for a lost length:
                # the same fields parse either inside the payload or after
                # it, at the outer level. Prefer the longest, which keeps the
                # fields nested (an envelope wrapping its whole body).
                viable.sort(key=lambda v: (-round(v[0], 6), -v[1]))
                total, end, score, assign, rest, desc = viable[0]
                merged = dict(rest.assign)
                merged.update(assign)
                if desc == "length":
                    others = sorted(
                        {v[1] - (end - self._len_of(assign, end)) for v in viable[1:] if v[5] == "length"},
                        reverse=True,
                    )
                    if others:
                        for pos in assign:
                            val, status, what = merged[pos]
                            if status == "exact" and what.startswith("length"):
                                shown = ", ".join(str(o) for o in others[:6]) + (", ..." if len(others) > 6 else "")
                                merged[pos] = (
                                    val,
                                    "guess",
                                    what + f"; other lengths also parse ({shown}): they leave the payload's"
                                    " last fields at the outer level",
                                )
                sol = _Sol(total, merged, rest.fields + 1)
        self.memo[key] = sol
        return sol

    @staticmethod
    def _len_of(assign, end):
        """Length recorded in a length assignment (for listing alternatives)."""
        for _, (_, _, what) in assign.items():
            if what.startswith("length of field"):
                return int(what.split("= ")[1].split(" ")[0])
        return 0

    def _options(self, p: int, e: int):
        """All ways a field can start at ``p`` and end by ``e``:
        (end, score, assignments, kind)."""
        t = self.t
        tag = self._varint(p, e)
        if tag is None:
            return []
        groups, pos = tag
        out = []
        unknown_tag = any(g is None for g in groups)
        wire_types = [groups[0] & 7] if groups[0] is not None else [0, 1, 2, 5]
        for wt in wire_types:
            g = list(groups)
            if g[0] is None:
                g[0] = wt  # unknown low field-number bits shown as 0
            for i in range(1, len(g)):
                if g[i] is None:
                    g[i] = 0
            number = self._value(g) >> 3
            if number == 0 and groups[0] is None:
                g[0] = (1 << 3) | wt  # smallest valid field number
                number = self._value(g) >> 3
            if number == 0 or number > MAX_FIELD_NUMBER:
                continue
            tag_assign = {}
            for i, orig in enumerate(groups):
                if orig is None:
                    what = (
                        f"tag of field {number} (wire type {wt}); field number uncertain"
                        if i == 0
                        else f"tag of field {number}; field number uncertain"
                    )
                    tag_assign[p + i] = (0x80 | g[i], "unknown", what)
            mult = 1.0 + _field_number_score(number) - (0.5 if unknown_tag else 0.0)
            if wt == 0:
                v = self._varint(pos, e)
                if v is None:
                    continue
                vg, end = v
                a = dict(tag_assign)
                for i, gg in enumerate(vg):
                    if gg is None:
                        a[pos + i] = (_PLACEHOLDER, "unknown", f"value of varint field {number} (shown value is a lower bound)")
                out.append((end, max(mult, -1.0) * (end - p), a, "varint"))
            elif wt in (1, 5):
                width = 8 if wt == 1 else 4
                end = pos + width
                if end > e:
                    continue
                a = dict(tag_assign)
                raw = t[pos:end]
                for i, b in enumerate(raw):
                    if b is None:
                        a[pos + i] = (_PLACEHOLDER, "unknown", f"value of fixed{width * 8} field {number}")
                m = mult
                if None not in raw and not _plausible_fixed(int.from_bytes(bytes(raw), "little"), width):
                    m -= 1.0
                out.append((end, max(m, -1.0) * (end - p), a, "fixed"))
            elif wt == 2:
                lv = self._varint(pos, e)
                if lv is None:
                    continue
                lg, data = lv
                for length, length_assign in self._lengths(lg, pos, data, e, number):
                    end = data + length
                    a = dict(tag_assign)
                    a.update(length_assign)
                    pscore, passign = self._payload(data, end, number)
                    a.update(passign)
                    header = max(mult, -1.0) * (data - p)
                    out.append((end, header + max(mult, 0.0) * pscore, a, "length" if length_assign else "len"))
            # Groups with lost bytes in their tags aren't attempted; known
            # group tags are rare enough in data like this to leave as is.
        return out

    def _lengths(self, lg, pos, data, e, number):
        unknown = [i for i, g in enumerate(lg) if g is None]
        base = self._value(lg)
        room = e - data
        if not unknown:
            if base <= room:
                yield base, {}
            return
        if len(unknown) == 1:
            i = unknown[0]
            for gval in range(128):
                length = base + (gval << (7 * i))
                if length <= room:
                    yield length, {pos + i: (0x80 | gval, "exact", f"length of field {number} = {length} bytes")}
            return
        # Several unknown groups: only consider the length that ends exactly
        # at the end of the enclosing region, if it is representable.
        length = room
        extra = length - base
        if extra < 0:
            return
        assign = {}
        for i in unknown:
            gval = (extra >> (7 * i)) & 0x7F
            assign[pos + i] = (0x80 | gval, "exact", f"length of field {number} = {length} bytes")
            extra -= gval << (7 * i)
        if extra == 0:
            yield length, assign

    def _payload(self, data, end, number):
        """Score a payload and assign its lost bytes."""
        length = end - data
        if length == 0:
            return 0.0, {}
        raw = self.t[data:end]
        if None not in raw:
            try:
                if is_printable_text(bytes(raw).decode("utf-8")):
                    return float(length), {}
            except UnicodeDecodeError:
                pass
        sub = self.best(data, end)
        opaque = 0.5 * length
        if sub is not None and sub.score >= _PLAUSIBLE_MESSAGE * length:
            return sub.score, sub.assign
        a = {
            data + i: (_PLACEHOLDER, "unknown", f"inside bytes field {number}")
            for i, b in enumerate(raw)
            if b is None
        }
        if sub is not None and sub.score > opaque:
            return sub.score, sub.assign
        return opaque, a


def repair(buf: bytes) -> Optional[RepairResult]:
    """Repair ``buf``. Returns None if no consistent structure exists, and
    raises RepairTooHard if there are too many possibilities to explore."""
    count = count_replacements(buf)
    t = _template(buf)
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 20000))
    try:
        sol = _Repairer(t).best(0, len(t))
    finally:
        sys.setrecursionlimit(old_limit)
    if sol is None:
        return None
    out = bytearray()
    lost: List[LostByte] = []
    for i, b in enumerate(t):
        if b is None:
            val, status, what = sol.assign.get(i, (_PLACEHOLDER, "unknown", "not covered by any field"))
            out.append(val)
            lost.append(LostByte(i, val, status, what))
        else:
            out.append(b)
    notes = [
        f"found {count} replacement character{'s' if count != 1 else ''} (EF BF BD); "
        "assuming each stands for one lost byte"
    ]
    return RepairResult(bytes(out), count, lost, notes)
