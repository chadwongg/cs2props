"""Cross-book line disagreement: the one edge that needs no forecasting.

The real-line backtest settled an uncomfortable question. Against 694
archived PrizePicks lines the model scores AUC 0.5128 — no discrimination at
any confidence level — because its inputs ARE the book's inputs. Historical
per-map stats run through a smaller model reproduce the book's number minus
the vig. Refining that model cannot fix it.

Disagreement is a different kind of claim. When PrizePicks posts 25.5 and
Underdog posts 27.5 on the same player in the same match, at least one of
them is wrong, and knowing that requires no opinion about the player at all.
The edge comes from the market contradicting itself, not from out-thinking
it.

Three separate things are measured here, because conflating them is the easy
mistake (an earlier ad-hoc probe reported "55.4%" that silently mixed all
three):

  OVER at the cheaper line   -- a directional bet that the low book is soft
  UNDER at the pricier line  -- a directional bet that the high book is soft
  MIDDLE                     -- the result lands BETWEEN the two lines, so
                                both sides win. Real, but it is a
                                two-book two-ticket play, not something a
                                single 4-man slip can harvest. Reported
                                separately so it never inflates a
                                directional hit rate.

Also tracked: whether one book is systematically softer on a stat. A
persistent signed bias is far more actionable than scattered disagreement,
because it tells you which book to shop for which market.

SAMPLE WARNING: the prop archive starts 2026-07-24. Everything here is days
of overlap. The point of this module is to accumulate the measurement, not
to act on today's version of it.

PRE-COMMITTED CHECKPOINT (2026-07-30): at 400 settled disagreements, the
under-at-higher lift over the books-agree control decides this module's
fate — >= 5pts with z >= 2 wires disagreement into the optimizer's leg
selection; anything less kills the hypothesis and drops live stakes to
paper-tracking. Fixed IN ADVANCE because the readings so far trace the
classic noise-mirage curve (57 settled -> +15.6pt z=1.93; 205 settled ->
+6.9pt z=1.53), and the whole failure mode this guards against is finding
a friendlier slice after the first one fades. Do not move the goalposts.

VERDICT (2026-08-27, applied as pre-committed): at 1,037 settled
disagreements the lift is +0.0pt — nowhere near the >=5pt bar. The
hypothesis is DEAD: line disagreement between these books carries no
directional edge. The curve completed exactly the noise-mirage arc the
checkpoint was written to catch (+15.6 -> +6.9 -> +0.0). This module
stays as a MONITORING tool only — line shopping still uses the gaps for
price, never for direction — and nothing here may feed leg selection.
"""

from __future__ import annotations

import logging
import math
import sqlite3

# The pre-committed decision point from the module docstring, importable so
# the dashboard's checkpoint tile and this module can never drift apart.
CHECKPOINT_N = 400
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cs2props.model.reallines import (
    MATCH_WINDOW_AFTER,
    MATCH_WINDOW_BEFORE,
    _epoch,
    _observed,
)
from cs2props.model.state_builder import clean_name
from cs2props.standins import _loose_key, _same_person

log = logging.getLogger(__name__)

PRIZEPICKS_LEAGUE = "265"
# Two postings describe the same match only if their start times agree. Books
# occasionally round a start differently, so a small tolerance is allowed --
# but nothing near the 10h grading window, which would pair a player's
# afternoon match with his evening one.
SAME_MATCH_TOLERANCE_S = 3 * 3600.0
MIN_DISAGREEMENT = 1.0  # below this, the gap is rounding, not opinion


@dataclass(frozen=True)
class Disagreement:
    player: str
    stat: str
    map_lo: int
    map_hi: int
    pp_line: float
    ud_line: float
    start_ts: float

    @property
    def gap(self) -> float:
        """Signed: positive when PrizePicks posts the HIGHER number."""
        return self.pp_line - self.ud_line

    @property
    def low_book(self) -> str:
        return "underdog" if self.pp_line > self.ud_line else "prizepicks"

    @property
    def low_line(self) -> float:
        return min(self.pp_line, self.ud_line)

    @property
    def high_line(self) -> float:
        return max(self.pp_line, self.ud_line)

    def describe(self) -> str:
        return (
            f"{self.player} {self.stat} maps {self.map_lo}-{self.map_hi}: "
            f"PP {self.pp_line:g} vs UD {self.ud_line:g} "
            f"({abs(self.gap):g} apart)"
        )


@dataclass(frozen=True)
class Graded(Disagreement):
    observed: float = 0.0

    @property
    def over_low_won(self) -> bool:
        return self.observed > self.low_line

    @property
    def under_high_won(self) -> bool:
        return self.observed < self.high_line

    @property
    def middled(self) -> bool:
        """Landed strictly between the lines — both directional bets win."""
        return self.low_line < self.observed < self.high_line


@dataclass
class CrossBookResult:
    paired: int = 0  # props both books posted
    graded: list[Graded] = field(default_factory=list)
    # Props both books posted at the SAME line, graded. This is the control
    # and it is not optional: without it, "unders at the higher line win 63%"
    # reads as edge when it might only be "unders win". The comparison is
    # what turns a rate into a finding.
    agreed: list[Graded] = field(default_factory=list)
    live: list[Disagreement] = field(default_factory=list)
    pp_minus_ud: list[tuple[str, float]] = field(default_factory=list)

    def control_under_rate(self) -> tuple[int, int, float, float]:
        """UNDER at the higher line, on props where the books AGREE."""
        return self._rate(
            sum(g.under_high_won for g in self.agreed), len(self.agreed)
        )

    def lift(self) -> tuple[float, float]:
        """(disagreement rate minus control rate, z-score of the gap).

        The claim this module exists to test is not "unders win" — it is
        "disagreement predicts which side wins". Only the DIFFERENCE speaks
        to that.
        """
        n1, _w1, r1, _ = self.under_at_higher()
        n0, _w0, r0, _ = self.control_under_rate()
        if not n1 or not n0:
            return float("nan"), float("nan")
        se = math.sqrt(
            max(r1 * (1 - r1), 1e-9) / n1 + max(r0 * (1 - r0), 1e-9) / n0
        )
        return r1 - r0, (r1 - r0) / se if se > 0 else float("nan")

    def _rate(self, wins: int, n: int) -> tuple[int, int, float, float]:
        rate = wins / n if n else float("nan")
        se = math.sqrt(max(rate * (1 - rate), 1e-9) / n) if n else float("nan")
        return n, wins, rate, 1.96 * se

    def over_at_lower(self) -> tuple[int, int, float, float]:
        return self._rate(
            sum(g.over_low_won for g in self.graded), len(self.graded)
        )

    def under_at_higher(self) -> tuple[int, int, float, float]:
        return self._rate(
            sum(g.under_high_won for g in self.graded), len(self.graded)
        )

    def middle_rate(self) -> float:
        if not self.graded:
            return float("nan")
        return sum(g.middled for g in self.graded) / len(self.graded)

    def book_bias(self) -> dict[str, tuple[int, float]]:
        """stat -> (n, mean PP line minus UD line).

        A persistent nonzero value means one book is systematically softer on
        that market — the most actionable pattern this module can find.
        """
        groups: dict[str, list[float]] = defaultdict(list)
        for stat, d in self.pp_minus_ud:
            groups[stat].append(d)
        return {
            k: (len(v), sum(v) / len(v)) for k, v in sorted(groups.items())
        }


def _book_of(league_id: str) -> str:
    return "prizepicks" if str(league_id) == PRIZEPICKS_LEAGUE else "underdog"


def _first_lines(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per (book, player, stat, range, match) at its first-seen line.

    First-seen rather than closing, matching the real-line backtest: it is
    the number the scanner prices and the bettor acts on.
    """
    rows = conn.execute(
        """
        SELECT player_name, stat_kind, map_lo, map_hi, start_time, league_id,
               line_score, MIN(scanned_at) AS seen
        FROM props
        WHERE board = 'standard' AND start_time IS NOT NULL
          AND map_lo IS NOT NULL AND map_hi IS NOT NULL
        GROUP BY player_name, stat_kind, map_lo, map_hi, start_time, league_id
        """
    ).fetchall()
    out = []
    for name, stat, lo, hi, start, lg, line, _seen in rows:
        ts = _epoch(str(start))
        if ts is None:
            continue
        out.append({
            "player": str(name), "key": _loose_key(str(name)),
            "stat": str(stat), "lo": int(lo), "hi": int(hi),
            "start_ts": ts, "book": _book_of(lg), "line": float(line),
        })
    return out


def pair_books(conn: sqlite3.Connection) -> list[Disagreement]:
    """Every prop both books posted for the same player, stat and match.

    Player names are matched with the same leet-tolerant comparison used for
    stand-in detection: the books spell nicknames differently (sh1ro vs
    sh1r0), and an exact join silently discards most of the overlap — which
    is the entire sample this module depends on.
    """
    rows = _first_lines(conn)
    by_market: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_market[(r["stat"], r["lo"], r["hi"])].append(r)

    out: list[Disagreement] = []
    for market, entries in by_market.items():
        pps = [e for e in entries if e["book"] == "prizepicks"]
        uds = [e for e in entries if e["book"] == "underdog"]
        used: set[int] = set()
        for pp in pps:
            best: dict[str, Any] | None = None
            for i, ud in enumerate(uds):
                if i in used:
                    continue
                if abs(ud["start_ts"] - pp["start_ts"]) > SAME_MATCH_TOLERANCE_S:
                    continue
                if not _same_person(pp["player"], ud["player"]):
                    continue
                if best is None or abs(
                    ud["start_ts"] - pp["start_ts"]
                ) < abs(best["start_ts"] - pp["start_ts"]):
                    best, best_i = ud, i
            if best is None:
                continue
            used.add(best_i)
            out.append(Disagreement(
                player=pp["player"], stat=market[0],
                map_lo=market[1], map_hi=market[2],
                pp_line=pp["line"], ud_line=best["line"],
                start_ts=pp["start_ts"],
            ))
    return out


def _outcome_index(
    conn: sqlite3.Connection,
) -> dict[str, list[tuple[float, str, int, int, float]]]:
    """loose player key -> [(match_ts, stat, lo, hi, observed)]."""
    rows = conn.execute(
        """
        SELECT match_id, map_number, player_id, player_name, team, opponent,
               event_tier, map_name, played_at, kills, rounds, headshots
        FROM player_maps
        WHERE rounds IS NOT NULL AND rounds > 10
        ORDER BY played_at, match_id, map_number
        """
    ).fetchall()
    series: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for r in rows:
        series[(r[0], r[2])].append(r)
    idx: dict[str, list[tuple[float, str, int, int, float]]] = defaultdict(list)
    for maps in series.values():
        maps = sorted(maps, key=lambda r: r[1])
        ts = _epoch(str(maps[0][8]))
        if ts is None:
            continue
        key = _loose_key(clean_name(str(maps[0][3])))
        for stat in ("kills", "headshots"):
            for lo, hi in ((1, 1), (1, 2), (2, 2), (1, 3)):
                obs = _observed(maps, stat, lo, hi)
                if obs is not None:
                    idx[key].append((ts, stat, lo, hi, obs))
    return idx


def run(conn: sqlite3.Connection) -> CrossBookResult:
    """Pair both books, grade what has settled, list what is still live."""
    res = CrossBookResult()
    pairs = pair_books(conn)
    res.paired = len(pairs)
    outcomes = _outcome_index(conn)

    for d in pairs:
        res.pp_minus_ud.append((d.stat, d.gap))
        hit = None
        for ts, stat, lo, hi, obs in outcomes.get(_loose_key(d.player), []):
            if stat != d.stat or lo != d.map_lo or hi != d.map_hi:
                continue
            if -MATCH_WINDOW_BEFORE <= ts - d.start_ts <= MATCH_WINDOW_AFTER:
                hit = obs
                break
        disagrees = abs(d.gap) >= MIN_DISAGREEMENT
        if hit is None:
            if disagrees:
                res.live.append(d)
            continue
        graded = Graded(
            player=d.player, stat=d.stat, map_lo=d.map_lo,
            map_hi=d.map_hi, pp_line=d.pp_line, ud_line=d.ud_line,
            start_ts=d.start_ts, observed=hit,
        )
        (res.graded if disagrees else res.agreed).append(graded)
    res.live.sort(key=lambda d: -abs(d.gap))
    return res


def format_report(res: CrossBookResult) -> str:
    out = [
        "=== cross-book line disagreement ===",
        f"props both books posted : {res.paired}",
        f"  disagreeing by {MIN_DISAGREEMENT:g}+ : "
        f"{len(res.graded)} settled, {len(res.live)} still live",
    ]
    if res.graded:
        out.append("")
        out.append("DIRECTIONAL (each is a bet you could actually place):")
        for label, fn in (
            ("OVER at the cheaper line ", res.over_at_lower),
            ("UNDER at the pricier line", res.under_at_higher),
        ):
            n, won, rate, ci = fn()
            out.append(f"  {label}: {won}/{n} = {rate:.1%}  (+/-{ci:.1%})")
        out.append(
            f"  MIDDLE (both win, needs 2 tickets on 2 books): "
            f"{res.middle_rate():.1%}"
        )
        if res.agreed:
            n0, w0, r0, ci0 = res.control_under_rate()
            gap, z = res.lift()
            out += [
                "",
                "CONTROL — the same bet where the books AGREE:",
                f"  UNDER at that line       : {w0}/{n0} = {r0:.1%} "
                f"(+/-{ci0:.1%})",
                f"  LIFT from disagreement   : {gap * 100:+.1f} pts, z={z:.2f}",
                "  (this is the actual claim. A high under-rate on its own",
                "   could just mean unders win; only the LIFT says the",
                "   disagreement itself carries information.)",
            ]
        out.append("")
        out.append("  break-even for a 4-man at 8.5x: 58.6% per leg")
    if res.pp_minus_ud:
        out += ["", "book bias (mean PrizePicks line MINUS Underdog line):"]
        for stat, (n, mean) in res.book_bias().items():
            softer = "PP posts higher" if mean > 0 else "UD posts higher"
            out.append(f"  {stat:<10} n={n:<5} {mean:+.3f}  ({softer})")
    if res.live:
        out += ["", f"LIVE disagreements, widest first ({len(res.live)}):"]
        for d in res.live[:12]:
            out.append(f"  {d.describe()}")
    out += [
        "",
        "The archive starts 2026-07-24. Treat every rate above as a running",
        "measurement, not a signal -- a 55% over 70 bets has a +/-12 point",
        "interval, which decides nothing. This becomes usable at ~400 graded",
        "disagreements, which is weeks of accumulation, not days.",
        "",
        "And note the multiple-comparisons trap: several slices were examined",
        "to find this one. A z near 2 on the slice that looked best is much",
        "weaker evidence than a z near 2 on a hypothesis fixed in advance.",
        "The honest use of this number is to PRE-COMMIT to it now and check",
        "whether it survives the next few hundred disagreements.",
    ]
    return "\n".join(out)
