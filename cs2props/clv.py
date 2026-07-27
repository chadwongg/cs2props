"""Closing-line value (CLV) for tracked slips.

CLV asks: after you bet, did the line move TOWARD your side? It is the
fastest honest read on whether this model beats the market — CLV converges in
roughly 20-30 bets where raw P&L needs hundreds, because it strips out the
variance of whether the legs happened to land.

Sign convention (positive = you beat the close):
- UNDER: you want a HIGH line, so CLV = your_line - closing_line.
  Bet under 14, closes at 13 -> +1.0: the market came to your view and you
  hold the easier number.
- OVER: you want a LOW line, so CLV = closing_line - your_line.

Closing lines come from the ``props`` snapshot table, which every scan and
import writes. The closing line is simply the LAST line observed for that
prop before kickoff — so CLV quality depends on scanning close to match time.
A scan run hours early gives a stale "close" and understates the signal.

Reading the output:
- consistently POSITIVE CLV = the model finds value the market later agrees
  with. That is real edge, and it shows up long before the P&L does.
- CLV around zero = the model is re-deriving the market. It may still win or
  lose, but there is no evidence of an edge.
- consistently NEGATIVE CLV = the model is systematically on the wrong side.
  Stop betting and fix the projections; the losses just have not arrived yet.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass

from cs2props.model.state_builder import clean_name

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegCLV:
    slip_id: str
    book: str
    player: str
    stat: str
    side: str
    bet_line: float
    closing_line: float
    clv: float  # signed, positive = beat the close


def signed_clv(side: str, bet_line: float, closing_line: float) -> float:
    """CLV in stat units. Positive means you hold the better number."""
    if side.lower() == "under":
        return bet_line - closing_line
    return closing_line - bet_line


def closing_line(
    conn: sqlite3.Connection,
    player: str,
    stat_kind: str,
    map_lo: int,
    map_hi: int,
    after: float,
) -> float | None:
    """Last line seen for this prop IN THE MATCH THAT WAS BET.

    Matching only on name + stat + map range was wrong: it happily returned a
    line from the player's NEXT match days later, producing impossible
    readings (Techno4k bet at 25.5, "closed" at 37.5 — a different game).
    The fix pins the match by its scheduled start: take the EARLIEST start
    time seen after placement, which is the game that was bet, then take the
    latest snapshot carrying that start time.

    Only STANDARD board lines count. Underdog posts alt/scorcher ladders for
    the same player and start time (25.5 / 29.5 / 31.5 / 34.5 / 37.5), and
    picking one of those as the "close" produced a -12.0 CLV reading on a
    line that never moved.
    """
    rows = conn.execute(
        """
        SELECT player_name, line_score, scanned_at, start_time
        FROM props
        WHERE stat_kind = ? AND map_lo = ? AND map_hi = ? AND scanned_at >= ?
          AND board = 'standard'
        ORDER BY scanned_at DESC
        """,
        (stat_kind, map_lo, map_hi, after),
    ).fetchall()
    target = clean_name(player)
    mine = [r for r in rows if clean_name(r[0]) == target and r[3]]
    if not mine:
        return None
    bet_match = min(r[3] for r in mine)  # the soonest game = the one bet on
    same = [r for r in mine if r[3] == bet_match]
    return float(same[0][1]) if same else None  # rows are newest-first


def leg_clvs(conn: sqlite3.Connection) -> list[LegCLV]:
    """CLV for every tracked leg that has a usable closing line."""
    legs = conn.execute(
        """
        SELECT l.slip_id, l.player_name, l.side, l.line, l.stat_kind,
               l.map_lo, l.map_hi, s.placed_at, s.book
        FROM slip_legs l JOIN slips s ON s.slip_id = l.slip_id
        ORDER BY s.placed_at, l.leg_no
        """
    ).fetchall()
    out: list[LegCLV] = []
    for slip_id, player, side, line, stat, lo, hi, placed_at, book in legs:
        close = closing_line(conn, player, stat, lo, hi, placed_at)
        if close is None:
            continue
        out.append(LegCLV(
            slip_id=slip_id, book=book, player=player, stat=stat, side=side,
            bet_line=float(line), closing_line=close,
            clv=signed_clv(side, float(line), close),
        ))
    return out


def _block(rows: list[LegCLV], label: str) -> list[str]:
    vals = [r.clv for r in rows]
    n = len(vals)
    mean = sum(vals) / n
    beat = sum(1 for v in vals if v > 0)
    tied = sum(1 for v in vals if v == 0)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    out = [f"{label}: {n} legs · mean CLV {mean:+.3f} · beat the close "
           f"{beat}/{n} ({beat / n:.0%}), tied {tied}"]
    if n >= 2:
        lo, hi = mean - 1.645 * se, mean + 1.645 * se
        verdict = ("beating the close" if lo > 0
                   else "LOSING to the close" if hi < 0
                   else "not yet distinguishable from zero")
        out.append(f"    90% band {lo:+.3f} to {hi:+.3f} -> {verdict}")
    return out


def format_report(rows: list[LegCLV]) -> str:
    """Per-book CLV, plus a pooled figure when both books are in play.

    CLV compares your line against the CLOSING LINE AT THE SAME BOOK. The two
    books post different numbers for the same prop, so mixing them measures
    nothing coherent — a book whose lines you consistently beat can be masked
    by one you do not.
    """
    if not rows:
        return (
            "No CLV yet — closing lines come from board snapshots taken AFTER "
            "a bet is placed. Run `cs2props scan` again closer to match time "
            "(the 8:00/15:00 automation does this) and re-check."
        )
    lines: list[str] = []
    books = sorted({r.book for r in rows})
    for bk in books:
        lines += _block([r for r in rows if r.book == bk], bk)
    if len(books) > 1:
        lines.append("")
        lines += _block(rows, "ALL BOOKS")
    lines.append("")
    lines.append("worst legs (line moved against you):")
    for r in sorted(rows, key=lambda x: x.clv)[:5]:
        lines.append(
            f"   {r.book:<11}{r.player:<14}{r.side:<6}{r.stat:<11}"
            f"bet {r.bet_line:>5.1f}  closed {r.closing_line:>5.1f}"
            f"   CLV {r.clv:+.1f}"
        )
    return "\n".join(lines)
