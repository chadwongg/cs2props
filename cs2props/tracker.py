"""Module 5: slip tracking and automatic grading.

Slips the user actually placed are recorded, then graded against the same
``player_maps`` table the daily top-up fills — results entry is never manual.

Grading semantics (Power play):
- A leg wins if the observed total beats its line on the chosen side.
- Exact-line landings (whole-number lines) push -> VOID, per book practice.
- A map range extending past the maps actually played -> VOID (books drop
  the leg), except when no maps were found at all -> still pending.
- Voids shrink the slip: an n-pick with v voided legs pays the (n-v)-pick
  multiplier from the payouts config; below 2 live legs the stake refunds.
- Any lost leg -> slip lost.

Live-calibration readout: mean claimed P(win) vs realized win rate, with a
binomial 90% band so a cold streak reads honestly.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from cs2props.config import load_payouts
from cs2props.model.state_builder import clean_name

log = logging.getLogger(__name__)

# "donk over 32.5 kills 1-2"  /  "sh1ro under 15.0 headshots 1-3"
_LEG_RE = re.compile(
    r"^\s*(?P<player>\S+)\s+(?P<side>over|under)\s+(?P<line>\d+(?:\.\d+)?)\s+"
    r"(?P<stat>kills|headshots)\s+(?P<lo>\d+)-(?P<hi>\d+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LegSpec:
    player_name: str
    side: str
    line: float
    stat_kind: str
    map_lo: int
    map_hi: int


def parse_leg(text: str) -> LegSpec:
    m = _LEG_RE.match(text)
    if not m:
        raise ValueError(
            f"cannot parse leg {text!r} — expected "
            "'<player> over|under <line> kills|headshots <lo>-<hi>'"
        )
    return LegSpec(
        player_name=m.group("player"),
        side=m.group("side").lower(),
        line=float(m.group("line")),
        stat_kind=m.group("stat").lower(),
        map_lo=int(m.group("lo")),
        map_hi=int(m.group("hi")),
    )


def track_slip(
    conn: sqlite3.Connection,
    book: str,
    stake: float,
    legs: list[LegSpec],
    claimed_p: float | None = None,
    placed_at: float | None = None,
    multiplier: float | None = None,
    product: str = "power",
) -> str:
    slip_id = uuid.uuid4().hex[:8]
    conn.execute(
        "INSERT INTO slips (slip_id, book, placed_at, stake, n_legs, claimed_p,"
        " multiplier, product) VALUES (?,?,?,?,?,?,?,?)",
        (slip_id, book, placed_at or time.time(), stake, len(legs), claimed_p,
         multiplier, product),
    )
    conn.executemany(
        "INSERT INTO slip_legs (slip_id, leg_no, player_name, side, line,"
        " stat_kind, map_lo, map_hi) VALUES (?,?,?,?,?,?,?,?)",
        [
            (slip_id, i, l.player_name, l.side, l.line, l.stat_kind,
             l.map_lo, l.map_hi)
            for i, l in enumerate(legs)
        ],
    )
    conn.commit()
    log.info("tracked slip %s: %s @ %.2f, %d legs", slip_id, book, stake, len(legs))
    return slip_id


def _epoch(ts: str) -> float | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def _prop_start_epoch(
    conn: sqlite3.Connection, player: str, stat: str, line: float,
    lo: int, hi: int, placed_at: float,
) -> float | None:
    """Start time of the prop this leg was taken from, as UTC epoch.

    Recovered from the board archive rather than stored on the leg, so it
    also repairs slips tracked before this mattered. Matching on the exact
    line as well as the player pins the right posting when a player has
    several markets up.
    """
    rows = conn.execute(
        "SELECT start_time FROM props WHERE player_name = ? AND stat_kind = ?"
        " AND map_lo = ? AND map_hi = ? AND line_score = ?"
        " AND start_time IS NOT NULL GROUP BY start_time",
        (player, stat, int(lo), int(hi), float(line)),
    ).fetchall()
    best, best_gap = None, None
    for (st,) in rows:
        ts = _epoch(str(st))
        if ts is None or ts < placed_at - 6 * 3600:
            continue
        gap = ts - placed_at
        if best_gap is None or gap < best_gap:
            best, best_gap = ts, gap
    return best


def _grade_leg(
    conn: sqlite3.Connection, placed_at: float, leg: "tuple[Any, ...]"
) -> tuple[str, float | None]:
    """-> (status, observed).

    Finds the player's first finished match played within
    [placed_at - 6h, placed_at + 60h] (name-joined in Python because steam
    nicknames are decorated). Only finished matches are ever ingested, so
    finding rows means the match is settled.
    """
    (_sid, _no, player, side, line, stat, lo, hi, _st, _obs) = tuple(leg)
    target = clean_name(player)
    # The match must START AFTER placement. A window reaching backwards
    # graded slips against the player's PREVIOUS match — it fabricated two
    # settled losses for games that had not been played (2026-07-24). Props
    # are pre-game only, so "after placement" is always correct.
    # Bounds are built in PYTHON, not with SQLite's datetime(): that returns
    # "YYYY-MM-DD HH:MM:SS" (space) while played_at is ISO with a "T". Since
    # 'T' > ' ' lexically, every past match slipped through the filter and
    # slips were graded against the players' PREVIOUS games (2026-07-24).
    t_lo = datetime.fromtimestamp(placed_at, tz=timezone.utc)
    t_hi = t_lo + timedelta(hours=60)
    fmt = "%Y-%m-%dT%H:%M:%S"
    rows = conn.execute(
        """
        SELECT player_name, map_number, kills, headshots, match_id, played_at
        FROM player_maps
        WHERE played_at >= ? AND played_at <= ?
        ORDER BY played_at, match_id, map_number
        """,
        (t_lo.strftime(fmt), t_hi.strftime(fmt)),
    ).fetchall()
    mine = [r for r in rows if clean_name(r[0]) == target]
    if not mine:
        # bo3.gg decorates nicknames the board does not ("Salazar_9" for the
        # board's "Salazar", found 2026-07-27 blocking a settled leg) — fall
        # back to the same loose comparison the stand-in detector uses.
        from cs2props.standins import _same_person

        mine = [r for r in rows if _same_person(r[0], player)]
    if not mine:
        return "pending", None
    # WHICH match? Taking the earliest one after placement is wrong: players
    # routinely play twice in a day, and the prop names a specific opponent.
    # Measured 2026-07-26 — frontales and kade0 both played INFINITE at 11:00
    # and Astralis at 14:47, the props were "@ Astralis", and the grader
    # scored both against INFINITE. It flipped two legs (frontales 8 vs the
    # real 11, kade0 16 vs the real 10) and so corrupted the leg hit rate,
    # which is the single number this project is being judged on.
    # The prop archive knows the start time, so use it.
    start = _prop_start_epoch(conn, player, stat, line, lo, hi, placed_at)
    if start is not None:
        best, best_gap = None, None
        for _p, _mn, _k, _h, mid, at in mine:
            ts = _epoch(str(at))
            if ts is None:
                continue
            # A match cannot FINISH before its prop's start time. bo3.gg's
            # clock skew only runs LATE (+~8h), so any candidate earlier
            # than the start is a different game — z1Nny's leg was graded
            # 44 off a morning match when his real evening match (25) was
            # simply absent from bo3 (2026-07-27).
            if ts < start - 2 * 3600:
                continue
            gap = abs(ts - start)
            if best_gap is None or gap < best_gap:
                best, best_gap = mid, gap
        # A match too far from the prop's start time is a DIFFERENT game.
        # The window was 12h, which silently rejected REAL results: bo3.gg
        # timestamps run hours ahead of true UTC (Salazar's settled match sat
        # at a 21h apparent gap — ~13h real plus ~8h of source skew, found
        # 2026-07-27). 26h absorbs the skew; the closest-match rule above
        # still picks the right game on double-header days because both of a
        # player's matches carry the same skew.
        if best is not None and best_gap is not None and best_gap <= 26 * 3600:
            match_id = best
        else:
            return "pending", None
    else:
        match_id = mine[0][4]  # no archived prop — fall back to earliest
    maps = [(mn, k, h) for _p, mn, k, h, mid, _at in mine if mid == match_id]
    n_played = max(mn for mn, _k, _h in maps)
    lo_i, hi_i = int(lo), int(hi)
    if n_played < lo_i:
        return "void", None  # range never started (e.g. map 3 in a sweep)
    in_range = [m for m in maps if lo_i <= m[0] <= min(hi_i, n_played)]
    total = float(
        sum((m[1] if stat == "kills" else (m[2] or 0)) for m in in_range)
    )
    if total == line:
        return "void", total  # exact landing on a whole line -> push
    over = total > line
    won = over if side == "over" else not over
    return ("won" if won else "lost"), total


def product_of(conn: sqlite3.Connection, slip_id: str) -> str:
    """"power" | "flex" for a tracked slip.

    Stored per slip because the two settle differently: a power slip dies on
    its first lost leg, a flex slip merely steps down a tier. Slips tracked
    before the column existed are power — that is all this app placed.
    """
    row = conn.execute(
        "SELECT product FROM slips WHERE slip_id=?", (slip_id,)
    ).fetchone()
    return str(row[0]) if row and row[0] else "power"


def manual_grade_leg(
    conn: sqlite3.Connection,
    slip_id: str,
    leg_no: int,
    observed: float | None,
    dnp: bool = False,
) -> str:
    """Grade one leg from a number the user read off the book's own app.

    Exists because auto-grading has hard limits found live 2026-07-27:
    bo3.gg does not cover every tier-C/qualifier event the books post
    (three settled legs had no results to grade against at all), and a
    player the book voids as "did not play" never gets a row, so his leg
    would sit pending forever. The book's app shows the real number either
    way — this records it. DNP grades as void, a total exactly on the line
    as a push (void), otherwise won/lost by side.
    """
    row = conn.execute(
        "SELECT side, line, status FROM slip_legs WHERE slip_id=? AND leg_no=?",
        (slip_id, leg_no),
    ).fetchone()
    if row is None:
        raise ValueError(f"no leg {slip_id}/{leg_no}")
    side, line, cur = str(row[0]), float(row[1]), str(row[2])
    if cur != "pending":
        return cur  # never silently overwrite an auto-graded result
    if dnp or observed is None:
        status: str = "void"
        observed = None
    elif observed == line:
        status = "void"  # push
    elif observed > line:
        status = "won" if side == "over" else "lost"
    else:
        status = "won" if side == "under" else "lost"
    conn.execute(
        "UPDATE slip_legs SET status=?, observed=? WHERE slip_id=? AND leg_no=?",
        (status, observed, slip_id, leg_no),
    )
    conn.commit()
    log.info("manually graded %s/%s -> %s (obs %s)",
             slip_id, leg_no, status, observed)
    grade_open_slips(conn)  # settle the slip if that was its last leg
    return status


def grade_open_slips(conn: sqlite3.Connection) -> int:
    """Grade all pending legs; settle slips whose legs are terminal.
    Returns number of slips settled this run."""
    settled = 0
    # Grade every ungraded leg, INCLUDING legs of slips already settled.
    # Early power settlement (a slip dies on its first lost leg) would
    # otherwise orphan the rest of that slip's legs forever — and those legs
    # are the live-calibration sample. Orphaning them biases the leg rate
    # DOWNWARD, because a slip settles early precisely when legs are losing:
    # measured 2026-07-26, the pooled rate fell 58.0% -> 52.6% purely from
    # dropping the ungraded remainder of three dead slips.
    for (sid, placed) in conn.execute(
        "SELECT DISTINCT s.slip_id, s.placed_at FROM slips s"
        " JOIN slip_legs l ON l.slip_id = s.slip_id"
        " WHERE s.status != 'pending' AND l.status = 'pending'"
    ).fetchall():
        for leg in conn.execute(
            "SELECT * FROM slip_legs WHERE slip_id=? AND status='pending'"
            " ORDER BY leg_no", (sid,),
        ).fetchall():
            status, observed = _grade_leg(conn, placed, leg)
            if status != "pending":
                conn.execute(
                    "UPDATE slip_legs SET status=?, observed=?"
                    " WHERE slip_id=? AND leg_no=?",
                    (status, observed, sid, leg[1]),
                )
                log.info("graded orphaned leg %s/%s -> %s", sid, leg[1], status)
    conn.commit()

    slips = conn.execute(
        "SELECT slip_id, book, placed_at, stake, n_legs, multiplier FROM slips "
        "WHERE status = 'pending'"
    ).fetchall()
    for slip_id, book, placed_at, stake, n_legs, stored_mult in slips:
        legs = conn.execute(
            "SELECT * FROM slip_legs WHERE slip_id = ? ORDER BY leg_no",
            (slip_id,),
        ).fetchall()
        for leg in legs:
            if leg[8] != "pending":
                continue
            status, observed = _grade_leg(conn, placed_at, leg)
            if status != "pending":
                conn.execute(
                    "UPDATE slip_legs SET status=?, observed=? "
                    "WHERE slip_id=? AND leg_no=?",
                    (status, observed, slip_id, leg[1]),
                )
        statuses = [
            r[0] for r in conn.execute(
                "SELECT status FROM slip_legs WHERE slip_id=?", (slip_id,)
            )
        ]
        if "pending" in statuses:
            # A POWER slip is decided the instant one leg loses — every leg
            # must hit, so the remaining ones cannot rescue it. Waiting for
            # them costs real things: P&L reads stale, and the held-player
            # exclusion keeps blocking that leg's MATCH from the scanner long
            # after the bet is dead. Observed 2026-07-26: two 4-picks with
            # two and three lost legs sat "pending" on a leg whose match had
            # not been published yet, and their match stayed excluded from
            # the board the whole time.
            # FLEX cannot settle early — losing a leg only drops it to a
            # lower paying tier — so this applies to power only.
            if product_of(conn, slip_id) != "power" or "lost" not in statuses:
                conn.commit()
                continue
            conn.execute(
                "UPDATE slips SET status='lost', payout=0.0 WHERE slip_id=?",
                (slip_id,),
            )
            log.info(
                "settled %s early: power slip with a lost leg cannot recover "
                "(%d leg(s) still ungraded)", slip_id, statuses.count("pending"),
            )
            settled += 1
            conn.commit()
            continue
        n_void = statuses.count("void")
        n_live = len(statuses) - n_void
        if product_of(conn, slip_id) == "flex":
            # FLEX pays on tiers, not all-or-nothing. Found live 2026-07-27:
            # a 5-pick flex that went 3/5 was settled here as "lost $0.00"
            # while PrizePicks paid $0.40 — the slip had been stored as
            # "power" (the UI never passed the product) and the power path
            # early-settled it on its first lost leg. A flex slip settles
            # only when every leg is terminal, and pays the tier its LIVE
            # wins land in (voids shrink the ladder, mirroring the book).
            wins = statuses.count("won")
            if n_live < 2:
                new_status, payout = "won", stake  # refund
            else:
                pay = load_payouts(book)
                if n_live in pay.flex:
                    tier = pay.flex_multiplier(n_live, wins)
                else:
                    # Voids can shrink a flex below the smallest flex tier
                    # (PrizePicks has none under 3 picks). The book then
                    # converts it to a standard all-must-hit play at that
                    # size — not zero, which is what a blind table lookup
                    # returns.
                    tier = (pay.power.get(n_live, 0.0)
                            if wins == n_live else 0.0)
                payout = stake * tier
                # A partial-return tier below the stake (e.g. 3/5 -> 0.4x)
                # is labeled "Win" in the app but is a net LOSS. Counting it
                # as a W would inflate the record; the payout row keeps the
                # P&L honest either way.
                new_status = "won" if payout >= stake else "lost"
        elif "lost" in statuses:
            new_status, payout = "lost", 0.0
        elif n_live < 2:
            new_status, payout = "won", stake  # too few live legs -> refund
        else:
            # the multiplier recorded at placement is authoritative — Arena
            # prices per-pick, so the config table cannot reproduce it
            mult = (
                stored_mult if stored_mult and n_live == n_legs
                else load_payouts(book).power.get(n_live, 0.0)
            )
            new_status, payout = "won", stake * mult
        conn.execute(
            "UPDATE slips SET status=?, payout=? WHERE slip_id=?",
            (new_status, payout, slip_id),
        )
        settled += 1
        conn.commit()
    return settled


def summary(conn: sqlite3.Connection) -> str:
    """Per-book results, plus a POOLED leg hit rate.

    The split matters: a PrizePicks 4-pick wins ~17% of the time and an
    Underdog 2-pick ~48%. Pooling their win rates produces a number that
    describes neither product, and pooling P&L hides which book is working.
    Payout ladders and lines differ per book, so anything money- or
    price-related is reported separately.

    Leg hit rate is the exception and is deliberately POOLED: it tests the
    PROJECTION MODEL, which is book-agnostic — the same probability estimate
    is used for both. Pooling it is the fastest honest read on calibration,
    and it accumulates ~4x faster than slip counts.
    """
    lines: list[str] = []
    books = [r[0] for r in conn.execute(
        "SELECT DISTINCT book FROM slips ORDER BY book")]
    for book in books:
        rows = conn.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(stake),0),"
            " COALESCE(SUM(payout),0) FROM slips WHERE book=? GROUP BY status",
            (book,),
        ).fetchall()
        by = {r[0]: r for r in rows}
        won = by.get("won", (0, 0, 0, 0))[1]
        lost = by.get("lost", (0, 0, 0, 0))[1]
        open_ = by.get("pending", (0, 0, 0, 0))[1]
        settled = won + lost
        staked = sum(r[2] for r in rows if r[0] != "pending")
        ret = sum(r[3] for r in rows if r[0] != "pending")
        sizes = sorted({r[0] for r in conn.execute(
            "SELECT n_legs FROM slips WHERE book=?", (book,))})
        tag = "/".join(f"{n}-pick" for n in sizes)
        head = (f"{book} ({tag}): {open_} open, {settled} settled "
                f"({won}W/{lost}L)")
        if settled:
            head += f" · ${staked:.2f} -> ${ret:.2f} (P&L {ret - staked:+.2f})"
        lines.append(head)
        claimed = conn.execute(
            "SELECT AVG(claimed_p) FROM slips WHERE book=? AND status!='pending'"
            " AND claimed_p IS NOT NULL", (book,)
        ).fetchone()[0]
        if claimed and settled:
            sd = math.sqrt(claimed * (1 - claimed) / settled)
            lines.append(
                f"    claimed {claimed:.1%} -> actual {won}/{settled} "
                f"({won / settled:.1%}); 90% band "
                f"{max(claimed - 1.645 * sd, 0):.1%}-"
                f"{claimed + 1.645 * sd:.1%}"
            )
    # pooled leg calibration — the model is the same across books
    legs = conn.execute(
        "SELECT l.status, COUNT(*) FROM slip_legs l JOIN slips s"
        " ON s.slip_id=l.slip_id WHERE s.status!='pending' GROUP BY l.status"
    ).fetchall()
    d = dict(legs)
    live = d.get("won", 0) + d.get("lost", 0)
    if live:
        lines.append(
            f"LEG hit rate (both books, tests the model): "
            f"{d.get('won', 0)}/{live} = {d.get('won', 0) / live:.1%}"
        )
    return "\n".join(lines) if lines else "no slips tracked yet"


# Everything tracked before this instant was a different experiment: 4-pick
# POWER on PrizePicks (the worst tier on the board, abandoned 0W/12L) and
# 2-pick Underdog, selected by a model that still carried the whole-number
# push bug (+6.1pt on exactly the legs it favoured) and graded by a tracker
# that could read the wrong match. The fixes and the product switch all
# landed 2026-07-26; the first slip placed under the corrected setup arrived
# 22:37 UTC that day. Records and the learned haircut start fresh from here —
# but the legacy rows are NEVER deleted: the -$12 was real money, and the
# report still shows it as a separate line.
TRACKING_EPOCH = 1_785_105_000.0  # 2026-07-26 ~22:30 UTC


@dataclass(frozen=True)
class BookStat:
    """One book's record. Never pooled across books: a PrizePicks 4-pick
    POWER wins ~10% of the time and an Underdog 3-pick ~26%, so a combined
    win rate describes neither, and combined P&L hides which is working."""

    book: str
    sizes: str          # "3-pick" / "2-pick/3-pick"
    n_open: int
    won: int
    lost: int
    staked: float
    returned: float
    claimed: float | None   # mean claimed P(win) over settled slips
    # Settled FLEX slips by "wins of live legs" — e.g. "3/5 ×2 · 2/3 ×1".
    # W/L alone hides flex economics: a 3-of-5 returning $0.40 and a 1-of-5
    # wipeout are both "L", and the tier distribution is exactly what the
    # flex pricing predicts, so showing it is a calibration readout.
    tiers: str = ""

    @property
    def settled(self) -> int:
        return self.won + self.lost

    @property
    def pnl(self) -> float:
        return self.returned - self.staked

    @property
    def actual(self) -> float | None:
        return self.won / self.settled if self.settled else None


def summary_rows(
    conn: sqlite3.Connection,
    since: float = TRACKING_EPOCH,
) -> tuple[list[BookStat], tuple[int, int], str]:
    """(per-book records since the epoch, (legs won, legs graded), legacy).

    Structured twin of :func:`summary`, so the dashboard can lay this out as
    a table instead of a 300-character run-on sentence. ``legacy`` is a
    one-line accounting of everything before the epoch — shown, not hidden,
    because the money was real even though the products were abandoned.
    """
    out: list[BookStat] = []
    for (book,) in conn.execute(
        "SELECT DISTINCT book FROM slips WHERE placed_at >= ? ORDER BY book",
        (since,),
    ).fetchall():
        rows = conn.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(stake),0),"
            " COALESCE(SUM(payout),0) FROM slips WHERE book=?"
            " AND placed_at >= ? GROUP BY status",
            (book, since),
        ).fetchall()
        by = {r[0]: r for r in rows}
        # The label is the book's CURRENT strategy, not the historical mix.
        # "4-pick/5-pick" was every size ever tracked — which read as if the
        # app still bets 4-pick power, the product whose 0W/12L is the reason
        # it does not. The record spans old products; the label should say
        # what a new slip will be.
        try:
            from cs2props.config import load_restrictions

            r_ = load_restrictions(book)
            sizes_label = f"{r_.default_slip_size}-man {r_.default_product}"
        except Exception:  # unknown book in the DB — fall back to history
            ns = sorted({r[0] for r in conn.execute(
                "SELECT n_legs FROM slips WHERE book=?", (book,))})
            sizes_label = "/".join(f"{n}-pick" for n in ns)
        claimed = conn.execute(
            "SELECT AVG(claimed_p) FROM slips WHERE book=? AND"
            " status!='pending' AND claimed_p IS NOT NULL"
            " AND placed_at >= ?", (book, since)
        ).fetchone()[0]
        tier_counts: dict[str, int] = {}
        for (sid,) in conn.execute(
            "SELECT slip_id FROM slips WHERE book=? AND status!='pending'"
            " AND product='flex' AND placed_at >= ?", (book, since)
        ).fetchall():
            st = [r[0] for r in conn.execute(
                "SELECT status FROM slip_legs WHERE slip_id=?", (sid,))]
            live = sum(1 for x in st if x in ("won", "lost"))
            wins = st.count("won")
            key = f"{wins}/{live}"
            tier_counts[key] = tier_counts.get(key, 0) + 1
        tiers = " · ".join(
            f"{k} ×{n}" for k, n in sorted(tier_counts.items(), reverse=True)
        )
        out.append(BookStat(
            book=book,
            sizes=sizes_label,
            tiers=tiers,
            n_open=by.get("pending", (0, 0, 0, 0))[1],
            won=by.get("won", (0, 0, 0, 0))[1],
            lost=by.get("lost", (0, 0, 0, 0))[1],
            staked=sum(r[2] for r in rows if r[0] != "pending"),
            returned=sum(r[3] for r in rows if r[0] != "pending"),
            claimed=claimed,
        ))
    d = dict(conn.execute(
        "SELECT l.status, COUNT(*) FROM slip_legs l JOIN slips s"
        " ON s.slip_id=l.slip_id WHERE s.status!='pending'"
        " AND s.placed_at >= ? GROUP BY l.status", (since,)
    ).fetchall())
    leg = conn.execute(
        "SELECT COALESCE(SUM(s.stake),0), COALESCE(SUM(s.payout),0),"
        " COUNT(*), SUM(s.status='won')"
        " FROM slips s WHERE s.status!='pending' AND s.placed_at < ?",
        (since,),
    ).fetchone()
    legacy = ""
    if leg and leg[2]:
        legacy = (
            f"legacy (pre-fix 4-pick/2-pick era, before 2026-07-26): "
            f"{leg[3] or 0}W/{leg[2] - (leg[3] or 0)}L, "
            f"P&L {leg[1] - leg[0]:+.2f} — excluded from the table above"
        )
    return (out, (d.get("won", 0), d.get("won", 0) + d.get("lost", 0)),
            legacy)


def tracked_for_report(conn: sqlite3.Connection, limit: int = 20) -> list[Any]:
    """Placed slips + their legs, newest first, for the HTML report.

    Suggested slips churn on every scan; a bet the user actually placed must
    stay visible regardless of what the board does afterwards.
    """
    from cs2props.report import TrackedLeg, TrackedSlip

    out: list[Any] = []
    rows = conn.execute(
        "SELECT slip_id, book, n_legs, stake, multiplier, claimed_p, status,"
        " payout FROM slips ORDER BY placed_at DESC LIMIT ?", (limit,)
    ).fetchall()
    # CLV per leg: closing-line value is the fastest signal that the model
    # actually beats the market, so surface it next to each placed leg.
    from cs2props.clv import leg_clvs

    clv_by_slip: dict[str, list[Any]] = {}
    for r in leg_clvs(conn):
        clv_by_slip.setdefault(r.slip_id, []).append(r)

    for sid, book, n_legs, stake, mult, cp, status, payout in rows:
        clv_rows = {
            (clean_name(c.player), c.side.lower()): c
            for c in clv_by_slip.get(sid, [])
        }
        legs = []
        for leg_no, r in enumerate(conn.execute(
            "SELECT side, player_name, line, stat_kind, status, observed"
            " FROM slip_legs WHERE slip_id=? ORDER BY leg_no", (sid,)
        )):
            c = clv_rows.get((clean_name(r[1]), r[0].lower()))
            legs.append(TrackedLeg(
                side=r[0], player=r[1], line=float(r[2]), stat=r[3],
                status=r[4], observed=r[5],
                clv=c.clv if c else None,
                closing_line=c.closing_line if c else None,
                slip_id=sid, leg_no=leg_no,
            ))
        have = [l.clv for l in legs if l.clv is not None]
        slip_clv = sum(have) / len(have) if have else None
        out.append(TrackedSlip(
            slip_id=sid, book=book, n_legs=int(n_legs), stake=float(stake),
            multiplier=float(mult) if mult else None,
            claimed_p=float(cp) if cp else None, status=status,
            payout=float(payout) if payout is not None else None,
            legs=tuple(legs), clv=slip_clv,
        ))
    return out


def committed_players(conn: sqlite3.Connection) -> set[str]:
    """Cleaned names of every player in an OPEN slip.

    A player you already hold must not reappear in a new suggestion: two
    slips sharing a leg are not two independent bets. They fail together,
    raise variance for the same stake, and halve the effective sample the
    live-calibration line is built from. Settled slips are excluded — once a
    match is over the player is free again.
    """
    return {
        clean_name(r[0])
        for r in conn.execute(
            "SELECT l.player_name FROM slip_legs l JOIN slips s"
            " ON s.slip_id = l.slip_id WHERE s.status = 'pending'"
        )
    }
