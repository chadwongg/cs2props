"""Backtest against REAL PrizePicks lines, not synthetic ones.

The existing backtest (:mod:`cs2props.model.backtest`) scores the projector
against lines it draws itself, at fixed offsets around its own predictive
median. That is the right test for one question — "is the distribution
honestly shaped?" — and it passes it: log loss 0.6365 against a 0.6927
base-rate baseline, reliability within a point across 90,000 observations.

It is the WRONG test for the question that decides whether this app makes
money. A book's line is not a neutral point near the median; it is set by
people paid to set it, and it carries a shade that a self-drawn line never
does. Beating your own line proves calibration. Beating PrizePicks' line
proves edge. Those are different claims, and only the second one pays.

This module answers the second one. Every standard-board projection the app
has ever imported is stored with its line and its timestamp, so once the
match settles the pair (real line, real outcome) is recoverable. State is
walked forward exactly as in the synthetic backtest — predict strictly
before the player's own maps are folded in — so no result informs the
prediction that scored it.

What it CANNOT tell you yet is anything about the long run: the prop archive
starts 2026-07-24, so the sample is days, not months, and every number here
carries an enormous standard error. It is a live instrument that sharpens as
the archive grows, not a verdict.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from cs2props.model import projector as pj
from cs2props.model.backtest import BIAS_CLIP, BIAS_HALF_LIFE_SERIES
from cs2props.model.state_builder import clean_name

log = logging.getLogger(__name__)

N_SAMPLES = 20_000
MIN_HISTORY_MAPS = 20
# Headshots are drawn as a beta-binomial thinning of the kill samples, the
# same construction the correlation engine uses. Kept identical on purpose:
# a backtest that models headshots differently from the live path would be
# measuring a model the app does not actually bet.
HS_DISPERSION_K = 144.0


@dataclass
class Scored:
    """One real line, priced before the match and graded after."""

    player: str
    stat: str
    book: str
    line: float
    p_over: float
    observed: float
    won_over: bool
    map_lo: int
    map_hi: int
    played_at: str
    tier: str | None


@dataclass
class RealLineResult:
    rows: list[Scored] = field(default_factory=list)
    skipped_no_history: int = 0
    skipped_unplayed: int = 0
    skipped_void: int = 0

    @property
    def n(self) -> int:
        return len(self.rows)

    def log_loss(self) -> float:
        if not self.rows:
            return float("nan")
        tot = 0.0
        for r in self.rows:
            p = min(max(r.p_over, 1e-9), 1 - 1e-9)
            tot += -(math.log(p) if r.won_over else math.log(1 - p))
        return tot / len(self.rows)

    def baseline_log_loss(self) -> float:
        """Log loss of always predicting the observed over-rate.

        This is the bar that matters. A book's line sits near a coin flip by
        construction, so the base rate is ~0.5 and the baseline is ~0.693.
        Failing to beat it means the model adds nothing to the line.
        """
        if not self.rows:
            return float("nan")
        rate = sum(r.won_over for r in self.rows) / len(self.rows)
        rate = min(max(rate, 1e-9), 1 - 1e-9)
        tot = 0.0
        for r in self.rows:
            tot += -(math.log(rate) if r.won_over else math.log(1 - rate))
        return tot / len(self.rows)

    def over_rate(self) -> float:
        return (
            sum(r.won_over for r in self.rows) / len(self.rows)
            if self.rows else float("nan")
        )

    def under_rate(self) -> tuple[float, int, int]:
        """(under win rate on LIVE legs, n_live, n_pushed).

        NOT ``1 - over_rate``. A whole-number line that lands exactly on the
        total is a PUSH: the book voids the leg, so it is neither an over win
        nor an under win. Computing the under rate by subtraction credits
        every push to the under side — the same error that inflated the
        engine's whole-number under legs by 6.1 points, repeated here in the
        measurement used to judge whether unders beat the book at all.
        """
        if not self.rows:
            return float("nan"), 0, 0
        pushed = sum(1 for r in self.rows if r.observed == r.line)
        wins = sum(1 for r in self.rows if r.observed < r.line)
        live = len(self.rows) - pushed
        return (wins / live if live else float("nan")), live, pushed

    def mean_pred(self) -> float:
        return (
            sum(r.p_over for r in self.rows) / len(self.rows)
            if self.rows else float("nan")
        )

    def picks_at(self, threshold: float) -> tuple[int, int, float]:
        """(n_picks, n_won, hit_rate) taking the side the model prefers.

        Only lines where the model's conviction clears ``threshold`` count —
        this is the closest thing to "would the legs the optimizer actually
        chooses have hit?", since MIN_LEG_P gates candidates the same way.
        """
        picks = [
            r for r in self.rows
            if r.p_over >= threshold or (1 - r.p_over) >= threshold
        ]
        won = sum(
            (r.won_over if r.p_over >= threshold else not r.won_over)
            for r in picks
        )
        rate = won / len(picks) if picks else float("nan")
        return len(picks), won, rate

    def reliability(self, n_bins: int = 5) -> list[tuple[float, float, int]]:
        buckets: dict[int, list[Scored]] = defaultdict(list)
        for r in self.rows:
            buckets[min(int(r.p_over * n_bins), n_bins - 1)].append(r)
        out = []
        for b in sorted(buckets):
            rs = buckets[b]
            out.append((
                sum(r.p_over for r in rs) / len(rs),
                sum(r.won_over for r in rs) / len(rs),
                len(rs),
            ))
        return out

    def by_book(self) -> dict[str, tuple[int, float, float]]:
        """book -> (n, mean predicted P(over), actual over-rate)."""
        groups: dict[str, list[Scored]] = defaultdict(list)
        for r in self.rows:
            groups[r.book].append(r)
        return {
            k: (len(v),
                sum(r.p_over for r in v) / len(v),
                sum(r.won_over for r in v) / len(v))
            for k, v in sorted(groups.items())
        }

    def by_stat(self) -> dict[str, tuple[int, float, float]]:
        """stat -> (n, mean predicted P(over), actual over-rate)."""
        groups: dict[str, list[Scored]] = defaultdict(list)
        for r in self.rows:
            groups[r.stat].append(r)
        return {
            k: (len(v),
                sum(r.p_over for r in v) / len(v),
                sum(r.won_over for r in v) / len(v))
            for k, v in sorted(groups.items())
        }


def _epoch(ts: str) -> float | None:
    """ISO timestamp -> UTC epoch seconds.

    Props carry a -04:00 offset and player_maps carry +00:00, so ANY
    date-string comparison between them is wrong across the evening
    boundary. Both sides are normalised to UTC epoch instead. This is the
    same class of bug that graded slips against players' previous matches on
    2026-07-24 (SQLite's space-separated datetime() vs ISO 'T'); it is not
    allowed to recur in a module whose whole job is deciding whether the
    model works.
    """
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


# A match's maps are logged as they finish, so a map's played_at runs LATER
# than the scheduled start. A prop belongs to the series whose first map
# falls inside this window after its start time.
MATCH_WINDOW_BEFORE = 2 * 3600.0
MATCH_WINDOW_AFTER = 10 * 3600.0


PRIZEPICKS_LEAGUE = "265"


def _real_lines(
    conn: sqlite3.Connection, book: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """clean player name -> the FIRST line seen for each distinct prop.

    First-seen rather than closing: that is the line the scanner prices and
    the bettor acts on. Grading against the closing line would test a
    decision nobody makes and would flatter or punish the model for line
    movement it never saw.

    MUST be grouped by book. Without ``league_id`` in the GROUP BY, a prop
    both books posted collapsed into one row and SQLite returned an
    ARBITRARY book's line_score for it — so results attributed to
    "PrizePicks lines" were a silent mix of two books at whichever number
    the query happened to pick. Books post systematically different lines
    (PrizePicks runs +0.108 higher on kills), so that is not a rounding
    difference; it is grading against a line nobody offered.
    """
    rows = conn.execute(
        """
        SELECT player_name, stat_kind, map_lo, map_hi, line_score,
               start_time, league_id, MIN(scanned_at) AS first_seen
        FROM props
        WHERE board = 'standard' AND start_time IS NOT NULL
          AND map_lo IS NOT NULL AND map_hi IS NOT NULL
        GROUP BY player_name, stat_kind, map_lo, map_hi, start_time,
                 league_id, line_score
        ORDER BY start_time
        """
    ).fetchall()
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, stat, lo, hi, line, start, lg, seen in rows:
        t = _epoch(str(start))
        if t is None:
            continue
        bk = "prizepicks" if str(lg) == PRIZEPICKS_LEAGUE else "underdog"
        if book is not None and bk != book:
            continue
        out[clean_name(name)].append({
            "stat": stat, "lo": int(lo), "hi": int(hi),
            "line": float(line), "start_ts": t, "seen": float(seen),
            "player": name, "book": bk, "used": False,
        })
    return out


def _observed(maps: list[Any], stat: str, lo: int, hi: int) -> float | None:
    """Total for the map range, or None if the range never completed.

    A prop on maps 1-2 that only played one map is a VOID at the book, not a
    loss — scoring it as a loss would manufacture failures the bettor never
    suffered, which is exactly the bug that fabricated eight graded losses in
    the tracker on 2026-07-24.
    """
    played = max(m[1] for m in maps)
    if played < hi:
        return None
    in_range = [m for m in maps if lo <= m[1] <= hi]
    if len(in_range) != (hi - lo + 1):
        return None
    idx = 9 if stat == "kills" else 11  # kills / headshots column
    vals = [m[idx] for m in in_range]
    if any(v is None for v in vals):
        return None
    return float(sum(vals))


def run_real_line_backtest(
    conn: sqlite3.Connection,
    min_history: int = MIN_HISTORY_MAPS,
    seed: int = 7,
    book: str | None = None,
) -> RealLineResult:
    """Score every stored line whose match has since settled.

    ``book`` restricts to one book's lines. Leaving it None mixes two books
    that post measurably different numbers, so the per-book runs are the
    meaningful ones.
    """
    rng = np.random.default_rng(seed)
    lines = _real_lines(conn, book)
    log.info("real lines archived for %d players", len(lines))

    rows = conn.execute(
        """
        SELECT match_id, map_number, player_id, player_name, team, opponent,
               event_tier, map_name, played_at, kills, rounds, headshots
        FROM player_maps
        WHERE rounds IS NOT NULL AND rounds > 10
        ORDER BY played_at, match_id, map_number
        """
    ).fetchall()

    players: dict[str, pj.PlayerState] = defaultdict(pj.PlayerState)
    teams: dict[str, pj.TeamState] = defaultdict(pj.TeamState)
    league = pj.League()
    bias = pj.EwMean(BIAS_HALF_LIFE_SERIES)
    res = RealLineResult()

    series: dict[tuple[str, str], list[Any]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for r in rows:
        key = (r[0], r[2])
        if key not in series:
            order.append(key)
        series[key].append(r)

    for key in order:
        maps = sorted(series[key], key=lambda r: r[1])
        first = maps[0]
        pid, pname, opp = first[2], first[3], first[5]
        tier, played_at = first[6], str(first[8])
        pstate = players[pid]
        cname = clean_name(pname)

        # ---- price every archived line for this player-match, BEFORE the
        # player's own maps update state ----
        # Match on TIMESTAMP, and consume each prop exactly once. Many
        # players play two matches in a day (verified: dozens on 2026-07-25),
        # so a date-only join scores one line against both series — inventing
        # observations and corrupting the very statistic this module exists
        # to produce.
        played_ts = _epoch(played_at)
        cands = []
        if played_ts is not None:
            for ln in lines.get(cname, []):
                if ln["used"]:
                    continue
                delta = played_ts - ln["start_ts"]
                if -MATCH_WINDOW_BEFORE <= delta <= MATCH_WINDOW_AFTER:
                    cands.append(ln)
            for ln in cands:
                ln["used"] = True
        if cands:
            if pstate.n_maps < min_history:
                res.skipped_no_history += len(cands)
            else:
                correction = min(
                    max(bias.value or 1.0, BIAS_CLIP[0]), BIAS_CLIP[1]
                )
                hs_rate = pj.shrunk_hs_rate(pstate)
                for ln in cands:
                    lo, hi = ln["lo"], ln["hi"]
                    obs = _observed(maps, ln["stat"], lo, hi)
                    if obs is None:
                        res.skipped_void += 1
                        continue
                    rng_maps = [m for m in maps if lo <= m[1] <= hi]
                    kprs = [
                        pj.expected_kpr(
                            pstate, teams.get(opp or ""), league, m[7]
                        ) * correction
                        for m in rng_maps
                    ]
                    samples = pj.sample_series_kills(
                        kprs, league, rng, N_SAMPLES
                    )
                    if ln["stat"] == "headshots":
                        a = max(hs_rate * HS_DISPERSION_K, 1e-3)
                        b = max((1.0 - hs_rate) * HS_DISPERSION_K, 1e-3)
                        samples = rng.binomial(
                            samples.astype(int),
                            rng.beta(a, b, size=samples.shape[0]),
                        ).astype(float)
                    res.rows.append(Scored(
                        player=ln["player"], stat=ln["stat"],
                        book=ln["book"], line=ln["line"],
                        p_over=pj.p_over(samples, ln["line"]),
                        observed=obs, won_over=obs > ln["line"],
                        map_lo=lo, map_hi=hi, played_at=played_at, tier=tier,
                    ))

        # ---- update state (after prediction: walk-forward) ----
        maps12 = [m for m in maps if m[1] in (1, 2)]
        if maps12 and pstate.n_maps >= min_history:
            pred = pj.sample_series_kills(
                [pj.expected_kpr(pstate, teams.get(opp or ""), league, m[7])
                 for m in maps12], league, rng, 2000,
            ).mean()
            if pred > 1:
                obs12 = float(sum(m[9] for m in maps12))
                bias.update(min(max(obs12 / pred, 0.5), 2.0))
        for r in maps:
            kills, rounds = int(r[9]), int(r[10])
            players[pid].update(kills, rounds, r[7], r[11])
            league.update(kills / rounds, rounds)
            if opp:
                teams[opp].update_allowed(kills / rounds)

    res.skipped_unplayed = sum(
        1 for v in lines.values() for ln in v if not ln["used"]
    )
    return res


def format_report(res: RealLineResult) -> str:
    if not res.rows:
        return (
            "no settled real lines yet — the prop archive only covers matches "
            "that have not finished. Re-run after tonight's results land."
        )
    ll, base = res.log_loss(), res.baseline_log_loss()
    verdict = "BEATS" if ll < base else "LOSES TO"
    out = [
        "=== backtest against REAL PrizePicks lines ===",
        f"scored lines     : {res.n}",
        f"  skipped: {res.skipped_void} void/incomplete range, "
        f"{res.skipped_no_history} thin history, "
        f"{res.skipped_unplayed} not yet played",
        f"log loss         : {ll:.4f}",
        f"  baseline (rate): {base:.4f}  ({verdict} baseline)",
        f"actual over-rate : {res.over_rate():.1%}  "
        f"(model predicted {res.mean_pred():.1%} on average)",
        (lambda t: f"blind UNDER rate : {t[0]:.1%} on {t[1]} live legs "
                   f"({t[2]} pushed — NOT counted as under wins)")(
            res.under_rate()),
        "",
        "if the model PICKED a side at each conviction level:",
    ]
    for th in (0.55, 0.60, 0.65, 0.70):
        n, won, rate = res.picks_at(th)
        if n:
            se = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
            out.append(
                f"  p >= {th:.2f}: {won}/{n} = {rate:.1%}  (+/-{1.96 * se:.1%})"
            )
        else:
            out.append(f"  p >= {th:.2f}: no lines")
    out += ["", "reliability (pred -> actual, n):"]
    for p, y, n in res.reliability():
        out.append(f"  {p:5.2f} -> {y:5.2f}  n={n}")
    out += ["", "by BOOK (n, mean predicted P(over), actual over-rate):"]
    for bk, (n, pred, act) in res.by_book().items():
        out.append(f"  {bk:<12} n={n:<5} pred {pred:.1%}  actual {act:.1%}")
    out += ["", "by stat (n, mean predicted P(over), actual over-rate):"]
    for stat, (n, pred, act) in res.by_stat().items():
        out.append(f"  {stat:<10} n={n:<5} pred {pred:.1%}  actual {act:.1%}")
    out += [
        "",
        "READ THIS BEFORE BELIEVING ANY OF IT: the prop archive starts",
        "2026-07-24. This is days of data, not months. A 60% hit rate over",
        "100 lines has a 95% interval of roughly +/-10 points, which spans",
        "'losing' to 'printing'. The number to watch is whether it holds as",
        "the archive grows -- not what it says today.",
    ]
    return "\n".join(out)
