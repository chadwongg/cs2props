"""Walk-forward backtest: calibration of maps-1-2 kill projections.

Chronological replay of ``player_maps``. For every (match, player) where the
player has enough prior history, the model predicts the distribution of kills
across maps 1+2 *before* seeing the match, then the realized total is scored.
State updates strictly after prediction — no leakage.

Real PrizePicks lines are not available historically, so P(over) calibration
uses synthetic lines at fixed offsets around the predictive median — this
tests exactly the quantity the optimizer will consume ("is model P(over) x%
right x% of the time?") across the line range that matters. Distributional
calibration is additionally checked with randomized PIT.

Outputs: log loss vs baselines, Brier, reliability table, PIT histogram,
per-tier residuals (to check whether tier needs to be an explicit factor).
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cs2props.model import projector as pj

log = logging.getLogger(__name__)

LINE_OFFSETS = (-4.5, -1.5, 1.5, 4.5)
N_SAMPLES = 4000
MIN_HISTORY_MAPS = 20
BIAS_HALF_LIFE_SERIES = 150.0  # online mean-bias corrector (obs/pred ratio)
BIAS_CLIP = (0.85, 1.20)


@dataclass
class Calibration:
    """Aggregated backtest results."""

    n_series: int = 0
    preds: list[float] = field(default_factory=list)  # P(over) per synthetic line
    outcomes: list[int] = field(default_factory=list)
    pits: list[float] = field(default_factory=list)
    tier_residuals: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # -- metrics ----------------------------------------------------------

    def log_loss(self) -> float:
        p = np.clip(np.asarray(self.preds), 1e-6, 1 - 1e-6)
        y = np.asarray(self.outcomes)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def baseline_log_loss(self) -> float:
        """Log loss of always predicting the empirical base rate."""
        y = np.asarray(self.outcomes)
        base = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        return float(-np.mean(y * math.log(base) + (1 - y) * math.log(1 - base)))

    def brier(self) -> float:
        p = np.asarray(self.preds)
        y = np.asarray(self.outcomes)
        return float(np.mean((p - y) ** 2))

    def reliability(self, n_bins: int = 10) -> list[tuple[float, float, int]]:
        """[(mean predicted, empirical frequency, n)] per bin."""
        p = np.asarray(self.preds)
        y = np.asarray(self.outcomes)
        out = []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
            if mask.sum() == 0:
                continue
            out.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
        return out

    def pit_histogram(self, n_bins: int = 10) -> list[float]:
        hist, _ = np.histogram(self.pits, bins=n_bins, range=(0, 1))
        return list(hist / max(len(self.pits), 1))

    def tier_bias(self) -> dict[str, tuple[float, int]]:
        """Mean (observed - predicted-mean) kills by tier."""
        return {
            t: (float(np.mean(v)), len(v))
            for t, v in sorted(self.tier_residuals.items())
        }


def run_backtest(
    conn: sqlite3.Connection,
    min_history: int = MIN_HISTORY_MAPS,
    seed: int = 7,
) -> Calibration:
    rng = np.random.default_rng(seed)
    rows = conn.execute(
        """
        SELECT match_id, map_number, player_id, player_name, team, opponent,
               event_tier, map_name, played_at, kills, rounds
        FROM player_maps
        WHERE rounds IS NOT NULL AND rounds > 10
        ORDER BY played_at, match_id, map_number
        """
    ).fetchall()
    log.info("backtest over %d map-rows", len(rows))

    players: dict[str, pj.PlayerState] = defaultdict(pj.PlayerState)
    teams: dict[str, pj.TeamState] = defaultdict(pj.TeamState)
    league = pj.League()
    cal = Calibration()
    # Online mean-bias correction: EW mean of observed/predicted, updated
    # AFTER each prediction is scored — future predictions only ever see
    # past residuals, so walk-forward validity is preserved.
    bias = pj.EwMean(BIAS_HALF_LIFE_SERIES)

    # group rows into series: (match_id, player_id) -> list of map rows
    series: dict[tuple[str, str], list[Any]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for r in rows:
        key = (r[0], r[2])
        if key not in series:
            order.append(key)
        series[key].append(r)

    for key in order:
        maps = sorted(series[key], key=lambda r: r[1])
        maps12 = [r for r in maps if r[1] in (1, 2)]
        first = maps[0]
        (_mid, _mn, pid, _pname, _team, opp, tier, _map, _at, _k, _rounds) = first
        pstate = players[pid]

        # ---- predict before updating state (no leakage) ----
        if pstate.n_maps >= min_history and len(maps12) == 2:
            correction = min(max(bias.value or 1.0, BIAS_CLIP[0]), BIAS_CLIP[1])
            kprs = [
                pj.expected_kpr(pstate, teams.get(opp or ""), league, r[7])
                * correction
                for r in maps12
            ]
            samples = pj.sample_series_kills(kprs, league, rng, N_SAMPLES)
            observed = float(sum(r[9] for r in maps12))
            pred_mean = float(samples.mean())
            if pred_mean > 1:
                bias.update(min(max(observed / pred_mean, 0.5), 2.0))
            cal.n_series += 1
            cal.pits.append(pj.pit_value(samples, observed, rng))
            cal.tier_residuals[tier or "?"].append(observed - float(samples.mean()))
            median = float(np.median(samples))
            for off in LINE_OFFSETS:
                line = round(median + off) + 0.5
                cal.preds.append(pj.p_over(samples, line))
                cal.outcomes.append(int(observed > line))

        # ---- update state ----
        for r in maps:
            kills, rounds = int(r[9]), int(r[10])
            kpr_obs = kills / rounds
            players[pid].update(kills, rounds, r[7])
            league.update(kpr_obs, rounds)
            if opp:
                teams[opp].update_allowed(kpr_obs)

    return cal


def format_report(cal: Calibration) -> str:
    if not cal.preds:
        return "Not enough history for any prediction — let the backfill run longer."
    lines = [
        f"series predicted : {cal.n_series}",
        f"scored lines     : {len(cal.preds)}",
        f"log loss         : {cal.log_loss():.4f}",
        f"  baseline (rate): {cal.baseline_log_loss():.4f}"
        f"  ({'BEATS' if cal.log_loss() < cal.baseline_log_loss() else 'LOSES TO'}"
        " baseline)",
        f"brier score      : {cal.brier():.4f}",
        "",
        "reliability (pred -> actual, n):",
    ]
    for p, y, n in cal.reliability():
        bar = "#" * int(y * 40)
        lines.append(f"  {p:5.2f} -> {y:5.2f}  n={n:<5} {bar}")
    lines.append("")
    lines.append("PIT histogram (flat = calibrated distribution):")
    for i, frac in enumerate(cal.pit_histogram()):
        lines.append(f"  {i / 10:.1f}-{(i + 1) / 10:.1f}  {frac:5.3f} "
                     + "#" * int(frac * 200))
    lines.append("")
    lines.append("tier residuals (obs - pred mean kills, n):")
    for t, (bias, n) in cal.tier_bias().items():
        lines.append(f"  tier {t}: {bias:+6.2f}  n={n}")
    return "\n".join(lines)
