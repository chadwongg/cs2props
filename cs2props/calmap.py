"""Probability calibration map: what a raw model probability REALLY hits.

The flat haircut assumed the model's error was a constant offset. The
real-line archive says otherwise: bucketing 5,490 settled kills lines by the
model's claimed pick probability showed ~54% realized whether the model
claimed 57% or 72% — the confidence above the threshold carries almost no
information, and the optimizer selects precisely that empty tail. A flat
subtraction (capped at 12pt) cannot represent a 13-18pt distortion that
GROWS with claimed confidence.

So the EV pipeline prices legs at f(p): the realized hit rate of archive
picks that claimed p, fitted as a shrunk, monotone, one-sided map:

- shrunk: each bucket's realized rate is blended toward the raw claim by
  sample size, so thin buckets do not swing the map;
- monotone: pool-adjacent-violators, because "more confident should not
  mean less likely" is enforced rather than assumed;
- one-sided: f(p) <= p always. The map may only DISCOUNT — a bucket that
  overperformed its claim is capped at the claim, for the same reason the
  adaptive haircut floors at zero: betting more because a slice of the
  sample looked good is how bankrolls die.

Fitting walks the whole archive (~seconds), so the map is fitted weekly by
the Sunday automation (`cs2props calmap`) and persisted to
calibration_map.json; the scan loads the file. The adaptive haircut remains
on the dashboard as a live-vs-claimed diagnostic, but no longer prices EV —
applying both would double-count the same error.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MAP_PATH = Path("calibration_map.json")
# raw pick-probability bucket edges; the region the optimizer selects from
EDGES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0)
BUCKET_PRIOR_N = 50.0  # pseudo-lines pulling a bucket toward its raw claim
STALE_AFTER_S = 21 * 86400.0
MIN_LINES_TO_FIT = 300


@dataclass(frozen=True)
class CalibrationMap:
    """Piecewise-linear map raw pick probability -> realized probability."""

    knots: tuple[tuple[float, float], ...]  # (raw, calibrated), raw ascending
    n_lines: int
    fitted: float  # epoch
    when: str

    def apply(self, p: float) -> float:
        """Calibrated probability for a raw pick probability. f(p) <= p."""
        if not self.knots or p <= 0.5:
            return p
        pts = self.knots
        if p <= pts[0][0]:
            cal = pts[0][1]
        elif p >= pts[-1][0]:
            cal = pts[-1][1]
        else:
            cal = p
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x0 <= p <= x1:
                    t = (p - x0) / (x1 - x0) if x1 > x0 else 0.0
                    cal = y0 + t * (y1 - y0)
                    break
        return min(cal, p)

    def delta(self, p: float) -> float:
        """The per-leg discount this map applies at raw probability p."""
        return p - self.apply(p)

    @property
    def stale(self) -> bool:
        return time.time() - self.fitted > STALE_AFTER_S

    def describe(self) -> str:
        mid = self.apply(0.65)
        return (f"calibration map {self.when} ({self.n_lines} lines, "
                f"raw 65% -> {mid:.0%})")


def fit(conn: sqlite3.Connection) -> CalibrationMap | None:
    """Fit the map from the real-line archive (kills, both books).

    Every settled line contributes its PICK side — the side the model put
    over 50% on — because that is the population the optimizer draws from.
    Returns None when the archive is too thin to say anything.
    """
    from cs2props.model.reallines import run_real_line_backtest

    res = run_real_line_backtest(conn)
    rows = [(max(r.p_over, 1 - r.p_over),
             r.won_over if r.p_over >= 0.5 else not r.won_over)
            for r in res.rows if r.stat == "kills"]
    if len(rows) < MIN_LINES_TO_FIT:
        log.warning("calibration map: only %d lines — not fitting", len(rows))
        return None

    knots: list[tuple[float, float]] = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        b = [(q, won) for q, won in rows if lo <= q < hi]
        if not b:
            continue
        raw = sum(q for q, _ in b) / len(b)
        realized = sum(won for _, won in b) / len(b)
        # shrink toward the raw claim by sample size
        w = len(b) / (len(b) + BUCKET_PRIOR_N)
        cal = w * realized + (1 - w) * raw
        knots.append((raw, cal))

    # isotonic in the calibrated value: pool adjacent violators
    pooled: list[list[float]] = []  # [sum_raw, sum_cal, n]
    for raw, cal in knots:
        pooled.append([raw, cal, 1.0])
        while len(pooled) > 1 and (pooled[-2][1] / pooled[-2][2]
                                   > pooled[-1][1] / pooled[-1][2]):
            a = pooled.pop()
            pooled[-1] = [pooled[-1][0] + a[0], pooled[-1][1] + a[1],
                          pooled[-1][2] + a[2]]
    mono = [(sr / n, sc / n) for sr, sc, n in pooled]

    return CalibrationMap(
        knots=tuple((raw, min(cal, raw)) for raw, cal in mono),
        n_lines=len(rows),
        fitted=time.time(),
        when=time.strftime("%Y-%m-%d"),
    )


def save(cm: CalibrationMap, path: Path = MAP_PATH) -> None:
    path.write_text(json.dumps({
        "when": cm.when, "fitted": cm.fitted, "n_lines": cm.n_lines,
        "knots": [list(k) for k in cm.knots],
    }, indent=2))


def load(path: Path = MAP_PATH) -> CalibrationMap | None:
    """Load the persisted map; None if absent or unreadable.

    A stale map is still returned (with a log warning): a three-week-old
    calibration beats pretending the model is honest.
    """
    try:
        raw = json.loads(path.read_text())
        cm = CalibrationMap(
            knots=tuple((float(a), float(b)) for a, b in raw["knots"]),
            n_lines=int(raw["n_lines"]),
            fitted=float(raw["fitted"]),
            when=str(raw["when"]),
        )
    except (OSError, ValueError, KeyError):
        return None
    if cm.stale:
        log.warning("calibration map is stale (%s) — refit with "
                    "`cs2props calmap`", cm.when)
    return cm
