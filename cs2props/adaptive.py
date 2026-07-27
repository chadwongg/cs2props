"""Learn the model's optimism from live results instead of assuming it.

The optimizer discounts every leg by a haircut before deciding a slip is
worth betting. That haircut started as a fixed 2 points, derived from an
INTERNAL comparison (engine marginals vs the calibrated projector). Live
results tell a different story: the first 28 graded legs hit 57.1% against a
claimed ~62%, implying nearly 5 points — but with a standard error of ~9
points, which is far too noisy to act on.

So the haircut is estimated the honest way: shrink the observed gap toward
the prior by sample size, exactly as team map-win rates and headshot rates
are shrunk elsewhere in this model. With no data it returns the prior; with
a lot of data it converges on what actually happened; in between it moves
gradually and never lurches on a handful of legs.

Deliberately one-sided in effect but not in measurement: if the model turns
out to be PESSIMISTIC (legs hitting above claimed), the haircut shrinks
toward zero but is floored there. Betting more because a small sample looked
good is how bankrolls die.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

log = logging.getLogger(__name__)

PRIOR_HAIRCUT = 0.02  # the internal engine-vs-projector estimate
PRIOR_WEIGHT_LEGS = 120.0  # pseudo-legs of prior; ~SE 4.4pt at this size
MAX_HAIRCUT = 0.12  # never discount a leg by more than this
MIN_LEGS_TO_LEARN = 20


@dataclass(frozen=True)
class HaircutEstimate:
    haircut: float
    n_legs: int
    observed_rate: float | None
    claimed_rate: float | None
    prior_weight: float

    @property
    def source(self) -> str:
        if self.n_legs < MIN_LEGS_TO_LEARN:
            return f"prior only ({self.n_legs} legs graded)"
        pull = self.n_legs / (self.n_legs + self.prior_weight)
        return (
            f"{self.n_legs} legs, observed {self.observed_rate:.1%} vs claimed "
            f"{self.claimed_rate:.1%}, {pull:.0%} weight on live data"
        )


def estimate_haircut(
    conn: sqlite3.Connection,
    prior: float = PRIOR_HAIRCUT,
    prior_weight: float = PRIOR_WEIGHT_LEGS,
) -> HaircutEstimate:
    """Per-leg optimism, learned from graded legs and shrunk to the prior.

    The claimed per-leg rate is reconstructed from each slip's claimed P(win)
    as ``claimed_p ** (1 / n_legs)`` — the geometric mean leg probability.
    That is exact under independence and close enough under the mild
    correlation these slips carry, and it avoids having to store per-leg
    probabilities retroactively.
    """
    # Only the current era: legs placed before the 2026-07-26 fixes were
    # selected by a model carrying the whole-number push bug (+6.1pt on the
    # very legs it favoured), so their observed-vs-claimed gap punishes the
    # corrected model for a defect it no longer has. Fresh era, fresh
    # evidence; until 20 new legs grade, this returns the prior.
    from cs2props.tracker import TRACKING_EPOCH

    rows = conn.execute(
        """
        SELECT s.claimed_p, s.n_legs, l.status
        FROM slip_legs l JOIN slips s ON s.slip_id = l.slip_id
        WHERE s.status != 'pending' AND l.status IN ('won', 'lost')
          AND s.claimed_p IS NOT NULL AND s.n_legs > 0
          AND s.placed_at >= ?
        """, (TRACKING_EPOCH,)
    ).fetchall()
    if not rows:
        return HaircutEstimate(prior, 0, None, None, prior_weight)

    n = len(rows)
    wins = sum(1 for r in rows if r[2] == "won")
    observed = wins / n
    claimed = sum(float(cp) ** (1.0 / int(nl)) for cp, nl, _ in rows) / n
    gap = claimed - observed  # positive = model was optimistic

    if n < MIN_LEGS_TO_LEARN:
        return HaircutEstimate(prior, n, observed, claimed, prior_weight)

    # shrink the observed gap toward the prior by sample size
    learned = (gap * n + prior * prior_weight) / (n + prior_weight)
    haircut = min(max(learned, 0.0), MAX_HAIRCUT)
    log.info(
        "haircut: observed gap %.3f over %d legs -> %.3f after shrinkage",
        gap, n, haircut,
    )
    return HaircutEstimate(haircut, n, observed, claimed, prior_weight)
