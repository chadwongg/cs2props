"""A whole-number line can PUSH, and a push is not a win.

23.5% of PrizePicks lines are whole numbers (Underdog posts none). In our
archive those pushed 6.1% of the time -- 7.6% on headshots, which is most of
what the optimizer was recommending. Scoring an UNDER as ``~hits`` folded
every push into the win column at full 4-leg payout, when the book voids the
leg and pays the 3-leg multiplier instead.

The error only ever ran one way: it inflated exactly the legs this app kept
choosing.
"""

from __future__ import annotations

import numpy as np

from cs2props.correlation.engine import SimResult
from cs2props.optimizer.search import Leg, _leg_arrays
from cs2props.ingest.prizepicks import Prop


def _prop(line: float) -> Prop:
    return Prop(
        projection_id="1", player_id="p", player_name="p", team="A",
        opponent="B", stat_type="Kills", stat_kind="kills",
        map_range=(1, 2), line_score=line, board="standard",
        start_time=None, league_id="265",
    )


class _Sim:
    def __init__(self, result: SimResult) -> None:
        self.result = result


def _result(totals: np.ndarray, line: float) -> SimResult:
    n = totals.shape[0]
    hits = (totals > line).reshape(n, 1)
    short = np.zeros((n, 1), dtype=bool)
    push = np.zeros((n, 1), dtype=bool)
    if float(line).is_integer():
        push[:, 0] = totals == line
    return SimResult(hits=hits, short=short, n_maps=np.full(n, 2), push=push)


def test_push_is_not_an_under_win() -> None:
    """20 kills against a line of 20 is a VOID, not a win for the under."""
    totals = np.array([18, 20, 22, 20])  # two pushes
    res = _result(totals, 20.0)
    sims = [_Sim(res)]
    leg = Leg(0, 0, "under", 0.5, _prop(20.0), "A")
    hit, live = _leg_arrays(sims, leg)  # type: ignore[arg-type]
    assert list(live) == [True, False, True, False]  # pushes are not live
    # index 0 (18 < 20) is a genuine under win; index 2 (22) is a loss
    assert hit[0] and not hit[2]
    # a voided leg "passes" the all-hit test but shrinks the payout
    assert hit[1] and hit[3]


def test_push_is_not_an_over_win_either() -> None:
    totals = np.array([18, 20, 22])
    sims = [_Sim(_result(totals, 20.0))]
    leg = Leg(0, 0, "over", 0.5, _prop(20.0), "A")
    hit, live = _leg_arrays(sims, leg)  # type: ignore[arg-type]
    assert list(live) == [True, False, True]
    assert not hit[0] and hit[2]


def test_half_point_lines_can_never_push() -> None:
    """A .5 line has no integer it can land on — every leg stays live."""
    totals = np.array([18, 20, 22, 25])
    sims = [_Sim(_result(totals, 20.5))]
    leg = Leg(0, 0, "under", 0.5, _prop(20.5), "A")
    _hit, live = _leg_arrays(sims, leg)  # type: ignore[arg-type]
    assert live.all()


def test_under_probability_excludes_pushes() -> None:
    """The measured overstatement was 6.1 points. With 10% of iterations
    landing on the line, the under's win rate must drop by exactly that
    10 -- not silently absorb it."""
    totals = np.array([10] * 40 + [20] * 10 + [30] * 50)  # 10% push at 20
    res = _result(totals, 20.0)
    sims = [_Sim(res)]
    leg = Leg(0, 0, "under", 0.5, _prop(20.0), "A")
    hit, live = _leg_arrays(sims, leg)  # type: ignore[arg-type]
    real_wins = (hit & live).sum() / len(totals)
    assert abs(real_wins - 0.40) < 1e-9  # not 0.50
    assert live.mean() == 0.90


def test_p_all_treats_a_push_as_a_refund_not_a_loss() -> None:
    totals = np.array([25, 20, 15])
    res = _result(totals, 20.0)
    # OVER on all three: iteration 0 hits, 1 pushes (refund), 2 misses
    assert abs(res.p_all([0]) - 2 / 3) < 1e-9


def test_missing_push_array_is_handled() -> None:
    """Older SimResults have no push array; they must not crash."""
    n = 3
    res = SimResult(
        hits=np.ones((n, 1), dtype=bool),
        short=np.zeros((n, 1), dtype=bool),
        n_maps=np.full(n, 2),
    )
    assert not res.pushes(0).any()
