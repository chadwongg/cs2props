"""Structure-comparison search: shapes priced at REAL multipliers.

The spec this implements assumed 2+2 pays 10x; the app was measured paying
~8x (second same-match pair costs ~20%), which moves the break-even from
10% to 12.5%. These tests pin the parts that make the comparison honest:
teammate-only grouping, per-structure break-even filtering, the two edge
sources reported separately, and never pricing a structure the config
carries no number for.
"""

from __future__ import annotations

import numpy as np

from cs2props.config import Payouts, load_payouts
from cs2props.correlation.engine import SimResult
from cs2props.ingest.prizepicks import Prop
from cs2props.optimizer.structures import (
    StructureSlip,
    _iter_structures,
    collect_structure_legs,
    search_structures,
)
from cs2props.optimizer.search import Leg

RNG = np.random.default_rng(5)
N = 20_000


def _prop(name: str, team: str, stat: str = "kills",
          maps: tuple[int, int] = (1, 2)) -> Prop:
    return Prop(
        projection_id=name, player_id=name, player_name=name, team=team,
        opponent="OPP", stat_type="x", stat_kind=stat, map_range=maps,
        line_score=25.5, board="standard", start_time=None, league_id="CS",
    )


class _Sim:
    def __init__(self, names: list[str], teams: list[str],
                 p: float = 0.62, rho: float = 0.5,
                 stat: str = "kills", maps: tuple[int, int] = (1, 2)) -> None:
        n = len(names)
        factor = RNG.uniform(size=N)
        hits = np.zeros((N, n), dtype=bool)
        for j in range(n):
            follow = RNG.uniform(size=N) < rho
            hits[:, j] = np.where(follow, factor < p, RNG.uniform(size=N) < p)
        self.result = SimResult(
            hits=hits, short=np.zeros((N, n), dtype=bool),
            n_maps=np.full(N, 2), push=np.zeros((N, n), dtype=bool),
            p_over=[float(hits[:, j].mean()) for j in range(n)],
        )
        self.props = [_prop(nm, tm, stat, maps)
                      for nm, tm in zip(names, teams)]


def _board() -> list:
    return [
        _Sim(["a1", "a2", "a3"], ["A", "A", "A"]),        # match 0
        _Sim(["b1", "b2", "c1", "c2"], ["B", "B", "C", "C"]),  # match 1
        _Sim(["d1", "d2"], ["D", "D"]),                   # match 2
        _Sim(["e1"], ["E"]),                              # match 3
    ]


def test_stat_and_map_filters_are_parameters() -> None:
    sims = [_Sim(["h1", "h2"], ["H", "H"], stat="headshots"),
            _Sim(["k1"], ["K"], maps=(1, 1))]
    assert collect_structure_legs(sims) == []          # kills 1-2 default
    hs = collect_structure_legs(sims, stats=frozenset({"headshots"}))
    assert {l.prop.player_name for l in hs} == {"h1", "h2"}
    m1 = collect_structure_legs(sims, maps=(1, 1))
    assert {l.prop.player_name for l in m1} == {"k1"}


def test_pairs_are_teammates_and_matches_never_repeat() -> None:
    legs = collect_structure_legs(_board())
    for structure, combo in _iter_structures(legs):
        by_match: dict[int, list[Leg]] = {}
        for l in combo:
            by_match.setdefault(l.sim_idx, []).append(l)
        sizes = sorted((len(v) for v in by_match.values()), reverse=True)
        if structure == "2+2":
            assert sizes == [2, 2]
            for group in by_match.values():
                assert len({l.team for l in group}) == 1  # TEAMMATES
        elif structure == "3+1":
            assert sizes == [3, 1]
        elif structure == "4":
            assert sizes == [4]
            teams = {l.team for l in combo}
            assert len(teams) == 2  # single-team slips are illegal
        elif structure == "2+1+1":
            assert sizes == [2, 1, 1]
        elif structure == "1+1+1+1":
            assert sizes == [1, 1, 1, 1]


def test_unpriced_structures_are_skipped_not_guessed() -> None:
    """Underdog has a trusted price only for 1+1+1+1 — the concentrated
    shapes must vanish there, not inherit PrizePicks' numbers."""
    slips = search_structures(_board(), load_payouts("underdog"),
                              filter_ratio=0.0, top=50)
    assert slips
    assert {s.structure for s in slips} == {"1+1+1+1"}


def test_filter_is_per_structure_breakeven_not_flat() -> None:
    """P(all)=0.115 clears 1.1x break-even at 10x but NOT at 8x (needs
    0.1375) — the flat 0.11 from the original spec would wrongly admit
    -EV 2+2 slips at the real multiplier."""
    pay = Payouts(power={}, flex={}, correlated={}, pair_penalty={},
                  structures_4pick={"2+2": 8.0, "1+1+1+1": 10.0})
    lo = StructureSlip("2+2", 8.0, (), 0.115, 0.115, ((1.0,),), "OK")
    assert lo.ev < 0.0  # at 8x, 11.5% is a losing bet
    slips = search_structures(_board(), pay, filter_ratio=1.1, top=50)
    for s in slips:
        assert s.p_all >= 1.1 / s.multiplier - 1e-9


def test_edge_sources_are_reported_separately() -> None:
    # top must be wide enough to reach the 8x shapes: every 10x combination
    # outranks them on EV in this synthetic board
    slips = search_structures(_board(), load_payouts("prizepicks"),
                              filter_ratio=0.0, top=500)
    assert slips
    s = next(x for x in slips if x.structure == "2+2")
    assert abs(s.corr_lift - (s.p_all - s.p_independent)) < 1e-12
    # shared-factor sims: teammates correlate, so the joint beats the product
    assert s.corr_lift > 0.0
    assert len(s.corr_matrix) == 4 and len(s.corr_matrix[0]) == 4
    assert all(abs(s.corr_matrix[i][i] - 1.0) < 1e-6 for i in range(4))


def test_confidence_defaults_to_ok_without_history() -> None:
    slips = search_structures(_board(), load_payouts("prizepicks"),
                              conn=None, filter_ratio=0.0, top=5)
    assert slips and all(s.confidence == "OK" for s in slips)


def test_json_payload_is_complete() -> None:
    slips = search_structures(_board(), load_payouts("prizepicks"),
                              filter_ratio=0.0, top=3)
    j = slips[0].to_json()
    for key in ("structure", "multiplier", "p_all", "p_independent",
                "corr_lift", "ev", "confidence", "corr_matrix", "legs"):
        assert key in j
    leg = j["legs"][0]
    for key in ("player", "team", "side", "line", "model_p", "implied_p",
                "edge_pts"):
        assert key in leg
