"""Flex slips: the payout reads the whole hit distribution, not P(all).

Flex is the best-priced product on the verified PrizePicks ladder — 5-pick
flex breaks even at 54.25% per leg, the lowest number available, against
57.74% for a 2-pick power. It earns that by paying partial tiers, which
means every calculation that treats a slip as win/lose is wrong for flex.

Two specific traps are pinned here: scoring a VOID leg as a hit (which would
pay the full-size tier on a slip the book shrank), and applying the optimism
haircut by scaling P(all) (which says nothing about what happens to "4 of
5", the tier carrying most of flex's value).
"""

from __future__ import annotations

import numpy as np

from cs2props.optimizer.search import Leg, Slip, _thin
from cs2props.ingest.prizepicks import Prop

FLEX5 = {5: 10.0, 4: 2.0, 3: 0.4}


def _prop(line: float = 20.5) -> Prop:
    return Prop(
        projection_id="1", player_id="p", player_name="p", team="A",
        opponent="B", stat_type="Kills", stat_kind="kills",
        map_range=(1, 2), line_score=line, board="standard",
        start_time=None, league_id="265",
    )


def _leg(name: str, p: float = 0.6, team: str = "A",
         sim: int = 0) -> Leg:
    return Leg(sim, 0, "under", p, Prop(
        projection_id=name, player_id=name, player_name=name, team=team,
        opponent="B", stat_type="Kills", stat_kind="kills",
        map_range=(1, 2), line_score=20.5, board="standard",
        start_time=None, league_id="265",
    ), team)


def _flex_slip(k_probs: tuple[float, ...], n: int = 5) -> Slip:
    ev = sum(q * FLEX5.get(k, 0.0) for k, q in enumerate(k_probs)) - 1.0
    return Slip(
        legs=[_leg(f"p{i}") for i in range(n)], p_all=k_probs[n],
        p_independent=0.0, ev=ev, multiplier=FLEX5[n], product="flex",
        k_probs=k_probs, _flex_table=dict(FLEX5),
    )


def test_flex_ev_uses_every_paying_tier() -> None:
    """A slip that never goes 5/5 can still be profitable on 4/5 and 3/5.
    Scoring flex off P(all) alone would price this at -100%."""
    s = _flex_slip((0.0, 0.0, 0.2, 0.5, 0.3, 0.0))
    assert s.p_all == 0.0
    assert abs(s.ev - (0.5 * 0.4 + 0.3 * 2.0 - 1.0)) < 1e-9
    assert s.ev > -1.0


def test_power_growth_is_not_zero_without_k_probs() -> None:
    """Power slips carry no hit distribution — every non-win pays zero, so
    the bet is two outcomes. Requiring k_probs made every power slip report
    zero growth and lose each product comparison it entered."""
    s = Slip(legs=[_leg(f"p{i}") for i in range(4)], p_all=0.20,
             p_independent=0.16, ev=1.0, multiplier=10.0, product="power")
    assert s.kelly_growth > 0.0


def test_flex_beats_power_on_growth_at_equal_edge() -> None:
    """The whole reason to prefer flex: lower variance lets a Kelly bettor
    stake more, and the bigger stake compounds past the EV it gave up."""
    k = (0.01, 0.05, 0.14, 0.28, 0.34, 0.18)
    flex = _flex_slip(k)
    power = Slip(legs=[_leg(f"p{i}") for i in range(5)], p_all=k[5],
                 p_independent=0.0, ev=20.0 * k[5] - 1.0, multiplier=20.0,
                 product="power")
    assert flex.ev < power.ev  # power wins on headline EV
    assert flex.kelly_growth > power.kelly_growth  # flex wins on growth


def test_haircut_is_applied_to_the_distribution_not_p_all() -> None:
    s = _flex_slip((0.01, 0.05, 0.14, 0.28, 0.34, 0.18))
    s.adjusted_ev_flex = 0.42
    assert s.adjusted_ev == 0.42  # uses the thinned figure, not a scaled p_all


def test_flex_falls_back_to_raw_ev_when_unthinned() -> None:
    s = _flex_slip((0.01, 0.05, 0.14, 0.28, 0.34, 0.18))
    assert s.adjusted_ev_flex is None
    assert s.adjusted_ev == s.ev


def test_thinning_lowers_a_leg_to_p_minus_haircut() -> None:
    """The haircut must land on the leg's MARGINAL, or the tiers move by an
    arbitrary amount instead of the measured one."""
    rng = np.random.default_rng(0)
    hit = rng.random(200_000) < 0.60
    thinned = _thin(hit, 0.60, 0.05, rng)
    assert abs(thinned.mean() - 0.55) < 0.005


def test_thinning_only_ever_removes_hits() -> None:
    rng = np.random.default_rng(0)
    hit = rng.random(10_000) < 0.6
    thinned = _thin(hit, 0.6, 0.05, rng)
    assert not (thinned & ~hit).any()  # never invents a win


def test_zero_haircut_is_a_no_op() -> None:
    rng = np.random.default_rng(0)
    hit = rng.random(1000) < 0.6
    assert (_thin(hit, 0.6, 0.0, rng) == hit).all()


def test_same_player_two_spellings_is_rejected() -> None:
    """'910' and '910-' were both live on the board 2026-07-26 and the
    optimizer built slips using both. The book rejects that entry."""
    from cs2props.config import load_restrictions
    from cs2props.optimizer.search import is_submittable

    restr = load_restrictions("prizepicks")
    # one leg per match, so the same-match cap is not what rejects
    legs = [_leg("910"), _leg("910-", team="B", sim=1),
            _leg("alpha", sim=2), _leg("bravo", team="B", sim=3)]
    assert not is_submittable(legs, restr)  # 910 / 910- are one player
    ok = [_leg("910"), _leg("charlie", team="B", sim=1),
          _leg("alpha", sim=2), _leg("bravo", team="B", sim=3)]
    assert is_submittable(ok, restr)


def test_flex_pricing_shrinks_the_table_on_voids() -> None:
    """User-verified 2026-07-29: a void (push on a whole-number line, or a
    DNP) converts a 5-flex into a 4-flex. Pricing had counted a voided leg
    as a plain miss — "4 of 5 -> 2x" — when the book actually pays
    "4 of 4 -> 6x". Settlement already did this right; the optimizer's EV
    has to agree with the tracker or slips are chosen on the wrong number."""
    import numpy as np

    from cs2props.config import load_payouts, load_restrictions
    from cs2props.correlation.engine import SimResult
    from cs2props.optimizer.search import _best_flex

    n = 40_000
    rng = np.random.default_rng(3)
    hits = np.zeros((n, 5), dtype=bool)
    short = np.zeros((n, 5), dtype=bool)
    push = np.zeros((n, 5), dtype=bool)
    for j in range(4):
        hits[:, j] = rng.random(n) < 0.999  # four near-certain winners
    push[:, 4] = True                       # fifth leg ALWAYS voids
    res = SimResult(hits=hits, short=short, n_maps=np.full(n, 2), push=push,
                    p_over=[0.999] * 4 + [0.5])

    class _Sim:
        result = res
        props = [_prop(20.5 + i) for i in range(5)]

    teams = ["A", "A", "B", "B", "C"]
    legs = [Leg(0, j, "over", 0.99 if j < 4 else 0.5,
                Leg(0, 0, "over", 0.5, _prop(), "A").prop, teams[j])
            for j in range(5)]
    # build legs against distinct matches so structure rules pass
    legs = [Leg(j, 0, "over", 0.99 if j < 4 else 0.5, _leg(f"p{j}").prop,
                teams[j]) for j in range(5)]

    class _S:
        def __init__(self, j): self.result = _mk_single(res, j)
        props = None

    def _mk_single(full, j):
        return SimResult(hits=full.hits[:, [j]], short=full.short[:, [j]],
                         n_maps=full.n_maps, push=full.push[:, [j]],
                         p_over=[full.p_over[j]])

    sims = [_S(j) for j in range(5)]
    slips = _best_flex(sims, legs, 5, load_payouts("prizepicks"),
                       load_restrictions("prizepicks"),
                       min_adjusted_ev=-10.0, haircut=0.0)
    assert slips, "the always-void slip must still price"
    s = slips[0]
    # four sure winners + one guaranteed void = a 4-flex paying 4-of-4 = 6x
    # (old pricing said 4-of-5 = 2x)
    assert s.ev > 4.0, f"void must shrink the table, got EV {s.ev:+.2f}"
