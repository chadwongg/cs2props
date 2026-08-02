"""Optimizer tests: refusal, conditional math, void payouts, restrictions."""

from __future__ import annotations

import numpy as np

from cs2props.config import load_payouts, load_restrictions
from cs2props.correlation.engine import SimResult
from cs2props.ingest.prizepicks import Prop
from cs2props.optimizer.search import Leg, collect_legs, evaluate, search_slips
from cs2props.pipeline import MatchSim

from cs2props.config import Payouts, Restrictions

RNG = np.random.default_rng(11)
N = 20_000
# Fixed classic table: tests pin construction math, not the live config
# (payouts.json is user-editable by design — e.g. Arena guarantees).
PP = Payouts(power={2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0}, flex={},
             correlated={}, pair_penalty={})
# Restrictions are PINNED here for the same reason payouts are: these tests
# check construction math, and restrictions.json is deliberately user-editable
# book policy. Wiring them to the live file made a policy change (adding
# max_same_match=2 on 2026-07-26) fail seven unrelated construction tests.
# The live values get their own dedicated tests below.
RESTR = Restrictions(
    max_legs_per_player=1, same_team_action="flag", max_same_team=3,
    boards_combinable={"standard": True}, min_distinct_teams=2,
    default_slip_size=4, max_same_match=99,
)


def _prop(name: str, team: str, line: float = 25.5) -> Prop:
    return Prop(
        projection_id=name, player_id=name, player_name=name, team=team,
        opponent="OPP", stat_type="MAPS 1-2 Kills", stat_kind="kills",
        map_range=(1, 2), line_score=line, board="standard",
        start_time="2026-07-25T18:00:00Z", league_id="CS",
    )


def _sim(
    p_hits: list[float], teams: list[str], rho: float = 0.0,
    short_p: list[float] | None = None, prefix: str = "p",
) -> MatchSim:
    """Synthetic MatchSim with controllable marginals + shared-factor corr."""
    n_props = len(p_hits)
    factor = RNG.uniform(size=N)
    hits = np.zeros((N, n_props), dtype=bool)
    for j, p in enumerate(p_hits):
        if rho > 0:
            # mixture: with prob rho follow the shared factor, else fresh draw
            follow = RNG.uniform(size=N) < rho
            fresh = RNG.uniform(size=N) < p
            hits[:, j] = np.where(follow, factor < p, fresh)
        else:
            hits[:, j] = RNG.uniform(size=N) < p
    short = np.zeros((N, n_props), dtype=bool)
    if short_p:
        for j, sp in enumerate(short_p):
            short[:, j] = RNG.uniform(size=N) < sp
    props = [_prop(f"{prefix}{j}", teams[j]) for j in range(n_props)]
    res = SimResult(hits=hits, short=short, n_maps=np.full(N, 2),
                    p_over=[float(hits[~short[:, j], j].mean())
                            for j in range(n_props)])
    return MatchSim(label="A vs B", start_time=None, props=props, result=res,
                    p_a_map=0.5, side_of={"A": "A", "B": "B"})


def test_collect_legs_takes_best_side_and_one_per_player() -> None:
    sim = _sim([0.70, 0.30, 0.50], ["A", "A", "B"])
    legs = collect_legs([sim])
    by_name = {l.prop.player_name: l for l in legs}
    assert by_name["p0"].side == "over"
    assert by_name["p1"].side == "under"  # 30% over -> 70% under
    assert "p2" not in by_name  # 50% clears nothing


def test_evaluate_joint_vs_independent_with_correlation() -> None:
    sim = _sim([0.62, 0.62, 0.62], ["A", "A", "A"], rho=0.5)
    legs = collect_legs([sim])[:3]
    p_all, p_ind, _ev, _m = evaluate([sim], legs, PP)
    assert p_all > p_ind + 0.03  # correlation lifts the joint


def test_void_shrinks_payout_not_kills_slip() -> None:
    """A leg that voids 100% of the time: slip becomes a 3-pick at 5x."""
    sim = _sim([0.60, 0.60, 0.60, 0.99], ["A", "A", "B", "B"],
               short_p=[0.0, 0.0, 0.0, 1.0])
    legs = collect_legs([sim])
    four = sorted(legs, key=lambda l: l.prop.player_name)
    _p, _i, ev4, _m = evaluate([sim], four, PP)
    three = [l for l in four if l.prop.player_name != "p3"]
    _p3, _i3, ev3, _m3 = evaluate([sim], three, PP)
    assert abs(ev4 - ev3) < 0.03  # voided leg contributes ~nothing


def test_search_refuses_filler_and_notes_rejection() -> None:
    """Underdog's 3->4 ratio is 10/6 -> the 4th-leg bar is ~62%+ margin.
    A 58% independent leg clears the pool filter but NOT the conditional
    bar -> the optimizer must refuse and show the rejected candidate."""
    ud = load_payouts("underdog")
    # Pinned, like RESTR: this test checks the CONDITIONAL 4th-leg bar, not
    # book policy. Against the live config the 4-leg slip is now illegal on
    # the same-match cap before the conditional test ever runs, which would
    # exercise a different refusal path than the one under test.
    ud_restr = Restrictions(
        max_legs_per_player=1, same_team_action="flag", max_same_team=4,
        boards_combinable={"standard": True}, min_distinct_teams=1,
        default_slip_size=4, max_same_match=99,
    )
    sim_a = _sim([0.66, 0.64, 0.62], ["A", "A", "A"], rho=0.6)
    sim_b = _sim([0.58], ["C"], prefix="q")  # decent, below UD's 4th-leg bar
    slips, reason = search_slips([sim_a, sim_b], ud, ud_restr)
    assert reason is None
    top = slips[0]
    assert len(top.legs) == 3  # refused to pad
    assert top.note is not None and "REFUSED" in top.note
    # No flag: this fixture's pinned restrictions allow the concentration,
    # and a flag saying "nothing is wrong" is noise. Flags now fire only when
    # a slip carries MORE pairs than the book gives away free — which
    # is_submittable rejects, so it is a canary for broken enforcement.
    assert not any("same-match" in f for f in top.flags)


def test_prizepicks_low_bar_accepts_independent_57() -> None:
    """PP's 3->4 ratio is 5/10 -> bar ~52%: an independent 57% leg is
    mathematically worth adding, so no refusal on the same board."""
    sim_a = _sim([0.66, 0.64, 0.62], ["A", "A", "A"], rho=0.6)
    sim_b = _sim([0.57], ["C"], prefix="q")
    slips, reason = search_slips([sim_a, sim_b], PP, RESTR)
    assert reason is None
    assert len(slips[0].legs) == 4


def test_search_accepts_qualifying_fourth() -> None:
    # 3-stack core + an independent 60% leg from another match: clears the
    # PP bar (52%) and the stack cap -> full legal 4-man, no refusal note
    sim_a = _sim([0.66, 0.65, 0.64], ["A", "A", "A"], rho=0.7)
    sim_b = _sim([0.60], ["C"], prefix="q")
    slips, reason = search_slips([sim_a, sim_b], PP, RESTR)
    assert reason is None
    assert len(slips[0].legs) == 4
    assert slips[0].note is None


def test_single_team_slip_never_offered_at_any_size() -> None:
    """PrizePicks' two-team rule applies to 3-leg slips too (verified live
    2026-07-24). With only one team's legs qualifying and no acceptable 4th,
    the optimizer must NOT emit the all-STATE 3-stack."""
    sim_a = _sim([0.66, 0.64, 0.62], ["A", "A", "A"], rho=0.7)
    sim_b = _sim([0.51], ["C"], prefix="q")  # too weak to qualify anywhere
    slips, _reason = search_slips([sim_a, sim_b], PP, RESTR)
    for s in slips:
        assert len({l.team for l in s.legs if l.team}) >= 2, s.legs


def test_is_submittable_rules() -> None:
    from cs2props.optimizer.search import is_submittable

    sim = _sim([0.66, 0.64, 0.62, 0.60], ["A", "A", "A", "B"], rho=0.5)
    legs = collect_legs([sim])
    by_team = {t: [l for l in legs if l.team == t] for t in ("A", "B")}
    assert not is_submittable(by_team["A"][:3], RESTR)  # all one team
    assert is_submittable(by_team["A"][:3] + by_team["B"][:1], RESTR)
    assert not is_submittable(by_team["A"][:1], RESTR)  # too few legs


def test_stack_cap_is_hard_constraint() -> None:
    """PrizePicks hard-rejects one-team slips (verified live 2026-07-24):
    with 4 strong correlated legs on one team plus a mediocre outsider, the
    optimizer must output 3-stack + outsider, never the 4-stack."""
    sim_a = _sim([0.68, 0.66, 0.65, 0.64], ["A", "A", "A", "A"], rho=0.7)
    sim_b = _sim([0.57], ["C"], prefix="q")
    slips, reason = search_slips([sim_a, sim_b], PP, RESTR)
    assert reason is None
    for s in slips:
        teams = [l.team for l in s.legs]
        assert max(teams.count(t) for t in set(teams)) <= 3
    assert len(slips[0].legs) == 4  # still built a 4-man, just legally


def test_no_slips_today_when_board_is_bad() -> None:
    sim = _sim([0.52, 0.51, 0.50, 0.49], ["A", "A", "B", "B"])
    slips, reason = search_slips([sim], PP, RESTR)
    assert slips == []
    assert reason is not None


def test_surfaced_slips_do_not_reuse_players() -> None:
    """Ranking by EV alone returns the same bet five ways — overlapping slips
    have nearly equal EV. Placing two of them shares model error, raises
    variance and shrinks the live-calibration sample, so surfaced slips must
    be substantially distinct."""
    from cs2props.optimizer.search import MAX_SHARED_PLAYERS

    sim_a = _sim([0.68, 0.66, 0.65, 0.64], ["A", "A", "B", "B"], rho=0.5)
    sim_b = _sim([0.67, 0.65, 0.64, 0.62], ["C", "C", "D", "D"], rho=0.5,
                 prefix="q")
    slips, _ = search_slips([sim_a, sim_b], PP, RESTR)
    assert len(slips) >= 2
    for i, s in enumerate(slips):
        for other in slips[:i]:
            shared = {l.prop.player_name for l in s.legs} & {
                l.prop.player_name for l in other.legs
            }
            assert len(shared) <= MAX_SHARED_PLAYERS, shared


def test_surfaced_slips_do_not_reuse_matches() -> None:
    """Distinct players is not enough — two slips drawing on the same GAME
    share its round count and die together (found 2026-07-24). Matches are
    the unit of shared fate, so surfaced slips must not overlap on them."""
    from cs2props.optimizer.search import MAX_SHARED_MATCHES

    sims = [
        _sim([0.68, 0.66, 0.65, 0.64], ["A", "A", "B", "B"], rho=0.5),
        _sim([0.67, 0.65, 0.64, 0.62], ["C", "C", "D", "D"], rho=0.5,
             prefix="q"),
        _sim([0.66, 0.64, 0.63, 0.61], ["E", "E", "F", "F"], rho=0.5,
             prefix="r"),
    ]
    slips, _ = search_slips(sims, PP, RESTR)
    for i, s in enumerate(slips):
        for other in slips[:i]:
            shared = {l.sim_idx for l in s.legs} & {
                l.sim_idx for l in other.legs
            }
            assert len(shared) <= MAX_SHARED_MATCHES, shared


def test_ranked_by_ev_and_deduped() -> None:
    sim = _sim([0.68, 0.66, 0.64, 0.62, 0.60], ["A", "A", "A", "B", "B"],
               rho=0.5)
    slips, _ = search_slips([sim], PP, RESTR)
    evs = [s.ev for s in slips]
    assert evs == sorted(evs, reverse=True)
    keys = [frozenset(f"{l.prop.player_name}|{l.side}" for l in s.legs)
            for s in slips]
    assert len(keys) == len(set(keys))


def test_target_size_two_returns_two_leg_slips() -> None:
    """Underdog defaults to 2-picks; the search must honour target_size."""
    sim = _sim([0.70, 0.68, 0.66, 0.64], ["A", "A", "B", "B"], rho=0.4)
    slips, reason = search_slips([sim], PP, RESTR, target_size=2)
    assert reason is None
    assert all(len(s.legs) == 2 for s in slips)


def test_min_ev_floor_drops_the_thin_tail() -> None:
    """The scanner ranks slips but must not invite betting the tail. On
    2026-07-25 the user placed seven Underdog slips whose post-haircut EV ran
    +50% down to +6%; the bottom ones added variance without expectation."""
    sims = [
        _sim([0.70, 0.68], ["A", "B"], rho=0.3),
        _sim([0.60, 0.58], ["C", "D"], rho=0.3, prefix="q"),
        _sim([0.56, 0.56], ["E", "F"], rho=0.3, prefix="r"),
    ]
    loose, _ = search_slips(sims, PP, RESTR, target_size=2,
                            min_adjusted_ev=-1.0)
    strict, _ = search_slips(sims, PP, RESTR, target_size=2,
                             min_adjusted_ev=0.20)
    assert len(strict) < len(loose)
    assert all(s.adjusted_ev >= 0.20 for s in strict)


def test_adjusted_ev_is_below_raw_ev() -> None:
    """The haircut must actually bite — a slip is only as good as it is when
    every leg is 2 points worse than claimed."""
    sim = _sim([0.68, 0.66], ["A", "B"], rho=0.3)
    slips, _ = search_slips([sim], PP, RESTR, target_size=2,
                            min_adjusted_ev=-1.0)
    s = slips[0]
    assert s.adjusted_ev < s.ev


def test_no_slips_message_names_the_threshold() -> None:
    """Legs clear the pool bar but no slip clears the EV floor — the reason
    must say so, not blame the pool."""
    sim = _sim([0.64, 0.62], ["A", "B"], rho=0.2)
    slips, reason = search_slips([sim], PP, RESTR, target_size=2,
                                 min_adjusted_ev=5.0)
    assert slips == []
    assert reason is not None and "haircut" in reason


def test_live_config_allows_only_one_same_match_pair() -> None:
    """Four in-app 5-pick readings. The shift is CUMULATIVE in same-match
    pairs, and team never enters into it:

        2+1+1+1 (1 pair)  -> 10x   full payout
        2+2+1   (2 pairs) ->  8x
        3+1+1   (3 pairs) ->  7x

    3+1+1 was measured twice — once as two teammates plus an opponent, once
    as three teammates — and paid 7x both times, which rules out an
    opposing-team penalty. 2+2+1 rules out a plain per-match cap.
    """
    live = load_restrictions("prizepicks")
    assert live.max_same_match == 2
    assert live.max_multi_leg_matches == 1


def test_two_matches_each_supplying_a_pair_is_rejected() -> None:
    """The 2+2+1 case: measured at 8x, not 10x. A per-match cap alone passes
    this, which is why the pair COUNT is capped too."""
    from cs2props.optimizer.search import is_submittable

    live = load_restrictions("prizepicks")
    a = _sim([0.66, 0.64], ["A", "A"], prefix="x")
    b = _sim([0.65, 0.63], ["C", "C"], prefix="y")
    c = _sim([0.62], ["E"], prefix="z")
    legs = collect_legs([a, b, c])
    assert len(legs) == 5
    assert not is_submittable(legs, live)
    # one pair plus singles is the shape that paid 10x
    ok = [l for l in legs if l.sim_idx == 0] + \
         [next(l for l in legs if l.sim_idx == 1)] + \
         [next(l for l in legs if l.sim_idx == 2)]
    assert is_submittable(ok, live)


def test_two_teammates_in_one_match_are_free() -> None:
    """The sjuush + stavn case: two legs, one match, full 10x."""
    from cs2props.optimizer.search import is_submittable

    live = load_restrictions("prizepicks")
    a = _sim([0.66, 0.64], ["A", "A"], prefix="x")   # two TEAMMATES
    b = _sim([0.63], ["C"], prefix="y")
    legs = collect_legs([a, b])
    assert is_submittable(legs, live)


def test_three_teammates_in_one_match_are_rejected() -> None:
    """Measured at 7x, not 10x — the case that disproved the opposing-team
    theory, since these three share a team."""
    from cs2props.optimizer.search import is_submittable

    live = load_restrictions("prizepicks")
    a = _sim([0.66, 0.64, 0.62], ["A", "A", "A"], prefix="x")
    b = _sim([0.63], ["C"], prefix="y")
    legs = collect_legs([a, b])
    assert not is_submittable(legs, live)


def test_one_pair_is_free_whether_teammates_or_opponents() -> None:
    """Team is irrelevant to the shift — 3+1+1 paid 7x both as two teammates
    plus an opponent AND as three teammates. So a single same-match pair is
    accepted regardless of which sides the two players are on."""
    from cs2props.optimizer.search import is_submittable

    live = load_restrictions("prizepicks")
    a = _sim([0.66, 0.64], ["A", "B"], prefix="x")   # OPPONENTS
    b = _sim([0.63], ["C"], prefix="y")
    legs = collect_legs([a, b])
    assert is_submittable(legs, live)


def test_unknown_team_in_a_shared_match_is_treated_as_opposing() -> None:
    """A leg whose team cannot be resolved must not be assumed a teammate —
    a false rejection costs one slip, a false pass costs a real payout cut."""
    from cs2props.optimizer.search import Leg, _has_opposing_pair

    known = Leg(0, 0, "over", 0.6, _prop("a", "A"), "A")
    unknown = Leg(0, 1, "over", 0.6, _prop("b", "A"), None)
    assert _has_opposing_pair([known, unknown])


def test_flag_text_matches_the_measured_payout_rule() -> None:
    """The flag describes the PAYOUT, so it must not outlive a disproven
    rule. It once read "only opposing players shift the payout" — killed the
    same day by three teammates shifting it exactly as much as two teammates
    plus an opponent.
    """
    from cs2props.optimizer.search import _flags

    live = load_restrictions("prizepicks")
    # sim_idx comes from POSITION in the list handed to collect_legs, so all
    # matches must be built in one call or their indices collide.
    sims = [
        _sim([0.66, 0.64], ["A", "A"], prefix="x"),   # match 0: a pair
        _sim([0.63], ["C"], prefix="y"),              # match 1: a single
        _sim([0.65, 0.62], ["E", "E"], prefix="z"),   # match 2: a pair
    ]
    legs = collect_legs(sims)
    # ONE pair is free and the header already says what the slip pays, so a
    # flag here would announce that nothing is wrong. Silence is correct.
    one_pair = [l for l in legs if l.sim_idx in (0, 1)]
    assert _flags(one_pair, live) == []

    # TWO pairs would shift the payout. is_submittable rejects that, so this
    # is a canary for broken enforcement rather than a routine warning.
    warn = " ".join(_flags(legs, live))
    assert "BUG" in warn and "2 same-match pairs" in warn
    assert "opposing" not in warn.lower(), "disproven rule resurfaced"


def test_no_flag_when_every_leg_is_a_different_match() -> None:
    from cs2props.optimizer.search import _flags

    live = load_restrictions("prizepicks")
    # distinct teams too — a 4-deep single-team slip trips the separate
    # "exceeds the configured max" flag, which is not what this checks
    teams = ["A", "B", "C", "D"]
    sims = [_sim([0.64], [teams[i]], prefix=f"m{i}") for i in range(4)]
    assert _flags(collect_legs(sims), live) == []


def _shaded_prop(name: str, team: str, over: float, under: float) -> object:
    from cs2props.ingest.prizepicks import Prop

    return Prop(
        projection_id=name, player_id=name, player_name=name, team=team,
        opponent="OPP", stat_type="MAPS 1-2 Kills", stat_kind="kills",
        map_range=(1, 2), line_score=25.5, board="standard",
        start_time="2026-07-26T18:00:00Z", league_id="CS",
        side_multipliers={"over": over, "under": under},
    )


def test_per_side_shade_scales_the_slip_payout() -> None:
    """Underdog prices each side independently even on "balanced" lines:
    Salazar 14.5 headshots was higher 1.03 / lower 0.82 on 2026-07-26. Taking
    that under turns a 6.5x 3-pick into 5.33x, which the EV must reflect —
    otherwise the app quotes a payout the book will not pay."""
    from cs2props.optimizer.search import Leg, slip_price_factor

    legs = [
        Leg(0, 0, "under", 0.6, _shaded_prop("a", "A", 1.03, 0.82), "A"),
        Leg(1, 0, "over", 0.6, _shaded_prop("b", "B", 1.0, 1.0), "B"),
        Leg(2, 0, "over", 0.6, _shaded_prop("c", "C", 1.0, 1.0), "C"),
    ]
    assert abs(slip_price_factor(legs) - 0.82) < 1e-9
    assert abs(6.5 * slip_price_factor(legs) - 5.33) < 0.01


def test_a_boosted_side_is_credited_not_discarded() -> None:
    """The old code tagged the whole prop "alt" and dropped it whenever
    EITHER side was off 1.0 — throwing away the 1.03 side, which pays a
    premium."""
    from cs2props.optimizer.search import Leg, slip_price_factor

    legs = [Leg(0, 0, "over", 0.6, _shaded_prop("a", "A", 1.03, 0.82), "A")]
    assert slip_price_factor(legs) == 1.03


def test_unpriced_props_are_treated_as_even_money() -> None:
    """PrizePicks exposes no per-side multiplier at all — absent means 1.0,
    never zero."""
    from cs2props.optimizer.search import Leg

    leg = Leg(0, 0, "over", 0.6, _prop("x", "A"), "A")
    assert leg.payout_multiplier == 1.0


def test_delta_is_suppressed_when_it_is_only_noise() -> None:
    """A fully diversified slip has a TRUE delta of zero — separate matches
    are simulated independently. Whatever prints is sampling error, and one
    live slip showed -0.1 pts: a negative correlation between independent
    events, which cannot happen."""
    from cs2props.optimizer.search import Slip

    s = Slip(legs=[], p_all=0.337, p_independent=0.335, ev=1.0,
             multiplier=6.5, n_iters=50_000)
    assert abs(s.delta_pts - 0.2) < 1e-9
    assert s.delta_noise_pts > 0.2
    assert not s.delta_is_real


def test_a_real_correlation_bonus_still_reports() -> None:
    """A same-match pair moves P(all) by 2-3 points — far outside the noise
    floor, and worth printing."""
    from cs2props.optimizer.search import Slip

    s = Slip(legs=[], p_all=0.180, p_independent=0.155, ev=1.0,
             multiplier=10.0, n_iters=50_000)
    assert s.delta_pts > 2.0
    assert s.delta_is_real


def test_delta_noise_floor_is_zero_without_iteration_count() -> None:
    """Slips built outside the search (tests, fixtures) carry no n_iters;
    they must not claim a noise floor they cannot compute."""
    from cs2props.optimizer.search import Slip

    s = Slip(legs=[], p_all=0.3, p_independent=0.3, ev=0.0, multiplier=6.5)
    assert s.delta_noise_pts == 0.0


def test_stat_filter_drops_headshot_legs_but_default_keeps_them() -> None:
    """The kills-only policy lives at slip selection: with a stats filter,
    headshot props never become legs; without one, nothing changes. The
    filter must sit HERE so headshot lines keep feeding the simulation and
    crossbook upstream."""
    from dataclasses import replace

    sim = _sim([0.70, 0.70], ["A", "B"])
    sim.props[1] = replace(sim.props[1], stat_kind="headshots")
    legs = collect_legs([sim], frozenset({"kills"}))
    assert {l.prop.player_name for l in legs} == {"p0"}
    legs_all = collect_legs([sim])
    assert {l.prop.player_name for l in legs_all} == {"p0", "p1"}


def _aace_restr(**kw: object) -> Restrictions:
    return Restrictions(
        max_legs_per_player=1, same_team_action="flag", max_same_team=3,
        boards_combinable={"standard": True}, min_distinct_teams=2,
        default_slip_size=4, max_same_match=2, max_multi_leg_matches=1,
        require_teammate_pair=True, **kw,  # type: ignore[arg-type]
    )


def test_aace_search_returns_only_one_teammate_pair_slips() -> None:
    """With require_teammate_pair, every returned 4-pick is exactly 2+1+1
    with the pair on ONE team — never fully diversified, never 2+2."""
    a = _sim([0.66, 0.65, 0.64], ["A", "A", "B"], rho=0.4, prefix="a")
    b = _sim([0.66, 0.65], ["C", "C"], rho=0.4, prefix="b")
    c = _sim([0.64], ["E"], prefix="c")
    d = _sim([0.63], ["G"], prefix="d")
    pay = Payouts(power={3: 6.0, 4: 10.0}, flex={},
                  correlated={4: {1: 10.0, 2: 9.5, 3: 6.75, 4: 5.0}},
                  pair_penalty={})
    slips, reason = search_slips([a, b, c, d], pay, _aace_restr(),
                                 target_size=4, min_adjusted_ev=0.0)
    assert reason is None and slips
    from cs2props.optimizer.search import (
        _has_exactly_one_teammate_pair, _same_match_count,
    )

    for s in slips:
        assert len(s.legs) in (3, 4)
        if len(s.legs) == 4:
            # the pair is required — and only priced right — at 4 picks
            assert _has_exactly_one_teammate_pair(s.legs)
            assert s.multiplier == 9.5
        else:
            # 3-man fallback must be diversified: PP charges 20.8% for a
            # 3-pick pair worth ~+16% (6x -> 4.75x, user-read 2026-08-02)
            assert _same_match_count(s.legs) == 1


def test_aace_rejects_cross_team_pair_in_same_match() -> None:
    """A same-match pair split across the two teams pays the same 9.5x but
    carries the weaker opposing correlation — not the structure we want."""
    from cs2props.optimizer.search import _has_exactly_one_teammate_pair

    sim = _sim([0.7, 0.7, 0.7, 0.7], ["A", "B", "A", "B"])
    legs = collect_legs([sim])
    # manufacture: two legs from this match (teams A and B) + fabricate
    # two singles from other "matches" by reusing Leg with new sim_idx
    l1 = [l for l in legs if l.team == "A"][0]
    l2 = [l for l in legs if l.team == "B"][0]
    other = _sim([0.7, 0.7], ["C", "E"], prefix="x")
    singles = collect_legs([other])[:2]
    singles = [Leg(8 + i, s.prop_idx, s.side, s.p, s.prop, s.team)
               for i, s in enumerate(singles)]
    assert not _has_exactly_one_teammate_pair([l1, l2, *singles])
    mate = [l for l in legs if l.team == "A"][1]
    assert _has_exactly_one_teammate_pair([l1, mate, *singles])


def test_recommended_slips_share_no_props() -> None:
    """User policy: no duplicate props across the recommended list — every
    slip is an independent bet. diversify enforces zero shared players and
    zero shared matches, INCLUDING against a shorter-slip fallback top."""
    from cs2props.optimizer.search import Slip, diversify

    def slip_of(sim_idx: int, names: list[str]) -> Slip:
        sim = _sim([0.7] * len(names), ["A"] * len(names), prefix="")
        legs = []
        for i, n in enumerate(names):
            p = _prop(n, "A")
            legs.append(Leg(sim_idx, i, "over", 0.7, p, "A"))
        return Slip(legs=legs, p_all=0.2, p_independent=0.2, ev=1.0,
                    multiplier=9.5)

    s1 = slip_of(0, ["p1", "p2"])
    s1_dup = slip_of(1, ["p2", "p9"])   # shares p2
    s2 = slip_of(2, ["p3", "p4"])
    s3 = slip_of(0, ["p5", "p6"])       # same MATCH as s1
    out = diversify([s1_dup, s2, s3], 5, taken=[s1])
    assert out == [s2]


def test_side_filter_drops_over_legs_but_default_keeps_them() -> None:
    """Unders-only policy: with a sides filter no over leg is ever a
    candidate, even when the over is the stronger side; without one both
    sides compete as before."""
    sim = _sim([0.70, 0.30], ["A", "B"])  # p0 best as OVER, p1 best as UNDER
    legs = collect_legs([sim], sides=frozenset({"under"}))
    assert {(l.prop.player_name, l.side) for l in legs} == {("p1", "under")}
    legs_all = collect_legs([sim])
    assert {(l.prop.player_name, l.side) for l in legs_all} == {
        ("p0", "over"), ("p1", "under"),
    }
