"""Target-shape search: price a slip at a multiplier that was OBSERVED.

Every EV number in this app rests on a multiplier PrizePicks does not
publish. The fitted pair_penalty is an extrapolation and it is wrong here —
it predicts 9.25x for the cross2x2 structure (10 - 2*0.375) where the app
consistently shows 8.5x. Quoting 9.25x on a slip that pays 8.5x overstates
EV by 9 points, which is more than the entire edge the optimizer is hunting.

So the shape must match EXACTLY. A combination that is one leg off the
structure is not the thing that was priced, and must not inherit its price.
"""

from __future__ import annotations

from cs2props.config import Shape, load_payouts
from cs2props.optimizer.search import Leg, matches_shape

CROSS2X2 = Shape(
    name="cross2x2", multiplier=8.5, n_legs=4, n_matches=2,
    legs_per_match=2, opposing_within_match=True,
    same_direction_within_match=True,
)


def _leg(sim: int, team: str, side: str, name: str = "") -> Leg:
    return Leg(sim, 0, side, 0.6, None, team)  # type: ignore[arg-type]


def test_the_shape_is_configured_and_loads() -> None:
    shape = load_payouts("prizepicks").shape("cross2x2")
    assert shape.multiplier == 8.5
    assert shape.n_matches == 2 and shape.legs_per_match == 2
    assert shape.opposing_within_match


def test_unknown_shape_names_what_is_available() -> None:
    try:
        load_payouts("prizepicks").shape("nope")
    except KeyError as exc:
        assert "cross2x2" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_canonical_cross_match_slip_matches() -> None:
    """One player from each side of two matches, each pair same-direction."""
    legs = [
        _leg(0, "FaZe", "under"), _leg(0, "Vitality", "under"),
        _leg(1, "G2", "over"), _leg(1, "Spirit", "over"),
    ]
    assert matches_shape(legs, CROSS2X2)


def test_the_two_pairs_may_point_opposite_ways() -> None:
    """Direction is constrained WITHIN a match, not across them — that is
    what the user described and it is the structure that was priced."""
    legs = [
        _leg(0, "FaZe", "under"), _leg(0, "Vitality", "under"),
        _leg(1, "G2", "over"), _leg(1, "Spirit", "over"),
    ]
    assert matches_shape(legs, CROSS2X2)


def test_same_team_pair_is_rejected() -> None:
    """Two teammates is the 7.75x structure, not the 8.5x one. Pricing it at
    8.5x would inflate EV on exactly the slips already known to pay less."""
    legs = [
        _leg(0, "FaZe", "under"), _leg(0, "FaZe", "under"),
        _leg(1, "G2", "over"), _leg(1, "Spirit", "over"),
    ]
    assert not matches_shape(legs, CROSS2X2)


def test_mixed_direction_inside_a_match_is_rejected() -> None:
    """The 7.25x slip the user placed was 2+2 with mixed directions inside
    each match. It is a different structure and a different price."""
    legs = [
        _leg(0, "FaZe", "under"), _leg(0, "Vitality", "over"),
        _leg(1, "G2", "over"), _leg(1, "Spirit", "over"),
    ]
    assert not matches_shape(legs, CROSS2X2)


def test_four_separate_matches_is_rejected() -> None:
    legs = [
        _leg(0, "FaZe", "under"), _leg(1, "Vitality", "under"),
        _leg(2, "G2", "under"), _leg(3, "Spirit", "under"),
    ]
    assert not matches_shape(legs, CROSS2X2)


def test_three_one_split_is_rejected() -> None:
    legs = [
        _leg(0, "FaZe", "under"), _leg(0, "Vitality", "under"),
        _leg(0, "FaZe", "under"), _leg(1, "Spirit", "under"),
    ]
    assert not matches_shape(legs, CROSS2X2)


def test_unknown_team_is_rejected() -> None:
    """A leg with no team cannot be verified as cross-team, so it cannot be
    priced at the cross-team multiplier."""
    legs = [
        _leg(0, "FaZe", "under"), Leg(0, 0, "under", 0.6, None, None),  # type: ignore[arg-type]
        _leg(1, "G2", "over"), _leg(1, "Spirit", "over"),
    ]
    assert not matches_shape(legs, CROSS2X2)


def test_wrong_leg_count_is_rejected() -> None:
    legs = [
        _leg(0, "FaZe", "under"), _leg(0, "Vitality", "under"),
        _leg(1, "G2", "over"),
    ]
    assert not matches_shape(legs, CROSS2X2)
