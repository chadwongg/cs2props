"""Config loader tests: payout tables + restrictions round-trip."""

from __future__ import annotations

import pytest

from cs2props.config import load_payouts, load_restrictions


def test_prizepicks_power_table() -> None:
    p = load_payouts("prizepicks")
    # 10x is the UNSTACKED 4-pick guarantee (verified in-app 2026-07-24);
    # the 6.5x seen earlier was the correlation penalty on a 3+1 slip.
    assert p.power_multiplier(4) == 10.0
    assert p.power_multiplier(2) == 3.0


def test_unconfigured_leg_count_raises() -> None:
    p = load_payouts("prizepicks")
    with pytest.raises(KeyError):
        p.power_multiplier(9)


def test_flex_partial_credit() -> None:
    p = load_payouts("prizepicks")
    # STANDARD rates, from the app's own payout card 2026-07-26. The Arena
    # contest pays less for the same lineup (4-of-4 is 5x there, 3-of-4
    # 1.25x) — an Arena screenshot briefly overwrote these with the reduced
    # numbers. `correlated`/`shapes` carry the Arena side; this is standard.
    assert p.flex_multiplier(4, 4) == 6.0
    assert p.flex_multiplier(4, 3) == 1.5
    assert p.flex_multiplier(4, 2) == 0.0  # bust pays nothing
    assert p.flex_multiplier(3, 3) == 3.0
    assert p.flex_multiplier(3, 2) == 1.0


def test_underdog_differs_from_prizepicks() -> None:
    ud, pp = load_payouts("underdog"), load_payouts("prizepicks")
    # both ladders are verified now and the 2-pick is where they part:
    # Underdog 3.5x vs PrizePicks 3.0x, a 12.5% hold against 25%.
    assert ud.power_multiplier(2) != pp.power_multiplier(2)


def test_restrictions_shape() -> None:
    r = load_restrictions("prizepicks")
    assert r.max_legs_per_player == 1
    assert r.same_team_action in ("allow", "flag", "forbid")
    assert r.boards_combinable["goblin"] is False


def test_correlated_payout_cuts_multiplier_for_stacks() -> None:
    """Books cut the payout when you stack teammates (user-verified in app
    2026-07-24: a 4-leg stack pays ~6-7x, not 10x). The optimizer must price
    the multiplier it will ACTUALLY be paid, or it recommends -EV stacks."""
    ud = load_payouts("underdog")
    diversified = ud.power_multiplier(4, max_same_match=1)
    stacked = ud.power_multiplier(4, max_same_match=4)
    assert diversified == 10.0
    assert stacked < diversified
    # break-even for a 4-stack vs a diversified 10x slip is ~8.0x
    assert stacked < 8.0, "stack payout above break-even would flip the maths"


def test_same_match_penalty_verified_in_app() -> None:
    """Verified in the PrizePicks app 2026-07-24 across three lineups: the
    penalty tracks how many legs come from the SAME MATCH, not the same team.
    A 2+2 split across the two teams of ONE match still pays only 5x."""
    pp = load_payouts("prizepicks")
    assert pp.power_multiplier(4, max_same_match=1) == 10.0
    # 9.5 and 6.75 from fresh in-app readings 2026-08-02 — on 4-pick entries
    # even the FIRST same-match pair is charged (2+1+1 pays 9.5x, not 10x),
    # unlike the 5-pick tests where one pair rode free; the earlier 6.5x on
    # 3+1 was misread or has moved.
    assert pp.power_multiplier(4, max_same_match=2) == 9.5
    assert pp.power_multiplier(4, max_same_match=3) == 6.75
    assert pp.power_multiplier(4, max_same_match=4) == 5.0


def test_payout_falls_back_when_no_correlated_table() -> None:
    pp = load_payouts("prizepicks")
    assert pp.power_multiplier(2, max_same_match=2) == pp.power_multiplier(2)


def test_underdog_alt_not_combinable() -> None:
    r = load_restrictions("underdog")
    assert r.boards_combinable["alt"] is False


def test_per_book_default_product_and_size() -> None:
    """The books get different defaults because their ladders differ.

    Underdog: 2-pick standard, 3.5x against a fair 4x — a 12.5% hold, the
    cheapest rung on either board, and there is no flex tier below 3 picks.
    PrizePicks: 5-pick FLEX — break-even 54.25%, +9.1% at the measured
    blind-under rate. The 6-pick edges it on paper (54.21%, +13.5%) but the
    gap is inside the +/-4.6pt error on that rate and it needs a sixth
    qualifying leg the board often cannot supply. The 4-pick power this app
    shipped with is the WORST standard tier on the board (56.23%).
    """
    ud = load_restrictions("underdog")
    # 3-pick standard: 6.5x against a fair 8x, an 18.75% hold and a 53.58%
    # break-even — second-cheapest rung on either board.
    assert ud.default_slip_size == 3
    assert ud.default_product == "power"
    pp = load_restrictions("prizepicks")
    assert pp.default_slip_size == 5
    assert pp.default_product == "flex"


def test_underdog_has_no_flex_below_three_picks() -> None:
    """The 2-pick default MUST be power — pricing it as flex would read a
    tier the book does not offer."""
    ud = load_payouts("underdog")
    assert 2 not in ud.flex
    assert ud.flex_multiplier(2, 2) == 0.0


def test_five_pick_flex_has_no_two_loss_tier_on_underdog() -> None:
    """Verified 2026-07-26: Underdog pays 2 losses only at 6 picks and up.
    The config credited 0.4x for 3-of-5, an outcome that pays nothing."""
    ud = load_payouts("underdog")
    assert ud.flex_multiplier(5, 5) == 10.0
    assert ud.flex_multiplier(5, 4) == 2.5
    assert ud.flex_multiplier(5, 3) == 0.0
    assert ud.flex_multiplier(6, 4) == 0.25  # but 6-pick does pay 2 losses


def test_underdog_two_pick_is_the_best_priced_rung() -> None:
    ud = load_payouts("underdog")
    holds = {n: 1 - ud.power_multiplier(n) / (2 ** n) for n in (2, 3, 4)}
    assert holds[2] < holds[3] < holds[4]
    assert holds[2] < 0.15  # ~12.5%


def test_both_books_require_two_teams() -> None:
    """Verified in-app 2026-07-24 for both books. On Underdog's 2-pick this
    makes teammate pairs illegal outright — every slip is cross-team."""
    assert load_restrictions("underdog").min_distinct_teams == 2
    assert load_restrictions("prizepicks").min_distinct_teams == 2


def test_underdog_is_bettable_again() -> None:
    """Underdog was briefly data-only on the claim its lines were worse than
    PrizePicks'. Both figures behind that came from a query that grouped
    props without league_id, scoring half the sample against the wrong
    book's line. Corrected, model picks hit 58.4% on Underdog vs 57.0% on
    PrizePicks — the books are equivalent and there was no basis to exclude.
    """
    assert load_restrictions("underdog").bet_enabled is True
    assert load_restrictions("prizepicks").bet_enabled is True


def test_bet_enabled_defaults_to_true() -> None:
    """A book with no explicit flag is bettable — data-only is opt-in, so a
    new book is never silently muted."""
    from cs2props.config import Restrictions

    assert Restrictions(1, "flag", 3, {}).bet_enabled is True


def test_underdog_forbids_any_same_match_pair() -> None:
    """Underdog shades harder than PrizePicks: a same-team pair shifts the
    payout there, while PrizePicks gives the first same-match pair away free
    (both measured in-app 2026-07-26). The observed Underdog pair was both
    same-team AND same-match, so the readings cannot be separated — the
    conservative rule satisfies either.
    """
    ud = load_restrictions("underdog")
    assert ud.max_multi_leg_matches == 0
    assert ud.max_same_match == 1
    pp = load_restrictions("prizepicks")
    assert pp.max_multi_leg_matches == 1   # first pair is free
    assert pp.max_same_match == 2

def test_kills_only_policy_on_both_books() -> None:
    """USER POLICY 2026-08-02: no headshot legs in live slips. This is a
    preference, not an evidence call — the backtest scored headshot picks
    at 54.6% vs kills 51.8% (p>=0.55) — so if the config ever loosens it,
    that is a deliberate edit, not a regression."""
    assert load_restrictions("prizepicks").bettable_stats == {"kills"}
    assert load_restrictions("underdog").bettable_stats == {"kills"}


def test_bettable_stats_defaults_to_all() -> None:
    """A book with no explicit list may bet every stat — kills-only is
    opt-in policy, never a silent default."""
    from cs2props.config import Restrictions

    assert Restrictions(1, "flag", 3, {}).bettable_stats == {
        "kills", "headshots"
    }
