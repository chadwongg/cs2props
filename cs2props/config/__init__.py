"""Typed loaders for payout tables and slip-construction restrictions.

Both live in JSON so book policy changes are config edits, not code changes.
The optimizer consumes these; nothing here decides anything by itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Shape:
    """An exact slip STRUCTURE whose real multiplier has been observed.

    The whole EV chain rests on a multiplier the public payload does not
    expose, and the fitted `pair_penalty` is only an approximation — it
    predicts 9.25x for `cross2x2` where the app consistently shows 8.5x.
    A shape pins the price for one structure the user has actually read off
    the app, so the optimizer can search WITHIN that structure and quote a
    number that survives contact with the checkout screen.
    """

    name: str
    multiplier: float
    n_legs: int
    n_matches: int
    legs_per_match: int
    opposing_within_match: bool
    same_direction_within_match: bool


@dataclass(frozen=True)
class Payouts:
    power: dict[int, float]  # n_legs -> multiplier (all must hit)
    flex: dict[int, dict[int, float]]  # n_legs -> {hits: multiplier}
    # Correlation-adjusted multipliers: n_legs -> max legs drawn from a
    # SINGLE MATCH -> multiplier. Verified in-app 2026-07-24: the penalty
    # tracks same-MATCH concentration, not same-team. A 4-pick pays 10x with
    # at most 2 legs per match, 6.5x at 3, and 5x when all four come from one
    # match — even split 2+2 across the two opposing teams.
    correlated: dict[int, dict[int, float]]
    # Per-correlated-pair penalty, fitted to multipliers read off the app.
    # keys: base, teammate, opponent, all_one_match, floor
    pair_penalty: dict[str, float]
    # Structures whose real multiplier has been observed in-app. Preferred
    # over pair_penalty wherever one applies.
    shapes: tuple[Shape, ...] = ()

    def shape(self, name: str) -> Shape:
        for s in self.shapes:
            if s.name == name:
                return s
        known = ", ".join(s.name for s in self.shapes) or "none configured"
        raise KeyError(f"unknown shape {name!r} (known: {known})")

    def pair_multiplier(
        self, n_legs: int, teammate_pairs: int, opponent_pairs: int,
        max_same_match: int,
    ) -> float | None:
        """Multiplier from correlated-pair counts, or None if unconfigured.

        Fitted 2026-07-24 to four lineups read directly off PrizePicks:
          2 teammate pairs, 2 matches                -> 7.75x
          1 teammate + 1 opposing pair               -> 8.50x
          3 teammate pairs (3 from one team)         -> 6.50x
          all four legs in one match (2+2 opposing)  -> 5.00x
        A teammate pair costs ~3x what an opposing pair does, mirroring the
        real correlation gap measured in our own history (+0.21 vs +0.13).
        The all-in-one-match case is super-additive, so it is pinned
        explicitly rather than extrapolated. APPROXIMATE — always let
        `cs2props ev --mult <what the app shows>` have the final word.
        """
        p = self.pair_penalty
        if not p or n_legs != 4:
            return None
        if max_same_match >= 4 and "all_one_match" in p:
            return p["all_one_match"]
        mult = (
            p.get("base", 10.0)
            - teammate_pairs * p.get("teammate", 0.0)
            - opponent_pairs * p.get("opponent", 0.0)
        )
        return max(mult, p.get("floor", 1.0))

    def power_multiplier(self, n_legs: int, max_same_match: int = 1) -> float:
        table = self.correlated.get(n_legs)
        if table:
            best = None
            for cap, mult in sorted(table.items()):
                if max_same_match >= cap:
                    best = mult
            if best is not None:
                return best
        if n_legs not in self.power:
            raise KeyError(f"no power payout configured for {n_legs} legs")
        return self.power[n_legs]

    def flex_multiplier(self, n_legs: int, hits: int) -> float:
        return self.flex.get(n_legs, {}).get(hits, 0.0)


@dataclass(frozen=True)
class Restrictions:
    max_legs_per_player: int
    same_team_action: str  # "allow" | "flag" | "forbid"
    max_same_team: int
    boards_combinable: dict[str, bool]
    min_distinct_teams: int = 1  # books may require >=2 teams per slip
    default_slip_size: int = 4  # per-book: payout ladders differ
    # "power" (every leg must hit) or "flex" (partial tiers pay). Per-book
    # because the ladders differ: PrizePicks' 6-pick flex is the best-priced
    # entry on either board, while Underdog has no flex tier below 3 picks.
    default_product: str = "power"
    # False = ingest the board (its lines feed cross-book comparison) but
    # never surface slips for it. A book can be worth READING long before it
    # is worth betting.
    bet_enabled: bool = True
    # Legs from ONE match. This is what books actually price — same-team is a
    # proxy that misses cross-team pairs inside the same game.
    max_same_match: int = 99
    # "allow" | "forbid". Taking players from BOTH sides of one match is what
    # actually shifts the payout — teammates in the same match are free.
    same_match_opposing: str = "allow"
    # How many matches may supply MORE THAN ONE leg. The payout shift is
    # cumulative in same-match pairs, and only the first pair is free.
    max_multi_leg_matches: int = 99


def load_payouts(book: str, path: Path | None = None) -> Payouts:
    raw = json.loads((path or _DIR / "payouts.json").read_text())[book]
    return Payouts(
        power={int(k): float(v) for k, v in raw.get("power", {}).items()},
        flex={
            int(n): {int(h): float(m) for h, m in tiers.items()}
            for n, tiers in raw.get("flex", {}).items()
        },
        correlated={
            int(n): {int(cap): float(m) for cap, m in caps.items()}
            for n, caps in raw.get("correlated", {}).items()
        },
        pair_penalty={
            str(k): float(v) for k, v in raw.get("pair_penalty", {}).items()
        },
        shapes=tuple(
            Shape(
                name=str(s["name"]),
                multiplier=float(s["multiplier"]),
                n_legs=int(s["n_legs"]),
                n_matches=int(s["n_matches"]),
                legs_per_match=int(s["legs_per_match"]),
                opposing_within_match=bool(s.get("opposing_within_match")),
                same_direction_within_match=bool(
                    s.get("same_direction_within_match")
                ),
            )
            for s in raw.get("shapes", [])
        ),
    )


def load_restrictions(book: str, path: Path | None = None) -> Restrictions:
    raw = json.loads((path or _DIR / "restrictions.json").read_text())[book]
    stack = raw.get("same_team_stack", {})
    return Restrictions(
        max_legs_per_player=int(raw.get("max_legs_per_player", 1)),
        same_team_action=str(stack.get("action", "allow")),
        max_same_team=int(stack.get("max_same_team", 6)),
        boards_combinable={
            str(k): bool(v) for k, v in raw.get("boards_combinable", {}).items()
        },
        min_distinct_teams=int(raw.get("min_distinct_teams", 1)),
        default_slip_size=int(raw.get("default_slip_size", 4)),
        default_product=str(raw.get("default_product", "power")),
        bet_enabled=bool(raw.get("bet_enabled", True)),
        max_same_match=int(raw.get("max_same_match", 99)),
        same_match_opposing=str(
            raw.get("same_match_opposing", {}).get("action", "allow")
        ),
        max_multi_leg_matches=int(raw.get("max_multi_leg_matches", 99)),
    )
