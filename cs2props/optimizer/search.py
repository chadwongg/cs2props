"""4-man power slip search over joint Monte Carlo samples.

Locked spec (user-approved):
- Fixed 4-man power target, stack-first construction.
- Marginal-leg test with REFUSAL: the 4th leg is admitted only when its
  conditional probability given the core — measured on the joint samples,
  never assumed — beats the payout ratio plus a safety margin. No qualifying
  4th leg -> emit the 3-man with the best rejected candidate shown.
  No qualifying core at all -> "no slips today" is the correct output.
- Restrictions config enforced: one leg per player, at most one same-match
  PAIR (the payout shift is cumulative in pairs — measured 2026-07-26:
  1 pair 10x, 2 pairs 8x, 3 pairs 7x, with team irrelevant), standard boards
  only (already filtered upstream).
- P(all) is always read off the joint hit matrix. Cross-match combinations
  AND independent iteration streams together, which is valid Monte Carlo
  because separate matches are simulated independently.
- Voids (map-3 in a sweep, whole-line pushes are settled by the tracker;
  here map-range voids) shrink the slip: payout uses the live-leg count's
  multiplier, below 2 live legs the stake refunds.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from cs2props.config import Payouts, Restrictions, Shape
from cs2props.ingest.prizepicks import Prop
from cs2props.pipeline import MatchSim

log = logging.getLogger(__name__)

MIN_LEG_P = 0.55  # a leg must clear this marginal probability to be a candidate
MARGINAL_SAFETY = 0.02  # 4th-leg conditional must beat mult ratio by this
MIN_SLIP_EV = 0.0  # slips at or below breakeven are refused
# The model is known to run optimistic — measured ~2 points per leg on the
# engine's marginals. A slip whose edge does not survive that haircut is not
# an edge, it is noise wearing a percentage. Surfacing the whole ranked list
# invites betting the tail: on 2026-07-25 the user placed seven Underdog
# slips whose post-haircut EV ran from +50% down to +6%, and the bottom ones
# added variance without adding expectation.
LEG_OPTIMISM = 0.02  # DEFAULT only — cs2props.adaptive learns this from
                     # graded legs and the CLI passes the learned value in
MIN_ADJUSTED_EV = 0.10  # required EV AFTER the haircut. Set to 0.10 (not
                     # 0.20) during the measurement phase: at $1 stakes the
                     # cost of a marginal bet is trivial, while a day with no
                     # bets is a day with no data, and CLV needs ~150 legs.
TOP_LEGS_PER_MATCH = 6  # search-width control
EXHAUSTIVE_POOL = 24  # C(24,4) = 10,626 combos — exact and fast enough
# Enumeration is exact but combinatorial: C(24,6) is 134,596 slips, each
# needing a full pass over the joint samples. The pool is narrowed for larger
# sizes so the search stays exhaustive over the legs that could plausibly
# make the cut, rather than turning greedy over all of them.
POOL_FOR_SIZE = {5: 18, 6: 15}  # C(18,5)=8,568  C(15,6)=5,005
MAX_SLIPS = 5
# Ranking by EV alone returns near-duplicate slips (they share legs, so they
# share EV). Placing two of them is NOT two independent shots: shared legs
# mean shared model error, higher variance, and a smaller validation sample
# for the same money. Every surfaced slip must therefore be substantially
# different from the ones above it.
# 0 = surfaced slips are fully independent bets. Set deliberately: during
# live validation, two entries sharing even one leg are not two data points,
# and the shared leg becomes the bettor's largest single exposure without
# them noticing. Raise to 1 only if a thin board leaves too few slips.
MAX_SHARED_PLAYERS = 0
# Zero shared MATCHES too. Distinct players is not enough: two slips can draw
# from the same games via different players (found 2026-07-24 — two "diverse"
# slips both rode Astralis-HOTU and Liquid-G2, so they would have died
# together). Matches are the unit of shared fate, because every leg in a game
# shares its round count.
MAX_SHARED_MATCHES = 0


@dataclass(frozen=True)
class Leg:
    sim_idx: int
    prop_idx: int
    side: str  # "over" | "under"
    p: float  # marginal, refund-excluded
    prop: Prop
    team: str | None

    @property
    def payout_multiplier(self) -> float:
        """Per-side price shade on this leg, 1.0 when the side is balanced.

        Underdog prices each side independently even on "balanced" lines, and
        the slip pays the ladder multiplier TIMES the product of its legs'
        shades. A single 0.82 leg turns a 6.5x 3-pick into 5.33x — enough to
        move a slip from +EV to -EV without anything about the projection
        changing.
        """
        return self.prop.side_multiplier(self.side)


@dataclass
class Slip:
    legs: list[Leg]
    p_all: float
    p_independent: float
    ev: float
    multiplier: float
    flags: list[str] = field(default_factory=list)
    note: str | None = None
    product: str = "power"  # "power" | "flex"
    # P(exactly k legs hit), k = 0..n, read off the joint samples. Only
    # populated for flex, where every tier matters, not just the top one.
    k_probs: tuple[float, ...] = ()
    # EV with the per-leg haircut applied by THINNING the joint hit matrix
    # rather than scaling a single probability — for flex there is no single
    # probability to scale, and the tiers do not move together.
    adjusted_ev_flex: float | None = None

    n_iters: int = 0  # Monte Carlo draws behind p_all, for the noise floor

    @property
    def delta_pts(self) -> float:
        return (self.p_all - self.p_independent) * 100

    @property
    def delta_noise_pts(self) -> float:
        """95% Monte Carlo noise floor on ``delta_pts``, in points.

        p_all is an average over n_iters draws, so it carries a standard
        error of sqrt(p(1-p)/n). On a fully diversified slip the true delta
        is ZERO — separate matches are simulated independently, so nothing
        links them — and whatever prints is that error. Slip #2 on
        2026-07-26 showed -0.1 pts, a NEGATIVE correlation between
        independent events, which is impossible and was the giveaway.
        """
        if self.n_iters <= 0:
            return 0.0
        se = math.sqrt(max(self.p_all * (1 - self.p_all), 0.0) / self.n_iters)
        return 1.96 * se * 100

    @property
    def delta_is_real(self) -> bool:
        """Is the correlation bonus distinguishable from sampling noise?

        Printing a noise value beside a real one invites reading it as a
        finding. It is only ever real when the slip carries a same-match
        pair, which the payout rules mostly forbid — so this is usually
        False, and that is the honest answer.
        """
        return abs(self.delta_pts) > self.delta_noise_pts

    haircut: float = LEG_OPTIMISM

    @property
    def adjusted_ev(self) -> float:
        """EV after shaving LEG_OPTIMISM off every leg.

        Scaling P(all) by the product of (p-haircut)/p per leg is an
        approximation — the joint is not a product of marginals — but it is
        the right direction and magnitude, and it is what separates a real
        edge from one that exists only if the model is exactly right.

        Flex cannot use that shortcut: its payout reads several tiers of the
        hit distribution, and shrinking the top one says nothing about what
        happens to "4 of 5". Flex slips therefore carry a precomputed
        ``adjusted_ev_flex`` measured on a thinned hit matrix.
        """
        if self.product == "flex":
            return (
                self.adjusted_ev_flex
                if self.adjusted_ev_flex is not None else self.ev
            )
        if not self.legs or self.p_all <= 0:
            return self.ev
        shrink = 1.0
        for leg in self.legs:
            shrink *= max(leg.p - self.haircut, 0.01) / leg.p
        return self.multiplier * self.p_all * shrink - 1.0

    @property
    def kelly_growth(self) -> float:
        """Log-growth per bet at the Kelly-optimal stake.

        The criterion that actually ranks products against each other. EV per
        dollar prefers whatever pays biggest on the top tier; growth prices
        the variance you carry to get it, which is why 5-pick flex beats
        4-pick power despite lower headline EV.
        """
        outcomes = self._outcomes()
        if not outcomes:
            return 0.0
        best = 0.0
        for i in range(1, 201):
            f = i / 200.0
            g = 0.0
            ok = True
            for q, mult in outcomes:
                if q <= 0:
                    continue
                w = 1.0 - f + f * mult
                if w <= 1e-12:
                    ok = False
                    break
                g += q * math.log(w)
            if ok:
                best = max(best, g)
        return best

    def _outcomes(self) -> list[tuple[float, float]]:
        """[(probability, payout multiple)] covering the whole outcome space.

        Power needs no hit distribution — every non-win pays zero, so the bet
        collapses to two outcomes. Requiring ``k_probs`` here made power slips
        silently report zero growth and lose every product comparison they
        were entered in.
        """
        n = len(self.legs)
        if self.product == "power":
            if self.p_all <= 0:
                return []
            return [(self.p_all, self.multiplier), (1.0 - self.p_all, 0.0)]
        if not self.k_probs:
            return []
        table = self._flex_table or {}
        return [(q, table.get(k, 0.0)) for k, q in enumerate(self.k_probs)]

    _flex_table: dict[int, float] | None = None

    @property
    def breakeven_multiplier(self) -> float:
        """Multiplier the book must pay for this slip to be +EV.

        This is the honest headline. PrizePicks Arena derives its multiplier
        from per-pick pricing that is NOT exposed in the public projections
        payload — verified 2026-07-24, when two slips with identical
        structure paid 7.75x and 8.00x while every visible field matched. Any
        EV we print rests on a GUESSED multiplier; the break-even does not.
        Read the real multiplier in the app and compare against this.
        """
        return float("inf") if self.p_all <= 0 else 1.0 / self.p_all


def _leg_arrays(sims: list[MatchSim], leg: Leg) -> tuple[np.ndarray, np.ndarray]:
    """(hit, live) boolean arrays for a leg. A voided leg counts as passed
    for the all-live-hit test but shrinks the live-leg count.

    Two things void a leg: the map range never completing, and the total
    landing EXACTLY on a whole-number line. The second was missing, and it
    is not rare — 23.5% of PrizePicks lines are whole numbers and they push
    6.1% of the time (7.6% on headshots). ``~hits`` treats a push as an
    under WIN at full 4-leg payout, when the book actually voids the leg and
    pays the 3-leg multiplier. Every under this app recommended on a
    whole-number line carried that 6.1-point overstatement.
    """
    res = sims[leg.sim_idx].result
    hits = res.hits[:, leg.prop_idx]
    pushed = res.pushes(leg.prop_idx)
    live = ~res.short[:, leg.prop_idx] & ~pushed
    hit = hits if leg.side == "over" else (~hits & ~pushed)
    return hit | ~live, live


def evaluate(
    sims: list[MatchSim], legs: list[Leg], payouts: Payouts
) -> tuple[float, float, float, float]:
    """-> (p_all_live_hit, p_independent, ev_per_unit, headline_multiplier)."""
    n_iters = sims[legs[0].sim_idx].result.hits.shape[0]
    all_hit = np.ones(n_iters, dtype=bool)
    n_live = np.zeros(n_iters, dtype=np.int16)
    for leg in legs:
        hit, live = _leg_arrays(sims, leg)
        all_hit &= hit
        n_live += live.astype(np.int16)
    # payout per iteration: full-live multiplier, shrunk on voids, refund <2
    # multiplier depends on STRUCTURE, not just leg count. The penalty tracks
    # SAME-MATCH concentration (verified in-app): 4 legs from one match pays
    # 5x where a 2-per-match spread pays 10x. Since same-match legs are also
    # the most correlated, the book is charging for exactly the dependency
    # the engine measures — so it must be priced, not maximised.
    stack = _same_match_count(legs)
    payout = np.zeros(n_iters)
    for k in range(len(legs), -1, -1):
        mask = all_hit & (n_live == k)
        if not mask.any():
            continue
        if k < 2:
            payout[mask] = 1.0
            continue
        try:
            payout[mask] = payouts.power_multiplier(
                k, max_same_match=min(stack, k)
            )
        except KeyError:
            payout[mask] = 0.0
    p_all = float(all_hit.mean())
    p_ind = float(np.prod([l.p for l in legs]))
    ev = float(payout.mean()) - 1.0
    try:
        headline = payouts.power_multiplier(len(legs), max_same_match=stack)
    except KeyError:
        headline = 0.0
    return p_all, p_ind, ev, headline


def collect_legs(sims: list[MatchSim]) -> list[Leg]:
    """Candidate legs: both sides of every modeled prop clearing MIN_LEG_P,
    best line per player (one leg per player, per restrictions)."""
    best_by_player: dict[str, Leg] = {}
    for si, sim in enumerate(sims):
        for pi, (prop, p_over) in enumerate(zip(sim.props, sim.result.p_over)):
            for side, p in (("over", p_over), ("under", 1.0 - p_over)):
                if p < MIN_LEG_P:
                    continue
                leg = Leg(si, pi, side, p, prop, prop.team)
                cur = best_by_player.get(prop.player_name)
                if cur is None or p > cur.p:
                    best_by_player[prop.player_name] = leg
    return sorted(best_by_player.values(), key=lambda l: -l.p)


def _same_team_count(legs: list[Leg]) -> int:
    teams = [l.team for l in legs if l.team]
    return max((teams.count(t) for t in set(teams)), default=0)


def slip_price_factor(legs: list[Leg]) -> float:
    """Product of the legs' per-side payout shades."""
    factor = 1.0
    for leg in legs:
        factor *= leg.payout_multiplier
    return factor


def _legs_per_match(legs: list[Leg]) -> dict[int, int]:
    out: dict[int, int] = {}
    for leg in legs:
        out[leg.sim_idx] = out.get(leg.sim_idx, 0) + 1
    return out


def _same_match_count(legs: list[Leg]) -> int:
    """Most legs drawn from any single match — drives the payout penalty."""
    counts = _legs_per_match(legs)
    return max(counts.values(), default=0)


def _has_opposing_pair(legs: list[Leg]) -> bool:
    """Any two legs drawn from opposite sides of the same match.

    This is the structure the book charges for. A leg whose team is unknown
    cannot be cleared, so it is treated as opposing anything else in its
    match — the conservative direction, since a false rejection costs one
    slip while a false pass costs an unpriced payout cut on a real bet.
    """
    by_match: dict[int, list[Leg]] = {}
    for leg in legs:
        by_match.setdefault(leg.sim_idx, []).append(leg)
    for group in by_match.values():
        if len(group) < 2:
            continue
        teams = {l.team for l in group}
        if None in teams or len(teams) > 1:
            return True
    return False


def matches_shape(legs: list[Leg], shape: Shape) -> bool:
    """Does this combination have exactly the observed structure?

    Structure is checked, not approximated: the point of a shape is that its
    real multiplier is KNOWN, and that only holds if the slip is genuinely
    the thing that was priced. A near-miss gets the fitted estimate instead,
    which is the very uncertainty shapes exist to remove.
    """
    if len(legs) != shape.n_legs:
        return False
    by_match: dict[int, list[Leg]] = {}
    for leg in legs:
        by_match.setdefault(leg.sim_idx, []).append(leg)
    if len(by_match) != shape.n_matches:
        return False
    for group in by_match.values():
        if len(group) != shape.legs_per_match:
            return False
        if shape.opposing_within_match:
            teams = {l.team for l in group}
            if None in teams or len(teams) != len(group):
                return False
        if shape.same_direction_within_match:
            if len({l.side for l in group}) != 1:
                return False
    return True


def is_submittable(legs: list[Leg], restrictions: Restrictions) -> bool:
    """Would the book actually accept this slip?

    Verified live 2026-07-24: PrizePicks rejects single-team slips with
    "Picks must be from at least two different teams" — a rule that applies
    at EVERY slip size, so a 3-leg one-team stack is just as illegal as a
    4-leg one. Checked on the final slip, never only on the core.
    """
    if len(legs) < 2:
        return False
    # One leg per player — matched LOOSELY. The board posts the same player
    # under more than one spelling ("910" and "910-" were both live on
    # 2026-07-26, and the optimizer happily built a slip using both). Exact
    # string comparison passes that, and the book rejects the entry.
    from cs2props.standins import _same_person

    names = [l.prop.player_name for l in legs]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _same_person(a, b):
                return False
    if _same_team_count(legs) > restrictions.max_same_team:
        return False
    if _same_match_count(legs) > restrictions.max_same_match:
        return False
    # The shift is CUMULATIVE in same-match pairs and ignores team entirely:
    # 2+1+1+1 pays 10x, 2+2+1 pays 8x, 3+1+1 pays 7x (measured in-app
    # 2026-07-26). Only the FIRST match supplying a second leg is free, so a
    # per-match cap alone is not enough — a slip can sit at two legs in every
    # match and still be shaded.
    multi = sum(1 for n in _legs_per_match(legs).values() if n > 1)
    if multi > restrictions.max_multi_leg_matches:
        return False
    known_teams = {l.team for l in legs if l.team}
    if restrictions.min_distinct_teams > 1:
        if len(known_teams) < restrictions.min_distinct_teams:
            return False
    return True


def _flags(legs: list[Leg], restrictions: Restrictions) -> list[str]:
    """Notes shown beside a slip. These describe the PAYOUT, so they have to
    track the measured rule — an earlier version said "only opposing players
    shift the payout", which was disproven the same day by a lineup of three
    teammates that shifted it exactly as much as two teammates plus an
    opponent. Team is irrelevant; the count of same-match PAIRS is what the
    book charges for.
    """
    out = []
    per_match = _legs_per_match(legs)
    pairs = sum(1 for n in per_match.values() if n > 1)
    biggest = max(per_match.values(), default=0)
    shade = slip_price_factor(legs)
    if abs(shade - 1.0) > 1e-9:
        worst = min(legs, key=lambda l: l.payout_multiplier)
        out.append(
            f"payout shaded x{shade:.2f} by per-side pricing "
            f"({worst.prop.player_name} {worst.side} pays "
            f"{worst.payout_multiplier:g}x) — already priced into the EV above"
        )
    # Measured in-app 2026-07-26 on 5-pick lineups:
    #   1 pair -> 10x (full)    2 pairs -> 8x    3 pairs / 3-in-one -> 7x
    # A slip carrying MORE than the configured allowance is rejected outright
    # by is_submittable, so this branch should be unreachable. It is kept as a
    # canary: if it ever prints, the structural enforcement has broken and the
    # quoted payout is wrong. The "1 pair is free" case is deliberately
    # SILENT — the header already states what the slip pays, and a flag
    # announcing that nothing is wrong is noise.
    if pairs > restrictions.max_multi_leg_matches or biggest > restrictions.max_same_match:
        out.append(
            f"BUG: {pairs} same-match pairs, up to {biggest} legs from one "
            "match — the payout WILL shift (2 pairs ~-20%, 3 ~-30%) and this "
            "slip should not have been surfaced. Read the real multiplier."
        )
    n_stack = _same_team_count(legs)
    if n_stack > restrictions.max_same_team:
        out.append(
            f"{n_stack} legs from one team — exceeds the configured max, "
            "the book may reject the entry"
        )
    return out


def _build_pool(sims: list[MatchSim], shape: Shape | None = None,
                size: int | None = None) -> list[Leg]:
    """Strongest candidate legs, capped per match and overall.

    When the target shape needs opposing teams inside a match, the per-match
    cap is split evenly between the two sides. Taking the top 6 legs outright
    would routinely hand back six legs from the stronger team, leaving no
    legal cross-team pair in that match at all.
    """
    legs = collect_legs(sims)
    by_match: dict[int, list[Leg]] = {}
    for l in legs:
        by_match.setdefault(l.sim_idx, []).append(l)
    pool: list[Leg] = []
    for _si, ls in by_match.items():
        if shape is not None and shape.opposing_within_match:
            per_team = max(TOP_LEGS_PER_MATCH // 2, shape.legs_per_match)
            by_team: dict[str | None, list[Leg]] = {}
            for l in ls:
                by_team.setdefault(l.team, []).append(l)
            for team, tls in by_team.items():
                if team is not None:
                    pool.extend(tls[:per_team])
        else:
            pool.extend(ls[:TOP_LEGS_PER_MATCH])
    cap = EXHAUSTIVE_POOL if shape is None else EXHAUSTIVE_POOL * 2
    if size is not None:
        cap = min(cap, POOL_FOR_SIZE.get(size, cap))
    return sorted(pool, key=lambda l: -l.p)[:cap]


def _best_of_size(
    sims: list[MatchSim],
    pool: list[Leg],
    size: int,
    payouts: Payouts,
    restrictions: Restrictions,
    min_adjusted_ev: float = MIN_ADJUSTED_EV,
    haircut: float = LEG_OPTIMISM,
    shape: Shape | None = None,
) -> list[Slip]:
    """Exhaustively score every submittable combination of `size` legs.

    Exhaustive rather than greedy: the old "best 3-core, then extend"
    heuristic ranked cores by P(all), which always surfaces same-match trios,
    so the highest-EV DIVERSIFIED slips were discarded before being priced
    (measured: greedy returned +150% where enumeration found +165%).

    With ``shape`` set, only combinations matching that structure are scored
    and its OBSERVED multiplier is used in place of the fitted table.
    """
    hit_of: dict[int, np.ndarray] = {}
    any_void = False
    for leg in pool:
        h, lv = _leg_arrays(sims, leg)
        hit_of[id(leg)] = h
        if not lv.all():
            any_void = True

    out: list[Slip] = []
    for combo in itertools.combinations(pool, size):
        legs = list(combo)
        if not is_submittable(legs, restrictions):
            continue
        if shape is not None and not matches_shape(legs, shape):
            continue
        if shape is not None:
            # observed price — no void handling needed beyond the live test,
            # since a voided leg would change the structure the price is for
            ok = hit_of[id(legs[0])]
            for leg in legs[1:]:
                ok = ok & hit_of[id(leg)]
            p_all = float(ok.mean())
            p_ind = float(np.prod([l.p for l in legs]))
            mult = shape.multiplier
            ev = mult * p_all - 1.0
        elif any_void:
            p_all, p_ind, ev, mult = evaluate(sims, legs, payouts)
        else:
            ok = hit_of[id(legs[0])]
            for leg in legs[1:]:
                ok = ok & hit_of[id(leg)]
            p_all = float(ok.mean())
            p_ind = float(np.prod([l.p for l in legs]))
            try:
                mult = payouts.power_multiplier(
                    size, max_same_match=_same_match_count(legs)
                )
            except KeyError:
                continue
            mult *= slip_price_factor(legs)
            ev = mult * p_all - 1.0
        if ev <= MIN_SLIP_EV:
            continue
        slip = Slip(
            legs=legs, p_all=p_all, p_independent=p_ind, ev=ev,
            multiplier=mult, flags=_flags(legs, restrictions),
            haircut=haircut, n_iters=int(hit_of[id(legs[0])].shape[0]),
        )
        if slip.adjusted_ev < min_adjusted_ev:
            continue  # edge does not survive the model's known optimism
        out.append(slip)
    out.sort(key=lambda s: -s.ev)
    return out


def _thin(hit: np.ndarray, p: float, haircut: float,
          rng: np.random.Generator) -> np.ndarray:
    """Drop a haircut's worth of a leg's hits, preserving correlation.

    Flex reads several tiers of the hit distribution, so the power trick of
    scaling one probability does not apply — shrinking P(5 of 5) says
    nothing about what happens to P(4 of 5). Instead each leg's hits are
    thinned at rate ``haircut / p``, which lowers that leg's marginal to
    ``p - haircut`` while leaving the joint structure intact, because the
    thinning is applied to the SAME iterations the correlation lives in.
    """
    if haircut <= 0 or p <= 0:
        return hit
    keep = 1.0 - min(haircut / p, 1.0)
    return hit & (rng.random(hit.shape[0]) < keep)


def _best_flex(
    sims: list[MatchSim],
    pool: list[Leg],
    size: int,
    payouts: Payouts,
    restrictions: Restrictions,
    min_adjusted_ev: float = MIN_ADJUSTED_EV,
    haircut: float = LEG_OPTIMISM,
    seed: int = 11,
) -> list[Slip]:
    """Score flex slips of ``size`` legs over the full hit distribution.

    Flex pays on "k of n correct", so the whole distribution matters, not
    P(all). That is also why it beats power on log-growth at 5 and 6 legs:
    the lower tiers cut variance, which lets a Kelly bettor stake more, and
    the larger stake compounds faster than the headline EV it gives up.
    """
    base_table = payouts.flex.get(size)
    if not base_table:
        return []
    rng = np.random.default_rng(seed)
    hit_of: dict[int, np.ndarray] = {}
    thin_of: dict[int, np.ndarray] = {}
    for leg in pool:
        h, live = _leg_arrays(sims, leg)
        # a voided leg neither hits nor counts against you; treating it as a
        # hit here would pay the full-size tier on a slip the book shrank
        won = h & live
        hit_of[id(leg)] = won
        thin_of[id(leg)] = _thin(won, leg.p, haircut, rng)

    out: list[Slip] = []
    for combo in itertools.combinations(pool, size):
        legs = list(combo)
        if not is_submittable(legs, restrictions):
            continue
        k = np.zeros(hit_of[id(legs[0])].shape[0], dtype=np.int8)
        kt = np.zeros_like(k)
        for leg in legs:
            k += hit_of[id(leg)]
            kt += thin_of[id(leg)]
        probs = tuple(float((k == j).mean()) for j in range(size + 1))
        # per-side shades scale every paying tier, not just the top one
        shade = slip_price_factor(legs)
        table = {j: m * shade for j, m in base_table.items()}
        ev = sum(probs[j] * table.get(j, 0.0) for j in range(size + 1)) - 1.0
        if ev <= MIN_SLIP_EV:
            continue
        adj = sum(
            float((kt == j).mean()) * table.get(j, 0.0)
            for j in range(size + 1)
        ) - 1.0
        if adj < min_adjusted_ev:
            continue
        out.append(Slip(
            legs=legs, p_all=probs[size],
            p_independent=float(np.prod([l.p for l in legs])),
            ev=ev, multiplier=table.get(size, 0.0),
            flags=_flags(legs, restrictions), haircut=haircut,
            product="flex", k_probs=probs, adjusted_ev_flex=adj,
            _flex_table=dict(table), n_iters=int(k.shape[0]),
        ))
    out.sort(key=lambda s: -s.kelly_growth)
    return out


def diversify(slips: list[Slip], limit: int) -> list[Slip]:
    """Greedily pick the best slips that don't reuse each other's players.

    EV ranking alone surfaces near-duplicates: overlapping slips have almost
    identical EV, so the top five are usually the same bet five ways. Taking
    two of those is not two independent shots — the shared legs carry shared
    model error, so they lose together, raise variance, and yield fewer
    effective data points for live calibration.
    """
    chosen: list[Slip] = []
    for slip in slips:
        names = {l.prop.player_name for l in slip.legs}
        matches = {l.sim_idx for l in slip.legs}
        clash = False
        for c in chosen:
            c_names = {l.prop.player_name for l in c.legs}
            c_matches = {l.sim_idx for l in c.legs}
            if len(names & c_names) > MAX_SHARED_PLAYERS:
                clash = True
                break
            if len(matches & c_matches) > MAX_SHARED_MATCHES:
                clash = True
                break
        if clash:
            continue
        chosen.append(slip)
        if len(chosen) >= limit:
            break
    return chosen


def search_slips(
    sims: list[MatchSim],
    payouts: Payouts,
    restrictions: Restrictions,
    target_size: int = 4,
    min_adjusted_ev: float = MIN_ADJUSTED_EV,
    haircut: float = LEG_OPTIMISM,
    shape: Shape | None = None,
    product: str = "power",
) -> tuple[list[Slip], str | None]:
    """-> (ranked slips at ``target_size``, refusal reason if none qualify).

    Targets ``target_size`` legs but REFUSES to pad: if the best legal slip
    one leg SHORTER carries more EV, that shorter slip is returned with a
    note explaining what was given up. Refusal compares realised EV (which
    prices the book's payout penalties), not a conditional-probability proxy.
    "No slips today" is a valid answer.

    Size is a per-book choice because the payout ladders differ sharply.
    Measured 2026-07-24 on Underdog: a 2-pick pays 3.5x against a fair 4x —
    a 12.5% hold — while the 3-pick (6x) charges 25% and the 4-pick (10x)
    charges 37.5%. Adding a third leg doubles the vig, a fourth triples it,
    and the bigger slips also decay faster under model error. PrizePicks'
    ladder does not have that break, so it stays on 4-picks.
    """
    if product == "flex":
        pool = _build_pool(sims, None, target_size)
        if len(pool) < target_size:
            return [], (
                f"only {len(pool)} legs clear the {MIN_LEG_P:.0%} bar — "
                f"need {target_size} for a {target_size}-pick flex"
            )
        flex = _best_flex(sims, pool, target_size, payouts, restrictions,
                          min_adjusted_ev, haircut)
        if not flex:
            return [], (
                f"no {target_size}-pick flex clears {min_adjusted_ev:.0%} EV "
                "after the model's optimism haircut — no slips today"
            )
        return diversify(flex, MAX_SLIPS), None

    pool = _build_pool(sims, shape)
    floor = max(target_size - 1, 2)
    if len(pool) < floor:
        return [], f"only {len(pool)} legs clear the {MIN_LEG_P:.0%} bar"

    if shape is not None:
        # A shape is a fixed structure at a known price. Shortening it would
        # break the structure, so the shorter-slip refusal below does not
        # apply — the honest answer when nothing fits is no slips.
        fixed = _best_of_size(sims, pool, shape.n_legs, payouts, restrictions,
                              min_adjusted_ev, haircut, shape)
        if not fixed:
            return [], (
                f"no {shape.name} slip ({shape.n_matches} matches x "
                f"{shape.legs_per_match} opposing legs) clears "
                f"{min_adjusted_ev:.0%} EV at the observed "
                f"{shape.multiplier:g}x — no slips today"
            )
        return diversify(fixed, MAX_SLIPS), None

    full = _best_of_size(sims, pool, target_size, payouts, restrictions,
                         min_adjusted_ev, haircut)
    shorter = (
        _best_of_size(sims, pool, target_size - 1, payouts, restrictions,
                      min_adjusted_ev, haircut)
        if target_size - 1 >= 2 else []
    )
    best_full = full[0].ev if full else float("-inf")
    best_short = shorter[0].ev if shorter else float("-inf")

    if shorter and best_short > best_full + MARGINAL_SAFETY:
        top = shorter[0]
        if full:
            lost = best_short - full[0].ev
            top.note = (
                f"wanted {target_size} — every legal extra leg LOWERS EV "
                f"(best {target_size}-man {full[0].ev * 100:+.0f}% at "
                f"{full[0].multiplier:g}x vs {best_short * 100:+.0f}% here, "
                f"-{lost * 100:.0f} pts) — REFUSED"
            )
        else:
            top.note = (
                f"wanted {target_size} — no legal {target_size}-leg slip "
                "clears breakeven"
            )
        return [top, *diversify(full, MAX_SLIPS - 1)], None

    if not full:
        return [], (
            f"no slip clears {min_adjusted_ev:.0%} EV after the model's "
            "optimism haircut — no slips today"
        )
    return diversify(full, MAX_SLIPS), None
