"""Generative Monte Carlo match simulator.

Explicitly NOT a correlation-matrix / Gaussian-copula approach: the series
length variable makes prop dependencies regime-switching (a maps-1-3 prop
behaves differently in a 2-map sweep than a 3-map series), so we simulate the
generative process per iteration:

1. **Series length** — map winners are drawn sequentially from the map-win
   probability implied by team strength; the series ends at the Bo clinch.
   This is the key latent variable and is exposed per-iteration in the result.
2. **Form shocks** — one Gamma shock per team per series (teams have good and
   bad days; this drives same-team correlation), one per player per series
   (continuity with the projection model), one per player per map.
3. **Within-map near-zero-sum** — raw Poisson kills for all ten players are
   rescaled so the map's total kills track rounds x total-kills-per-round.
   Kills compete: an opponent's hot map eats into everyone else's counts,
   which is what makes cross-team OVER/OVER weaker than same-team stacks.
4. Every prop is evaluated on every iteration, honoring its map range and
   whether those maps were actually played.

Output is the joint hit matrix (iterations x props) the optimizer consumes —
P(all legs) is a column-AND, never a product of marginals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cs2props.model.projector import PHI_MAP, PHI_SERIES

PHI_TEAM = 0.04  # variance of the per-series team form shock
DEFAULT_TOTAL_KPR = 6.5  # total kills per round, both teams (league-typical)
DEFAULT_HS_RATE = 0.45
ROUNDS_MEAN, ROUNDS_SD = 21.5, 4.0
SCALE_CLIP = (0.4, 2.5)  # bounds on the per-team zero-sum rescale factor
# Game-script coupling (added 2026-07-24). A stomp is SHORT: 13-3 is 16 rounds
# against 24 for a 13-11, so the winner's higher kill rate is largely offset
# by lost rounds while the loser is hit twice. Without this link the engine
# under-modelled the single strongest correlation channel in CS2 props.
# Constants below are FITTED to correlations measured on our own 153k
# map-rows (see tests/test_correlation.py::test_matches_measured_coupling):
#   real phi(teammate pairs) = +0.210, phi(cross-team pairs) = +0.132,
#   real CV of maps-1-2 kill totals = 0.275.
# Note both real couplings are POSITIVE and similar in size: for a maps-1-2
# prop the dominant shared factor is TOTAL ROUNDS PLAYED, which lifts all ten
# players together and outweighs within-map kill competition.
TEAM_FORM_TO_WIN = 0.6  # how hard the series form shock tilts map wins
# 0.35, not 0.50: kills ~ rounds x share are negatively correlated, so a
# larger beta drags E[kills] down through the covariance term (-1.4% mean at
# 0.50, measured). A mean bias corrupts every leg's probability, so it is
# worth more than the sharper cross-team correlation a higher beta buys.
KILL_SHARE_BETA = 0.35  # how hard round share pulls kill share
TEAM_KILL_SD = 0.11  # team-level "good kill night" noise, survives rescale
TOTAL_KPR_SD = 0.30  # match-level pace noise on the total-kills anchor
SHARE_CLIP = (0.15, 0.85)
# Full per-team rescale. Softening this (0.4-0.8) was tried on the theory
# that it would restore symmetric Poisson noise and cut the engine's excess
# right-skew; MEASURED RESULT: shape did not move (0.441 -> 0.443) and the
# mean drifted 1.3% below the projector, because raising a non-log-centred
# scale to a fractional power shifts its mean. Keep at 1.0.
RESCALE_STRENGTH = 1.0
# Headshots are OVERDISPERSED relative to binomial: measured on 771 players
# with 30+ series, the observed variance is 1.20x what binomial(kills,
# hs_rate) allows. Players have hot and cold headshot nights beyond simple
# per-kill randomness. Modelling them as plain binomial made big headshot
# games rarer than reality and so made every headshot UNDER look better than
# it was — visible in the first 28 graded legs, where losing headshot unders
# missed by ~6.5 while winners cleared by ~3.3.
# Beta-binomial with concentration K gives var = n*p*(1-p)*(1+(n-1)/(K+1));
# at a typical n≈30 kills, K=144 reproduces the measured 1.20x.
HS_DISPERSION_K = 144.0
REG_MAX_ROUNDS = 24  # MR12: 13-11 ends regulation
WIN_ROUNDS_REG = 13


_LOGISTIC_VAR = math.pi**2 / 3.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_mu(shift: np.ndarray, p_target: float) -> float:
    """Find mu with mean(sigmoid(mu + shift)) == p_target.

    Keeps P(A wins a map) exactly at the strength-implied value even though
    the form shock now perturbs individual maps — so series-length behaviour
    (and its analytic test) is unchanged by the coupling.
    """
    mu = float(np.log(p_target / (1.0 - p_target)))
    for _ in range(8):
        p = _sigmoid(mu + shift)
        err = float(p.mean()) - p_target
        deriv = float((p * (1.0 - p)).mean())
        if deriv < 1e-9:
            break
        mu -= err / deriv
        if abs(err) < 1e-9:
            break
    return mu


@dataclass(frozen=True)
class PlayerSpec:
    player_id: str
    name: str
    side: str  # "A" | "B"
    kpr: float  # projected kills per round (module 2 output)
    hs_rate: float = DEFAULT_HS_RATE


@dataclass(frozen=True)
class PropSpec:
    player_id: str
    stat: str  # "kills" | "headshots"
    map_lo: int  # 1-based inclusive
    map_hi: int
    line: float


@dataclass(frozen=True)
class MatchSpec:
    bo: int  # 3 or 5
    p_a_map: float  # P(team A wins any given map) from strength differential
    players: tuple[PlayerSpec, ...]
    # Zero-sum anchor: None (default) pins each map's total kills to the
    # SUM of the players' own projected rates, which preserves every
    # calibrated marginal while still forcing kills to compete. A fixed
    # league constant here would silently re-level star-heavy rosters and
    # manufacture phantom under/over edges (found the hard way, 2026-07-24).
    total_kpr: float | None = None
    rounds_pool: tuple[int, ...] = ()  # empirical rounds; falls back to normal


@dataclass
class SimResult:
    """Joint simulation output.

    hits[i, j]   — did prop j's OVER hit on iteration i
    short[i, j]  — did prop j's map range extend past the maps actually
                   played (books void/refund these; optimizer must not count
                   them as wins or losses)
    push[i, j]   — did the total land EXACTLY on the line. 23.5% of
                   PrizePicks lines are whole numbers (Underdog posts none),
                   and those push 6.1% of the time in our archive -- 7.6% on
                   headshots. A push VOIDS the leg: it is not a win. Before
                   this was tracked, an UNDER was scored as ``~hits``, which
                   silently folded every push into the win column and
                   overstated whole-number under legs by 6.1 points at full
                   4-leg payout. That is exactly the bet this app kept
                   recommending.
    n_maps[i]    — series length per iteration
    """

    hits: np.ndarray
    short: np.ndarray
    n_maps: np.ndarray
    p_over: list[float] = field(default_factory=list)
    push: np.ndarray | None = None

    def pushes(self, j: int) -> np.ndarray:
        """Push mask for prop j (all-False when the line cannot push)."""
        if self.push is None:
            return np.zeros(self.hits.shape[0], dtype=bool)
        return self.push[:, j]

    def p_all(self, prop_idx: list[int]) -> float:
        """P(all selected OVERs hit), refund-aware: void legs drop out."""
        ok = np.ones(self.hits.shape[0], dtype=bool)
        for j in prop_idx:
            ok &= self.hits[:, j] | self.short[:, j] | self.pushes(j)
        return float(ok.mean())

    def p_all_independent(self, prop_idx: list[int]) -> float:
        return float(np.prod([self.p_over[j] for j in prop_idx]))


def simulate_match(
    match: MatchSpec,
    props: list[PropSpec],
    n_iters: int = 50_000,
    seed: int | None = None,
) -> SimResult:
    rng = np.random.default_rng(seed)
    n_players = len(match.players)
    max_maps = match.bo
    need = match.bo // 2 + 1

    # ---- 1. form shocks (drawn first: they now tilt map wins) --------------
    team_shock = {
        "A": rng.gamma(1.0 / PHI_TEAM, PHI_TEAM, size=n_iters),
        "B": rng.gamma(1.0 / PHI_TEAM, PHI_TEAM, size=n_iters),
    }
    player_series_shock = rng.gamma(
        1.0 / PHI_SERIES, PHI_SERIES, size=(n_players, n_iters)
    )
    # a team having a good night wins maps AND wins them by more
    form_tilt = TEAM_FORM_TO_WIN * (
        np.log(team_shock["A"]) - np.log(team_shock["B"])
    )
    mu = _solve_mu(form_tilt, match.p_a_map)

    is_a = np.array([p.side == "A" for p in match.players])
    anchor = (
        match.total_kpr
        if match.total_kpr is not None
        else sum(p.kpr for p in match.players)
    )
    kpr_vec = np.array([p.kpr for p in match.players])
    natural_share_a = float(kpr_vec[is_a].sum() / max(anchor, 1e-9))

    rounds_sorted = (
        np.sort(np.asarray(match.rounds_pool, dtype=float))
        if match.rounds_pool
        else None
    )

    # ---- 2. maps: winner, length, kill split, kills ------------------------
    a_wins = np.zeros(n_iters, dtype=np.int64)
    b_wins = np.zeros(n_iters, dtype=np.int64)
    map_played = np.zeros((max_maps, n_iters), dtype=bool)
    kills = np.zeros((max_maps, n_players, n_iters), dtype=np.int64)

    for m in range(max_maps):
        active = (a_wins < need) & (b_wins < need)
        map_played[m] = active

        # latent dominance: sign picks the winner, magnitude sets the margin.
        # ONE uniform per draw — two would give a Laplace, whose spread breaks
        # the quantile transform below and inflates round counts (~+6%).
        u = rng.uniform(1e-9, 1 - 1e-9, size=n_iters)
        e = np.log(u / (1.0 - u))  # standard logistic
        z = mu + form_tilt + e
        a_win = z > 0
        a_wins += (a_win & active).astype(np.int64)
        b_wins += (~a_win & active).astype(np.int64)

        # margin -> length: big |z| (stomp) maps to the SHORT tail of the
        # empirical rounds distribution. z is rescaled to standard-logistic
        # spread first so q stays UNIFORM for an even matchup — that is what
        # preserves the empirical round distribution exactly. A lopsided
        # matchup keeps its shift, so mismatches still produce short maps.
        w = z * float(np.sqrt(_LOGISTIC_VAR / max(z.var(), 1e-9)))
        q = np.clip(2.0 * _sigmoid(-np.abs(w)), 1e-4, 1 - 1e-4)
        if rounds_sorted is not None:
            idx = (q * (len(rounds_sorted) - 1)).astype(np.int64)
            rounds = rounds_sorted[idx]
        else:
            rounds = np.clip(
                ROUNDS_MEAN + ROUNDS_SD * (np.log(q / (1 - q)) / 1.81), 16, 46
            ).round()

        # winner's round count (MR12), then each side's round share
        wr = np.where(
            rounds <= REG_MAX_ROUNDS, float(WIN_ROUNDS_REG),
            np.floor(rounds / 2.0) + 2.0,
        )
        wr = np.minimum(np.maximum(wr, rounds / 2.0 + 0.5), rounds - 1.0)
        share_rounds_a = np.where(a_win, wr / rounds, 1.0 - wr / rounds)

        # kill share follows round share, centred so the MARGINAL split stays
        # at the rosters' natural rate — this adds correlation, not bias.
        ref = share_rounds_a[active] if active.any() else share_rounds_a
        share_a = np.clip(
            natural_share_a
            + KILL_SHARE_BETA * (share_rounds_a - float(ref.mean())),
            *SHARE_CLIP,
        )
        # NOTE: kills ~ rounds x share and those are negatively correlated (a
        # stomp means few rounds but a big share), so E[kills] carries a small
        # covariance term. Renormalising each side independently to remove it
        # was tried and REJECTED: it breaks share_a + share_b == 1 and shifted
        # the mean 1.4% off the projector. The residual is left in place.
        share_b = 1.0 - share_a
        total_target = rounds * rng.normal(anchor, TOTAL_KPR_SD, size=n_iters)
        # team-level kill-night noise: survives the per-team rescale, so it
        # is the lever that lifts teammate coupling without killing the
        # match-wide (cross-team) coupling that rounds provide.
        # symmetric (not lognormal) so it adds spread without adding skew
        night_a = np.clip(
            1.0 + rng.normal(0.0, TEAM_KILL_SD, size=n_iters), 0.4, 1.6
        )
        night_b = np.clip(
            1.0 + rng.normal(0.0, TEAM_KILL_SD, size=n_iters), 0.4, 1.6
        )

        raw = np.empty((n_players, n_iters))
        for i, p in enumerate(match.players):
            g_map = rng.gamma(1.0 / PHI_MAP, PHI_MAP, size=n_iters)
            raw[i] = rng.poisson(
                p.kpr
                * rounds
                * team_shock[p.side]
                * player_series_shock[i]
                * g_map
            )
        # rescale per TEAM: kills compete inside a team (kill-share) and the
        # two teams split the map's kills by game script.
        raw_a = np.maximum(raw[is_a].sum(axis=0), 1.0)
        raw_b = np.maximum(raw[~is_a].sum(axis=0), 1.0)
        scale_a = np.clip(
            total_target * share_a * night_a / raw_a, *SCALE_CLIP
        ) ** RESCALE_STRENGTH
        scale_b = np.clip(
            total_target * share_b * night_b / raw_b, *SCALE_CLIP
        ) ** RESCALE_STRENGTH
        scale = np.where(is_a[:, None], scale_a[None, :], scale_b[None, :])
        scaled = np.floor(raw * scale + rng.uniform(size=raw.shape)).astype(
            np.int64
        )
        kills[m] = scaled * map_played[m]

    n_maps = map_played.sum(axis=0)

    # ---- 4. evaluate every prop -------------------------------------------
    idx_of = {p.player_id: i for i, p in enumerate(match.players)}
    hs_rate = {p.player_id: p.hs_rate for p in match.players}
    hits = np.zeros((n_iters, len(props)), dtype=bool)
    short = np.zeros((n_iters, len(props)), dtype=bool)
    push = np.zeros((n_iters, len(props)), dtype=bool)
    p_over: list[float] = []
    for j, prop in enumerate(props):
        i = idx_of[prop.player_id]
        lo, hi = prop.map_lo - 1, min(prop.map_hi, max_maps) - 1
        totals = kills[lo : hi + 1, i, :].sum(axis=0)
        if prop.stat == "headshots":
            # beta-binomial: per-series headshot rate wobbles around the
            # player's mean, widening the distribution to match reality
            rate = hs_rate[prop.player_id]
            a = max(rate * HS_DISPERSION_K, 1e-3)
            b = max((1.0 - rate) * HS_DISPERSION_K, 1e-3)
            totals = rng.binomial(totals, rng.beta(a, b, size=n_iters))
        hits[:, j] = totals > prop.line
        short[:, j] = n_maps < prop.map_hi
        # A whole-number line can land exactly on the total. The book voids
        # that leg; counting it as an under win is the 6.1-point error.
        if float(prop.line).is_integer():
            push[:, j] = totals == prop.line
        # marginal excludes refunded AND pushed iterations
        live = ~short[:, j] & ~push[:, j]
        p_over.append(
            float(hits[live, j].mean()) if live.any() else 0.0
        )
    return SimResult(hits=hits, short=short, n_maps=n_maps, p_over=p_over,
                     push=push)


def p_three_maps(p_map: float) -> float:
    """Bo3 helper: P(series goes 3) = 1 - p^2 - (1-p)^2, maximized at p=.5."""
    return 1.0 - p_map**2 - (1.0 - p_map) ** 2
