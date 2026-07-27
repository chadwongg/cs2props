"""Walk-forward projection state and distribution sampling.

Design notes (every constant is a named module attribute so backtest tuning
is traceable):

- Player skill is an exponentially weighted mean of kills-per-round (KPR),
  half-life ``HALF_LIFE_MAPS`` maps, with bias-corrected warmup.
- Opponent strength enters as the opponent team's *allowed* KPR (EW mean of
  what enemy players score against them) relative to league mean, tempered by
  ``OPP_EXPONENT``. Event tier is deliberately not a separate factor in v1:
  opponent-allowed-KPR already encodes opposition quality at team resolution.
  The backtest reports per-tier residuals; if they show bias, add the factor.
- Map-specific form is a per-(player, map) EW ratio to the player's overall
  KPR, shrunk toward 1 by n/(n+MAP_SHRINK_N).
- Kills on a map are Poisson conditional on a Gamma "form" multiplier
  (i.e. negative-binomial-like overdispersion). Part of the form shock is
  shared across maps of the same series — that is what makes a player's
  map-1 and map-2 kills co-move, and later drives same-player correlation
  in the slip engine.
- Rounds per map are sampled from a rolling empirical distribution.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

HALF_LIFE_MAPS = 20.0
TEAM_HALF_LIFE_MAPS = 30.0
LEAGUE_HALF_LIFE_MAPS = 500.0
OPP_EXPONENT = 0.7
MAP_SHRINK_N = 8.0
# Gamma form-shock variances (validated via PIT in the backtest; tightened
# 2026-07-24 after full-depth PIT showed thin tails / top-bin underconfidence):
PHI_SERIES = 0.008  # shared across maps of one series
PHI_MAP = 0.005  # idiosyncratic per map
ROUNDS_WINDOW = 2000
DEFAULT_ROUNDS = 21.5
DEFAULT_KPR = 0.62  # league-ish prior for a debuting player


def _decay(half_life: float) -> float:
    return float(0.5 ** (1.0 / half_life))


@dataclass
class EwMean:
    """Bias-corrected exponentially weighted mean."""

    half_life: float
    _s: float = 0.0
    _w: float = 0.0

    def update(self, x: float) -> None:
        d = _decay(self.half_life)
        self._s = d * self._s + (1 - d) * x
        self._w = d * self._w + (1 - d)

    @property
    def value(self) -> float | None:
        return self._s / self._w if self._w > 0 else None

    @property
    def n_effective(self) -> float:
        return self._w / (1 - _decay(self.half_life)) if self._w > 0 else 0.0


@dataclass
class PlayerState:
    kpr: EwMean = field(default_factory=lambda: EwMean(HALF_LIFE_MAPS))
    hs_rate: EwMean = field(default_factory=lambda: EwMean(HALF_LIFE_MAPS))
    map_ratio: dict[str, EwMean] = field(default_factory=dict)
    map_counts: dict[str, int] = field(default_factory=dict)
    n_maps: int = 0

    def update(
        self,
        kills: int,
        rounds: float,
        map_name: str | None,
        headshots: int | None = None,
    ) -> None:
        kpr_obs = kills / rounds
        base = self.kpr.value
        self.kpr.update(kpr_obs)
        self.n_maps += 1
        if headshots is not None and kills > 0:
            self.hs_rate.update(min(headshots / kills, 1.0))
        if map_name and base and base > 0:
            r = self.map_ratio.setdefault(map_name, EwMean(HALF_LIFE_MAPS))
            r.update(kpr_obs / base)
            self.map_counts[map_name] = self.map_counts.get(map_name, 0) + 1

    def map_factor(self, map_name: str | None) -> float:
        if not map_name or map_name not in self.map_ratio:
            return 1.0
        ratio = self.map_ratio[map_name].value or 1.0
        n = self.map_counts.get(map_name, 0)
        shrink = n / (n + MAP_SHRINK_N)
        return 1.0 + (ratio - 1.0) * shrink


@dataclass
class TeamState:
    """Opposition quality + strength.

    allowed_kpr: EW mean of KPR that enemies post against this team.
    map_win: EW map-win rate — feeds the series-length model via
    :func:`p_map_win`.
    """

    allowed_kpr: EwMean = field(default_factory=lambda: EwMean(TEAM_HALF_LIFE_MAPS))
    map_win: EwMean = field(default_factory=lambda: EwMean(TEAM_HALF_LIFE_MAPS))

    def update_allowed(self, enemy_kpr: float) -> None:
        self.allowed_kpr.update(enemy_kpr)

    def update_result(self, won: bool) -> None:
        self.map_win.update(1.0 if won else 0.0)


# Headshot-rate role clusters, fitted by 2-means over 976 players with 50+
# maps in our own history (2026-07-24):
#   AWPer-ish  mean 0.364  (n=214)     rifler  mean 0.556  (n=762)
# Confirmed against known players: m0NESY .387, ZywOo .414, broky .336,
# torzsi .325 vs donk .611, Twistzz .611, jcobbb .653, malbsMd .667.
# bo3.gg exposes no usable role field, so role is inferred from the rate
# itself and used only as a SHRINKAGE TARGET — a thin-sample AWPer is pulled
# toward 0.364 instead of the league's 0.514, rather than toward a rifler's
# profile. 33% of players have <10 maps and 31 of those carried impossible
# rates (<25% or >75%) before this was applied.
HS_AWP_CLUSTER = 0.364
HS_RIFLE_CLUSTER = 0.556
HS_CLUSTER_SD = 0.09  # softness of the role assignment
HS_PRIOR_MAPS = 15.0  # pseudo-maps of prior mixed into a player's own rate
HS_LEAGUE_MEAN = 0.514

MAP_WIN_PRIOR_MAPS = 25.0  # pseudo-maps of 50/50 prior mixed into map-win


def shrunk_hs_rate(player: PlayerState) -> float:
    """Headshot rate shrunk toward a ROLE-AWARE prior by sample size.

    The player's own rate softly assigns them between the two measured
    clusters (AWPer 0.364 / rifler 0.556); the shrinkage target is that
    blend, not the league mean. This matters for the middle of the sample
    range: a player with 10 maps at 0.35 is very likely a genuine AWPer, and
    shrinking him toward the league's 0.514 would systematically overstate
    his headshot props. Ambiguous rates fall back toward the league mean.
    """
    raw = player.hs_rate.value
    if raw is None:
        return HS_LEAGUE_MEAN
    n = player.hs_rate.n_effective
    # soft cluster weights — closer cluster dominates, ties blend
    d_awp = (raw - HS_AWP_CLUSTER) / HS_CLUSTER_SD
    d_rif = (raw - HS_RIFLE_CLUSTER) / HS_CLUSTER_SD
    w_awp = math.exp(-0.5 * d_awp * d_awp)
    w_rif = math.exp(-0.5 * d_rif * d_rif)
    if w_awp + w_rif < 1e-9:
        prior = HS_LEAGUE_MEAN
    else:
        prior = (w_awp * HS_AWP_CLUSTER + w_rif * HS_RIFLE_CLUSTER) / (
            w_awp + w_rif
        )
    return float((raw * n + prior * HS_PRIOR_MAPS) / (n + HS_PRIOR_MAPS))


def is_awper(player: PlayerState) -> bool:
    """Role read for display/diagnostics — the boundary sits between the two
    fitted clusters."""
    rate = shrunk_hs_rate(player)
    return rate < (HS_AWP_CLUSTER + HS_RIFLE_CLUSTER) / 2


def shrunk_map_win(team: TeamState | None) -> float:
    """Map-win rate shrunk toward 0.5 by sample size.

    Without this a team with 2 maps and 2 wins reads as a 100% map-winner —
    measured on our own history, 48% of team aliases have <10 maps and 24%
    sit at exactly 0% or 100%. Those bogus strengths propagate into series
    length and kill share, and manufactured whole slips (found 2026-07-24
    when DENDELE, 2 maps played, was priced as a 66% favourite over FaZe).
    """
    if team is None or team.map_win.value is None:
        return 0.5
    n = team.map_win.n_effective
    return float(
        (team.map_win.value * n + 0.5 * MAP_WIN_PRIOR_MAPS)
        / (n + MAP_WIN_PRIOR_MAPS)
    )


def p_map_win(team_a: TeamState | None, team_b: TeamState | None) -> float:
    """P(A beats B on a map) from shrunk EW win rates, clipped away from the
    extremes — Bradley-Terry-style ratio of strengths."""
    wa = max(shrunk_map_win(team_a), 0.05)
    wb = max(shrunk_map_win(team_b), 0.05)
    return float(min(max(wa / (wa + wb), 0.15), 0.85))


@dataclass
class League:
    mean_kpr: EwMean = field(default_factory=lambda: EwMean(LEAGUE_HALF_LIFE_MAPS))
    rounds_hist: deque[int] = field(default_factory=lambda: deque(maxlen=ROUNDS_WINDOW))

    def update(self, kpr: float, rounds: int) -> None:
        self.mean_kpr.update(kpr)
        self.rounds_hist.append(rounds)

    def sample_rounds(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if len(self.rounds_hist) < 50:
            return np.full(size, DEFAULT_ROUNDS)
        arr = np.asarray(self.rounds_hist, dtype=float)
        return rng.choice(arr, size=size)


def expected_kpr(
    player: PlayerState,
    opponent: TeamState | None,
    league: League,
    map_name: str | None = None,
) -> float:
    """Point forecast of the player's KPR for one upcoming map."""
    base = player.kpr.value if player.kpr.value is not None else DEFAULT_KPR
    league_mean = league.mean_kpr.value or DEFAULT_KPR
    opp_factor = 1.0
    if opponent is not None and opponent.allowed_kpr.value is not None:
        opp_factor = (opponent.allowed_kpr.value / league_mean) ** OPP_EXPONENT
    return base * opp_factor * player.map_factor(map_name)


def sample_series_kills(
    kpr_per_map: list[float],
    league: League,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Sample total kills across the given maps of one series.

    Returns array of shape (n_samples,). One Gamma form shock is shared across
    all maps (PHI_SERIES); each map adds an idiosyncratic shock (PHI_MAP);
    kills are then Poisson conditional on the composed rate.
    """
    g_series = rng.gamma(1.0 / PHI_SERIES, PHI_SERIES, size=n_samples)
    total = np.zeros(n_samples)
    for kpr in kpr_per_map:
        rounds = league.sample_rounds(rng, n_samples)
        g_map = rng.gamma(1.0 / PHI_MAP, PHI_MAP, size=n_samples)
        lam = kpr * rounds * g_series * g_map
        total += rng.poisson(lam)
    return total


def p_over(samples: np.ndarray, line: float) -> float:
    return float(np.mean(samples > line))


def pit_value(samples: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    """Randomized PIT: uniform on [0,1] iff the predictive distribution is
    calibrated. Randomization breaks ties on the discrete kill counts."""
    below = float(np.mean(samples < observed))
    at = float(np.mean(samples == observed))
    return below + float(rng.uniform()) * at
