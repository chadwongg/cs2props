"""Model math tests: EWMA, factors, sampler moments, PIT self-consistency."""

from __future__ import annotations

import numpy as np

from cs2props.model import projector as pj
from cs2props.model.backtest import Calibration


# ---------------------------------------------------------------- EwMean


def test_ewmean_constant_series() -> None:
    m = pj.EwMean(half_life=20)
    for _ in range(50):
        m.update(0.7)
    assert m.value is not None
    assert abs(m.value - 0.7) < 1e-12


def test_ewmean_half_life_weighting() -> None:
    """After H updates of b following many of a, weight on b is ~50%."""
    m = pj.EwMean(half_life=10)
    for _ in range(500):
        m.update(0.0)
    for _ in range(10):
        m.update(1.0)
    assert m.value is not None
    assert 0.45 < m.value < 0.55


def test_ewmean_unbiased_warmup() -> None:
    """Bias correction: a single observation gives exactly that value."""
    m = pj.EwMean(half_life=20)
    m.update(0.83)
    assert m.value is not None
    assert abs(m.value - 0.83) < 1e-12


# ---------------------------------------------------------------- factors


def test_map_factor_shrinks_toward_one() -> None:
    p = pj.PlayerState()
    # establish a baseline then one hot map on de_dust2
    for _ in range(30):
        p.update(kills=14, rounds=20, map_name="de_mirage")
    p.update(kills=28, rounds=20, map_name="de_dust2")  # ratio ~2 but n=1
    f = p.map_factor("de_dust2")
    assert 1.0 < f < 1.2  # heavily shrunk, not the raw ~2x
    assert p.map_factor("de_unknown") == 1.0


def _player_with_hs(rate: float, n_maps: int) -> pj.PlayerState:
    p = pj.PlayerState()
    for _ in range(n_maps):
        p.update(kills=20, rounds=22, map_name=None, headshots=int(20 * rate))
    return p


def test_awper_shrinks_toward_awp_cluster_not_league_mean() -> None:
    """A thin-sample AWPer must not be dragged to the league's 0.514 — that
    would systematically overstate his headshot props. Role is inferred from
    the rate itself because bo3.gg exposes no usable role field."""
    awper = _player_with_hs(0.35, 10)
    got = pj.shrunk_hs_rate(awper)
    assert got < 0.45, got  # stayed in AWPer territory
    assert got > 0.35  # but still shrunk toward the prior
    assert abs(got - pj.HS_LEAGUE_MEAN) > 0.05  # NOT pulled to league mean
    assert pj.is_awper(awper)


def test_rifler_shrinks_toward_rifle_cluster() -> None:
    rifler = _player_with_hs(0.62, 10)
    got = pj.shrunk_hs_rate(rifler)
    assert 0.56 < got < 0.62, got
    assert not pj.is_awper(rifler)


def test_impossible_thin_sample_rate_is_pulled_back_hard() -> None:
    """31 players in our history had rates <25% or >75% off <10 maps. A rate
    of 0.95 from 2 maps must not survive into a headshot projection."""
    freak = _player_with_hs(0.95, 2)
    got = pj.shrunk_hs_rate(freak)
    assert got < 0.70, got


def test_well_sampled_player_keeps_his_own_rate() -> None:
    """Shrinkage must fade as evidence accumulates."""
    veteran = _player_with_hs(0.36, 200)
    assert abs(pj.shrunk_hs_rate(veteran) - 0.36) < 0.02


def test_unknown_player_gets_league_mean() -> None:
    assert pj.shrunk_hs_rate(pj.PlayerState()) == pj.HS_LEAGUE_MEAN


def test_map_win_shrinks_on_tiny_samples() -> None:
    """A team with 2 maps and 2 wins must NOT read as unbeatable. Found in
    production 2026-07-24: DENDELE (2 maps, both won) was priced as a 66%
    favourite over FaZe, which shortened the simulated series and
    manufactured UNDER edges on every FaZe player."""
    tiny = pj.TeamState()
    for _ in range(2):
        tiny.update_result(True)
    solid = pj.TeamState()
    for i in range(120):
        solid.update_result(i % 2 == 0)  # a true 50% team

    assert tiny.map_win.value == 1.0  # raw rate is still 100%
    assert pj.shrunk_map_win(tiny) < 0.60  # but shrunk hard toward 0.5
    assert abs(pj.shrunk_map_win(solid) - 0.5) < 0.05
    # and a 2-map wonder must not dominate a well-sampled opponent
    assert pj.p_map_win(tiny, solid) < 0.56


def test_map_win_still_respects_real_strength() -> None:
    """Shrinkage must not flatten genuinely strong teams."""
    strong, weak = pj.TeamState(), pj.TeamState()
    for i in range(150):
        strong.update_result(i % 10 != 0)  # ~90% winner
        weak.update_result(i % 10 == 0)  # ~10% winner
    assert pj.p_map_win(strong, weak) > 0.70


def test_opponent_factor_direction() -> None:
    league = pj.League()
    for _ in range(200):
        league.update(kpr=0.65, rounds=21)
    soft = pj.TeamState()
    for _ in range(50):
        soft.update_allowed(0.80)  # leaky team
    player = pj.PlayerState()
    for _ in range(30):
        player.update(kills=13, rounds=20, map_name=None)  # 0.65 kpr
    vs_soft = pj.expected_kpr(player, soft, league)
    vs_avg = pj.expected_kpr(player, None, league)
    assert vs_soft > vs_avg  # leaky opponent -> higher projection


# ---------------------------------------------------------------- sampler


def test_sampler_mean_matches_rate() -> None:
    rng = np.random.default_rng(1)
    league = pj.League()
    for _ in range(300):
        league.update(kpr=0.65, rounds=21)
    s = pj.sample_series_kills([0.7, 0.7], league, rng, 40_000)
    # E[total] = kpr * E[rounds] * 2 maps; gamma shocks have mean 1
    expected = 0.7 * 21 * 2
    assert abs(float(s.mean()) - expected) < 0.6
    # overdispersed relative to pure Poisson
    assert float(s.var()) > expected * 1.3


def test_p_over_monotone_in_line() -> None:
    rng = np.random.default_rng(2)
    league = pj.League()
    for _ in range(300):
        league.update(kpr=0.65, rounds=21)
    s = pj.sample_series_kills([0.65], league, rng, 20_000)
    assert pj.p_over(s, 10.5) > pj.p_over(s, 13.5) > pj.p_over(s, 16.5)


def test_pit_self_consistency() -> None:
    """Observations drawn from the model itself must yield ~uniform PIT."""
    rng = np.random.default_rng(3)
    league = pj.League()
    for _ in range(300):
        league.update(kpr=0.65, rounds=int(rng.integers(16, 31)))
    pits = []
    for _ in range(400):
        samples = pj.sample_series_kills([0.68, 0.68], league, rng, 2000)
        observed = float(rng.choice(samples))  # draw truth from the model
        pits.append(pj.pit_value(samples, observed, rng))
    hist, _ = np.histogram(pits, bins=5, range=(0, 1))
    frac = hist / len(pits)
    assert all(0.10 < f < 0.30 for f in frac), frac  # flat-ish across bins


# ---------------------------------------------------------------- metrics


def test_reliability_binning_and_log_loss() -> None:
    cal = Calibration()
    rng = np.random.default_rng(4)
    # perfectly calibrated predictions
    for _ in range(5000):
        p = float(rng.uniform(0.05, 0.95))
        cal.preds.append(p)
        cal.outcomes.append(int(rng.uniform() < p))
    rel = cal.reliability()
    for pred_mean, actual, n in rel:
        if n > 200:
            assert abs(pred_mean - actual) < 0.08
    assert cal.log_loss() < cal.baseline_log_loss()


def test_prediction_precedes_update_no_leakage() -> None:
    """expected_kpr must not see the current match's own maps."""
    league = pj.League()
    for _ in range(100):
        league.update(kpr=0.6, rounds=20)
    p = pj.PlayerState()
    for _ in range(25):
        p.update(kills=12, rounds=20, map_name=None)  # 0.60 kpr baseline
    before = pj.expected_kpr(p, None, league)
    p.update(kills=30, rounds=20, map_name=None)  # monster game arrives after
    after = pj.expected_kpr(p, None, league)
    assert abs(before - 0.60) < 0.01
    assert after > before  # state moved only once updated
