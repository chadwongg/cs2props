"""Calibration map: fitting, monotonicity, one-sidedness, persistence."""

from __future__ import annotations

from cs2props.calmap import CalibrationMap


def _map() -> CalibrationMap:
    return CalibrationMap(
        knots=((0.57, 0.53), (0.62, 0.545), (0.70, 0.55)),
        n_lines=5000, fitted=0.0, when="2026-08-27",
    )


def test_apply_interpolates_and_never_exceeds_raw() -> None:
    m = _map()
    assert m.apply(0.57) == 0.53
    assert 0.53 < m.apply(0.60) < 0.545  # between knots
    assert m.apply(0.80) == 0.55  # clamped to last knot
    for p in (0.55, 0.60, 0.65, 0.72, 0.90):
        assert m.apply(p) <= p  # one-sided: may only discount


def test_delta_is_the_per_leg_discount() -> None:
    m = _map()
    assert abs(m.delta(0.70) - 0.15) < 1e-9


def test_below_half_passes_through() -> None:
    # the map is fitted on pick probabilities (>0.5); anything else is not
    # a pick and must not be touched
    assert _map().apply(0.40) == 0.40


def test_slip_adjusted_ev_uses_map_over_flat_haircut() -> None:
    from cs2props.optimizer.search import Slip

    legs = []  # empty legs -> adjusted_ev falls back to ev; use effective_p
    s = Slip(legs=legs, p_all=0.2, p_independent=0.2, ev=1.0,
             multiplier=10.0, calmap=_map())
    # raw 70% leg is priced at 55%, not 70%-haircut
    assert abs(s._leg_effective_p(0.70) - 0.55) < 1e-9
    flat = Slip(legs=legs, p_all=0.2, p_independent=0.2, ev=1.0,
                multiplier=10.0, haircut=0.05)
    assert abs(flat._leg_effective_p(0.70) - 0.65) < 1e-9


def test_save_load_roundtrip(tmp_path) -> None:
    from pathlib import Path

    from cs2props.calmap import load, save

    m = _map()
    path = Path(tmp_path) / "cm.json"
    save(m, path)
    back = load(path)
    assert back is not None
    assert back.knots == m.knots
    assert back.n_lines == 5000
    assert load(Path(tmp_path) / "missing.json") is None
