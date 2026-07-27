"""The haircut must LEARN from graded legs — carefully.

A model that lurches on a handful of results is worse than a fixed guess:
it would have swung to a 5-point haircut on 28 legs whose standard error was
9 points. Shrinkage toward the prior is what makes this safe.
"""

from __future__ import annotations

from pathlib import Path

from cs2props import db
from cs2props.adaptive import (
    MAX_HAIRCUT,
    MIN_LEGS_TO_LEARN,
    PRIOR_HAIRCUT,
    estimate_haircut,
)
from cs2props.tracker import parse_leg, track_slip


def _slip_with_results(
    conn: "db.sqlite3.Connection", claimed_p: float, results: list[str]
) -> None:
    """Track a slip and force its legs to the given won/lost statuses."""
    legs = [parse_leg(f"pl{i}_{id(results)} under 20.5 kills 1-2")
            for i in range(len(results))]
    sid = track_slip(conn, "prizepicks", 1.0, legs, claimed_p=claimed_p)
    for i, r in enumerate(results):
        conn.execute(
            "UPDATE slip_legs SET status=? WHERE slip_id=? AND leg_no=?",
            (r, sid, i),
        )
    conn.execute("UPDATE slips SET status='lost' WHERE slip_id=?", (sid,))
    conn.commit()


def test_no_data_returns_the_prior(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    est = estimate_haircut(conn)
    assert est.haircut == PRIOR_HAIRCUT
    assert est.n_legs == 0
    assert "prior only" in est.source
    conn.close()


def test_tiny_sample_does_not_move_the_haircut(tmp_path: Path) -> None:
    """28 legs at 57% vs 62% claimed implied a 5-point bias, but its standard
    error was 9 points. A handful of legs must not move the estimate."""
    conn = db.connect(tmp_path / "a.db")
    for _ in range(3):  # 12 legs, all lost -> a very bad-looking sample
        _slip_with_results(conn, 0.15, ["lost"] * 4)
    est = estimate_haircut(conn)
    assert est.n_legs < MIN_LEGS_TO_LEARN
    assert est.haircut == PRIOR_HAIRCUT  # unmoved
    conn.close()


def test_large_biased_sample_raises_the_haircut(tmp_path: Path) -> None:
    """With real evidence of optimism, the haircut must grow beyond the
    prior — that is the whole point of learning it."""
    conn = db.connect(tmp_path / "a.db")
    for i in range(60):  # 240 legs, ~50% hit against a ~62% claim
        _slip_with_results(conn, 0.15, ["won", "lost", "won", "lost"])
    est = estimate_haircut(conn)
    assert est.n_legs >= 200
    assert est.haircut > PRIOR_HAIRCUT
    assert est.haircut <= MAX_HAIRCUT
    conn.close()


def test_haircut_never_goes_negative(tmp_path: Path) -> None:
    """If legs beat their claim, the haircut floors at zero. Betting MORE
    because a sample looked good is how bankrolls die."""
    conn = db.connect(tmp_path / "a.db")
    for i in range(60):
        _slip_with_results(conn, 0.15, ["won"] * 4)  # everything hits
    est = estimate_haircut(conn)
    assert est.haircut >= 0.0
    conn.close()


def test_haircut_is_capped(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    for i in range(80):
        _slip_with_results(conn, 0.40, ["lost"] * 4)  # catastrophic
    assert estimate_haircut(conn).haircut <= MAX_HAIRCUT
    conn.close()


def test_source_string_reports_the_evidence(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    for i in range(60):
        _slip_with_results(conn, 0.15, ["won", "lost", "won", "lost"])
    src = estimate_haircut(conn).source
    assert "legs" in src and "observed" in src and "weight on live data" in src
    conn.close()
