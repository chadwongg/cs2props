"""Cross-book disagreement: the three outcomes must never be conflated.

An earlier ad-hoc probe reported "55.4%" that silently mixed directional
bets with middles — a number that looked like edge and described nothing you
could place. The whole value of this module is keeping those apart, and
keeping the AGREE control alongside the disagreement rate so a high under-
rate cannot be mistaken for a finding.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from cs2props import db
from cs2props.crossbook import Graded, pair_books, run

START = datetime(2026, 7, 25, 12, tzinfo=timezone.utc)


def _g(pp: float, ud: float, obs: float) -> Graded:
    return Graded(player="p", stat="kills", map_lo=1, map_hi=2,
                  pp_line=pp, ud_line=ud, start_ts=0.0, observed=obs)


def test_below_both_lines_is_an_under_win_only() -> None:
    g = _g(pp=27.5, ud=25.5, obs=20)
    assert g.under_high_won and not g.over_low_won and not g.middled


def test_above_both_lines_is_an_over_win_only() -> None:
    g = _g(pp=27.5, ud=25.5, obs=30)
    assert g.over_low_won and not g.under_high_won and not g.middled


def test_between_the_lines_is_a_middle() -> None:
    """Both directional bets win — but only across two tickets on two books.
    Counting it toward a single-slip hit rate invents an edge."""
    g = _g(pp=27.5, ud=25.5, obs=26)
    assert g.over_low_won and g.under_high_won and g.middled


def test_low_book_is_identified_regardless_of_order() -> None:
    assert _g(27.5, 25.5, 0).low_book == "underdog"
    assert _g(25.5, 27.5, 0).low_book == "prizepicks"
    assert _g(25.5, 27.5, 0).low_line == 25.5
    assert _g(25.5, 27.5, 0).high_line == 27.5


def test_gap_sign_says_which_book_posts_higher() -> None:
    assert _g(27.5, 25.5, 0).gap > 0  # PrizePicks higher
    assert _g(25.5, 27.5, 0).gap < 0


def _add_prop(
    conn: "db.sqlite3.Connection", name: str, line: float, book: str,
    start: datetime, stat: str = "kills",
) -> None:
    conn.execute(
        "INSERT INTO props (scanned_at, projection_id, player_id, "
        "player_name, team, opponent, stat_type, stat_kind, map_lo, map_hi, "
        "line_score, board, start_time, league_id) VALUES "
        "(1,?,?,?,'A','B','MAPS 1-2 Kills',?,1,2,?,'standard',?,?)",
        (f"{name}{book}{line}{stat}", name, name, stat, line,
         start.isoformat(), "265" if book == "pp" else "CS"),
    )
    conn.commit()


def _add_result(
    conn: "db.sqlite3.Connection", name: str, kills: int, played: datetime,
) -> None:
    for m in (1, 2):
        conn.execute(
            "INSERT INTO player_maps (player_id, player_name, team, opponent,"
            " event_tier, map_name, played_at, kills, deaths, adr, rating,"
            " rounds, headshots, won, match_id, map_number) VALUES "
            "(?,?,'A','B','b','de_dust2',?,?,15,80,1.1,24,10,1,?,?)",
            (name, name, played.isoformat(), kills, f"{name}-match", m),
        )
    conn.commit()


def test_books_are_paired_across_leetspeak_spellings(tmp_path: Path) -> None:
    """PrizePicks writes sh1ro where Underdog writes sh1r0. An exact join
    discards most of the overlap — which is the entire sample."""
    conn = db.connect(tmp_path / "a.db")
    _add_prop(conn, "sh1ro", 27.5, "pp", START)
    _add_prop(conn, "sh1r0", 25.5, "ud", START)
    pairs = pair_books(conn)
    assert len(pairs) == 1
    assert pairs[0].pp_line == 27.5 and pairs[0].ud_line == 25.5
    conn.close()


def test_different_matches_are_not_paired(tmp_path: Path) -> None:
    """A player's afternoon and evening games are different props."""
    conn = db.connect(tmp_path / "a.db")
    _add_prop(conn, "alpha", 27.5, "pp", START)
    _add_prop(conn, "alpha", 25.5, "ud", START + timedelta(hours=9))
    assert pair_books(conn) == []
    conn.close()


def test_agreeing_props_land_in_the_control_not_the_finding(
    tmp_path: Path,
) -> None:
    """Equal lines are the CONTROL group. Letting them into `graded` would
    dilute the very comparison the module exists to make."""
    conn = db.connect(tmp_path / "a.db")
    _add_prop(conn, "alpha", 25.5, "pp", START)
    _add_prop(conn, "alpha", 25.5, "ud", START)
    _add_result(conn, "alpha", kills=10, played=START + timedelta(hours=1))
    res = run(conn)
    assert len(res.graded) == 0
    assert len(res.agreed) == 1
    conn.close()


def test_a_settled_disagreement_is_graded(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    _add_prop(conn, "bravo", 27.5, "pp", START)
    _add_prop(conn, "bravo", 25.5, "ud", START)
    _add_result(conn, "bravo", kills=10, played=START + timedelta(hours=1))
    res = run(conn)
    assert len(res.graded) == 1
    assert res.graded[0].observed == 20.0  # 10 + 10, below both lines
    n, won, rate, _ = res.under_at_higher()
    assert (n, won, rate) == (1, 1, 1.0)
    conn.close()


def test_unplayed_disagreements_are_live_not_graded(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    _add_prop(conn, "charlie", 27.5, "pp", START)
    _add_prop(conn, "charlie", 25.5, "ud", START)
    res = run(conn)
    assert len(res.graded) == 0 and len(res.live) == 1
    assert "charlie" in res.live[0].describe()
    conn.close()


def test_lift_is_nan_without_a_control(tmp_path: Path) -> None:
    """No control means no claim. It must not silently report the raw rate
    as if the comparison had been made."""
    conn = db.connect(tmp_path / "a.db")
    _add_prop(conn, "delta", 27.5, "pp", START)
    _add_prop(conn, "delta", 25.5, "ud", START)
    _add_result(conn, "delta", kills=10, played=START + timedelta(hours=1))
    gap, z = run(conn).lift()
    assert gap != gap and z != z  # NaN
    conn.close()
