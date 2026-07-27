"""Real-line backtest: the resolution logic must be beyond suspicion.

This module decides whether the model works. If it matches a prop to the
wrong match, or scores one prop twice, it produces a confident number that
is simply false — and a false PASS here would keep real money on the table.
Two bugs of exactly this shape have already shipped in this project (the
tracker's 6h look-back that fabricated eight losses, and the CLV lookup that
crossed matches), so the joins are tested, not trusted.
"""

from __future__ import annotations

from pathlib import Path

from cs2props import db
from cs2props.model.reallines import (
    _epoch,
    _observed,
    run_real_line_backtest,
)


def _map(match_id: str, n: int, kills: int, hs: int, at: str) -> tuple:
    # mirrors the SELECT column order in run_real_line_backtest
    return (match_id, n, "p1", "Player", "A", "B", "b", "de_dust2", at,
            kills, 24, hs)


def test_epoch_normalises_offsets() -> None:
    """Props carry -04:00 and player_maps carry +00:00. A 20:00 ET match is
    00:00 UTC the NEXT day — any date-string compare gets it wrong."""
    et = _epoch("2026-07-25T20:00:00.000-04:00")
    utc = _epoch("2026-07-26T00:00:00.000+00:00")
    assert et is not None and utc is not None
    assert abs(et - utc) < 1.0


def test_epoch_returns_none_on_junk() -> None:
    assert _epoch("not a timestamp") is None


def test_observed_sums_the_map_range() -> None:
    maps = [_map("m1", 1, 20, 10, "x"), _map("m1", 2, 15, 8, "x")]
    assert _observed(maps, "kills", 1, 2) == 35.0
    assert _observed(maps, "headshots", 1, 2) == 18.0
    assert _observed(maps, "kills", 1, 1) == 20.0
    assert _observed(maps, "kills", 2, 2) == 15.0


def test_incomplete_range_is_void_not_a_loss() -> None:
    """A maps-1-2 prop where only map 1 was played is a VOID at the book.
    Scoring it as a loss manufactures failures the bettor never took — the
    exact bug that invented eight settled losses on 2026-07-24."""
    maps = [_map("m1", 1, 20, 10, "x")]
    assert _observed(maps, "kills", 1, 2) is None
    assert _observed(maps, "kills", 1, 1) == 20.0


def test_empty_db_reports_no_lines_rather_than_a_verdict(
    tmp_path: Path,
) -> None:
    conn = db.connect(tmp_path / "a.db")
    res = run_real_line_backtest(conn)
    assert res.n == 0
    conn.close()


def _seed(conn: "db.sqlite3.Connection", start: str, played: str) -> None:
    """One prop and one settled match for the same player."""
    conn.execute(
        "INSERT INTO props (scanned_at, projection_id, player_id, "
        "player_name, team, opponent, stat_type, stat_kind, map_lo, map_hi, "
        "line_score, board, start_time, league_id) VALUES "
        "(1,'p1','pid','Player','A','B','MAPS 1-2 Kills','kills',1,2,"
        "30.5,'standard',?,'265')",
        (start,),
    )
    for n in (1, 2):
        conn.execute(
            "INSERT INTO player_maps (player_id, player_name, team, opponent,"
            " event_tier, map_name, played_at, kills, deaths, adr, rating,"
            " rounds, headshots, won, match_id, map_number) VALUES "
            "('pid','Player','A','B','b','de_dust2',?,20,15,80,1.1,24,10,1,"
            "'match1',?)",
            (played, n),
        )
    conn.commit()


def test_prop_matches_its_match_across_the_utc_boundary(
    tmp_path: Path,
) -> None:
    """A 20:00 ET prop and its 00:30 UTC maps are the SAME match. A date
    join would drop this pairing entirely and silently shrink the sample."""
    conn = db.connect(tmp_path / "a.db")
    _seed(conn, "2026-07-25T20:00:00.000-04:00",
          "2026-07-26T00:30:00.000+00:00")
    res = run_real_line_backtest(conn, min_history=0)
    assert res.n == 1
    assert res.rows[0].observed == 40.0  # 20 + 20
    assert res.rows[0].won_over is True  # 40 > 30.5
    conn.close()


def test_a_prop_is_never_scored_twice(tmp_path: Path) -> None:
    """Players routinely play two matches a day (dozens did on 2026-07-25).
    One archived line must produce at most one scored row, or the sample
    inflates with observations that never existed."""
    conn = db.connect(tmp_path / "a.db")
    _seed(conn, "2026-07-25T12:00:00.000-04:00",
          "2026-07-25T16:30:00.000+00:00")
    for n in (1, 2):  # a SECOND match the same day
        conn.execute(
            "INSERT INTO player_maps (player_id, player_name, team, opponent,"
            " event_tier, map_name, played_at, kills, deaths, adr, rating,"
            " rounds, headshots, won, match_id, map_number) VALUES "
            "('pid','Player','A','C','b','de_inferno',"
            "'2026-07-25T20:30:00.000+00:00',25,15,80,1.1,24,12,1,"
            "'match2',?)",
            (n,),
        )
    conn.commit()
    res = run_real_line_backtest(conn, min_history=0)
    assert res.n == 1, "one archived line must not score against both matches"
    conn.close()


def test_far_away_match_does_not_claim_the_prop(tmp_path: Path) -> None:
    """A match three days after the prop's start time is a different game."""
    conn = db.connect(tmp_path / "a.db")
    _seed(conn, "2026-07-25T12:00:00.000-04:00",
          "2026-07-28T16:30:00.000+00:00")
    res = run_real_line_backtest(conn, min_history=0)
    assert res.n == 0
    assert res.skipped_unplayed == 1
    conn.close()


def test_baseline_uses_the_realised_over_rate(tmp_path: Path) -> None:
    """The bar is 'always predict the base rate'. Books set lines near a coin
    flip, so beating this is the whole claim — it must not be hand-set."""
    conn = db.connect(tmp_path / "a.db")
    _seed(conn, "2026-07-25T12:00:00.000-04:00",
          "2026-07-25T16:30:00.000+00:00")
    res = run_real_line_backtest(conn, min_history=0)
    assert res.over_rate() == 1.0
    conn.close()


def test_under_rate_does_not_credit_pushes(tmp_path: Path) -> None:
    """`1 - over_rate` counts a push as an under WIN. It is not. This is the
    same error that inflated whole-number under legs by 6.1 points inside the
    engine — repeated in the measurement used to decide whether betting
    unders beats the book at all, where it moved the headline figure from
    52.2% to 52.9% and helped hold up an edge that was not there.
    """
    conn = db.connect(tmp_path / "a.db")
    _seed(conn, "2026-07-25T12:00:00.000-04:00",
          "2026-07-25T16:30:00.000+00:00")
    conn.execute("UPDATE props SET line_score = 40.0")   # whole number
    conn.commit()
    res = run_real_line_backtest(conn, min_history=0)
    assert res.n == 1
    assert res.rows[0].observed == 40.0                  # lands ON the line
    rate, live, pushed = res.under_rate()
    assert pushed == 1
    assert live == 0            # nothing left to score
    assert 1 - res.over_rate() == 1.0   # the WRONG way says "under won"
    conn.close()
