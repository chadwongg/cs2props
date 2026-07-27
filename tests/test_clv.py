"""CLV tests. The sign convention is the whole feature — inverting it would
report a losing model as a winning one, so it is pinned from both sides."""

from __future__ import annotations

import time
from pathlib import Path

from cs2props import db
from cs2props.clv import format_report, leg_clvs, signed_clv
from cs2props.ingest.prizepicks import Prop
from cs2props.tracker import parse_leg, track_slip


def test_under_beats_close_when_line_drops() -> None:
    """Bet UNDER 14, closes at 13: you hold the EASIER number -> positive."""
    assert signed_clv("under", 14.0, 13.0) == 1.0
    assert signed_clv("under", 14.0, 15.0) == -1.0


def test_over_beats_close_when_line_rises() -> None:
    """Bet OVER 25.5, closes at 27: you hold the easier number -> positive."""
    assert signed_clv("over", 25.5, 27.0) == 1.5
    assert signed_clv("over", 25.5, 24.0) == -1.5


def _prop(name: str, line: float) -> Prop:
    return Prop(
        projection_id=f"{name}{line}", player_id=name, player_name=name,
        team="T", opponent="O", stat_type="MAPS 1-2 Kills", stat_kind="kills",
        map_range=(1, 2), line_score=line, board="standard",
        start_time="2026-07-25T18:00:00Z", league_id="CS",
    )


def test_end_to_end_clv_from_snapshots(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "c.db")
    placed = time.time()
    track_slip(conn, "prizepicks", 1.0,
               [parse_leg("donk under 30.5 kills 1-2")],
               placed_at=placed)
    # a snapshot taken AFTER placement is the close
    db.save_props(conn, [_prop("donk", 29.0)])
    rows = leg_clvs(conn)
    assert len(rows) == 1
    assert rows[0].closing_line == 29.0
    assert rows[0].clv == 1.5  # under 30.5 vs close 29.0 -> beat the close
    assert "beat the close" in format_report(rows)
    conn.close()


def test_snapshot_before_placement_is_not_a_close(tmp_path: Path) -> None:
    """A line seen before the bet cannot be its closing line."""
    conn = db.connect(tmp_path / "c.db")
    db.save_props(conn, [_prop("donk", 29.0)])
    time.sleep(0.01)
    track_slip(conn, "prizepicks", 1.0,
               [parse_leg("donk under 30.5 kills 1-2")],
               placed_at=time.time())
    assert leg_clvs(conn) == []
    conn.close()


def test_report_flags_a_losing_model(tmp_path: Path) -> None:
    """Consistently negative CLV must say so in plain words — that is the
    signal to stop betting, and it arrives long before the P&L does."""
    conn = db.connect(tmp_path / "c.db")
    placed = time.time()
    for i in range(12):
        track_slip(conn, "prizepicks", 1.0,
                   [parse_leg(f"p{i} under 20.5 kills 1-2")],
                   placed_at=placed)
    db.save_props(conn, [_prop(f"p{i}", 23.0) for i in range(12)])
    out = format_report(leg_clvs(conn))
    assert "LOSING to the close" in out
    conn.close()


def _prop_at(name: str, line: float, start: str, board: str = "standard") -> Prop:
    return Prop(
        projection_id=f"{name}{line}{board}", player_id=name, player_name=name,
        team="T", opponent="O", stat_type="MAPS 1-2 Kills", stat_kind="kills",
        map_range=(1, 2), line_score=line, board=board,
        start_time=start, league_id="CS",
    )


def test_alt_ladder_lines_are_not_closing_lines(tmp_path: Path) -> None:
    """Underdog posts alt ladders for the same player and start time
    (25.5/29.5/.../37.5). Treating one as the close produced a -12.0 CLV on a
    line that never moved (2026-07-25)."""
    conn = db.connect(tmp_path / "c.db")
    placed = time.time()
    track_slip(conn, "underdog", 1.0,
               [parse_leg("Techno4k under 25.5 kills 1-2")], placed_at=placed)
    db.save_props(conn, [
        _prop_at("Techno4k", 25.5, "2026-07-26T15:00:00Z"),
        _prop_at("Techno4k", 37.5, "2026-07-26T15:00:00Z", board="alt"),
        _prop_at("Techno4k", 31.5, "2026-07-26T15:00:00Z", board="alt"),
    ])
    rows = leg_clvs(conn)
    assert len(rows) == 1
    assert rows[0].closing_line == 25.5   # the standard line, not an alt
    assert rows[0].clv == 0.0
    conn.close()


def test_closing_line_comes_from_the_match_that_was_bet(tmp_path: Path) -> None:
    """A later match for the same player must not supply the closing line."""
    conn = db.connect(tmp_path / "c.db")
    placed = time.time()
    track_slip(conn, "underdog", 1.0,
               [parse_leg("donk under 30.5 kills 1-2")], placed_at=placed)
    db.save_props(conn, [
        _prop_at("donk", 29.5, "2026-07-26T15:00:00Z"),   # the game bet on
        _prop_at("donk", 40.5, "2026-07-29T15:00:00Z"),   # a later game
    ])
    rows = leg_clvs(conn)
    assert rows[0].closing_line == 29.5
    assert rows[0].clv == 1.0
    conn.close()
