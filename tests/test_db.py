"""Snapshot persistence round-trip."""

from __future__ import annotations

from pathlib import Path

from cs2props import db
from cs2props.ingest.prizepicks import load_fixture

FIXTURE = Path(__file__).parent / "fixtures" / "prizepicks_projections.json"


def test_save_and_read_snapshot(tmp_path: Path) -> None:
    props = load_fixture(FIXTURE)
    conn = db.connect(tmp_path / "test.db")
    ts = db.save_props(conn, props)

    rows = conn.execute(
        "SELECT player_name, stat_kind, map_lo, map_hi, line_score, board "
        "FROM props WHERE scanned_at = ? ORDER BY projection_id",
        (ts,),
    ).fetchall()
    assert len(rows) == len(props)
    donk = rows[0]
    assert donk == ("donk", "kills", 1, 2, 32.5, "standard")
    # full-series prop stores NULL map range
    series = conn.execute(
        "SELECT map_lo, map_hi FROM props WHERE projection_id = '5301007'"
    ).fetchone()
    assert series == (None, None)
    conn.close()
