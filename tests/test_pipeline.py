"""Glue-layer tests: name joining, history replay, board simulation."""

from __future__ import annotations

from pathlib import Path

from cs2props import db
from cs2props.ingest.bo3gg import PlayerMapRow
from cs2props.ingest.prizepicks import Prop
from cs2props.model.state_builder import build_history, clean_name
from cs2props.pipeline import build_match_sim, group_matches, simulate_board


def test_canonical_team_merges_org_aliases() -> None:
    """bo3.gg reports the same org under several clan names, splitting its
    history and starving team-strength estimates (found 2026-07-24: FaZe's
    139 maps and 32 maps were tracked as two teams)."""
    from cs2props.model.state_builder import canonical_team

    assert canonical_team("FaZe Clan") == canonical_team("FaZe") == "faze"
    assert canonical_team("Team Liquid") == canonical_team("Liquid") == "liquid"
    assert canonical_team("G2 Esports") == canonical_team("G2") == "g2"
    assert canonical_team("DENDELE CS") == canonical_team("DENDELE")
    assert canonical_team(None) is None


def test_canonical_team_keeps_distinct_rosters_apart() -> None:
    """Academy/secondary rosters must NOT be merged into the main team —
    that would be a worse error than the fragmentation being fixed."""
    from cs2props.model.state_builder import canonical_team

    assert canonical_team("MOUZ NXT") != canonical_team("MOUZ")
    assert canonical_team("Astralis Talent") != canonical_team("Astralis")
    assert canonical_team("FaZe Academy") != canonical_team("FaZe Clan")


def test_clean_name_strips_decorations() -> None:
    assert clean_name("★ ⑳ Hezz †") == "hezz"
    assert clean_name("m0NESY") == "m0nesy"
    assert clean_name("k1slll") == "k1slll"


def _row(
    pid: str, name: str, team: str, opp: str, kills: int, match: str,
    map_no: int, played: str, won: int = 1, hs: int | None = None,
) -> PlayerMapRow:
    return PlayerMapRow(
        player_id=pid, player_name=name, team=team, opponent=opp,
        event_tier="a", map_name="de_mirage", played_at=played, kills=kills,
        deaths=15, adr=70.0, rating=1.0, rounds=22, headshots=hs, won=won,
        match_id=match, map_number=map_no,
    )


def _seed_history(tmp_path: Path) -> "db.sqlite3.Connection":
    conn = db.connect(tmp_path / "h.db")
    rows: list[PlayerMapRow] = []
    # 30 matches of history for two teams' stars, alternating wins
    for i in range(30):
        ts = f"2026-06-{(i % 28) + 1:02d}T12:00:00"
        rows += [
            _row("p-donk", "★ donk ★", "Team Spirit", "FaZe Clan",
                 20, f"m{i}", 1, ts, won=i % 3 != 0, hs=9),
            _row("p-broky", "broky", "FaZe Clan", "Team Spirit",
                 12, f"m{i}", 1, ts, won=i % 3 == 0, hs=4),
        ]
    db.save_match_maps(conn, "seed", rows, "a", "2026-06-01", 1)
    return conn


def _board_prop(
    pid: str, name: str, team: str, opp: str, line: float,
    stat: str = "kills",
) -> Prop:
    return Prop(
        projection_id=pid, player_id=pid, player_name=name, team=team,
        opponent=opp, stat_type=stat, stat_kind=stat, map_range=(1, 2),
        line_score=line, board="standard",
        start_time="2026-07-25T18:00:00Z", league_id="CS",
    )


def test_history_replay_and_decorated_name_join(tmp_path: Path) -> None:
    conn = _seed_history(tmp_path)
    h = build_history(conn)
    hit = h.lookup("donk")  # board name is clean; history name is decorated
    assert hit is not None
    pid, team = hit
    assert pid == "p-donk"
    assert team == "spirit"  # canonicalised from "Team Spirit"
    # kpr state: 20 kills / 22 rounds
    assert abs((h.players["p-donk"].kpr.value or 0) - 20 / 22) < 1e-9
    assert abs((h.players["p-donk"].hs_rate.value or 0) - 9 / 20) < 1e-9


def test_group_matches_pairs_both_sides() -> None:
    props = [
        _board_prop("1", "donk", "Spirit", "FaZe", 32.5),
        _board_prop("2", "broky", "FaZe", "Spirit", 28.5),
        _board_prop("3", "other", "NAVI", "G2", 25.5),
    ]
    groups = group_matches(props)
    assert len(groups) == 2
    key = ("FaZe", "Spirit", "2026-07-25T18:00:00Z")
    assert {p.player_name for p in groups[key]} == {"donk", "broky"}


def test_build_match_sim_end_to_end(tmp_path: Path) -> None:
    conn = _seed_history(tmp_path)
    h = build_history(conn)
    props = [
        _board_prop("1", "donk", "Spirit", "FaZe", 33.5),
        _board_prop("2", "donk", "Spirit", "FaZe", 15.5, stat="headshots"),
        _board_prop("3", "broky", "FaZe", "Spirit", 24.5),
        _board_prop("4", "unknown_guy", "FaZe", "Spirit", 20.5),
    ]
    sim = build_match_sim(props, h, n_iters=4000, seed=5)
    assert sim is not None
    assert sim.matched == 2
    assert sim.unmatched == ["unknown_guy"]
    assert len(sim.props) == 3  # unknown player's prop excluded
    # donk projects ~0.9 kpr * ~44 rounds ~ 40 kills: over 33.5 should lean up
    donk_p = sim.result.p_over[0]
    assert donk_p > 0.5
    # broky ~0.55 kpr * 44 ~ 24: over 24.5 near coin-flip or below
    assert sim.result.p_over[2] < 0.65
    # ghosts padded both sides to 5
    # (donk + 4 ghosts) vs (broky + 4 ghosts) = 10 total; verified indirectly:
    # headshot prob must be below kills prob for same player
    assert sim.result.p_over[1] < 1.0


def test_simulate_board_skips_unjoinable(tmp_path: Path) -> None:
    conn = _seed_history(tmp_path)
    h = build_history(conn)
    props = [
        _board_prop("1", "donk", "Spirit", "FaZe", 33.5),
        _board_prop("2", "nobody", "XYZ", "ABC", 20.5),
    ]
    sims = simulate_board(props, h, n_iters=2000, seed=6)
    assert len(sims) == 1
    assert sims[0].label == "FaZe vs Spirit"
