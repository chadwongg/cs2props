"""Stand-in detection must be conservative in both directions.

A missed stand-in prices a match as if a different team were playing. A
FALSE stand-in is worse: it moves projections on a roster that never
changed, and it would fire constantly on tier-C teams whose history is thin
simply because we have not seen their players before. Both failure modes are
tested; the "cannot judge" path matters as much as the detection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from cs2props import db
from cs2props.standins import (
    REGULAR_KPR_FACTOR,
    STANDIN_MAP_WIN_DELTA,
    adjust_map_win,
    detect,
    kpr_factor,
    team_regulars,
)

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)
ROSTER = ["alpha", "bravo", "charlie", "delta", "echo"]


def _play(
    conn: "db.sqlite3.Connection", team: str, players: list[str],
    days_ago: int, match_id: str, n_maps: int = 2,
) -> None:
    at = (NOW - timedelta(days=days_ago)).isoformat()
    for p in players:
        for m in range(1, n_maps + 1):
            conn.execute(
                "INSERT OR REPLACE INTO player_maps (player_id, player_name,"
                " team, opponent, event_tier, map_name, played_at, kills,"
                " deaths, adr, rating, rounds, headshots, won, match_id,"
                " map_number) VALUES (?,?,?,'OPP','b','de_dust2',?,20,15,80,"
                "1.1,24,10,1,?,?)",
                (p, p, team, at, f"{match_id}-{m}", m),
            )
    conn.commit()


def _seed_regular_team(conn: "db.sqlite3.Connection") -> None:
    """Enough recent history that the team can be judged at all."""
    for i in range(6):
        _play(conn, "TeamA", ROSTER, days_ago=5 + i * 3, match_id=f"m{i}")


def test_usual_roster_is_not_flagged(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    _seed_regular_team(conn)
    rep = detect(conn, {"TeamA": ROSTER}, NOW.timestamp())
    lu = rep.for_team("TeamA")
    assert lu is not None and lu.judged
    assert not lu.has_standin
    assert rep.teams_with_standins() == []
    conn.close()


def test_a_new_face_is_flagged_as_a_standin(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    _seed_regular_team(conn)
    lineup = ["alpha", "bravo", "charlie", "delta", "zulu"]  # echo -> zulu
    rep = detect(conn, {"TeamA": lineup}, NOW.timestamp())
    lu = rep.for_team("TeamA")
    assert lu is not None and lu.has_standin
    assert lu.standins == ("zulu",)
    assert "echo" in lu.missing_regulars
    assert "STAND-IN zulu" in lu.describe()
    conn.close()


def test_thin_history_is_unjudged_not_flagged(tmp_path: Path) -> None:
    """A team we have barely seen must NOT have all five called stand-ins.
    That would fire on most of a tier-C board and move every projection in
    it on the strength of our own missing data."""
    conn = db.connect(tmp_path / "a.db")
    _play(conn, "NewTeam", ROSTER, days_ago=3, match_id="only")
    rep = detect(conn, {"NewTeam": ROSTER}, NOW.timestamp())
    lu = rep.for_team("NewTeam")
    assert lu is not None
    assert not lu.judged
    assert not lu.has_standin
    conn.close()


def test_regulars_are_judged_only_on_the_past(tmp_path: Path) -> None:
    """The match being assessed must not define who counts as a regular."""
    conn = db.connect(tmp_path / "a.db")
    _seed_regular_team(conn)
    _play(conn, "TeamA", ["zulu"] * 1, days_ago=-1, match_id="future")
    regs, _ = team_regulars(conn, "TeamA", NOW.timestamp())
    assert "zulu" not in regs
    conn.close()


def test_stale_players_age_out_of_the_regular_set(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    _seed_regular_team(conn)
    _play(conn, "TeamA", ["ancient"], days_ago=200, match_id="old")
    regs, _ = team_regulars(conn, "TeamA", NOW.timestamp())
    assert "ancient" not in regs
    assert "alpha" in regs
    conn.close()


def test_leetspeak_spelling_is_not_a_standin(tmp_path: Path) -> None:
    """Caught on the first live run: the board writes sh1ro and zont1x where
    bo3.gg writes sh1r0 and zontix. Flagging those invents a roster change on
    Team Spirit's actual starting five."""
    conn = db.connect(tmp_path / "a.db")
    hist = ["sh1r0", "zontix", "d0nk", "magixx", "chopper"]
    for i in range(6):
        _play(conn, "Spirit", hist, days_ago=5 + i * 3, match_id=f"s{i}")
    board = ["sh1ro", "zont1x", "donk", "magixx", "chopper"]
    rep = detect(conn, {"Spirit": board}, NOW.timestamp())
    lu = rep.for_team("Spirit")
    assert lu is not None and lu.judged
    assert not lu.has_standin, f"false stand-in: {lu.standins}"
    conn.close()


def test_appended_tag_is_not_a_standin(tmp_path: Path) -> None:
    """kursy vs kursyssj4 — players append tags to their handle."""
    conn = db.connect(tmp_path / "a.db")
    hist = ["kursyssj4", "bravo", "charlie", "delta", "echo"]
    for i in range(6):
        _play(conn, "TeamB", hist, days_ago=5 + i * 3, match_id=f"b{i}")
    rep = detect(
        conn, {"TeamB": ["kursy", "bravo", "charlie", "delta", "echo"]},
        NOW.timestamp(),
    )
    lu = rep.for_team("TeamB")
    assert lu is not None and not lu.has_standin
    conn.close()


def test_a_genuinely_different_name_still_flags(tmp_path: Path) -> None:
    """The loose matching must not swallow real substitutions."""
    conn = db.connect(tmp_path / "a.db")
    _seed_regular_team(conn)
    rep = detect(
        conn, {"TeamA": ["alpha", "bravo", "charlie", "delta", "zulu"]},
        NOW.timestamp(),
    )
    lu = rep.for_team("TeamA")
    assert lu is not None and lu.standins == ("zulu",)
    conn.close()


def test_map_win_is_shaded_down_for_a_standin() -> None:
    from cs2props.standins import TeamLineup

    normal = TeamLineup("T", judged=True)
    withsub = TeamLineup("T", standins=("zulu",), judged=True)
    assert adjust_map_win(0.55, normal) == 0.55
    assert adjust_map_win(0.55, withsub) == 0.55 + STANDIN_MAP_WIN_DELTA
    # never leaves probability space
    assert 0.0 < adjust_map_win(0.01, withsub) < 1.0


def test_unjudged_lineup_changes_nothing() -> None:
    from cs2props.standins import TeamLineup

    unjudged = TeamLineup("T", standins=("zulu",), judged=False)
    assert adjust_map_win(0.55, unjudged) == 0.55
    assert kpr_factor("alpha", unjudged) == 1.0
    assert kpr_factor("alpha", None) == 1.0


def test_regulars_gain_kill_share_but_the_sub_does_not() -> None:
    """Measured: regulars +1.08% (z=3.2); the substitute himself +1.1%
    +/-1.3%, indistinguishable from zero. Penalising him would be modelling
    a prejudice rather than a finding."""
    from cs2props.standins import TeamLineup

    lu = TeamLineup("T", standins=("zulu",), judged=True)
    assert kpr_factor("alpha", lu) == REGULAR_KPR_FACTOR
    assert kpr_factor("zulu", lu) == 1.0
