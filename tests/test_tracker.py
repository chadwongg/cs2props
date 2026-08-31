"""Module 5 grading tests: wins, losses, voids, pushes, void-adjusted payout."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from pathlib import Path

from cs2props import db
from cs2props.ingest.bo3gg import PlayerMapRow
from cs2props.tracker import (
    grade_open_slips,
    parse_leg,
    summary,
    track_slip,
)

PLACED = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc).timestamp()


def _row(name: str, kills: int, hs: int, map_no: int, n_maps_match: str = "m1",
         when: str = "2026-07-20T18:00:00") -> PlayerMapRow:
    return PlayerMapRow(
        player_id=f"p-{name}", player_name=name, team="T", opponent="O",
        event_tier="a", map_name="de_mirage", played_at=when, kills=kills,
        deaths=10, adr=70.0, rating=1.0, rounds=22, headshots=hs, won=1,
        match_id=n_maps_match, map_number=map_no,
    )


def _seed(tmp_path: Path, rows: list[PlayerMapRow]) -> "db.sqlite3.Connection":
    conn = db.connect(tmp_path / "t.db")
    db.save_match_maps(conn, "m1", rows, "a", "2026-07-20", 2)
    return conn


def test_parse_leg() -> None:
    leg = parse_leg("donk over 32.5 kills 1-2")
    assert (leg.player_name, leg.side, leg.line) == ("donk", "over", 32.5)
    assert (leg.stat_kind, leg.map_lo, leg.map_hi) == ("kills", 1, 2)


def test_win_and_loss_grading(tmp_path: Path) -> None:
    conn = _seed(tmp_path, [
        _row("donk", 20, 9, 1), _row("donk", 15, 7, 2),   # 35 kills maps 1-2
        _row("broky", 10, 4, 1), _row("broky", 11, 5, 2),  # 21 kills
    ])
    track_slip(conn, "prizepicks", 10.0, [
        parse_leg("donk over 32.5 kills 1-2"),      # 35 > 32.5 -> win
        parse_leg("broky under 24.5 kills 1-2"),    # 21 < 24.5 -> win
    ], claimed_p=0.30, placed_at=PLACED)
    track_slip(conn, "prizepicks", 10.0, [
        parse_leg("donk under 32.5 kills 1-2"),     # loss
        parse_leg("broky under 24.5 kills 1-2"),
    ], claimed_p=0.30, placed_at=PLACED)
    settled = grade_open_slips(conn)
    assert settled == 2
    rows = conn.execute(
        "SELECT status, payout FROM slips ORDER BY payout DESC"
    ).fetchall()
    assert rows[0] == ("won", 30.0)   # 2-pick power pays 3x
    assert rows[1] == ("lost", 0.0)


def test_map3_void_shrinks_slip_payout(tmp_path: Path) -> None:
    """4 legs, one is a map-3 prop in a 2-map sweep -> void -> pays as 3-pick."""
    conn = _seed(tmp_path, [
        _row("a1", 20, 9, 1), _row("a1", 15, 7, 2),
        _row("a2", 18, 8, 1), _row("a2", 16, 7, 2),
        _row("a3", 14, 6, 1), _row("a3", 13, 6, 2),
        _row("a4", 12, 5, 1), _row("a4", 11, 5, 2),   # match had only 2 maps
    ])
    track_slip(conn, "prizepicks", 10.0, [
        parse_leg("a1 over 30.5 kills 1-2"),   # 35 -> win
        parse_leg("a2 over 30.5 kills 1-2"),   # 34 -> win
        parse_leg("a3 over 25.5 kills 1-2"),   # 27 -> win
        parse_leg("a4 over 5.5 kills 3-3"),    # map 3 never played -> void
    ], placed_at=PLACED)
    assert grade_open_slips(conn) == 1
    status, payout = conn.execute(
        "SELECT status, payout FROM slips"
    ).fetchone()
    assert status == "won"
    assert payout == 60.0  # 3-pick power 6x, not 4-pick 10x
    void = conn.execute(
        "SELECT status FROM slip_legs WHERE player_name='a4'"
    ).fetchone()[0]
    assert void == "void"


def test_whole_line_push_voids_leg(tmp_path: Path) -> None:
    conn = _seed(tmp_path, [
        _row("x", 15, 8, 1), _row("x", 14, 9, 2),   # 17 HS exactly
        _row("y", 20, 9, 1), _row("y", 15, 8, 2),
        _row("z", 18, 8, 1), _row("z", 15, 7, 2),
    ])
    track_slip(conn, "prizepicks", 10.0, [
        parse_leg("x over 17 headshots 1-2"),     # exactly 17 -> push/void
        parse_leg("y over 30.5 kills 1-2"),       # 35 -> win
        parse_leg("z over 30.5 kills 1-2"),       # 33 -> win
    ], placed_at=PLACED)
    grade_open_slips(conn)
    status, payout = conn.execute("SELECT status, payout FROM slips").fetchone()
    assert status == "won"
    assert payout == 30.0  # shrank to 2-pick at 3x


def test_does_not_grade_against_an_earlier_match(tmp_path: Path) -> None:
    """A slip must never settle against a game played BEFORE it was placed.
    The original 6h look-back graded two live slips against the players'
    previous matches and reported fabricated losses (2026-07-24)."""
    earlier = [
        _row("donk", 40, 20, 1, when="2026-07-20T06:00:00"),
        _row("donk", 40, 20, 2, when="2026-07-20T06:00:00"),
    ]
    conn = _seed(tmp_path, earlier)  # played 6h BEFORE PLACED
    track_slip(conn, "prizepicks", 10.0, [
        parse_leg("donk under 32.5 kills 1-2"),
        parse_leg("donk over 30.5 kills 1-2"),
    ], placed_at=PLACED)
    assert grade_open_slips(conn) == 0
    assert conn.execute("SELECT status FROM slips").fetchone()[0] == "pending"


def test_unplayed_match_stays_pending(tmp_path: Path) -> None:
    conn = _seed(tmp_path, [_row("someone", 20, 9, 1)])
    track_slip(conn, "prizepicks", 10.0, [
        parse_leg("ghost_player over 30.5 kills 1-2"),
    ], placed_at=PLACED)
    assert grade_open_slips(conn) == 0
    assert conn.execute("SELECT status FROM slips").fetchone()[0] == "pending"


def test_summary_reports_live_calibration(tmp_path: Path) -> None:
    conn = _seed(tmp_path, [
        _row("donk", 20, 9, 1), _row("donk", 15, 7, 2),
    ])
    track_slip(conn, "prizepicks", 10.0,
               [parse_leg("donk over 32.5 kills 1-2"),
                parse_leg("donk over 30.5 kills 1-2")],
               claimed_p=0.4, placed_at=PLACED)
    grade_open_slips(conn)
    out = summary(conn)
    # results are reported PER BOOK: a PrizePicks 4-pick wins ~17% and an
    # Underdog 2-pick ~48%, so a pooled win rate describes neither
    assert "prizepicks" in out
    assert "1/1" in out                     # claimed vs actual for that book
    assert "LEG hit rate" in out            # pooled — tests the model itself


def test_committed_players_covers_open_slips_only(tmp_path: Path) -> None:
    """A player in an OPEN slip must be blocked from new suggestions — two
    slips sharing a leg fail together. Once a slip settles the player is free
    again, because the shared-fate problem is gone."""
    from cs2props.tracker import committed_players

    conn = _seed(tmp_path, [_row("donk", 20, 9, 1), _row("donk", 15, 7, 2)])
    track_slip(conn, "prizepicks", 1.0,
               [parse_leg("HooXi under 14 headshots 1-2"),
                parse_leg("EliGE under 30.5 kills 1-2")],
               placed_at=PLACED)
    held = committed_players(conn)
    assert {"hooxi", "elige"} <= held

    conn.execute("UPDATE slips SET status='lost'")
    conn.commit()
    assert committed_players(conn) == set()
    conn.close()


def test_committed_players_normalises_decorated_names(tmp_path: Path) -> None:
    from cs2props.tracker import committed_players

    conn = _seed(tmp_path, [_row("x", 10, 5, 1)])
    track_slip(conn, "prizepicks", 1.0,
               [parse_leg("ataraXia under 20.5 kills 1-2"),
                parse_leg("donk under 30.5 kills 1-2")], placed_at=PLACED)
    assert "ataraxia" in committed_players(conn)
    conn.close()


def test_power_slip_settles_on_its_first_lost_leg(tmp_path: Path) -> None:
    """A power slip cannot recover from a lost leg, so waiting on the rest
    is dead time — and the wait keeps that leg's MATCH excluded from the
    scanner long after the bet is decided."""
    conn = _seed(tmp_path, [
        _row("b1", 20, 9, 1), _row("b1", 15, 7, 2),
    ])
    track_slip(conn, "prizepicks", 1.0, [
        parse_leg("b1 under 10.5 kills 1-2"),      # 35 -> LOST
        parse_leg("ghost over 5.5 kills 1-2"),     # never played -> pending
    ], placed_at=PLACED)
    assert grade_open_slips(conn) == 1
    status, payout = conn.execute(
        "SELECT status, payout FROM slips"
    ).fetchone()
    assert status == "lost" and payout == 0.0


def test_settling_early_does_not_orphan_the_remaining_legs(
    tmp_path: Path,
) -> None:
    """The ungraded legs of a dead slip are still calibration data. Dropping
    them biases the leg rate DOWNWARD, because a slip settles early exactly
    when its legs are losing — measured at 58.0% -> 52.6% when three dead
    slips left their remainder ungraded."""
    conn = _seed(tmp_path, [
        _row("c1", 20, 9, 1), _row("c1", 15, 7, 2),
        _row("c2", 30, 9, 1), _row("c2", 25, 7, 2),
    ])
    track_slip(conn, "prizepicks", 1.0, [
        parse_leg("c1 under 10.5 kills 1-2"),   # 35 -> LOST, kills the slip
        parse_leg("c2 over 10.5 kills 1-2"),    # 55 -> WON, must still count
    ], placed_at=PLACED)
    grade_open_slips(conn)
    grade_open_slips(conn)  # second pass reaches the settled slip's legs
    statuses = [r[0] for r in conn.execute(
        "SELECT status FROM slip_legs ORDER BY leg_no")]
    assert "pending" not in statuses, f"orphaned leg: {statuses}"
    assert statuses == ["lost", "won"]


def test_flex_slip_pays_the_tier_not_all_or_nothing(tmp_path: Path) -> None:
    """Found live 2026-07-27: a 5-pick flex that went 3/5 was settled as
    "lost $0.00" while PrizePicks paid $0.40 — the UI had stored it as
    "power" and the power path early-settled it on the first lost leg."""
    conn = _seed(tmp_path, [
        _row("f1", 20, 9, 1), _row("f1", 15, 7, 2),
        _row("f2", 20, 9, 1), _row("f2", 15, 7, 2),
        _row("f3", 20, 9, 1), _row("f3", 15, 7, 2),
        _row("f4", 20, 9, 1), _row("f4", 15, 7, 2),
        _row("f5", 20, 9, 1), _row("f5", 15, 7, 2),
    ])
    track_slip(conn, "prizepicks", 1.0, [
        parse_leg("f1 over 30.5 kills 1-2"),   # 35 -> win
        parse_leg("f2 over 30.5 kills 1-2"),   # 35 -> win
        parse_leg("f3 over 30.5 kills 1-2"),   # 35 -> win
        parse_leg("f4 under 30.5 kills 1-2"),  # 35 -> LOSS
        parse_leg("f5 under 30.5 kills 1-2"),  # 35 -> LOSS
    ], placed_at=PLACED, product="flex")
    assert grade_open_slips(conn) == 1
    status, payout = conn.execute(
        "SELECT status, payout FROM slips").fetchone()
    # 3-of-5 flex pays 0.4x: the app calls it a Win, but a $0.40 return on a
    # $1 stake is a net loss — recorded as such, with the payout kept so the
    # P&L stays honest.
    assert payout == 0.4
    assert status == "lost"


def test_flex_slip_never_settles_early(tmp_path: Path) -> None:
    """One lost leg decides a power slip; a flex slip can still cash a lower
    tier, so it must wait for every leg."""
    conn = _seed(tmp_path, [
        _row("g1", 20, 9, 1), _row("g1", 15, 7, 2),
    ])
    track_slip(conn, "prizepicks", 1.0, [
        parse_leg("g1 under 30.5 kills 1-2"),   # 35 -> LOSS
        parse_leg("ghost over 5.5 kills 1-2"),  # unplayed -> pending
    ], placed_at=PLACED, product="flex")
    grade_open_slips(conn)
    assert conn.execute(
        "SELECT status FROM slips").fetchone()[0] == "pending"


def test_manual_grade_settles_what_auto_grade_cannot(tmp_path: Path) -> None:
    """bo3.gg skips whole tier-C events (three settled legs had nothing to
    grade against, 2026-07-27) and a did-not-play void never gets a row. The
    user reads the settled number in the book's app; recording it must grade
    the leg and settle the slip through the normal path."""
    from cs2props.tracker import manual_grade_leg

    conn = _seed(tmp_path, [])
    track_slip(conn, "prizepicks", 1.0, [
        parse_leg("ghostA over 24.5 kills 1-2"),
        parse_leg("ghostB under 20.5 kills 1-2"),
        parse_leg("ghostC over 30.5 kills 1-2"),
    ], placed_at=PLACED, product="flex")
    assert manual_grade_leg(conn, _only_slip(conn), 0, 29.0) == "won"
    assert manual_grade_leg(conn, _only_slip(conn), 1, 25.0) == "lost"
    assert manual_grade_leg(conn, _only_slip(conn), 2, None, dnp=True) == "void"
    status, payout = conn.execute(
        "SELECT status, payout FROM slips").fetchone()
    # the DNP void shrinks the 3-flex below the smallest flex tier, so it
    # converts to an all-must-hit play — 1 win of 2 live pays nothing
    assert status == "lost" and payout == 0.0


def test_manual_grade_never_overwrites_an_auto_grade(tmp_path: Path) -> None:
    from cs2props.tracker import manual_grade_leg

    conn = _seed(tmp_path, [_row("h1", 20, 9, 1), _row("h1", 15, 7, 2)])
    track_slip(conn, "prizepicks", 1.0, [
        parse_leg("h1 over 30.5 kills 1-2"),   # 35 -> auto-won
        parse_leg("ghost over 5.5 kills 1-2"),
    ], placed_at=PLACED)
    grade_open_slips(conn)
    assert manual_grade_leg(conn, _only_slip(conn), 0, 1.0) == "won"


def _only_slip(conn: "sqlite3.Connection") -> str:
    return str(conn.execute("SELECT slip_id FROM slips").fetchone()[0])


def test_wrong_opponent_match_stays_pending(tmp_path: Path) -> None:
    """The prop names an opponent; a time-window match against anyone else
    is a DIFFERENT game. controlez (2026-08-28, vs Rare Atom): the real
    match never reached bo3.gg and the grader scored his NEXT match (vs
    Alter Ego, 24h later, inside the 26h skew window). With the opponent
    guard, that leg stays pending for manual grading."""
    conn = db.connect(tmp_path / "t.db")
    # archived prop: controlez vs RA, starting shortly after placement
    conn.execute(
        "INSERT INTO props (scanned_at, projection_id, player_id,"
        " player_name, team, opponent, stat_type, stat_kind, map_lo, map_hi,"
        " line_score, board, start_time, league_id) VALUES"
        " (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (PLACED - 600, "x1", "c1", "controlez", "HUNS", "RA",
         "MAPS 1-2 Kills", "kills", 1, 2, 27.5, "standard",
         "2026-07-20T13:00:00Z", "u"),
    )
    # only ingested match in the window: NEXT DAY, wrong opponent
    db.save_match_maps(conn, "m9", [
        _row("controlez--", 24, 9, 1, "m9", "2026-07-21T05:20:00+00:00"),
        _row("controlez--", 9, 3, 2, "m9", "2026-07-21T06:20:00+00:00"),
    ], "c", "2026-07-21", 2)
    conn.execute("UPDATE player_maps SET team='The Huns Esports',"
                 " opponent='Alter Ego'")
    conn.commit()
    track_slip(conn, "underdog", 1.0,
               [parse_leg("controlez under 27.5 kills 1-2")],
               placed_at=PLACED)
    grade_open_slips(conn)
    st = conn.execute("SELECT status FROM slip_legs").fetchone()[0]
    assert st == "pending"  # wrong-opponent match must NOT grade the leg
    # same match but with the RIGHT opponent: grades normally
    conn.execute("UPDATE player_maps SET opponent='Rare Atom'")
    conn.commit()
    grade_open_slips(conn)
    st, obs = conn.execute(
        "SELECT status, observed FROM slip_legs").fetchone()
    assert (st, obs) == ("lost", 33.0)  # 33 > 27.5 on an under


def test_team_abbreviation_matching() -> None:
    from cs2props.tracker import _team_matches

    assert _team_matches("RA", "Rare Atom")        # initials
    assert _team_matches("EXR", "ex-RUSTEC")       # prefix after cleaning
    assert not _team_matches("RA", "Alter Ego")
    assert not _team_matches("NL", "Iowa Stormboar")
