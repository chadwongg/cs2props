"""One-click tracking: signature matching is what stops double-entry."""

from __future__ import annotations

from pathlib import Path

from cs2props import db
from cs2props.server import placed_signatures, slip_signature
from cs2props.tracker import parse_leg, track_slip

LEGS = [
    "HooXi under 14 headshots 1-2",
    "malbsMd under 20 headshots 1-2",
    "EliGE under 30.5 kills 1-2",
    "phzy under 29 kills 1-2",
]


def test_signature_ignores_leg_order() -> None:
    a = slip_signature([parse_leg(t) for t in LEGS])
    b = slip_signature([parse_leg(t) for t in reversed(LEGS)])
    assert a == b


def test_signature_distinguishes_a_changed_line() -> None:
    a = slip_signature([parse_leg(t) for t in LEGS])
    swapped = LEGS[:2] + ["EliGE under 31.5 kills 1-2"] + LEGS[3:]
    assert a != slip_signature([parse_leg(t) for t in swapped])


def test_signature_distinguishes_side() -> None:
    a = slip_signature([parse_leg(t) for t in LEGS])
    flipped = ["HooXi over 14 headshots 1-2"] + LEGS[1:]
    assert a != slip_signature([parse_leg(t) for t in flipped])


def test_placed_signatures_round_trip(tmp_path: Path) -> None:
    """A tracked slip must be recognisable afterwards, so the report can hide
    it and the user cannot enter the same bet twice."""
    dbp = tmp_path / "s.db"
    conn = db.connect(dbp)
    track_slip(conn, "prizepicks", 1.0, [parse_leg(t) for t in LEGS])
    conn.close()

    sigs = placed_signatures(dbp)
    assert slip_signature([parse_leg(t) for t in LEGS]) in sigs
    other = ["nilo under 20.5 headshots 1-2"] + LEGS[1:]
    assert slip_signature([parse_leg(t) for t in other]) not in sigs


def test_decorated_names_match_the_same_slip(tmp_path: Path) -> None:
    dbp = tmp_path / "s.db"
    conn = db.connect(dbp)
    track_slip(conn, "prizepicks", 1.0,
               [parse_leg("ataraXia under 20.5 kills 1-2"),
                parse_leg("donk under 30.5 kills 1-2")])
    conn.close()
    same = slip_signature([parse_leg("ataraxia under 20.5 kills 1-2"),
                           parse_leg("donk under 30.5 kills 1-2")])
    assert same in placed_signatures(dbp)
