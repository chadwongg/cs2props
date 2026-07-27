"""Roster verification tests.

The false-positive guard is the important part: bo3.gg does not cover every
match the books post, so a player missing from the index means nothing unless
the match is demonstrably covered. Flagging uncovered matches would reject
most of a tier-B/C board.
"""

from __future__ import annotations

from cs2props.roster import RosterIndex, benched_players


def _index(*players: str) -> RosterIndex:
    return RosterIndex(players={p.lower() for p in players})


def test_flags_a_benched_player_when_match_is_covered() -> None:
    idx = _index("twistzz", "frozen", "jcobbb", "broky")
    # rain is absent while four team-mates are announced -> genuinely benched
    out = benched_players(["Twistzz", "frozen", "jcobbb", "broky", "rain"], idx)
    assert out == ["rain"]


def test_uncovered_match_flags_nobody() -> None:
    """No player from this match is in the index — bo3.gg simply doesn't
    cover it, so we cannot conclude anyone is benched."""
    idx = _index("twistzz", "frozen")
    assert benched_players(["obscure1", "obscure2", "obscure3"], idx) == []


def test_empty_index_flags_nobody() -> None:
    assert benched_players(["anyone"], RosterIndex()) == []


def test_decorated_names_still_match() -> None:
    idx = _index("ataraxia", "donk")
    assert benched_players(["★ ⑳ ataraXia †", "donk"], idx) == []


def test_all_announced_flags_nobody() -> None:
    idx = _index("a", "b", "c", "d", "e")
    assert benched_players(["A", "B", "C", "D", "E"], idx) == []
