"""Line-shopping tests: never bet a side at the worse number.

Measured on a live board 2026-07-24: 65% of props listed on both books had
DIFFERENT lines, gaps up to 2.0 kills — worth ~7 percentage points of win
probability per leg, more than every correlation refinement combined.
"""

from __future__ import annotations

from cs2props.ingest.prizepicks import Prop
from cs2props.lineshop import (
    best_price,
    build_index,
    market_gap,
    prop_key,
    shop,
)


def _prop(name: str, line: float, stat: str = "kills") -> Prop:
    return Prop(
        projection_id=f"{name}-{line}", player_id=name, player_name=name,
        team="T", opponent="O", stat_type=stat, stat_kind=stat,
        map_range=(1, 2), line_score=line, board="standard",
        start_time="2026-07-25T18:00:00Z", league_id="CS",
    )


def test_under_wants_the_highest_line() -> None:
    books = {"underdog": _prop("rdnzao", 28.5),
             "prizepicks": _prop("rdnzao", 27.0)}
    bp = best_price("under", books)
    assert bp.book == "underdog"
    assert bp.line == 28.5
    assert bp.edge_lines == 1.5


def test_over_wants_the_lowest_line() -> None:
    books = {"underdog": _prop("podi", 25.5),
             "prizepicks": _prop("podi", 23.5)}
    bp = best_price("over", books)
    assert bp.book == "prizepicks"
    assert bp.line == 23.5
    assert bp.edge_lines == 2.0


def test_single_book_has_no_alternative() -> None:
    bp = best_price("under", {"underdog": _prop("solo", 20.5)})
    assert bp.other_book is None
    assert bp.edge_lines == 0.0


def test_index_matches_players_across_books_and_skips_alts() -> None:
    alt = _prop("donk", 40.5)
    alt = Prop(**{**alt.__dict__, "board": "alt"})
    idx = build_index({
        "underdog": [_prop("donk", 30.5), alt],
        "prizepicks": [_prop("donk", 29.0)],
    })
    key = prop_key(_prop("donk", 0))
    assert set(idx[key]) == {"underdog", "prizepicks"}  # alt excluded
    assert market_gap(idx[key]) == 1.5


def test_market_gap_zero_when_books_agree() -> None:
    idx = build_index({
        "underdog": [_prop("x", 26.5)], "prizepicks": [_prop("x", 26.5)],
    })
    assert market_gap(idx[prop_key(_prop("x", 0))]) == 0.0


def test_shop_reports_better_book_sorted_by_gap() -> None:
    boards = {
        "underdog": [_prop("a", 28.5), _prop("b", 20.5)],
        "prizepicks": [_prop("a", 27.0), _prop("b", 20.5)],
    }
    leans = {prop_key(_prop("a", 0)): "under",
             prop_key(_prop("b", 0)): "under"}
    rows = shop(boards, leans)
    assert rows[0].player == "a"  # widest gap first
    assert rows[0].best_book == "underdog" and rows[0].gap == 1.5
    assert rows[1].gap == 0.0
