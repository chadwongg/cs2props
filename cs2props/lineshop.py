"""Cross-book line shopping.

Measured on a live board 2026-07-24: 65% of props listed on BOTH books carry
DIFFERENT lines, with gaps up to 2.0 kills. Taking the same side at the better
number is worth roughly 7 percentage points of win probability per leg on a
~29-kill line — larger than every correlation refinement in this codebase
combined, and it requires no modelling at all.

Two uses, both implemented here:

1. **Best price.** For a chosen side, one book's line is strictly better. An
   UNDER wants the HIGHEST line; an OVER wants the LOWEST. Betting the worse
   number is pure donation.

2. **Consensus as a sanity check.** Where the two books agree, the market is
   confident and a model disagreeing with both is more likely wrong than
   right. Where they disagree, the market itself is unsure — that is where a
   projection edge can actually live. Exposed as ``market_gap`` so the
   optimizer (or the reader) can prefer legs the books can't price.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cs2props.ingest.prizepicks import Prop
from cs2props.model.state_builder import clean_name

log = logging.getLogger(__name__)


def prop_key(p: Prop) -> tuple[str, str, tuple[int, int] | None]:
    """Identity of a prop across books: player, stat, map range."""
    return (clean_name(p.player_name), p.stat_kind, p.map_range)


@dataclass(frozen=True)
class BestPrice:
    """Where to bet one side of one prop, and what it is worth."""

    book: str
    line: float
    other_book: str | None
    other_line: float | None

    @property
    def edge_lines(self) -> float:
        """How many stat-units better than the alternative book."""
        if self.other_line is None:
            return 0.0
        return abs(self.line - self.other_line)


def best_price(
    side: str,
    boards: dict[str, Prop],
) -> BestPrice:
    """Pick the book offering the better number for ``side``.

    UNDER wants the highest line (more room below), OVER the lowest.
    ``boards`` maps book name -> that book's version of the same prop.
    """
    if not boards:
        raise ValueError("no books offer this prop")
    reverse = side.lower() == "under"
    ranked = sorted(
        boards.items(), key=lambda kv: kv[1].line_score, reverse=reverse
    )
    book, prop = ranked[0]
    if len(ranked) > 1:
        other_book, other = ranked[1]
        return BestPrice(book, prop.line_score, other_book, other.line_score)
    return BestPrice(book, prop.line_score, None, None)


def build_index(
    boards: dict[str, list[Prop]],
) -> dict[tuple[str, str, tuple[int, int] | None], dict[str, Prop]]:
    """-> {prop key: {book: prop}} across every supplied board."""
    index: dict[tuple[str, str, tuple[int, int] | None], dict[str, Prop]] = {}
    for book, props in boards.items():
        for p in props:
            if p.board != "standard":
                continue  # alt/demon/goblin ladders are not comparable
            index.setdefault(prop_key(p), {})[book] = p
    return index


def market_gap(books: dict[str, Prop]) -> float:
    """Spread between books on this prop. 0 when they agree or only one lists
    it. A wide gap means the market itself is unsure — the best hunting
    ground for a projection edge."""
    if len(books) < 2:
        return 0.0
    lines = [p.line_score for p in books.values()]
    return max(lines) - min(lines)


@dataclass(frozen=True)
class ShopRow:
    player: str
    stat: str
    maps: str
    side: str
    best_book: str
    best_line: float
    other_book: str | None
    other_line: float | None
    gap: float


def shop(
    boards: dict[str, list[Prop]],
    leans: dict[tuple[str, str, tuple[int, int] | None], str],
) -> list[ShopRow]:
    """For each prop we have a lean on, report where to bet it.

    ``leans`` maps prop key -> "over"/"under" (the model's chosen side).
    Only props listed on more than one book produce a meaningful row, but
    single-book props are returned too so the caller sees the full slate.
    """
    index = build_index(boards)
    rows: list[ShopRow] = []
    for key, books in index.items():
        side = leans.get(key)
        if side is None:
            continue
        bp = best_price(side, books)
        any_prop = next(iter(books.values()))
        maps = (
            f"{any_prop.map_range[0]}-{any_prop.map_range[1]}"
            if any_prop.map_range else "series"
        )
        rows.append(ShopRow(
            player=any_prop.player_name, stat=any_prop.stat_kind, maps=maps,
            side=side.upper(), best_book=bp.book, best_line=bp.line,
            other_book=bp.other_book, other_line=bp.other_line,
            gap=bp.edge_lines,
        ))
    rows.sort(key=lambda r: -r.gap)
    return rows
