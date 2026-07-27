"""Report renderer: per-book sections + safety states."""

from __future__ import annotations

from cs2props.report import mock_data, render


def test_renders_both_book_sections() -> None:
    out = render(mock_data())
    assert 'class="book-badge prizepicks"' in out
    assert 'class="book-badge underdog"' in out
    assert out.index("PrizePicks") < out.index("Underdog")


def test_mock_banner_present_when_mock() -> None:
    out = render(mock_data())
    assert "MOCK DATA" in out
    assert "NOT CALIBRATED" in out


def test_freshness_shown_per_book() -> None:
    out = render(mock_data())
    assert "your last pp.json save" in out
    assert "fetched live" in out


def test_delta_is_rendered_per_slip() -> None:
    out = render(mock_data())
    assert out.count('class="delta"') == 4  # 2 PP + 2 UD mock slips


def test_placing_a_slip_retires_other_cards_sharing_a_player() -> None:
    """The server-side held-player exclusion only applies to the NEXT scan.
    A page left open will happily sell the same leg twice — it did on
    2026-07-26, when MahaR 13.5 went into two slips clicked from one render.
    Two slips sharing a leg are not two bets: same model error, they fail
    together, and the calibration sample gets a duplicate.
    """
    import json

    from cs2props.report import (
        BookView, LegView, ReportData, SlipView, render,
    )

    def leg(p: str) -> LegView:
        return LegView("OVER", p, "T", "kills", "1-1", 13.5, 0.62, "vs X")

    def slip(rank: int, players: list[str]) -> SlipView:
        return SlipView(
            rank=rank, n_legs=3, multiplier=6.5, ev_pct=51.0,
            ev_adj_pct=19.0, p_correlated=0.232, p_independent=0.232,
            legs=tuple(leg(p) for p in players), breakeven=2.95,
            legs_json=json.dumps(
                [f"{p} over 13.5 kills 1-1" for p in players]
            ),
            book="underdog",
        )

    out = render(ReportData(
        generated="now", calibration_label="test", is_mock=False,
        books=(BookView(
            book="underdog", display="Underdog", board_label="100 props",
            freshness="live",
            slips=(slip(1, ["MahaR", "Majky", "KWERTZZ"]),
                   slip(2, ["kronkzz", "eSx", "MahaR"]),
                   slip(3, ["aidKiT", "hfah", "kodak"])),
        ),),
    ))
    # the retirement pass must exist and run on a successful lock-in
    assert "function retireSlipsUsing" in out
    assert "retireSlipsUsing(JSON.parse" in out
    assert ".slip.superseded" in out
    # every card carries the leg specs the pass compares on
    assert out.count("data-legs=") == 3
    assert "MahaR" in out and "kronkzz" in out
