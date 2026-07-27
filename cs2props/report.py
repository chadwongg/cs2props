"""HTML report renderer — the app's face.

Renders scan output to a self-contained HTML file, organized as one section
per book (PrizePicks / Underdog). Slips are book-specific by nature: payout
tables differ, and each book's board has its own freshness (Underdog is
fetched live; PrizePicks is only as fresh as the user's last import).

Until module 4 exists this runs on clearly-labeled mock data; the renderer
is the contract, the numbers are placeholders.

Design: dark analyst-terminal. The correlation delta is the headline number
on every slip card, calibration provenance is pinned to the header, and
mock/uncalibrated states are loud amber banners.
"""

from __future__ import annotations

import html
from typing import Any
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class LegView:
    side: str  # OVER / UNDER
    player: str
    team: str
    stat: str
    maps: str
    line: float
    p_hit: float
    context: str  # "vs FaZe · Sat 14:00"


@dataclass(frozen=True)
class SlipView:
    rank: int
    n_legs: int
    multiplier: float
    ev_pct: float
    p_correlated: float
    p_independent: float
    legs: tuple[LegView, ...]
    flags: tuple[str, ...] = ()
    # Multiplier needed to be +EV. No longer shown on the card: with the
    # ladder verified, the pair rule enforced and per-side shades priced, the
    # payout is KNOWN, so "profitable above 2.95x" beside "pays 6.5x" and
    # "EV +120%" was the same fact three ways. Retained because PrizePicks
    # ARENA entries genuinely cannot be priced ahead of time — identical
    # lineups there paid 7.25x / 7.75x / 8.25x / 9.25x.
    breakeven: float = 0.0
    track_cmd: str = ""  # ready-to-paste command to log this slip
    legs_json: str = "[]"  # leg specs for the one-click tracker
    book: str = ""
    ev_adj_pct: float = 0.0  # EV after the model's optimism haircut
    delta_is_real: bool = False  # correlation bonus above Monte Carlo noise
    product: str = "power"  # settles all-or-nothing (power) or by tier (flex)


@dataclass(frozen=True)
class TrackedLeg:
    side: str
    player: str
    line: float
    stat: str
    status: str  # pending | won | lost | void
    observed: float | None
    clv: float | None = None  # + = line moved your way after you bet
    closing_line: float | None = None


@dataclass(frozen=True)
class TrackedSlip:
    slip_id: str
    book: str
    n_legs: int
    stake: float
    multiplier: float | None
    claimed_p: float | None
    status: str  # pending | won | lost
    payout: float | None
    legs: tuple[TrackedLeg, ...]
    clv: float | None = None  # mean CLV across this slip's legs


@dataclass(frozen=True)
class BookView:
    book: str  # "prizepicks" | "underdog"
    display: str  # "PrizePicks"
    board_label: str  # "425 props · 12 matches"
    freshness: str  # "imported 2026-07-24 00:59" | "fetched live 07:32 UTC"
    slips: tuple[SlipView, ...]
    legs: tuple[LegView, ...] = ()


@dataclass(frozen=True)
class ReportData:
    generated: str
    calibration_label: str
    is_mock: bool
    books: tuple[BookView, ...]
    tracked: tuple[TrackedSlip, ...] = ()
    tracked_summary: str = ""
    # structured twin of tracked_summary — rendered as a table instead of a
    # 300-character run-on sentence
    book_stats: tuple[Any, ...] = ()
    leg_record: tuple[int, int] = (0, 0)
    legacy_note: str = ""  # pre-epoch record, shown small but never hidden


def _mock_legs() -> dict[str, LegView]:
    return {
        "donk": LegView("OVER", "donk", "Spirit", "kills", "1-2", 32.5, 0.64,
                        "vs FaZe · Sat 14:00"),
        "sh1ro": LegView("OVER", "sh1ro", "Spirit", "kills", "1-2", 28.5, 0.61,
                         "vs FaZe · Sat 14:00"),
        "zont1x": LegView("OVER", "zont1x", "Spirit", "kills", "1-2", 25.5,
                          0.58, "vs FaZe · Sat 14:00"),
        "m0NESY": LegView("OVER", "m0NESY", "Falcons", "kills", "1-2", 29.5,
                          0.62, "vs Vitality · Sat 17:00"),
        "ropz": LegView("UNDER", "ropz", "Vitality", "kills", "1-3", 44.5,
                        0.60, "vs Falcons · Sat 17:00"),
        "broky": LegView("UNDER", "broky", "FaZe", "kills", "1-2", 27.5, 0.57,
                         "vs Spirit · Sat 14:00"),
    }


def mock_data() -> ReportData:
    """Same fake slate as `scan --mock` — one source of placeholder truth."""
    L = _mock_legs()
    pp_slips = (
        SlipView(1, 4, 10.0, 18.4, 0.128, 0.094,
                 (L["donk"], L["sh1ro"], L["zont1x"], L["m0NESY"]),
                 ("3x same-team stack — review book rules",)),
        SlipView(2, 4, 10.0, 11.7, 0.112, 0.098,
                 (L["donk"], L["m0NESY"], L["ropz"], L["broky"])),
    )
    ud_slips = (
        SlipView(1, 3, 6.0, 14.9, 0.192, 0.164,
                 (L["donk"], L["sh1ro"], L["m0NESY"]),
                 ("2x same-team stack",)),
        SlipView(2, 4, 10.0, 9.6, 0.110, 0.099,
                 (L["donk"], L["sh1ro"], L["ropz"], L["m0NESY"])),
    )
    legs = tuple(L.values())
    return ReportData(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        calibration_label="NOT CALIBRATED — backtest gate pending",
        is_mock=True,
        books=(
            BookView("prizepicks", "PrizePicks", "425 props · 12 matches",
                     "imported 2026-07-24 00:59 (your last pp.json save)",
                     pp_slips, legs),
            BookView("underdog", "Underdog", "681 props · 14 matches",
                     "fetched live", ud_slips, legs),
        ),
    )


def _leg_row(leg: LegView) -> str:
    side_cls = leg.side.lower()
    return (
        f'<div class="leg"><span class="side {side_cls}">{leg.side}</span>'
        f'<span class="lname">{html.escape(leg.player)}</span>'
        f'<span class="lteam">{html.escape(leg.team)}</span>'
        f'<span class="lstat">{leg.line:g} {html.escape(leg.stat)} '
        f'({html.escape(leg.maps)})</span>'
        f'<span class="lp">{leg.p_hit:.0%}</span>'
        f'<span class="lctx">{html.escape(leg.context)}</span></div>'
    )


def _slip_card(s: SlipView) -> str:
    """One slip card.

    The correlation delta used to be the headline, rendered at 22px. On a
    diversified slip its true value is ZERO — separate matches are simulated
    independently — so the biggest number on the card was Monte Carlo error,
    sitting beside "vs independent 33.8%", which said the same thing again.
    The headline is now what the book PAYS: the ladder is verified, the
    same-match pair rule is verified, and per-side shades are priced, so the
    old "estimate only" hedge no longer applies to standard entries.
    """
    delta = (s.p_correlated - s.p_independent) * 100
    flags = "".join(
        f'<span class="flag">{html.escape(f)}</span>' for f in s.flags
    )
    corr = (
        f'<span class="corr">Δ {delta:+.1f} pts from correlation</span>'
        if s.delta_is_real else
        '<span class="corr muted-note">no correlation '
        '(legs from separate matches)</span>'
    )
    return f"""
<article class="slip">
  <header>
    <span class="slip-id">SLIP #{s.rank} · {s.n_legs}-PICK</span>
    <span class="ev pos">pays {s.multiplier:g}x</span>
  </header>
  <div class="delta-line">
    <span class="delta">{s.p_correlated:.1%}</span>
    <span class="delta-detail">chance all {s.n_legs} hit &middot;
      EV <b>{s.ev_pct:+.0f}%</b>, <b>{s.ev_adj_pct:+.0f}%</b> after the
      model's optimism haircut<br>{corr}</span>
  </div>
  <div class="legs">{''.join(_leg_row(l) for l in s.legs)}</div>
  <div class="lockrow">
    <button class="lockin" data-legs="{html.escape(s.legs_json, quote=True)}"
            data-book="{s.book}" data-p="{s.p_correlated:.4f}"
            data-product="{s.product}"
            onclick="lockIn(this)"
            title="I placed this — $1 at {s.multiplier:g}x">✓</button>
    <button class="adjust" onclick="toggleAdjust(this)"
            title="different stake, or the app quoted another multiplier">
      adjust</button>
    <span class="adjustables" hidden>
      <input class="stake" type="number" step="0.5" min="0.5" value="1"
             title="stake" oninput="syncLabel(this)"><span class="lbl">$ at</span>
      <input class="mult" type="number" step="0.25" min="1"
             value="{s.multiplier:g}" oninput="syncLabel(this)"
             title="multiplier the app actually quoted">
      <span class="lbl">x</span>
    </span>
  </div>
  {f'<footer>{flags}</footer>' if flags else ''}
</article>"""


def _book_section(b: BookView) -> str:
    slips = "".join(_slip_card(s) for s in b.slips) or (
        '<p class="empty">No qualifying slips on this board right now.</p>'
    )
    board_rows = "".join(
        f"<tr><td><b>{html.escape(l.player)}</b> <span class='mut'>"
        f"{html.escape(l.team)}</span></td>"
        f"<td>{html.escape(l.stat)}</td><td>{html.escape(l.maps)}</td>"
        f"<td class='num'>{l.line:g}</td>"
        f"<td><span class='side {l.side.lower()}'>{l.side}</span></td>"
        f"<td class='num'>{l.p_hit:.0%}</td>"
        f"<td class='mut'>{html.escape(l.context)}</td></tr>"
        for l in sorted(b.legs, key=lambda x: -x.p_hit)
    )
    table = (
        f"""<details class="board"><summary>Board marginals
        ({len(b.legs)} legs)</summary><div class="tblwrap"><table>
        <thead><tr><th>Player</th><th>Stat</th><th>Maps</th><th>Line</th>
        <th>Lean</th><th>P(hit)</th><th>Match</th></tr></thead>
        <tbody>{board_rows}</tbody></table></div></details>"""
        if b.legs else ""
    )
    return f"""
<section class="book {b.book}">
  <div class="book-head">
    <span class="book-badge {b.book}">{html.escape(b.display)}</span>
    <span class="book-meta">{html.escape(b.board_label)} ·
      {html.escape(b.freshness)}</span>
  </div>
  {slips}
  {table}
</section>"""


def _tracked_card(t: TrackedSlip) -> str:
    cls = {"won": "won", "lost": "lost"}.get(t.status, "open")
    legs = ""
    for l in t.legs:
        mark = {"won": "✓", "lost": "✗", "void": "∅"}.get(l.status, "·")
        got = f" → {l.observed:g}" if l.observed is not None else ""
        if l.clv is None:
            clv_html = ""
        else:
            c = "good" if l.clv > 0 else "bad" if l.clv < 0 else "flat"
            clv_html = (f'<span class="clv {c}" title="closed at '
                        f'{l.closing_line:g}">CLV {l.clv:+.1f}</span>')
        legs += (
            f'<div class="tleg {l.status}"><span class="tmark">{mark}</span>'
            f'<span class="side {l.side.lower()}">{l.side.upper()}</span>'
            f'<b>{html.escape(l.player)}</b> {l.line:g} '
            f'{html.escape(l.stat)}{got}{clv_html}</div>'
        )
    payout = (f"returned ${t.payout:.2f}" if t.payout is not None else "pending")
    claim = f"claimed {t.claimed_p:.1%}" if t.claimed_p else ""
    if t.clv is not None:
        c = "good" if t.clv > 0 else "bad" if t.clv < 0 else "flat"
        claim += f' · <span class="clv {c}">slip CLV {t.clv:+.2f}</span>'
    mult = f"{t.multiplier:g}x" if t.multiplier else "mult ?"
    return f"""
<article class="slip tracked {cls}">
  <header>
    <span class="slip-id">{html.escape(t.book)} · {t.n_legs}-pick · {mult}
      · ${t.stake:.2f}</span>
    <span class="tstatus {cls}">{t.status.upper()}</span>
  </header>
  <div class="tmeta">{claim} · {payout}</div>
  <div class="legs">{legs}</div>
</article>"""


def _record_table(data: ReportData) -> str:
    """Per-book record as a table.

    Was a single line running past 300 characters, splicing two books' P&L,
    calibration bands and the pooled leg rate together with middots. Books
    are never pooled — different ladders, different break-evens — so they get
    their own rows, and the leg rate gets its own line because it is the one
    figure that tests the MODEL rather than a product.
    """
    if not data.book_stats:
        return (f'<div class="book-meta">'
                f'{html.escape(data.tracked_summary)}</div>')
    rows = ""
    for b in data.book_stats:
        pnl_cls = "pos" if b.pnl > 0 else ("neg" if b.pnl < 0 else "")
        acc = (
            f"{b.claimed:.0%} &rarr; {b.actual:.0%}"
            if b.claimed is not None and b.actual is not None else "&mdash;"
        )
        pnl = f"{b.pnl:+.2f}" if b.settled else "&mdash;"
        rows += (
            f'<tr><td class="bk">{html.escape(b.book)} '
            f'<span class="sz">{html.escape(b.sizes)}</span></td>'
            f'<td>{b.n_open}</td>'
            f'<td>{b.won}W/{b.lost}L</td>'
            f'<td class="{pnl_cls}">{pnl}</td>'
            f'<td class="muted-note">{acc}</td></tr>'
        )
    won, total = data.leg_record
    legline = (
        f'<div class="legrate">LEG HIT RATE <b>{won}/{total} = '
        f'{won / total:.1%}</b> <span class="muted-note">— pooled on '
        f'purpose: it tests the model, which is the same for both books'
        f'</span></div>' if total else
        '<div class="legrate"><span class="muted-note">no graded legs yet '
        'in the current era — the count restarted when the products and '
        'model fixes landed</span></div>'
    )
    if data.legacy_note:
        legline += (f'<div class="legrate"><span class="muted-note">'
                    f'{html.escape(data.legacy_note)}</span></div>')
    return f"""<table class="record">
  <thead><tr><th>book</th><th>open</th><th>record</th><th>P&amp;L</th>
    <th>claimed &rarr; actual</th></tr></thead>
  <tbody>{rows}</tbody>
</table>{legline}"""


def _tracked_section(data: ReportData) -> str:
    """Placed slips, collapsible and split by status.

    Once a few dozen bets accumulate this section would otherwise bury the
    suggestions below it, so each group is a <details>: open bets expanded
    (they still matter), settled history collapsed (it does not).
    """
    if not data.tracked:
        return "<!--TRACKED_START--><!--TRACKED_END-->"
    open_ = [t for t in data.tracked if t.status == "pending"]
    done = [t for t in data.tracked if t.status != "pending"]

    def group(title: str, items: list[TrackedSlip], is_open: bool) -> str:
        if not items:
            return ""
        attr = " open" if is_open else ""
        cards = "".join(_tracked_card(t) for t in items)
        return (f'<details class="tgroup"{attr}><summary>{title} '
                f'({len(items)})</summary>{cards}</details>')

    # Open bets are split PER BOOK. The books run different ladders, different
    # break-evens and different payout-shift rules, so a pooled list invites
    # comparing bets that are not comparable — the same mistake that made a
    # PrizePicks 4-pick POWER record (0W/12L, 56.23% bar) look like evidence
    # against the BOOK rather than against the product. Settled history stays
    # pooled and collapsed; the per-book split there already lives in the
    # summary line above.
    books = sorted({t.book for t in open_})
    open_groups = "".join(
        group(f"Open · {bk}", [t for t in open_ if t.book == bk],
              len(open_) <= 8)
        for bk in books
    )
    return f"""<!--TRACKED_START-->
<section class="book tracked-book">
  <div class="book-head">
    <span class="book-badge tracked">My slips</span>
  </div>
  {_record_table(data)}
  {open_groups}
  {group("Settled", done, False)}
</section><!--TRACKED_END-->"""


def render(data: ReportData) -> str:
    banner = (
        '<div class="banner">MOCK DATA — layout preview only. No number on '
        "this page is real; the model has not passed its calibration gate."
        "</div>"
        if data.is_mock
        else ""
    )
    sections = "".join(_book_section(b) for b in data.books)
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cs2props — slip scanner</title>
<style>
:root{{--bg:#0c0f15;--panel:#141926;--panel2:#1a2133;--line:#232c40;
--ink:#d6dcea;--muted:#7e8aa3;--faint:#57627a;--accent:#39e2c8;
--violet:#8b7bff;--good:#41d183;--warn:#ffb64a;--bad:#ff5673;
--pp:#b39dff;--ud:#f2c14e;
--mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace}}
@media(prefers-color-scheme:light){{:root{{--bg:#edf0f6;--panel:#fff;
--panel2:#f3f5fa;--line:#dce1ec;--ink:#1b2334;--muted:#5b6680;
--faint:#8b94a9;--accent:#0da596;--violet:#6252e6;--pp:#6a4fd8;--ud:#a97d10}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);line-height:1.5;
font-family:system-ui,-apple-system,'Segoe UI',sans-serif}}
.wrap{{max-width:1060px;margin:0 auto;padding:26px 18px 70px}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.22em;
text-transform:uppercase;color:var(--accent)}}
h1{{font-size:clamp(22px,4vw,32px);letter-spacing:-.02em;margin:4px 0 2px}}
.sub{{color:var(--muted);font-size:13px}}
.cal{{font-family:var(--mono);font-size:11px;margin-top:6px;color:var(--warn)}}
.banner{{background:color-mix(in srgb,var(--warn) 14%,transparent);
border:1px solid color-mix(in srgb,var(--warn) 45%,transparent);
color:var(--warn);border-radius:10px;padding:10px 14px;margin:18px 0 0;
font-family:var(--mono);font-size:12px;letter-spacing:.04em}}
.book{{margin-top:34px}}
.book-head{{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
margin-bottom:14px;border-bottom:1px solid var(--line);padding-bottom:10px}}
.book-badge{{font-family:var(--mono);font-size:13px;font-weight:800;
letter-spacing:.12em;text-transform:uppercase;padding:4px 12px;
border-radius:7px}}
.book-badge.prizepicks{{background:color-mix(in srgb,var(--pp) 16%,transparent);
color:var(--pp);border:1px solid color-mix(in srgb,var(--pp) 40%,transparent)}}
.book-badge.underdog{{background:color-mix(in srgb,var(--ud) 14%,transparent);
color:var(--ud);border:1px solid color-mix(in srgb,var(--ud) 40%,transparent)}}
.book-meta{{font-size:12px;color:var(--muted)}}
.slip{{background:var(--panel);border:1px solid var(--line);
border-radius:14px;padding:16px 18px;margin-bottom:14px}}
.book.prizepicks .slip{{border-left:3px solid
color-mix(in srgb,var(--pp) 55%,transparent)}}
.book.underdog .slip{{border-left:3px solid
color-mix(in srgb,var(--ud) 55%,transparent)}}
.slip header{{display:flex;justify-content:space-between;align-items:baseline;
flex-wrap:wrap;gap:8px}}
.slip-id{{font-family:var(--mono);font-size:12px;letter-spacing:.1em;
color:var(--muted)}}
.ev{{font-family:var(--mono);font-weight:700;font-size:15px}}
.ev.pos{{color:var(--good)}} .ev.neg{{color:var(--bad)}}
.delta-line{{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
margin:10px 0 12px;padding:10px 12px;background:var(--panel2);
border-radius:9px}}
.delta{{font-family:var(--mono);font-size:22px;font-weight:800;
color:var(--accent);font-variant-numeric:tabular-nums}}
.delta-detail{{font-size:13px;color:var(--muted)}}
.adjust{{background:none;border:none;color:var(--muted);font-size:11px;
  text-decoration:underline;cursor:pointer;opacity:.6}}
.adjust:hover{{opacity:1}}
.adjustables[hidden]{{display:none}}
.corr{{font-size:12px;color:var(--muted)}}
.record{{border-collapse:collapse;margin:6px 0 4px;font-family:var(--mono);
  font-size:12px}}
.record th{{text-align:left;font-weight:400;color:var(--faint);
  font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  padding:2px 18px 4px 0}}
.record td{{padding:3px 18px 3px 0;color:var(--ink);white-space:nowrap}}
.record td.bk{{color:var(--muted)}}
.record .sz{{color:var(--faint);font-size:10px;margin-left:8px}}
.record td.pos{{color:var(--good)}}
.record td.neg{{color:var(--bad)}}
.legrate{{font-family:var(--mono);font-size:12px;color:var(--muted);
  margin-top:6px}}
#rescan,#grade{{margin-left:14px;font-family:var(--mono);font-size:11px;
  letter-spacing:.05em;color:var(--accent);background:transparent;
  border:1px solid color-mix(in srgb,var(--accent) 40%,transparent);
  border-radius:16px;padding:4px 12px;cursor:pointer;vertical-align:1px}}
#rescan:hover,#grade:hover{{background:color-mix(in srgb,var(--accent) 12%,
  transparent)}}
#rescan:disabled,#grade:disabled{{color:var(--faint);border-color:var(--line);
  cursor:default}}
.muted-note{{opacity:.65;font-style:italic}}
.delta-detail b{{color:var(--ink)}}
.leg{{display:grid;grid-template-columns:58px 92px 74px 1fr 46px auto;
gap:8px;align-items:baseline;padding:5px 0;font-size:13px}}
@media(max-width:640px){{.leg{{grid-template-columns:52px 1fr 44px}}
.lteam,.lstat,.lctx{{display:none}}}}
.side{{font-family:var(--mono);font-size:10px;font-weight:700;
padding:2px 0;border-radius:5px;text-align:center}}
.side.over{{background:color-mix(in srgb,var(--accent) 15%,transparent);
color:var(--accent)}}
.side.under{{background:color-mix(in srgb,var(--violet) 18%,transparent);
color:var(--violet)}}
.lname{{font-weight:650}} .lteam{{color:var(--muted);font-size:12px}}
.lstat{{color:var(--ink)}}
.lp{{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}}
.lctx{{color:var(--faint);font-size:11px}}
.flag{{display:inline-block;font-family:var(--mono);font-size:10px;
letter-spacing:.05em;color:var(--warn);border:1px solid
color-mix(in srgb,var(--warn) 40%,transparent);border-radius:20px;
padding:2px 9px;margin-top:10px}}
.empty{{color:var(--faint);font-size:13px}}
details.board{{margin-top:4px}}
details.tgroup{{margin-bottom:10px}}
details.tgroup summary{{cursor:pointer;font-family:var(--mono);font-size:11px;
letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
padding:8px 0;user-select:none}}
details.tgroup[open] summary{{color:var(--good)}}
details.board summary{{cursor:pointer;font-family:var(--mono);font-size:11px;
letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
padding:8px 0}}
details.board[open] summary{{color:var(--accent)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
.tblwrap{{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
border-radius:14px;padding:6px 14px 10px}}
th{{font-family:var(--mono);font-size:10px;letter-spacing:.08em;
text-transform:uppercase;color:var(--faint);text-align:left;
padding:10px 8px 6px;border-bottom:1px solid var(--line)}}
td{{padding:7px 8px;border-bottom:1px solid var(--line)}}
tr:last-child td{{border-bottom:none}}
td.num{{font-family:var(--mono);font-variant-numeric:tabular-nums;
text-align:right}}
.mut{{color:var(--faint);font-size:11px}}
.lockrow{{display:flex;align-items:center;gap:6px;margin-top:12px;
flex-wrap:wrap}}
.lockrow input{{width:62px;background:var(--panel2);border:1px solid var(--line);
color:var(--ink);border-radius:6px;padding:5px 8px;font-family:var(--mono);
font-size:12px}}
.lockrow .lbl{{color:var(--faint);font-size:11px;font-family:var(--mono)}}
.slip.placed{{opacity:.35}}
.slip.superseded{{opacity:.4}}
.slip.superseded .lockin{{color:var(--faint);border-color:var(--line);
  cursor:not-allowed}}
.lockin{{margin-top:0;font-family:var(--mono);font-size:15px;line-height:1;
color:var(--accent);background:transparent;
border:1px solid color-mix(in srgb,var(--accent) 40%,transparent);
border-radius:50%;width:32px;height:32px;padding:0;cursor:pointer;
display:inline-flex;align-items:center;justify-content:center}}
/* saving/tracked states carry words, so the pill has to be able to grow */
.lockin.wide{{border-radius:20px;width:auto;height:auto;padding:6px 14px;
font-size:11px;letter-spacing:.06em}}
.lockin:hover{{background:color-mix(in srgb,var(--accent) 12%,transparent)}}
.lockin.done{{color:var(--good);border-color:var(--good)}}
.book-badge.tracked{{background:color-mix(in srgb,var(--good) 14%,transparent);
color:var(--good);border:1px solid color-mix(in srgb,var(--good) 40%,transparent)}}
.slip.tracked{{border-left:3px solid var(--muted)}}
.slip.tracked.won{{border-left-color:var(--good)}}
.slip.tracked.lost{{border-left-color:var(--bad)}}
.tstatus{{font-family:var(--mono);font-size:11px;font-weight:700;
padding:2px 9px;border-radius:20px;background:var(--panel2);color:var(--muted)}}
.tstatus.won{{color:var(--good)}} .tstatus.lost{{color:var(--bad)}}
.tmeta{{font-size:12px;color:var(--muted);margin:6px 0 8px}}
.tleg{{font-size:13px;padding:3px 0;display:flex;gap:8px;align-items:baseline}}
.tleg.lost{{color:var(--bad)}} .tleg.won{{color:var(--good)}}
.tleg.void{{color:var(--faint)}}
.tmark{{font-family:var(--mono);width:14px}}
.clv{{font-family:var(--mono);font-size:10px;font-weight:700;padding:1px 7px;
border-radius:20px;margin-left:6px}}
.clv.good{{background:color-mix(in srgb,var(--good) 16%,transparent);
color:var(--good)}}
.clv.bad{{background:color-mix(in srgb,var(--bad) 16%,transparent);
color:var(--bad)}}
.clv.flat{{background:var(--panel2);color:var(--faint)}}
.foot{{margin-top:26px;font-size:11px;color:var(--faint);
border-top:1px solid var(--line);padding-top:14px;line-height:1.6}}
</style>
<div class="wrap">
  <div class="eyebrow">cs2props · CS2 player props</div>
  <h1>Slip Scanner</h1>
  <p class="sub">generated {data.generated}
    <button id="rescan" onclick="runJob(this, '/api/rescan', 'scan')"
      title="fetch fresh boards from both books and rebuild the slips —
takes about 90 seconds (the PrizePicks client waits 60s between its two
requests on purpose)">&#8635; refresh board</button>
    <button id="grade" onclick="runJob(this, '/api/grade', 'grade')"
      title="pull the latest finished matches from bo3.gg, then grade every
open slip against them — can take a few minutes of polite paging">&#10003;
grade slips</button></p>
  <p class="cal">model: {html.escape(data.calibration_label)}</p>
  {banner}
  {_tracked_section(data)}
  {sections}
  <p class="foot">Analysis tool only — no bets are placed by this software.
  P(all N) comes from joint Monte Carlo simulation, not multiplied marginals;
  Δ is the gap between them. Slips are per-book because payout tables differ.
  PrizePicks freshness depends on your last pp.json import. A +EV slip still
  loses most of the time at high multipliers. Verify every line and payout in
  the app before entering. If it stops being fun: 1-800-522-4700.</p>
</div>
<script>
// The stake and multiplier are hidden by default: with the ladder verified,
// the same-match pair rule enforced and per-side shades priced in, the
// payout is KNOWN, and every slip this app has recommended was $1. The
// inputs still exist because the multiplier CAN differ — a leg shaded after
// the scan, or a PrizePicks Arena entry, which prices per-pick and cannot be
// predicted at all. Revealing them on demand keeps that without making the
// common case a three-field form.
// The header buttons start a job server-side and poll until it finishes,
// then reload. Both jobs are slow on purpose — the scan sleeps 60s between
// PrizePicks requests, grading pages bo3.gg politely — so the button counts
// up rather than leaving a dead spinner.
function runJob(btn, endpoint, job) {{
  var original = btn.textContent;
  btn.disabled = true;
  var t0 = Date.now();
  var verb = job === "grade" ? "grading" : "scanning";
  var tick = setInterval(function () {{
    btn.textContent = verb + "\u2026 " +
      Math.round((Date.now() - t0) / 1000) + "s";
  }}, 1000);
  fetch(endpoint, {{method: "POST"}}).then(function () {{
    var poll = setInterval(function () {{
      fetch("/api/scan-status?job=" + job)
        .then(function (r) {{ return r.json(); }})
        .then(function (d) {{
          if (d.state === "done") {{
            clearInterval(poll); clearInterval(tick);
            location.reload();
          }} else if (d.state && d.state.indexOf("failed") === 0) {{
            clearInterval(poll); clearInterval(tick);
            btn.disabled = false;
            btn.textContent = job + " failed \u2014 retry";
          }}
        }});
    }}, 3000);
  }}).catch(function () {{
    clearInterval(tick);
    btn.disabled = false;
    btn.textContent = original;
  }});
}}

function toggleAdjust(btn) {{
  var box = btn.parentNode.querySelector(".adjustables");
  box.hidden = !box.hidden;
  btn.textContent = box.hidden ? "adjust" : "done";
}}

function syncLabel(input) {{
  // The button stays a checkmark; what it will record lives in the tooltip
  // and in the adjust fields, which are open whenever this fires.
  var row = input.closest(".lockrow");
  var stake = parseFloat(row.querySelector(".stake").value) || 1;
  var mult = parseFloat(row.querySelector(".mult").value) || 0;
  row.querySelector(".lockin").title =
    "I placed this \u2014 $" + stake + " at " + mult + "x";
}}

function playerOf(spec) {{
  // leg specs look like "MahaR over 13.5 kills 1-2"
  return String(spec).trim().split(/\s+/)[0].toLowerCase();
}}

function retireSlipsUsing(placedLegs, placedCard) {{
  var taken = {{}};
  placedLegs.forEach(function (l) {{ taken[playerOf(l)] = true; }});
  document.querySelectorAll(".slip").forEach(function (card) {{
    if (card === placedCard || card.classList.contains("placed")) return;
    var btn = card.querySelector(".lockin");
    if (!btn) return;
    var legs = JSON.parse(btn.dataset.legs || "[]");
    var clash = legs.filter(function (l) {{ return taken[playerOf(l)]; }});
    if (!clash.length) return;
    card.classList.add("superseded");
    btn.disabled = true;
    btn.classList.add("wide");
    btn.textContent = "shares " + playerOf(clash[0]) + " \u2014 stale";
    btn.title = "A slip you just placed already uses this player. Two slips "
              + "sharing a leg are not two bets: same model error, they fail "
              + "together. Rescan for a fresh board.";
  }});
}}

function lockIn(btn) {{
  var row = btn.parentElement, card = btn.closest(".slip");
  var body = {{
    book: btn.dataset.book,
    product: btn.dataset.product || "power",
    claimed_p: parseFloat(btn.dataset.p),
    legs: JSON.parse(btn.dataset.legs),
    stake: parseFloat(row.querySelector(".stake").value) || 1,
    mult: parseFloat(row.querySelector(".mult").value) || null
  }};
  btn.disabled = true;
  btn.classList.add("wide"); btn.textContent = "saving\u2026";
  fetch("/api/track", {{
    method: "POST", headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify(body)
  }}).then(function (r) {{ return r.json(); }}).then(function (d) {{
    if (d.ok) {{
      btn.textContent = "\u2713 tracked \u2014 in My Slips";
      btn.classList.add("done");
      card.classList.add("placed");
      // Retire every OTHER card that reuses a player from this slip. The
      // server-side exclusion only applies to the NEXT scan, so a page that
      // has been sitting open will happily sell the same leg twice — it did
      // on 2026-07-26, when MahaR 13.5 went into two slips clicked from one
      // render. Two slips sharing a leg are not two bets: they carry the
      // same model error, fail together, and put a duplicate into the
      // calibration sample.
      retireSlipsUsing(JSON.parse(btn.dataset.legs), card);
      // reload so the bet appears in My Slips (the server re-renders that
      // section live) and the card is gone from the board
      setTimeout(function () {{ location.reload(); }}, 700);
    }} else {{
      btn.textContent = d.error === "already tracked"
        ? "already in My Slips" : "failed: " + d.error;
      btn.disabled = false;
    }}
  }}).catch(function (e) {{
    btn.textContent = "server not running \u2014 use cs2props serve";
    btn.disabled = false;
  }});
}}
</script>"""


def write_report(data: ReportData, path: str) -> None:
    from pathlib import Path

    Path(path).write_text(render(data))
