"""CLI entry point.

    cs2props board --fixture tests/fixtures/prizepicks_projections.json
    cs2props board --live
    cs2props scan          (modules 2-4; not built yet)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from cs2props import db
from cs2props.ingest.prizepicks import (
    CloudflareBlocked,
    LeagueNotFound,
    Prop,
    PrizePicksClient,
    load_fixture,
    parse_projections,
)

log = logging.getLogger("cs2props")


def _print_table(props: list[Prop]) -> None:
    if not props:
        print("No props parsed.")
        return
    hdr = (
        f"{'PLAYER':<16}{'TEAM':<10}{'OPP':<10}{'STAT':<12}{'MAPS':<8}"
        f"{'LINE':>7}  {'BOARD':<9}{'START':<22}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(props, key=lambda x: (x.start_time or "", x.player_name)):
        maps = f"{p.map_range[0]}-{p.map_range[1]}" if p.map_range else "series"
        print(
            f"{p.player_name:<16}{(p.team or '—'):<10}{(p.opponent or '—'):<10}"
            f"{p.stat_kind:<12}{maps:<8}{p.line_score:>7.1f}  "
            f"{p.board:<9}{(p.start_time or '—'):<22}"
        )
    print(f"\n{len(props)} props")


def cmd_board(args: argparse.Namespace) -> int:
    if args.fixture:
        props = load_fixture(Path(args.fixture))
    elif args.source == "underdog":
        from cs2props.ingest.underdog import UnderdogClient

        props = UnderdogClient(cache_dir=Path(args.cache_dir) / "underdog").fetch_board()
    else:
        client = PrizePicksClient(cache_dir=Path(args.cache_dir) / "prizepicks")
        try:
            props = client.fetch_board()
        except CloudflareBlocked as e:
            log.error("%s", e)
            return 2
        except LeagueNotFound as e:
            log.error("%s", e)
            return 3
    _print_table(props)
    if args.db:
        conn = db.connect(Path(args.db))
        db.save_props(conn, props)
        conn.close()
    return 0


def cmd_import_board(args: argparse.Namespace) -> int:
    """Ingest a browser-saved /projections payload into the client cache."""
    import json
    import time

    src = Path(args.payload)
    payload = json.loads(src.read_text())
    props = parse_projections(payload)
    if not props:
        log.error("no props parsed from %s — is this a /projections payload?", src)
        return 1
    league_id = props[0].league_id or "unknown"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"projections_{league_id}.json").write_text(
        json.dumps({"fetched_at": time.time(), "payload": payload})
    )
    log.info("imported %d props into cache (league %s)", len(props), league_id)
    _print_table(props)
    if args.db:
        conn = db.connect(Path(args.db))
        db.save_props(conn, props)
        conn.close()
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Walk bo3.gg finished matches and fill player_maps. Resumable."""
    from cs2props.ingest.bo3gg import Bo3Client, since_months_ago, stats_to_rows

    tiers = frozenset(t.strip().lower() for t in args.tiers.split(","))
    if getattr(args, "days", None):
        # Grading only needs matches as old as the oldest open slip — a few
        # days. Walking a whole month of finished-match pages at the polite
        # 2s-per-page pace made every grade-button press ~10x slower than
        # the data it was actually after.
        from datetime import datetime, timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(days=args.days)
    else:
        since = since_months_ago(args.months)
    client = Bo3Client(delay_s=args.delay)
    conn = db.connect(Path(args.db))
    log.info(
        "backfill: tiers=%s since=%s delay=%.1fs db=%s",
        sorted(tiers), since.date(), args.delay, args.db,
    )
    n_new = n_skipped = 0
    try:
        for match in client.iter_finished_matches(since, tiers):
            mid = str(match["id"])
            if db.is_match_ingested(conn, mid):
                n_skipped += 1
                continue
            games = client.fetch_games(mid)
            rows = []
            for game in games:
                stats = client.fetch_players_stats(game["id"])
                rows.extend(stats_to_rows(match, game, stats))
            db.save_match_maps(
                conn, mid, rows, match.get("tier"),
                match.get("start_date"), len(games),
            )
            n_new += 1
            if n_new % 25 == 0:
                m, r, p = db.history_summary(conn)
                log.info(
                    "progress: +%d matches this run (%d skipped) | db: "
                    "%d matches, %d map-rows, %d players",
                    n_new, n_skipped, m, r, p,
                )
            if args.limit and n_new >= args.limit:
                log.info("hit --limit %d, stopping", args.limit)
                break
    except KeyboardInterrupt:
        log.warning("interrupted — backfill is resumable, just rerun")
    finally:
        m, r, p = db.history_summary(conn)
        log.info(
            "done: +%d new, %d already ingested | db totals: %d matches, "
            "%d map-rows, %d distinct players", n_new, n_skipped, m, r, p,
        )
        conn.close()
    return 0


def cmd_crossbook(args: argparse.Namespace) -> int:
    """Track where the two books disagree, and whether the gap pays.

    This is the only edge hypothesis left that does not require beating a
    book's own number: when two books post different lines, one is wrong
    regardless of what any model thinks.
    """
    from cs2props.crossbook import format_report, run

    conn = db.connect(Path(args.db))
    print(format_report(run(conn)))
    return 0


def cmd_reallines(args: argparse.Namespace) -> int:
    """Score archived PrizePicks lines whose matches have settled.

    The synthetic backtest asks "is the distribution honestly shaped?". This
    asks the only question that pays: "does the model beat the book's own
    number?" They are not the same test and the first one passing says
    nothing about the second.
    """
    from cs2props.model.reallines import format_report, run_real_line_backtest

    conn = db.connect(Path(args.db))
    res = run_real_line_backtest(conn, min_history=args.min_history)
    print(format_report(res))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Module 2 gate: walk-forward backtest + calibration report."""
    from cs2props.model.backtest import format_report, run_backtest

    conn = db.connect(Path(args.db))
    m, r, p = db.history_summary(conn)
    print(f"history: {m} matches, {r} map-rows, {p} players\n")
    cal = run_backtest(conn, min_history=args.min_history)
    print(format_report(cal))
    if cal.preds:
        import json
        import time

        Path("calibration.json").write_text(json.dumps({
            "ts": time.time(),
            "when": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
            "log_loss": round(cal.log_loss(), 4),
            "baseline": round(cal.baseline_log_loss(), 4),
            "n_series": cal.n_series,
            "n_lines": len(cal.preds),
        }))
        print("\nwrote calibration.json (scan provenance stamp)")
    conn.close()
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Board x history x correlation engine — per-leg marginals only.

    Pre-calibration: shows what the model thinks, not what to bet. Slip
    ranking stays locked behind the module-2 calibration gate.
    """
    from cs2props.ingest.underdog import UnderdogClient
    from cs2props.model.state_builder import build_history
    from cs2props.pipeline import simulate_board

    conn = db.connect(Path(args.db))
    m, r, p = db.history_summary(conn)
    print(f"history: {m} matches, {r} map-rows, {p} players")
    history = build_history(conn)
    conn.close()
    props = UnderdogClient(cache_dir=Path(args.cache_dir) / "underdog").fetch_board()
    sims = simulate_board(props, history, n_iters=args.iters)
    if not sims:
        print("\nNo board matches could be joined to history yet — the "
              "backfill may not cover these teams. Try again as it grows.")
        return 0
    print(f"\n{len(sims)} matches simulated — PREVIEW ONLY, model not yet "
          "calibrated:\n")
    for sim in sims:
        print(f"── {sim.label}  (P(A map win) {sim.p_a_map:.0%}, "
              f"{sim.matched} players matched"
              + (f", unmatched: {', '.join(sim.unmatched)}" if sim.unmatched
                 else "") + ")")
        rows = sorted(
            zip(sim.props, sim.result.p_over), key=lambda t: -abs(t[1] - 0.5)
        )
        for prop, p_over in rows:
            lean = "OVER " if p_over >= 0.5 else "UNDER"
            conf = max(p_over, 1 - p_over)
            maps = (f"{prop.map_range[0]}-{prop.map_range[1]}"
                    if prop.map_range else "series")
            print(f"   {prop.player_name:<16}{prop.stat_kind:<11}"
                  f"maps {maps:<7}{prop.line_score:>5.1f}  -> {lean} {conf:.0%}")
        print()
    return 0


def cmd_ev(args: argparse.Namespace) -> int:
    """Re-price a slip against the multiplier the APP actually shows.

    The scanner filters on an ESTIMATED multiplier, so a slip that cleared
    the EV floor at an assumed 10x can be marginal at the real 7.25x. This
    re-runs the same test the scanner ran, using the real price, and says
    plainly when the slip no longer clears the bar it was surfaced under.
    """
    from cs2props.adaptive import estimate_haircut

    p, mult, n = args.p, args.mult, args.legs
    conn = db.connect(Path(args.db))
    hc = estimate_haircut(conn)
    conn.close()

    ev = mult * p - 1.0
    # same haircut the optimizer applies: shave each leg, recompute
    per_leg = p ** (1.0 / max(n, 1))
    shrink = (max(per_leg - hc.haircut, 0.01) / per_leg) ** n
    adj_ev = mult * p * shrink - 1.0

    verdict = "TAKE" if adj_ev >= args.min_ev else (
        "MARGINAL" if adj_ev > 0 else "PASS")
    print(f"P(win) {p:.1%} at {mult:g}x  ->  EV {ev:+.1%}")
    print(f"  after {hc.haircut:.1%}/leg haircut : {adj_ev:+.1%}   [{verdict}]")
    print(f"  break-even multiplier   : {1.0 / p:.2f}x"
          f"   (you have {mult:g}x)")
    if adj_ev < args.min_ev:
        print(f"\n  ⚠ this does NOT clear your {args.min_ev:.0%} floor at "
              f"{mult:g}x.")
        need = (1 + args.min_ev) / (p * shrink) if p * shrink > 0 else 0
        print(f"    it would need {need:.2f}x to qualify. The scanner "
              "surfaced it on an ESTIMATED multiplier; the app's real price "
              "is lower, so the edge it was chosen for is not there.")
    else:
        ratio = (1.0 / (mult * p)) ** (1.0 / max(n, 1))
        print(f"  each of {n} legs tolerates a ~{(1 - ratio) * 100:.0f}% "
              "relative error before break-even")
    print(f"\nhaircut basis: {hc.source}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Capture board lines shortly before YOUR open bets start.

    CLV is measured against the last line seen before kickoff. Snapshots
    taken ~17h early are useless — 50 of the first 54 legs came back "tied"
    simply because lines had not moved yet. This runs cheaply on a short
    interval and only fetches when an open slip's match is imminent.
    """
    import json
    import time as _time
    from datetime import datetime, timezone

    from cs2props.ingest.underdog import UnderdogClient
    from cs2props.model.state_builder import clean_name as _cn

    conn = db.connect(Path(args.db))
    held = {
        _cn(r[0]) for r in conn.execute(
            "SELECT l.player_name FROM slip_legs l JOIN slips s"
            " ON s.slip_id=l.slip_id WHERE s.status='pending'"
        )
    }
    if not held:
        print("no open legs — nothing to snapshot")
        conn.close()
        return 0
    starts: list[float] = []
    for name, st in conn.execute(
        "SELECT player_name, start_time FROM props WHERE start_time IS NOT NULL"
    ):
        if _cn(name) not in held:
            continue
        try:
            t = datetime.fromisoformat(st.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if t > _time.time():
            starts.append(t)
    conn.close()
    if not starts:
        print("no upcoming start times found for open legs")
        return 0
    mins = (min(starts) - _time.time()) / 60
    if mins > args.window:
        print(f"next open-slip match in {mins:.0f} min "
              f"(> {args.window} min window) — skipping")
        return 0

    props = UnderdogClient(
        cache_dir=Path(args.cache_dir) / "underdog", cache_ttl_s=0
    ).fetch_board()
    pp_dir = Path(args.cache_dir) / "prizepicks"
    pp_files = sorted(pp_dir.glob("projections_*.json"),
                      key=lambda p: -p.stat().st_mtime)
    if pp_files:
        w = json.loads(pp_files[0].read_text())
        props += parse_projections(w["payload"])
    conn = db.connect(Path(args.db))
    db.save_props(conn, props)
    conn.close()
    print(f"snapshotted {len(props)} props — next match in {mins:.0f} min")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the local dashboard with one-click slip tracking."""
    from cs2props.server import serve

    serve(Path.cwd(), Path(args.db), port=args.port)
    return 0


def cmd_clv(args: argparse.Namespace) -> int:
    """Closing-line value across tracked legs — the fastest read on edge."""
    from cs2props.clv import format_report, leg_clvs

    conn = db.connect(Path(args.db))
    print(format_report(leg_clvs(conn)))
    conn.close()
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    """Record a slip you actually placed. Legs like 'donk over 32.5 kills 1-2'."""
    from cs2props.tracker import parse_leg, track_slip

    legs = [parse_leg(t) for t in args.leg]
    conn = db.connect(Path(args.db))
    slip_id = track_slip(
        conn, args.book, args.stake, legs, claimed_p=args.claimed_p,
        multiplier=args.mult,
    )
    print(f"tracked slip {slip_id} ({args.book}, ${args.stake:.2f}, "
          f"{len(legs)} legs) — grade with: cs2props grade")
    conn.close()
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    """Grade open slips against ingested results; print the running summary."""
    from cs2props.tracker import grade_open_slips, summary

    conn = db.connect(Path(args.db))
    settled = grade_open_slips(conn)
    print(f"settled {settled} slip(s) this run\n")
    print(summary(conn))
    conn.close()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from cs2props.report import mock_data, write_report

    write_report(mock_data(), args.out)
    print(f"wrote {args.out} (mock)")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    if args.mock:
        _print_mock_scan()
        return 0
    import json
    import time

    from cs2props.config import load_payouts, load_restrictions
    from cs2props.ingest.underdog import UnderdogClient
    from cs2props.model.state_builder import build_history
    from cs2props.optimizer.search import Slip
    from cs2props.pipeline import simulate_board
    from cs2props.report import (
        BookView, LegView, ReportData, SlipView, write_report,
    )

    # calibration provenance — scan refuses to run blind
    cal_path = Path("calibration.json")
    if not cal_path.exists():
        log.error("no calibration.json — run `cs2props calibrate` first")
        return 2
    cal = json.loads(cal_path.read_text())
    cal_label = (f"calibrated {cal['when']} · log loss {cal['log_loss']} "
                 f"(baseline {cal['baseline']}) · {cal['n_series']} series")

    from cs2props.adaptive import estimate_haircut

    conn = db.connect(Path(args.db))
    hc = estimate_haircut(conn)
    history = build_history(conn)

    boards: list[tuple[str, str, list[Prop], str]] = []
    feed_only: list[list[Prop]] = []
    try:
        _ud = UnderdogClient(
            cache_dir=Path(args.cache_dir) / "underdog"
        ).fetch_board()
        # bet_enabled is the switch; a book that is off still feeds the
        # cross-book comparison, so the fetch happens either way.
        from cs2props.config import load_restrictions as _lr

        if _lr("underdog").bet_enabled:
            boards.append(("underdog", "Underdog", _ud, "fetched live"))
        else:
            feed_only.append(_ud)
    except Exception as exc:  # a reference feed must never break a scan
        log.warning("underdog feed unavailable (%s)", exc)
    # PrizePicks: fetch live from the partner host, which is not behind the
    # Cloudflare gate that forced two days of hand-saving pp.json. The disk
    # cache is now a FALLBACK for when the fetch fails, not the primary path
    # — it used to be the only path, so a scan silently reported whatever a
    # previous manual import had left behind.
    pp_dir = Path(args.cache_dir) / "prizepicks"
    pp_props: list[Prop] | None = None
    pp_label = ""
    try:
        pp_props = PrizePicksClient(cache_dir=pp_dir).fetch_board()
        pp_label = "fetched live"
    except (CloudflareBlocked, LeagueNotFound, OSError) as exc:
        log.warning("PrizePicks live fetch failed (%s) — trying cache", exc)
    if pp_props is None:
        pp_files = sorted(pp_dir.glob("projections_*.json"),
                          key=lambda p: -p.stat().st_mtime)
        if pp_files:
            wrapper = json.loads(pp_files[0].read_text())
            age_h = (time.time() - wrapper["fetched_at"]) / 3600
            if age_h <= 12:
                from cs2props.ingest.prizepicks import parse_projections

                pp_props = parse_projections(wrapper["payload"])
                pp_label = f"CACHED {age_h:.1f}h ago — live fetch failed"
            else:
                print(f"note: PrizePicks live fetch failed and the cache is "
                      f"{age_h:.0f}h old — skipped.")
    if pp_props is not None:
        boards.append(("prizepicks", "PrizePicks", pp_props, pp_label))

    def leg_view(l: object) -> LegView:
        from cs2props.optimizer.search import Leg as _L

        assert isinstance(l, _L)
        mr = l.prop.map_range or (1, 2)
        return LegView(
            side=l.side.upper(), player=l.prop.player_name,
            team=l.prop.team or "?", stat=l.prop.stat_kind,
            maps=f"{mr[0]}-{mr[1]}", line=l.prop.line_score, p_hit=l.p,
            context=f"vs {l.prop.opponent or '?'}",
        )

    # cross-book index: 65% of shared props carry different lines, and the
    # better number is worth ~7 pts of win probability per leg — more than
    # any modelling refinement here. Never bet a side at the worse price.
    from cs2props.lineshop import build_index, prop_key
    shop_index = build_index({b[0]: b[2] for b in boards})

    book_views: list[BookView] = []
    print(f"\ncs2props scan — model: {cal_label}")
    print(f"  leg haircut {hc.haircut:.1%} ({hc.source}) · "
          f"min EV after haircut {args.min_ev:.0%}\n")
    from cs2props.roster import RosterIndex, fetch_rosters

    rosters: RosterIndex | None
    try:
        rosters = fetch_rosters(Path(args.cache_dir) / "roster")
        print(f"announced lineups: {len(rosters.players)} players")
    except Exception as e:  # roster check must never block a scan
        log.warning("roster verification unavailable: %s", e)
        rosters = None

    # hide suggestions the user has already placed — the guard against
    # entering the same bet twice
    from cs2props.server import placed_signatures, slip_signature
    from cs2props.tracker import parse_leg as _pl

    already = placed_signatures(Path(args.db))

    # Players and matches already committed in OPEN slips are removed from
    # the board entirely. Reusing a player across slips shares model error
    # and concentrates exposure without adding information; reusing a MATCH
    # does the same one level up, because every leg in a game shares its
    # round count.
    from cs2props.model.state_builder import clean_name as _cn
    from cs2props.tracker import committed_players

    _tc = db.connect(Path(args.db))
    held = committed_players(_tc)
    _tc.close()

    for book, display, props, freshness in boards:
        if held:
            held_matches = {
                (p.team, p.opponent, p.start_time) for p in props
                if _cn(p.player_name) in held
            }
            before = len(props)
            props = [
                p for p in props
                if _cn(p.player_name) not in held
                and (p.team, p.opponent, p.start_time) not in held_matches
            ]
            if before != len(props):
                print(f"    (excluded {before - len(props)} props from "
                      f"{len(held_matches)} match(es) you already have open)")
        sims = simulate_board(props, history, n_iters=args.iters,
                              rosters=rosters, conn=conn)
        from cs2props.optimizer.search import slip_price_factor

        shape_name = getattr(args, "shape", None)
        product = getattr(args, "product", None)
        size = getattr(args, "size", None)
        slips, reason = search_slips_for(book, sims, args.min_ev, hc.haircut,
                                         shape_name, product, size)
        n_props = sum(len(s.props) for s in sims)
        print(f"=== {display}: {len(props)} props, {len(sims)} matches "
              f"simulated, {n_props} legs modeled ({freshness})")
        for sim in sims:
            for note in sim.standins:
                print(f"    ⚠ {sim.label} — {note}")
        slip_views: list[SlipView] = []
        if reason:
            print(f"    {reason}\n")
        def _leg_specs(sl: Any) -> list[str]:
            out = []
            for l in sl.legs:
                mr = l.prop.map_range or (1, 2)
                out.append(f"{l.prop.player_name} {l.side} "
                           f"{l.prop.line_score:g} {l.prop.stat_kind} "
                           f"{mr[0]}-{mr[1]}")
            return out

        slips = [
            s for s in slips
            if slip_signature([_pl(t) for t in _leg_specs(s)]) not in already
        ]
        for rank, s in enumerate(slips, 1):
            n = len(s.legs)
            if s.product == "flex":
                # A flex slip has no single break-even multiplier: it is paid
                # across several tiers, and quoting 1/P(all) invites reading
                # "needs 11.25x" on a bet whose top tier is 10x — a slip that
                # looks impossible while actually being fine, because most of
                # its value sits in the tiers below the top one.
                print(f"\n┌─ SLIP #{rank} · {n}-PICK FLEX")
                tiers = " ".join(
                    f"{k}/{n}={s.k_probs[k]:.0%}"
                    for k in range(n, max(n - 3, 1), -1)
                    if k < len(s.k_probs)
                )
                print(f"│  hit tiers  {tiers}")
                print(f"│  EV {s.ev * 100:+.0f}%, "
                      f"{s.adjusted_ev * 100:+.0f}% after haircut   "
                      f"{_delta_text(s)}")
            else:
                # Lead with the payout, not a break-even. The "needs >= X"
                # framing came from PrizePicks ARENA, where structurally
                # identical slips paid 7.25x / 7.75x / 8.25x / 9.25x and the
                # multiplier genuinely could not be predicted. On STANDARD
                # entries it can: the ladder is verified, the same-match pair
                # rule is verified, and per-side shades are priced in. So say
                # what the book will pay and keep break-even as the margin.
                from cs2props.optimizer.search import slip_price_factor

                shade = slip_price_factor(s.legs)
                if abs(shade - 1.0) > 1e-9:
                    price = (f"pays {s.multiplier * shade:.2f}x "
                             f"({s.multiplier:g}x shaded x{shade:.2f})")
                else:
                    price = f"pays {s.multiplier:g}x"
                print(f"\n┌─ SLIP #{rank} · {n}-PICK POWER · {price}")
                print(f"│  P(win) {s.p_all:.1%} -> EV {s.ev * 100:+.0f}%, "
                      f"{s.adjusted_ev * 100:+.0f}% after haircut"
                      f"   · profitable above {s.breakeven_multiplier:.2f}x"
                      f"   · {_delta_text(s)}")
            for l in s.legs:
                shop_note = ""
                alts = shop_index.get(prop_key(l.prop), {})
                if len(alts) > 1:
                    want_high = l.side == "under"
                    best_bk, best_pr = max(
                        alts.items(),
                        key=lambda kv: kv[1].line_score if want_high
                        else -kv[1].line_score,
                    )
                    if abs(best_pr.line_score - l.prop.line_score) > 1e-9:
                        shop_note = (
                            f"  ⇒ BETTER on {best_bk} @ "
                            f"{best_pr.line_score:g}"
                        )
                print(f"│   {l.side.upper():<6}{l.prop.player_name:<14}"
                      f"{(l.prop.team or '?'):<12}{l.prop.line_score:>5.1f} "
                      f"{l.prop.stat_kind:<11}P {l.p:.0%}  "
                      f"vs {l.prop.opponent or '?'}{shop_note}")
            for f in s.flags:
                print(f"│   ⚑ {f}")
            if s.note:
                print(f"│   note: {s.note}")
            legs_cmd = " ".join(
                f'--leg "{l.prop.player_name} {l.side} '
                f'{l.prop.line_score:g} {l.prop.stat_kind} '
                f'{(l.prop.map_range or (1, 2))[0]}-'
                f'{(l.prop.map_range or (1, 2))[1]}"'
                for l in s.legs
            )
            print(f"│   track: cs2props track --book {book} --stake 20 "
                  f"--claimed-p {s.p_all:.3f} {legs_cmd}")
            print("└" + "─" * 66)
            slip_views.append(SlipView(
                track_cmd=(
                    f"uv run cs2props track --book {book} --stake 1 "
                    f"--claimed-p {s.p_all:.3f} " + legs_cmd
                ),
                rank=rank, n_legs=len(s.legs),
                # what the book actually pays, per-side shades included
                multiplier=s.multiplier * slip_price_factor(s.legs),
                ev_pct=round(s.ev * 100, 1),
                ev_adj_pct=round(s.adjusted_ev * 100, 1),
                p_correlated=s.p_all,
                p_independent=s.p_independent,
                delta_is_real=s.delta_is_real,
                product=s.product,
                breakeven=s.breakeven_multiplier,
                legs_json=__import__("json").dumps(_leg_specs(s)),
                book=book,
                legs=tuple(leg_view(l) for l in s.legs),
                flags=tuple(s.flags + ([s.note] if s.note else [])),
            ))
        all_legs = tuple(
            leg_view(l)
            for sim in sims
            for l in _sim_legs(sim)
        )
        book_views.append(BookView(
            book=book, display=display,
            board_label=f"{len(props)} props · {len(sims)} matches",
            freshness=freshness, slips=tuple(slip_views), legs=all_legs,
        ))
        print()

    # snapshot every board we saw: the LAST line before kickoff is the
    # closing line, and closing-line value is the fastest read on whether the
    # model actually beats the market (converges in ~20-30 bets, where P&L
    # needs hundreds). No snapshots, no CLV.
    for _bk, _disp, _props, _fresh in boards:
        db.save_props(conn, _props)
    for _props in feed_only:  # reference lines: stored, never surfaced
        db.save_props(conn, _props)
    conn.close()

    from datetime import datetime, timezone

    from cs2props.tracker import summary as tracker_summary
    from cs2props.tracker import summary_rows, tracked_for_report

    tconn = db.connect(Path(args.db))
    tracked = tracked_for_report(tconn)
    tsummary = tracker_summary(tconn).replace("\n", " · ")
    from cs2props.clv import format_report as clv_report
    from cs2props.clv import leg_clvs as _lc

    _rows = _lc(tconn)
    if _rows:
        _v = [r.clv for r in _rows]
        _beat = sum(1 for x in _v if x > 0)
        tsummary += (f" · CLV {sum(_v) / len(_v):+.2f} over {len(_v)} legs "
                     f"({_beat} beat the close)")
    _bstats, _legrec, _legacy = summary_rows(tconn)
    tconn.close()

    data = ReportData(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        calibration_label=cal_label, is_mock=False,
        books=tuple(book_views),
        tracked=tuple(tracked), tracked_summary=tsummary,
        book_stats=tuple(_bstats), leg_record=_legrec,
        legacy_note=_legacy,
    )
    write_report(data, args.out)
    print(f"report written: {args.out}")
    return 0


def _delta_text(s: "Any") -> str:
    """Correlation bonus, or an honest statement that there is not one.

    On a fully diversified slip the true delta is zero — separate matches
    are simulated independently — so the printed value is Monte Carlo error.
    Showing "+0.2 pts" beside a real EV reads as a finding; one slip even
    printed a NEGATIVE correlation between independent events. Suppressed
    below the noise floor rather than dressed up.
    """
    if s.delta_is_real:
        return f"Δ {s.delta_pts:+.1f} pts from correlation"
    return "no correlation (all legs from separate matches)"


def search_slips_for(
    book: str, sims: "list[Any]", min_ev: float = 0.10,
    haircut: float = 0.02, shape_name: str | None = None,
    product: str | None = None, size: int | None = None,
) -> "tuple[list[Any], str | None]":
    from cs2props.config import load_payouts, load_restrictions
    from cs2props.optimizer.search import search_slips

    restr = load_restrictions(book)
    if not restr.bet_enabled:
        return [], (
            f"{book} is DATA-ONLY — its lines still feed cross-book "
            "comparison, but its measured hit rate does not clear its own "
            "cheapest product. Run `cs2props crossbook` to see what the "
            "feed is being used for."
        )
    payouts = load_payouts(book)
    shape = None
    if shape_name:
        try:
            shape = payouts.shape(shape_name)
        except KeyError:
            # Shapes are per-book observations. A structure priced on
            # PrizePicks says nothing about Underdog's ladder, so skip the
            # book rather than failing the scan or, worse, borrowing a
            # multiplier that was never seen there.
            return [], (
                f"shape '{shape_name}' is not configured for {book} — "
                "shapes are per-book observed prices, so this book is "
                "skipped rather than priced off another book's multiplier"
            )
    return search_slips(sims, payouts, restr,
                        target_size=size or restr.default_slip_size,
                        min_adjusted_ev=min_ev, haircut=haircut, shape=shape,
                        product=product or restr.default_product)


def _sim_legs(sim: "Any") -> "list[Any]":
    """Top displayable legs of a match sim for the board-marginals table."""
    from cs2props.optimizer.search import Leg

    out = []
    for pi, (prop, p_over) in enumerate(zip(sim.props, sim.result.p_over)):
        side, p = ("over", p_over) if p_over >= 0.5 else ("under", 1 - p_over)
        out.append(Leg(0, pi, side, p, prop, prop.team))
    return out


def _print_mock_scan() -> None:
    """Preview of the scan output format with FAKE numbers.

    Exists so the interface can be reviewed before the optimizer is built;
    delete once module 4 lands.
    """
    print("=" * 74)
    print("  cs2props scan — MOCK OUTPUT (fake numbers, format preview only)")
    print("=" * 74)
    print(
        "\nboard: 84 props | 6 matches | model: calibrated 2026-07-24 "
        "(log loss 0.658 vs 0.693 baseline)\n"
    )
    MockLeg = tuple[str, str, str, float, str, float, str]
    mock_slips: list[tuple[int, float, float, float, int, list[MockLeg], str]] = [
        (
            1, 18.4, 0.128, 0.094, 10,
            [
                ("donk",   "Spirit",  "OVER",  32.5, "kills 1-2", 0.64, "vs FaZe"),
                ("sh1ro",  "Spirit",  "OVER",  28.5, "kills 1-2", 0.61, "vs FaZe"),
                ("zont1x", "Spirit",  "OVER",  25.5, "kills 1-2", 0.58, "vs FaZe"),
                ("m0NESY", "Falcons", "OVER",  29.5, "kills 1-2", 0.62, "vs Vitality"),
            ],
            "3x Spirit stack — same-team, same-direction. FLAGGED for "
            "PrizePicks correlation rules review.",
        ),
        (
            2, 11.7, 0.112, 0.098, 10,
            [
                ("donk",   "Spirit",   "OVER", 32.5, "kills 1-2", 0.64, "vs FaZe"),
                ("m0NESY", "Falcons",  "OVER", 29.5, "kills 1-2", 0.62, "vs Vitality"),
                ("ropz",   "Vitality", "UNDER", 44.5, "kills srs", 0.60, "vs Falcons"),
                ("broky",  "FaZe",     "UNDER", 27.5, "kills 1-2", 0.57, "vs Spirit"),
            ],
            "cross-match; ropz UNDER + m0NESY OVER are same-match "
            "opposite sides (negatively coupled via rounds).",
        ),
    ]
    for rank, ev, corr, ind, mult, legs, note in mock_slips:
        delta = (corr - ind) * 100
        print(f"┌─ SLIP #{rank} ─ 4-PICK POWER @ {mult}x ─ EV {ev:+.1f}%")
        print(f"│  CORRELATION DELTA: {delta:+.1f} pts   "
              f"P(all 4) correlated {corr:.1%}  vs  independent {ind:.1%}")
        print("│")
        for name, team, side, line, stat, p, ctx in legs:
            print(f"│   {side:<6}{name:<9}{team:<10}{line:>5} {stat:<11}"
                  f"P(hit) {p:.0%}   {ctx}")
        print(f"│   note: {note}")
        print("└" + "─" * 72 + "\n")
    print(
        "MOCK — every number above is invented. Real output requires the "
        "calibrated model (module 2 gate) and optimizer (module 4)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="cs2props")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_board = sub.add_parser("board", help="fetch + parse the CS2 props board")
    src = p_board.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixture", help="parse a saved projections JSON payload")
    src.add_argument("--live", action="store_true", help="hit the live API")
    p_board.add_argument(
        "--source", choices=["underdog", "prizepicks"], default="underdog",
        help="lines source for --live (underdog is open; prizepicks needs "
        "import-board due to Cloudflare)",
    )
    p_board.add_argument("--cache-dir", default=".cache")
    p_board.add_argument("--db", help="also persist snapshot to this SQLite file")
    p_board.set_defaults(func=cmd_board)

    p_imp = sub.add_parser(
        "import-board",
        help="ingest a /projections JSON payload saved from a browser "
        "(bypasses Cloudflare; feeds the same cache --live reads)",
    )
    p_imp.add_argument("payload", help="path to saved projections JSON")
    p_imp.add_argument("--cache-dir", default=".cache/prizepicks")
    p_imp.add_argument("--db", help="also persist snapshot to this SQLite file")
    p_imp.set_defaults(func=cmd_import_board)

    p_bf = sub.add_parser(
        "backfill", help="ingest historical per-map stats from bo3.gg"
    )
    p_bf.add_argument("--months", type=int, default=12)
    p_bf.add_argument(
        "--days", type=float, default=None,
        help="walk only this many days back (overrides --months); what the "
             "dashboard grade button uses",
    )
    p_bf.add_argument("--tiers", default="s,a,b")
    p_bf.add_argument("--delay", type=float, default=2.0)
    p_bf.add_argument("--db", default="cs2props.db")
    p_bf.add_argument("--limit", type=int, help="stop after N new matches (smoke)")
    p_bf.set_defaults(func=cmd_backfill)

    p_cal = sub.add_parser(
        "calibrate", help="walk-forward backtest with calibration report"
    )
    p_cal.add_argument("--db", default="cs2props.db")
    p_cal.add_argument("--min-history", type=int, default=20)
    p_cal.set_defaults(func=cmd_calibrate)

    p_real = sub.add_parser(
        "reallines",
        help="backtest against REAL archived PrizePicks lines (not synthetic)",
    )
    p_real.add_argument("--db", default="cs2props.db")
    p_real.add_argument("--min-history", type=int, default=20)
    p_real.set_defaults(func=cmd_reallines)

    p_xb = sub.add_parser(
        "crossbook",
        help="track PrizePicks vs Underdog line disagreements and whether "
             "the cheap side pays",
    )
    p_xb.add_argument("--db", default="cs2props.db")
    p_xb.set_defaults(func=cmd_crossbook)

    p_prev = sub.add_parser(
        "preview",
        help="simulate the live board against history (pre-calibration)",
    )
    p_prev.add_argument("--db", default="cs2props.db")
    p_prev.add_argument("--cache-dir", default=".cache")
    p_prev.add_argument("--iters", type=int, default=50_000)
    p_prev.set_defaults(func=cmd_preview)

    p_ev = sub.add_parser(
        "ev", help="re-price a slip using the multiplier the app really shows"
    )
    p_ev.add_argument("--p", type=float, required=True,
                      help="claimed-p from the slip card, e.g. 0.188")
    p_ev.add_argument("--mult", type=float, required=True,
                      help="multiplier shown in the app, e.g. 8.5")
    p_ev.add_argument("--legs", type=int, default=4)
    p_ev.add_argument("--min-ev", type=float, default=0.10,
                      help="the floor the scanner used (default 0.10)")
    p_ev.add_argument("--db", default="cs2props.db")
    p_ev.set_defaults(func=cmd_ev)

    p_trk = sub.add_parser("track", help="record a slip you placed")
    p_trk.add_argument("--book", choices=["prizepicks", "underdog"],
                       required=True)
    p_trk.add_argument("--stake", type=float, required=True)
    p_trk.add_argument("--claimed-p", type=float,
                       help="model P(win) at placement, e.g. 0.128")
    p_trk.add_argument("--mult", type=float,
                       help="multiplier the APP showed (authoritative for P&L)")
    p_trk.add_argument("--leg", action="append", required=True,
                       help="'<player> over|under <line> kills|headshots "
                       "<lo>-<hi>' (repeat per leg)")
    p_trk.add_argument("--db", default="cs2props.db")
    p_trk.set_defaults(func=cmd_track)

    p_snap = sub.add_parser(
        "snapshot",
        help="capture closing lines when an open slip's match is imminent",
    )
    p_snap.add_argument("--window", type=int, default=75,
                        help="minutes before kickoff to start snapshotting")
    p_snap.add_argument("--db", default="cs2props.db")
    p_snap.add_argument("--cache-dir", default=".cache")
    p_snap.set_defaults(func=cmd_snapshot)

    p_srv = sub.add_parser(
        "serve", help="local dashboard with one-click slip tracking"
    )
    p_srv.add_argument("--port", type=int, default=8742)
    p_srv.add_argument("--db", default="cs2props.db")
    p_srv.set_defaults(func=cmd_serve)

    p_clv = sub.add_parser(
        "clv", help="closing-line value on tracked legs (edge signal)"
    )
    p_clv.add_argument("--db", default="cs2props.db")
    p_clv.set_defaults(func=cmd_clv)

    p_grd = sub.add_parser("grade", help="grade open slips vs results")
    p_grd.add_argument("--db", default="cs2props.db")
    p_grd.set_defaults(func=cmd_grade)

    p_rep = sub.add_parser(
        "report", help="render the HTML report (currently mock data only)"
    )
    p_rep.add_argument("--mock", action="store_true", required=True,
                       help="render with placeholder data (real data needs "
                       "the optimizer)")
    p_rep.add_argument("-o", "--out", default="cs2report.html")
    p_rep.set_defaults(func=cmd_report)

    p_scan = sub.add_parser("scan", help="rank +EV 4-man slips on live boards")
    p_scan.add_argument(
        "--mock", action="store_true",
        help="preview the output format with fake numbers",
    )
    p_scan.add_argument("--db", default="cs2props.db")
    p_scan.add_argument("--cache-dir", default=".cache")
    p_scan.add_argument("--iters", type=int, default=50_000)
    p_scan.add_argument(
        "--min-ev", type=float, default=0.10,
        help="minimum EV AFTER the model's optimism haircut (default 0.20). "
             "Below ~0.20 the edge does not survive known model error.",
    )
    p_scan.add_argument(
        "--shape", default=None,
        help="restrict the search to a slip STRUCTURE whose real multiplier "
             "has been observed in-app, and price it at that multiplier "
             "instead of the fitted estimate. 'cross2x2' = one player from "
             "each side of two matches, each pair same-direction, 8.5x.",
    )
    p_scan.add_argument(
        "--product", choices=("power", "flex"), default=None,
        help="power = every leg must hit; flex pays partial tiers. On the "
             "verified PrizePicks ladder flex wins on Kelly growth at 5 and "
             "6 legs (5-pick flex break-even 54.25%%, the lowest available).",
    )
    p_scan.add_argument(
        "--size", type=int, default=None,
        help="legs per slip; defaults to the book's configured size",
    )
    p_scan.add_argument("-o", "--out", default="cs2report.html")
    p_scan.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        stream=sys.stderr,
    )
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
