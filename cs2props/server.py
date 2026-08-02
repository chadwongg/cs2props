"""Local web server for the report: one-click slip tracking.

The static report can only copy a command to the clipboard; logging a bet
still meant a trip to the terminal, and nothing stopped the same slip being
entered twice. This server adds a single POST endpoint so the page can write
straight to the tracker, and the report then HIDES any suggested slip that
has already been placed.

Deliberately tiny and local-only: binds 127.0.0.1, no auth, no external
dependencies. It is a personal dashboard, not a service.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cs2props import db
from cs2props.tracker import LegSpec, parse_leg, track_slip

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
# background jobs the UI buttons may have in flight, keyed by name.
# One per name at a time: the PrizePicks rate limiter is per-process, and
# two backfills would hammer bo3.gg for the same matches.
_JOBS: dict[str, "subprocess.Popen[bytes]"] = {}
# when the background top-up last finished a backfill+grade (epoch). The
# grade button skips its own backfill while this is fresh, which is what
# makes it feel instant — the 7s of polite bo3.gg paging already happened.
_LAST_TOPUP = 0.0
_TOPUP_EVERY_S = 1800.0
_TOPUP_FRESH_S = 1500.0


def _topup_loop(db_path: Path) -> None:
    """Keep match results warm so grading needs no on-demand download.

    Every 30 minutes: backfill yesterday-and-today's finished matches and
    grade open slips. Skipped while a button-triggered job is running (the
    two would page bo3.gg for the same matches and contend for the DB).
    Failures are logged and retried next cycle — this thread must never die.
    """
    import time

    global _LAST_TOPUP
    exe = str(Path(sys.executable).parent / "cs2props")
    while True:
        try:
            with _LOCK:
                busy = any(j.poll() is None for j in _JOBS.values())
            if not busy:
                subprocess.run(
                    [exe, "backfill", "--db", str(db_path),
                     "--days", "1", "--limit", "40"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=600, check=True,
                )
                subprocess.run(
                    [exe, "grade", "--db", str(db_path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=300, check=True,
                )
                _LAST_TOPUP = time.time()
                log.info("background top-up done")
        except Exception as e:  # noqa: BLE001 — heartbeat thread
            log.warning("background top-up failed (will retry): %s", e)
        time.sleep(_TOPUP_EVERY_S)


def slip_signature(legs: list[LegSpec]) -> frozenset[tuple[str, str, float, str]]:
    """Identity of a slip: its set of legs, order-independent.

    Used to recognise a suggested slip the user has already placed so it can
    be hidden — the guard against entering the same bet twice.
    """
    from cs2props.model.state_builder import clean_name

    return frozenset(
        (clean_name(l.player_name), l.side.lower(), float(l.line), l.stat_kind)
        for l in legs
    )


def placed_signatures(db_path: Path) -> set[frozenset[tuple[str, str, float, str]]]:
    """Signatures of every slip already tracked."""
    from cs2props.model.state_builder import clean_name

    conn = db.connect(db_path)
    out: set[frozenset[tuple[str, str, float, str]]] = set()
    for (sid,) in conn.execute("SELECT slip_id FROM slips"):
        legs = conn.execute(
            "SELECT player_name, side, line, stat_kind FROM slip_legs "
            "WHERE slip_id = ?", (sid,)
        ).fetchall()
        out.add(frozenset(
            (clean_name(p), s.lower(), float(ln), st) for p, s, ln, st in legs
        ))
    conn.close()
    return out


class ReportHandler(SimpleHTTPRequestHandler):
    """Serves the report directory and accepts POST /api/track."""

    db_path: Path = Path("cs2props.db")

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter
        log.debug(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        """Serve the report with a FRESHLY rendered My Slips section.

        The report file is written at scan time, so a slip tracked from the
        UI afterwards would not appear until the next scan — the card would
        vanish from the board and show up nowhere. Re-rendering that one
        section on each request keeps a reload always truthful.
        """
        if self.path.startswith("/api/scan-status"):
            self._scan_status()
            return
        if self.path.rstrip("/") in ("", "/cs2report.html", "/index.html"):
            page = Path(self.directory) / "cs2report.html"
            if page.exists():
                try:
                    self._serve_with_live_tracked(page)
                    return
                except Exception as e:  # fall back to the static file
                    log.warning("live tracked-section render failed: %s", e)
        super().do_GET()

    def _scan_status(self) -> None:
        from urllib.parse import parse_qs, urlparse

        name = parse_qs(urlparse(self.path).query).get("job", ["scan"])[0]
        with _LOCK:
            job = _JOBS.get(name)
            if job is None:
                state = "idle"
            elif job.poll() is None:
                state = "running"
            elif job.returncode == 0:
                state = "done"
            else:
                state = f"failed ({job.returncode})"
        self._json({"ok": True, "state": state})

    def _serve_with_live_tracked(self, page: Path) -> None:
        from cs2props.report import (
            ReportData, _tracked_section, build_stat_cards,
        )
        from cs2props.tracker import summary as tracker_summary
        from cs2props.tracker import summary_rows, tracked_for_report

        html = page.read_text()
        start, end = "<!--TRACKED_START-->", "<!--TRACKED_END-->"
        i, j = html.find(start), html.find(end)
        if i == -1 or j == -1:
            super().do_GET()
            return
        with _LOCK:
            conn = db.connect(self.db_path)
            tracked = tracked_for_report(conn)
            summary = tracker_summary(conn).replace("\n", " · ")
            # The structured stats must be passed too. This block RE-RENDERS
            # the tracked section on every request so a bet shows up without
            # a rescan — which means anything the scan-time renderer needs
            # has to be supplied here as well, or the live version silently
            # falls back to the old layout and the page looks unchanged no
            # matter how many times the scan is re-run.
            bstats, legrec, legacy = summary_rows(conn)
            stats = build_stat_cards(conn)
            conn.close()
        fresh = _tracked_section(ReportData(
            generated="", calibration_label="", is_mock=False, books=(),
            tracked=tuple(tracked), tracked_summary=summary,
            book_stats=tuple(bstats), leg_record=legrec,
            legacy_note=legacy, stats=stats,
        ))
        body = (html[:i] + fresh + html[j + len(end):]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path == "/api/grade-leg":
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                from cs2props.tracker import manual_grade_leg

                with _LOCK:
                    conn = db.connect(self.db_path)
                    status = manual_grade_leg(
                        conn,
                        slip_id=str(payload["slip_id"]),
                        leg_no=int(payload["leg_no"]),
                        observed=(float(payload["observed"])
                                  if payload.get("observed") not in (None, "")
                                  else None),
                        dnp=bool(payload.get("dnp")),
                    )
                    conn.close()
                self._json({"ok": True, "status": status})
            except Exception as e:
                log.warning("manual grade failed: %s", e)
                self._json({"ok": False, "error": str(e)}, 400)
            return
        if self.path == "/api/rescan":
            self._start_job("scan", ["scan", "--db", str(self.db_path)])
            return
        if self.path == "/api/grade":
            # Results first, then grading: legs grade against ingested
            # matches, so grading alone right after games end would just
            # report "still pending" and look broken. BUT the background
            # top-up runs that backfill every 30 minutes — while its data is
            # fresh, grading alone is both correct and sub-second, which is
            # what makes this button feel instant.
            import time

            if time.time() - _LAST_TOPUP < _TOPUP_FRESH_S:
                self._start_job("grade", ["grade", "--db", str(self.db_path)])
                return
            exe = str(Path(sys.executable).parent / "cs2props")
            cmd = (f"{exe} backfill --db {self.db_path} --days 3 "
                   f"--limit 60 && {exe} grade --db {self.db_path}")
            self._start_job("grade", cmd, shell=True)
            return
        if self.path != "/api/track":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            legs = [parse_leg(t) for t in payload["legs"]]
            with _LOCK:  # one writer at a time; SQLite is not concurrent
                conn = db.connect(self.db_path)
                already = placed_signatures(self.db_path)
                if slip_signature(legs) in already:
                    conn.close()
                    self._json({"ok": False, "error": "already tracked"}, 409)
                    return
                slip_id = track_slip(
                    conn,
                    book=str(payload["book"]),
                    stake=float(payload.get("stake") or 1.0),
                    legs=legs,
                    claimed_p=(float(payload["claimed_p"])
                               if payload.get("claimed_p") else None),
                    multiplier=(float(payload["mult"])
                                if payload.get("mult") else None),
                    product=str(payload.get("product") or "power"),
                )
                conn.close()
            log.info("tracked %s from the report UI", slip_id)
            self._json({"ok": True, "slip_id": slip_id})
        except Exception as e:  # never take the dashboard down
            log.warning("track failed: %s", e)
            self._json({"ok": False, "error": str(e)}, 400)

    def _start_job(
        self, name: str, cmd: "list[str] | str", shell: bool = False,
    ) -> None:
        """Run a cs2props subcommand in the background, one per job name.

        Long jobs cannot block the request: a scan takes ~90s (the
        PrizePicks client sleeps 60s between its two requests by design) and
        a backfill+grade can take a few minutes of polite bo3.gg paging. The
        page polls /api/scan-status?job=... and reloads on completion.

        Not `uv run`: launchd starts this server with a minimal PATH that
        has no uv on it (FileNotFoundError, found live 2026-07-26). The
        server already runs inside the project venv, so the sibling
        `cs2props` console script is the same environment with no PATH
        assumptions at all.
        """
        with _LOCK:
            job = _JOBS.get(name)
            if job is not None and job.poll() is None:
                self._json({"ok": True, "state": "already-running"})
                return
            if not shell:
                exe = str(Path(sys.executable).parent / "cs2props")
                cmd = [exe, *cmd]
            _JOBS[name] = subprocess.Popen(
                cmd, cwd=str(self.directory), shell=shell,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        log.info("%s started from the report UI (pid %s)",
                 name, _JOBS[name].pid)
        self._json({"ok": True, "state": "started"})

    def _json(self, body: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(directory: Path, db_path: Path, port: int = 8742) -> None:
    ReportHandler.db_path = db_path
    handler = partial(ReportHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(
        target=_topup_loop, args=(db_path,), daemon=True,
        name="cs2props-topup",
    ).start()
    print(f"cs2props dashboard: http://127.0.0.1:{port}/cs2report.html")
    print("  one-click tracking is live — Ctrl-C to stop")
    server.serve_forever()
