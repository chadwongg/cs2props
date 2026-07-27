"""Underdog Fantasy lines ingestion (open API, no Cloudflare).

Feed: GET https://api.underdogfantasy.com/beta/v5/over_under_lines
Shape: flat arrays — ``over_under_lines`` (line + options), ``appearances``
(player↔match join), ``players`` (sport_id "CS"), ``games`` (match meta with
``abbreviated_title`` like "RRQ @ ZETA").

Output is the same :class:`~cs2props.ingest.prizepicks.Prop` dataclass, so
everything downstream is source-agnostic. Underdog stat keys look like
``kills_on_maps_1_2`` and are normalized to the same (stat_kind, map_range)
semantics as PrizePicks stat types.

Note: Underdog prices each side (american_price / payout_multiplier). Those
prices are not part of ``Prop`` v1 — the optimizer consumes model
probabilities — but the raw payload is cached to disk, so pricing can be
joined in later without refetching.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from cs2props.ingest.prizepicks import Prop

log = logging.getLogger(__name__)

BASE_URL = "https://api.underdogfantasy.com"
LINES_PATH = "/beta/v5/over_under_lines"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
MIN_REQUEST_INTERVAL_S = 10.0
DEFAULT_CACHE_TTL_S = 600.0

_STAT_KEY_RE = re.compile(r"^(.+?)_on_maps?_(\d+)(?:_(\d+))?(?:_(\d+))?$")


def normalize_stat_key(stat_key: str) -> tuple[str, tuple[int, int] | None]:
    """Underdog stat key -> (stat_kind, map_range).

    "kills_on_maps_1_2"      -> ("kills", (1, 2))
    "headshots_on_maps_1_2_3"-> ("headshots", (1, 3))
    "kills"                  -> ("kills", None)
    """
    m = _STAT_KEY_RE.match(stat_key.strip().lower())
    if not m:
        return stat_key.strip().lower().replace("_", " "), None
    kind = m.group(1).replace("_", " ")
    nums = [int(g) for g in m.groups()[1:] if g is not None]
    return kind, (nums[0], nums[-1])


def _team_names(game: dict[str, Any]) -> tuple[str | None, str | None]:
    """("away", "home") abbreviations from 'AWAY @ HOME' or 'AWAY vs HOME'."""
    title = game.get("abbreviated_title") or ""
    for sep in (" @ ", " vs ", " VS "):
        if sep in title:
            away, home = title.split(sep, 1)
            return away.strip(), home.strip()
    return None, None


def parse_lines(payload: dict[str, Any], sport_id: str = "CS") -> list[Prop]:
    """Parse an over_under_lines payload into Props (CS only by default)."""
    players = {p["id"]: p for p in payload.get("players", [])}
    appearances = {a["id"]: a for a in payload.get("appearances", [])}
    games = {g["id"]: g for g in payload.get("games", [])}
    props: list[Prop] = []
    for line in payload.get("over_under_lines", []):
        if line.get("status") not in (None, "active"):
            continue
        ou = line.get("over_under") or {}
        ast = ou.get("appearance_stat") or {}
        app = appearances.get(ast.get("appearance_id"))
        if not app:
            continue
        player = players.get(app.get("player_id"))
        if not player or player.get("sport_id") != sport_id:
            continue
        game = games.get(app.get("match_id"), {})
        away, home = _team_names(game)
        if app.get("team_id") == game.get("away_team_id"):
            team, opponent = away, home
        elif app.get("team_id") == game.get("home_team_id"):
            team, opponent = home, away
        else:
            team = opponent = None
        stat_key = str(ast.get("stat", ""))
        stat_kind, map_range = normalize_stat_key(stat_key)
        value = line.get("stat_value")
        if value is None:
            log.warning("skipping line %s: no stat_value", line.get("id"))
            continue
        name = (player.get("last_name") or player.get("first_name") or "?").strip()
        # Underdog prices each SIDE independently, even on lines it labels
        # "balanced": Salazar 14.5 headshots was higher 1.03 / lower 0.82 on
        # 2026-07-26. Carry both so the optimizer can price the side it
        # actually takes — a 0.82 leg turns a 6.5x 3-pick into 5.33x, which
        # is the difference between a good slip and a bad one.
        side_mults: dict[str, float] = {}
        for o in line.get("options", []):
            choice = str(o.get("choice", "")).lower()
            side = {"higher": "over", "lower": "under"}.get(choice)
            if side:
                side_mults[side] = float(o.get("payout_multiplier") or 1)
        # "alt" is now reserved for the SCORCHER ladders — alternate lines
        # sold at a big premium. A main line with a mild per-side shade stays
        # on the standard board and gets priced, not discarded.
        board = (
            "alt" if any(m >= 1.5 or m <= 0.5 for m in side_mults.values())
            else "standard"
        )
        props.append(
            Prop(
                projection_id=str(line["id"]),
                player_id=str(app.get("player_id")),
                player_name=name,
                team=team,
                opponent=opponent,
                stat_type=str(ast.get("display_stat", stat_key)),
                stat_kind=stat_kind,
                map_range=map_range,
                line_score=float(value),
                board=board,
                start_time=game.get("scheduled_at"),
                league_id=sport_id,
                side_multipliers=side_mults,
            )
        )
    log.info("parsed %d %s props from underdog feed", len(props), sport_id)
    return props


class UnderdogClient:
    """Disk-cached Underdog lines client (same policy shape as PrizePicks)."""

    def __init__(
        self,
        cache_dir: Path,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_ttl_s = cache_ttl_s
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stamp = cache_dir / ".last_request"
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=30.0,
            transport=transport,
        )

    def fetch_lines(self) -> dict[str, Any]:
        cache = self.cache_dir / "over_under_lines.json"
        if cache.exists():
            wrapper = json.loads(cache.read_text())
            age = time.time() - wrapper["fetched_at"]
            if age <= self.cache_ttl_s:
                log.info("underdog cache hit (age %.0fs)", age)
                payload: dict[str, Any] = wrapper["payload"]
                return payload
        if self._stamp.exists():
            elapsed = time.time() - self._stamp.stat().st_mtime
            wait = MIN_REQUEST_INTERVAL_S - elapsed
            if wait > 0:
                log.info("rate limit: sleeping %.1fs", wait)
                time.sleep(wait)
        self._stamp.touch()
        log.info("GET %s%s", BASE_URL, LINES_PATH)
        resp = self._client.get(LINES_PATH)
        resp.raise_for_status()
        payload_live: dict[str, Any] = resp.json()
        cache.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload_live})
        )
        return payload_live

    def fetch_board(self, sport_id: str = "CS") -> list[Prop]:
        return parse_lines(self.fetch_lines(), sport_id)
