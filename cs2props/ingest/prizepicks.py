"""PrizePicks board ingestion.

JSON:API format: ``data[]`` holds projection resources, ``included[]`` holds
``new_player`` (and league/game) resources. Projections join to players via
``relationships.new_player.data.id``.

Network policy:
- 1 request per 60s, enforced via a timestamp file in the cache dir.
- Every response cached to disk with a fetched-at timestamp; cache is served
  when fresh (default TTL 10 min) without touching the network.
- Cloudflare 403 raises :class:`CloudflareBlocked` immediately — no retries.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.prizepicks.com"
# The public api.prizepicks.com host sits behind Cloudflare and returns 403 to
# any non-browser client — confirmed live 2026-07-24, which is why the board
# had to be hand-saved as pp.json for two days. partner-api.prizepicks.com
# serves the same JSON:API payload without that gate (verified 2026-07-26:
# HTTP 200, 335 props parsing byte-identically to a manual save, and /leagues
# works too so the league id is still resolved, never hardcoded).
PARTNER_BASE_URL = "https://partner-api.prizepicks.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
MIN_REQUEST_INTERVAL_S = 60.0
DEFAULT_CACHE_TTL_S = 600.0
# /leagues answers one question — "what id is CS2?" — and the answer has
# never changed. Re-asking every scan cost the full 60s rate-limit gap
# between it and /projections, doubling every board refresh. A day-long
# cache keeps the resolve dynamic (never hardcoded, per spec) while making
# the common scan a single request.
LEAGUES_CACHE_TTL_S = 24 * 3600.0

# League names PrizePicks has used for Counter-Strike, lowercased.
CS_LEAGUE_NAMES = {"cs2", "csgo", "cs:go", "counter-strike 2", "counter-strike"}

_STAT_RE = re.compile(r"^maps?\s+(\d+)(?:\s*-\s*(\d+))?\s+(.+)$", re.IGNORECASE)
# live description field looks like "Keyd Stars MAPS 1-2" — opponent + stat
# context glued together; strip the trailing MAPS clause
_DESC_MAPS_RE = re.compile(r"\s+maps?\s+\d+(?:\s*-\s*\d+)?\s*$", re.IGNORECASE)


class CloudflareBlocked(RuntimeError):
    """PrizePicks returned 403 — blocked by Cloudflare. Do not retry in a loop."""


class LeagueNotFound(RuntimeError):
    """No CS league is currently on the PrizePicks board."""


@dataclass(frozen=True)
class Prop:
    """One parsed projection joined with its player."""

    projection_id: str
    player_id: str
    player_name: str
    team: str | None
    opponent: str | None
    stat_type: str  # raw, e.g. "MAPS 1-2 Kills"
    stat_kind: str  # normalized, e.g. "kills"
    map_range: tuple[int, int] | None  # (1, 2); None = full series / unspecified
    line_score: float
    board: str  # "standard" | "demon" | "goblin"
    start_time: str | None  # ISO 8601 as provided
    league_id: str
    # Per-SIDE payout multipliers, keyed "over"/"under". Underdog prices each
    # side independently even on lines it calls "balanced" — Salazar's 14.5
    # headshots was higher 1.03 / lower 0.82 on 2026-07-26 — and the slip
    # payout is the ladder multiplier TIMES the product of the chosen sides'
    # multipliers. Taking that under turns a 6.5x 3-pick into 5.33x. Tagging
    # the whole prop "alt" and dropping it was both too strict (it discards
    # the 1.03 side, which is a bonus) and too loose (the split moves after a
    # scan, so a prop balanced at scan time can be shaded by kickoff).
    # Empty means unpriced -> treated as 1.0.
    side_multipliers: dict[str, float] = field(default_factory=dict)

    def side_multiplier(self, side: str) -> float:
        return float(self.side_multipliers.get(side, 1.0))


def normalize_stat_type(stat_type: str) -> tuple[str, tuple[int, int] | None]:
    """Split a raw stat_type into (stat_kind, map_range).

    "MAPS 1-2 Kills"  -> ("kills", (1, 2))
    "MAP 3 Headshots" -> ("headshots", (3, 3))
    "Kills"           -> ("kills", None)   # full-series semantics differ; the
                                           # model must handle None explicitly.
    """
    m = _STAT_RE.match(stat_type.strip())
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        kind = m.group(3)
    else:
        lo = hi = 0
        kind = stat_type
    kind_norm = re.sub(r"\s+", " ", kind.strip().lower())
    return kind_norm, ((lo, hi) if m else None)


def parse_projections(payload: dict[str, Any]) -> list[Prop]:
    """Parse a raw /projections JSON:API payload into Props.

    Projections without a resolvable player are skipped with a warning —
    a prop we cannot attribute to a player is unusable downstream.
    """
    players: dict[str, dict[str, Any]] = {
        item["id"]: item.get("attributes", {})
        for item in payload.get("included", [])
        if item.get("type") == "new_player"
    }
    props: list[Prop] = []
    for item in payload.get("data", []):
        if item.get("type") != "projection":
            continue
        attrs = item.get("attributes", {})
        rel = (
            item.get("relationships", {})
            .get("new_player", {})
            .get("data")
        )
        if not rel or rel.get("id") not in players:
            log.warning(
                "skipping projection %s: unresolvable player relationship",
                item.get("id"),
            )
            continue
        player = players[rel["id"]]
        league_rel = (
            item.get("relationships", {}).get("league", {}).get("data") or {}
        )
        raw_stat = str(attrs.get("stat_type", ""))
        stat_kind, map_range = normalize_stat_type(raw_stat)
        line = attrs.get("line_score")
        if line is None:
            log.warning("skipping projection %s: no line_score", item.get("id"))
            continue
        props.append(
            Prop(
                projection_id=str(item["id"]),
                player_id=str(rel["id"]),
                player_name=str(
                    player.get("display_name") or player.get("name") or "?"
                ),
                team=player.get("team") or player.get("team_name"),
                opponent=_DESC_MAPS_RE.sub("", attrs.get("description") or "")
                or None,
                stat_type=raw_stat,
                stat_kind=stat_kind,
                map_range=map_range,
                line_score=float(line),
                board=str(attrs.get("odds_type", "standard")),
                start_time=attrs.get("start_time"),
                league_id=str(league_rel.get("id", "")),
            )
        )
    log.info("parsed %d props (%d players on board)", len(props), len(players))
    return props


class PrizePicksClient:
    """Rate-limited, disk-cached PrizePicks API client."""

    def __init__(
        self,
        cache_dir: Path,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        transport: httpx.BaseTransport | None = None,
        base_url: str = PARTNER_BASE_URL,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_ttl_s = cache_ttl_s
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stamp = cache_dir / ".last_request"
        self._client = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=30.0,
            transport=transport,
        )

    # -- cache / rate limit ------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(
        self, key: str, ttl_s: float | None = None
    ) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        wrapper = json.loads(path.read_text())
        age = time.time() - wrapper["fetched_at"]
        if age > (ttl_s if ttl_s is not None else self.cache_ttl_s):
            return None
        log.info("cache hit for %s (age %.0fs)", key, age)
        payload: dict[str, Any] = wrapper["payload"]
        return payload

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        self._cache_path(key).write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload})
        )

    def _respect_rate_limit(self) -> None:
        if self._stamp.exists():
            elapsed = time.time() - self._stamp.stat().st_mtime
            wait = MIN_REQUEST_INTERVAL_S - elapsed
            if wait > 0:
                log.info("rate limit: sleeping %.0fs before next request", wait)
                time.sleep(wait)
        self._stamp.touch()

    # -- requests ----------------------------------------------------------

    def _get(
        self, path: str, params: dict[str, Any], cache_key: str,
        cache_ttl_s: float | None = None,
    ) -> dict[str, Any]:
        cached = self._read_cache(cache_key, cache_ttl_s)
        if cached is not None:
            return cached
        self._respect_rate_limit()
        log.info("GET %s params=%s", path, params)
        resp = self._client.get(path, params=params)
        if resp.status_code == 403:
            raise CloudflareBlocked(
                f"PrizePicks returned 403 (Cloudflare) from "
                f"{self._client.base_url}. Not retrying — a retry loop against "
                "a bot gate is how an IP gets banned outright. Fall back to a "
                "browser-saved payload: open the projections URL, save it as "
                "pp.json, then `cs2props import-board pp.json`."
            )
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        self._write_cache(cache_key, payload)
        return payload

    def resolve_cs_league_id(self) -> str:
        """Hit /leagues and find the CS league id. Never hardcoded."""
        payload = self._get("/leagues", {}, cache_key="leagues",
                            cache_ttl_s=LEAGUES_CACHE_TTL_S)
        for item in payload.get("data", []):
            name = str(item.get("attributes", {}).get("name", "")).lower()
            if name in CS_LEAGUE_NAMES:
                league_id = str(item["id"])
                log.info("resolved CS league: name=%r id=%s", name, league_id)
                return league_id
        raise LeagueNotFound(
            "No CS2/CSGO league found on /leagues — CS may be off-board today."
        )

    def fetch_projections(self, league_id: str) -> dict[str, Any]:
        return self._get(
            "/projections",
            {"league_id": league_id, "per_page": 500, "single_stat": "true"},
            cache_key=f"projections_{league_id}",
        )

    def fetch_board(self) -> list[Prop]:
        """Resolve league, fetch projections, parse. The one-call entry point."""
        league_id = self.resolve_cs_league_id()
        return parse_projections(self.fetch_projections(league_id))


def load_fixture(path: Path) -> list[Prop]:
    """Parse a saved /projections payload from disk (no network)."""
    return parse_projections(json.loads(path.read_text()))
