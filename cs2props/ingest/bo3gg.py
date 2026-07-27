"""bo3.gg historical CS2 stats ingestion.

Endpoint map (all verified live 2026-07-23):
- ``/api/v1/matches`` — finished matches; ``tier`` (s/a/b/c), ``bo_type``,
  ``start_date``; sorted ``-start_date`` and walked with offset pagination.
- ``/api/v1/games?filter[games.match_id][eq]={id}`` — maps of a match.
- ``/api/v1/games/{game_id}/players_stats`` — 10 rows per map with kills,
  deaths, ADR, rating, headshots and an embedded ``steam_profile`` carrying
  ``nickname`` + ``player_id``, so the join is self-contained.

Network policy: fixed delay between every request (default 2s), no retries on
4xx/5xx — fail loudly and resume later; ingestion is match-idempotent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.bo3.gg/api/v1"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
PAGE_SIZE = 50


@dataclass(frozen=True)
class PlayerMapRow:
    """One player's stat line on one map — mirrors the player_maps table."""

    player_id: str
    player_name: str
    team: str | None
    opponent: str | None
    event_tier: str | None
    map_name: str | None
    played_at: str
    kills: int
    deaths: int
    adr: float | None
    rating: float | None
    rounds: int | None  # winner+loser score; kills-per-round normalization
    headshots: int | None
    won: int | None  # 1 if this player's team won the map (team strength)
    match_id: str
    map_number: int


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def stats_to_rows(
    match: dict[str, Any], game: dict[str, Any], stats: list[dict[str, Any]]
) -> list[PlayerMapRow]:
    """Pure transform: one map's players_stats payload -> PlayerMapRows.

    Rows without an embedded steam_profile are dropped (unattributable).
    """
    played_at = game.get("begin_at") or match.get("start_date") or ""
    map_number = game.get("number") or (int(game.get("ov_index") or 0) + 1)
    w, l = game.get("winner_clan_score"), game.get("loser_clan_score")
    rounds = (int(w) + int(l)) if (w is not None and l is not None) else None
    rows: list[PlayerMapRow] = []
    for s in stats:
        profile = s.get("steam_profile") or {}
        pid = profile.get("player_id")
        nick = profile.get("nickname")
        if pid is None or not nick:
            log.warning(
                "match %s game %s: stat row %s has no steam_profile join",
                match.get("id"), game.get("id"), s.get("id"),
            )
            continue
        rows.append(
            PlayerMapRow(
                player_id=str(pid),
                player_name=str(nick),
                team=s.get("clan_name"),
                opponent=s.get("enemy_clan_name"),
                event_tier=match.get("tier"),
                map_name=game.get("map_name"),
                played_at=str(played_at),
                kills=int(s.get("kills") or 0),
                deaths=int(s.get("death") or 0),
                adr=float(s["adr"]) if s.get("adr") is not None else None,
                rating=(
                    float(s["player_rating"])
                    if s.get("player_rating") is not None
                    else None
                ),
                rounds=rounds,
                headshots=(
                    int(s["headshots"]) if s.get("headshots") is not None else None
                ),
                won=int(s["win"]) if s.get("win") is not None else None,
                match_id=str(match["id"]),
                map_number=int(map_number),
            )
        )
    return rows


class Bo3Client:
    """Politely-paced bo3.gg API client."""

    def __init__(
        self,
        delay_s: float = 2.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.delay_s = delay_s
        self._last_request = 0.0
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=30.0,
            transport=transport,
        )

    MAX_TRANSPORT_RETRIES = 3

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """One GET with pacing.

        Transient transport failures (connection resets, timeouts) get a
        bounded backoff retry — a 10-hour backfill must survive a flaky
        socket. HTTP errors (4xx/5xx) still fail loudly with no retry:
        those mean something is actually wrong.
        """
        for attempt in range(1, self.MAX_TRANSPORT_RETRIES + 1):
            wait = self.delay_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self._client.get(path, params=params or {})
            except httpx.TransportError as e:
                if attempt == self.MAX_TRANSPORT_RETRIES:
                    raise
                backoff = 5.0 * attempt
                log.warning(
                    "transport error on %s (%s) — retry %d/%d in %.0fs",
                    path, e, attempt, self.MAX_TRANSPORT_RETRIES, backoff,
                )
                time.sleep(backoff)
                continue
            resp.raise_for_status()  # fail loudly; backfill is resumable
            return resp.json()
        raise AssertionError("unreachable")

    def iter_finished_matches(
        self, since: datetime, tiers: frozenset[str]
    ) -> Iterator[dict[str, Any]]:
        """Walk finished matches newest-first until older than ``since``.

        Tier filtering is client-side (the server-side tier filter is not
        reliably honored); the date cutoff stops pagination.
        """
        offset = 0
        while True:
            page = self._get(
                "/matches",
                {
                    "filter[matches.status][eq]": "finished",
                    "sort": "-start_date",
                    "page[limit]": PAGE_SIZE,
                    "page[offset]": offset,
                },
            )
            results: list[dict[str, Any]] = page.get("results") or []
            if not results:
                return
            for m in results:
                start = m.get("start_date")
                if start and parse_iso(start) < since:
                    return
                if (m.get("tier") or "").lower() in tiers:
                    yield m
            offset += PAGE_SIZE

    def fetch_games(self, match_id: int | str) -> list[dict[str, Any]]:
        page = self._get(
            "/games", {"filter[games.match_id][eq]": match_id, "page[limit]": 10}
        )
        results: list[dict[str, Any]] = page.get("results") or []
        return [g for g in results if g.get("status") == "finished"]

    def fetch_players_stats(self, game_id: int | str) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = self._get(f"/games/{game_id}/players_stats")
        return data


def since_months_ago(months: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=months * 30)
