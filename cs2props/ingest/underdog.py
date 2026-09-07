"""Underdog lines ingestion (open API, no Cloudflare).

Feed (since the 2026-09 underdogsports.com rebrand — the old
``/beta/v5/over_under_lines`` returns 426 Upgrade Required):
GET /v1/lobbies/content/lines?filter_id=<market>&filter_type=PickemStat
with the ``product`` / ``product_experience_id`` / ``state_config_id``
query params the web app sends. Works unauthenticated; the endpoint and
params were captured off the live app's network traffic 2026-09-07. One
request per market filter (kills, headshots), merged.

Shape: same entities as v5 — ``over_under_lines`` (line + options),
``appearances`` (player-match join), ``players``, ``games``
(``abbreviated_title`` like "ALL vs FaZe") — but containers are now DICTS
keyed by id instead of arrays; ``_vals`` normalizes.

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
LINES_PATH = "/v1/lobbies/content/lines"
# Query params the web app sends. product_experience_id / state_config_id
# look session-ish but are stable app config (identical across sessions and
# work from a cold curl); if Underdog rotates them, capture fresh ones from
# the app's network tab and fail loudly in the meantime.
LOBBY_PARAMS = {
    "include_live": "true",
    "product": "fantasy",
    "product_experience_id": "c7ade3c1-71ae-4593-a7e1-07f63c7e94ae",
    "show_mass_option_markets": "false",
    "sport_id": "CS",
    "state_config_id": "725014ef-3570-4e93-871d-d69674ab3521",
}
# Per-market filter ids (the market_filters discovery endpoint requires app
# tokens, so these are pinned from the app's own requests; a retired id
# yields zero lines, which fetch_lines treats as an error, never as an
# empty board).
MARKET_FILTERS = {
    "kills": "17f53e7e-4ea7-4400-ac0b-658178fb2320",
    "headshots": "24ed46e4-7baf-4a00-8190-c721709a0c52",
}


def _vals(container: "list[Any] | dict[str, Any] | None") -> list[Any]:
    """v5 shipped arrays, v1/lobbies ships id-keyed dicts — accept both."""
    if isinstance(container, dict):
        return list(container.values())
    return list(container or [])
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
    players = {p["id"]: p for p in _vals(payload.get("players"))}
    appearances = {a["id"]: a for a in _vals(payload.get("appearances"))}
    games = {g["id"]: g for g in _vals(payload.get("games"))}
    props: list[Prop] = []
    for line in _vals(payload.get("over_under_lines")):
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
        # The feed says which lines are alternates — believe it. A magnitude
        # threshold ("alt if any multiplier is ±50%") let a 1.49x alternate
        # ladder rung through as "standard": the optimizer then bet the
        # UNDER of NAF 32.5 — an alternate that only SELLS the over — while
        # the real balanced line sat at 25.5. The phantom 7-kill cushion was
        # the whole edge on that card (2026-07-29).
        board = (
            "standard" if str(line.get("line_type", "")) == "balanced"
            else "alt"
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
        """One merged payload across the pinned market filters.

        Entities are merged by id (a player appearing in both the kills and
        headshots response is the same row); lines never collide because
        each belongs to exactly one market."""
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
        merged: dict[str, dict[str, Any]] = {
            k: {} for k in ("over_under_lines", "appearances", "players",
                            "games")
        }
        for market, filter_id in MARKET_FILTERS.items():
            params = dict(LOBBY_PARAMS)
            params.update({"filter_id": filter_id,
                           "filter_type": "PickemStat"})
            log.info("GET %s%s (%s)", BASE_URL, LINES_PATH, market)
            resp = self._client.get(LINES_PATH, params=params)
            resp.raise_for_status()
            part = resp.json()
            n = len(_vals(part.get("over_under_lines")))
            if n == 0:
                # a retired filter id must be loud, not an empty board
                raise RuntimeError(
                    f"underdog returned ZERO {market} lines — the pinned "
                    f"filter_id {filter_id} has likely been rotated; "
                    "capture a fresh one from the app's network tab"
                )
            for key in merged:
                for item in _vals(part.get(key)):
                    merged[key][str(item["id"])] = item
            time.sleep(1.0)  # polite inter-market gap
        payload_live: dict[str, Any] = {
            k: list(v.values()) for k, v in merged.items()
        }
        cache.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload_live})
        )
        return payload_live

    def fetch_board(self, sport_id: str = "CS") -> list[Prop]:
        return parse_lines(self.fetch_lines(), sport_id)
