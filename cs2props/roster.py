"""Roster verification against announced lineups.

A prop on a player who does not actually play is dead weight: the book voids
the leg, which shrinks a 4-pick to a 3-pick payout — you lose the multiplier
you were betting for even when your other three legs land. Worse, a STAND-IN
changes the projections for everyone else in the match, and the model would
happily price the game as if the usual roster were playing.

bo3.gg publishes full 5v5 lineups for upcoming matches, so this is checkable
rather than guessable.

False-positive guard: bo3.gg does not cover every match the books post. A
player missing from the index therefore means nothing on its own. We only
flag a player as benched when at least one OTHER player from the SAME board
match IS in an announced lineup — that proves the match is covered and the
absence is real.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cs2props.ingest.bo3gg import Bo3Client
from cs2props.model.state_builder import clean_name

log = logging.getLogger(__name__)

ROSTER_TTL_S = 3600.0
DEFAULT_MAX_MATCHES = 40


@dataclass
class RosterIndex:
    """Announced lineups for upcoming matches, keyed by cleaned nickname."""

    players: set[str] = field(default_factory=set)
    team_of: dict[str, str] = field(default_factory=dict)  # player -> team id
    fetched_at: float = 0.0

    def has(self, player_name: str) -> bool:
        return clean_name(player_name) in self.players


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / "rosters.json"


def load_cached(cache_dir: Path, ttl_s: float = ROSTER_TTL_S) -> RosterIndex | None:
    path = _cache_path(cache_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if time.time() - raw.get("fetched_at", 0) > ttl_s:
        return None
    return RosterIndex(
        players=set(raw["players"]),
        team_of={k: str(v) for k, v in raw.get("team_of", {}).items()},
        fetched_at=float(raw["fetched_at"]),
    )


def fetch_rosters(
    cache_dir: Path,
    client: Bo3Client | None = None,
    max_matches: int = DEFAULT_MAX_MATCHES,
    ttl_s: float = ROSTER_TTL_S,
) -> RosterIndex:
    """Announced lineups for upcoming/current matches, cached to disk.

    One API call for the match list plus one per match, so it is paced and
    cached — a full refresh is ~40 polite requests.
    """
    cached = load_cached(cache_dir, ttl_s)
    if cached is not None:
        log.info("roster cache hit (%d players)", len(cached.players))
        return cached

    cli = client or Bo3Client()
    idx = RosterIndex(fetched_at=time.time())
    page: dict[str, Any] = cli._get(  # noqa: SLF001 - internal paced getter
        "/matches",
        {
            "filter[matches.status][in]": "upcoming,current",
            "sort": "start_date",
            "page[limit]": max_matches,
        },
    )
    for match in (page.get("results") or [])[:max_matches]:
        slug = match.get("slug")
        if not slug:
            continue
        try:
            detail: dict[str, Any] = cli._get(  # noqa: SLF001
                f"/matches/{slug}", {"with": "players"}
            )
        except Exception as e:  # a single bad match must not kill the scan
            log.warning("roster fetch failed for %s: %s", slug, e)
            continue
        for p in detail.get("players") or []:
            nick = p.get("nickname")
            if not nick:
                continue
            key = clean_name(nick)
            idx.players.add(key)
            if p.get("team_id") is not None:
                idx.team_of[key] = str(p["team_id"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir).write_text(json.dumps({
        "fetched_at": idx.fetched_at,
        "players": sorted(idx.players),
        "team_of": idx.team_of,
    }))
    log.info("fetched rosters: %d announced players", len(idx.players))
    return idx


def benched_players(
    match_players: list[str], index: RosterIndex
) -> list[str]:
    """Players from ONE board match who are not in the announced lineup.

    Returns [] when the match is not covered by bo3.gg at all — absence of
    evidence is not evidence of a benching, and flagging uncovered matches
    would reject most of a tier-B/C board for no reason.
    """
    if not index.players:
        return []
    known = [p for p in match_players if index.has(p)]
    if not known:
        return []  # match not covered — cannot conclude anything
    return sorted(p for p in match_players if not index.has(p))
