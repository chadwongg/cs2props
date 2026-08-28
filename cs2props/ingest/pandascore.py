"""PandaScore roster confirmation: a second, independent lineup source.

The stand-in detector compares the board's names against bo3.gg history and
deliberately errs toward flagging — but a flag built on one source inherits
that source's name spellings, and CS2 players re-tag constantly ("vision"
posts as "ozonevvision", "s1n" as "nlgs1n"). Through August's tier-C slate
most boards had a majority of matches flagged, many of them exactly these
variant-name false positives, and the no-bet rule starved the scanner.

This module asks PandaScore (free tier, token on disk) for each team's
ACTIVE roster and re-checks flagged matches: when every board player on
both teams matches an active-roster player, the match is CONFIRMED clean
and becomes bettable again. Confirmation can only ever act on matches the
detector flagged — an unflagged match needs no help, and a flagged match
that PandaScore cannot confirm stays excluded. Two sources agreeing is the
green light; one source's silence never is.

Team resolution is by PLAYER OVERLAP, not team-name search alone: board
team names are abbreviations ("WW", "NEW") and name search alone would
happily return the wrong org. A candidate team counts only when at least
two board players match its roster.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from cs2props.standins import MIN_PREFIX_MATCH, _loose_key, _same_person


def _same_player(a: str, b: str) -> bool:
    """Looser than the detector's ``_same_person``: suffix tags count too.

    Players prepend tags as often as they append them ("nlg"+"s1n",
    "ozonev"+"vision") — the detector stays prefix-only because looseness
    there suppresses flags, but HERE extra looseness is safe: confirmation
    additionally requires resolving the right org by 2-player overlap and
    matching the FULL lineup, so a coincidental suffix can't flip a match
    on its own."""
    if _same_person(a, b):
        return True
    ka, kb = _loose_key(a), _loose_key(b)
    # 3, not the detector's 4: short tags ("s1n" -> "sin") are common and
    # the extra guards above make a 3-char coincidence harmless
    if min(len(ka), len(kb)) >= 3:
        return (ka.endswith(kb) or kb.endswith(ka)
                or ka.startswith(kb) or kb.startswith(ka))
    return False

log = logging.getLogger(__name__)

BASE = "https://api.pandascore.co/csgo/teams"
TOKEN_PATH = Path(".pandascore_token")
CACHE_TTL_S = 12 * 3600.0
# a resolved team must share at least this many players with the board side
MIN_PLAYER_OVERLAP = 2


def _token() -> str | None:
    import os

    env = os.environ.get("PANDASCORE_TOKEN")
    if env:
        return env.strip()
    try:
        return TOKEN_PATH.read_text().strip()
    except OSError:
        return None


class PandaScoreRosters:
    """Fetch-and-cache active rosters, keyed by board team name."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._token = _token()

    def _cache_path(self, team: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in team.lower())
        return self.cache_dir / f"teams_{safe}.json"

    def _search(self, team: str) -> list[dict[str, Any]]:
        """Candidate PandaScore teams for a board team name (cached)."""
        path = self._cache_path(team)
        try:
            raw = json.loads(path.read_text())
            if time.time() - raw["fetched_at"] < CACHE_TTL_S:
                return list(raw["teams"])
        except (OSError, ValueError, KeyError):
            pass
        if not self._token:
            return []
        try:
            r = httpx.get(
                BASE,
                params={"search[name]": team, "per_page": 10},
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=15.0,
            )
            r.raise_for_status()
            teams = r.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("pandascore search %r failed: %s", team, e)
            return []
        path.write_text(json.dumps(
            {"fetched_at": time.time(), "teams": teams}
        ))
        return list(teams)

    def confirm_side(self, team: str, board_players: list[str]) -> bool:
        """Is every board player on this team's ACTIVE PandaScore roster?

        False means "could not confirm", never "confirmed dirty" — the
        caller treats unconfirmed exactly as it treated the match before
        this module existed.
        """
        for cand in self._search(team):
            roster = [str(p.get("name", "")) for p in cand.get("players", [])
                      if p.get("active", True)]
            if not roster:
                continue
            matched = [b for b in board_players
                       if any(_same_player(b, r) for r in roster)]
            if len(matched) < min(MIN_PLAYER_OVERLAP, len(board_players)):
                continue  # wrong org, or roster too different
            if len(matched) == len(board_players):
                return True
        return False


def confirm_flagged_sims(sims: list[Any], cache_dir: Path) -> int:
    """Clear stand-in flags on sims whose FULL lineups PandaScore confirms.

    Mutates ``sim.standins`` in place (that is the field the optimizer's
    no-bet rule reads). Returns how many sims were confirmed clean.
    """
    rosters = PandaScoreRosters(cache_dir)
    cleared = 0
    for sim in sims:
        if not sim.standins:
            continue
        by_team: dict[str, list[str]] = {}
        for p in sim.props:
            if p.team:
                by_team.setdefault(p.team, []).append(p.player_name)
        if not by_team:
            continue
        if all(rosters.confirm_side(team, sorted(set(players)))
               for team, players in by_team.items()):
            log.info("roster CONFIRMED clean via pandascore: %s "
                     "(was: %s)", sim.label, "; ".join(sim.standins))
            sim.standins.clear()
            cleared += 1
    return cleared
