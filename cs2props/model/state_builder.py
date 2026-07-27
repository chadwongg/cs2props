"""Replay history into final model states + a board-name join index.

Steam nicknames are decorated ("★ ⑳ Hezz †"); board names are clean
("Hezz"). Both sides are normalized to lowercase alphanumerics before
joining, and collisions keep the most recently active player.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from cs2props.model.projector import League, PlayerState, TeamState

log = logging.getLogger(__name__)

_CLEAN_RE = re.compile(r"[^a-z0-9]+")

# Org suffixes/prefixes that bo3.gg uses inconsistently across matches, so
# the same org lands in the DB as several "teams" ("FaZe" vs "FaZe Clan",
# "Liquid" vs "Team Liquid", "DENDELE" vs "DENDELE CS"). Splitting an org's
# history this way starves every team-strength estimate — 48% of aliases had
# under 10 maps before this was applied.
#
# Deliberately NOT stripped: "academy", "talent", "youth", "junior", "nxt",
# "fe", "2" — those denote genuinely DIFFERENT rosters and merging them
# would be worse than the fragmentation it fixes.
_ORG_NOISE = {"esports", "esport", "esports.", "gaming", "clan", "cs", "cs2",
              "csgo", "team"}


def clean_name(name: str) -> str:
    return _CLEAN_RE.sub("", name.lower())


def canonical_team(name: str | None) -> str | None:
    """Collapse an org's name variants to one key.

    "FaZe Clan"/"FaZe" -> "faze";  "Team Liquid"/"Liquid" -> "liquid";
    "MOUZ NXT" stays distinct from "MOUZ".
    """
    if not name:
        return None
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    while tokens and tokens[0] in _ORG_NOISE:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _ORG_NOISE:
        tokens = tokens[:-1]
    return " ".join(tokens) if tokens else name.strip().lower()


@dataclass
class History:
    players: dict[str, PlayerState] = field(
        default_factory=lambda: defaultdict(PlayerState)
    )
    teams: dict[str, TeamState] = field(
        default_factory=lambda: defaultdict(TeamState)
    )
    league: League = field(default_factory=League)
    # clean nickname -> (player_id, last team, last played_at)
    name_index: dict[str, tuple[str, str | None, str]] = field(
        default_factory=dict
    )

    def lookup(self, board_name: str) -> tuple[str, str | None] | None:
        hit = self.name_index.get(clean_name(board_name))
        return (hit[0], hit[1]) if hit else None


def build_history(conn: sqlite3.Connection) -> History:
    """Chronological replay of player_maps into final EW states."""
    rows = conn.execute(
        """
        SELECT player_id, player_name, team, opponent, map_name, played_at,
               kills, rounds, headshots, won
        FROM player_maps
        WHERE rounds IS NOT NULL AND rounds > 10
        ORDER BY played_at
        """
    ).fetchall()
    h = History()
    # team map results must be updated once per (match map, team), not per
    # player row — dedupe on the fly
    seen_map_results: set[tuple[str, str]] = set()
    for pid, pname, team, opp, map_name, played_at, kills, rounds, hs, won in (
        conn.execute(
            """
            SELECT player_id, player_name, team, opponent, map_name,
                   played_at, kills, rounds, headshots, won
            FROM player_maps
            WHERE rounds IS NOT NULL AND rounds > 10
            ORDER BY played_at, match_id, map_number
            """
        )
    ):
        kpr_obs = kills / rounds
        h.players[pid].update(kills, rounds, map_name, headshots=hs)
        h.league.update(kpr_obs, rounds)
        # canonical keys: an org's aliases must share one strength estimate
        c_team, c_opp = canonical_team(team), canonical_team(opp)
        if c_opp:
            h.teams[c_opp].update_allowed(kpr_obs)
        if c_team and won is not None:
            key = (f"{played_at}|{map_name}", c_team)
            if key not in seen_map_results:
                seen_map_results.add(key)
                h.teams[c_team].update_result(bool(won))
        cn = clean_name(pname)
        prev = h.name_index.get(cn)
        if prev is None or played_at >= prev[2]:
            h.name_index[cn] = (pid, c_team, played_at)
    log.info(
        "history: %d players, %d teams, %d name keys (%d rows)",
        len(h.players), len(h.teams), len(h.name_index), len(rows),
    )
    return h
