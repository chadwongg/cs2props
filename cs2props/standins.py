"""Stand-in detection, and the measured consequences of one playing.

The roster module already answers "is this player in the announced lineup?"
and drops props on players who are not. That is a VOID guard. It says
nothing about the case that actually distorts a projection: the lineup is
full, but one of the five is not the team's usual player.

Every state in the model — a player's KPR, a team's map-win rate, opponent
quality — is estimated from history that assumes roster continuity. A team
fielding a substitute is, for projection purposes, a different team, and the
model prices it as if nothing happened.

MEASURED over 10,396 team-matches in our own history (2026-07-25), using a
60-day/5-map definition of "regular":

  stand-in present in 1,753 team-matches (16.9% -- this is common, not edge
  case)

  team map-win rate      0.490 with a stand-in  vs  0.518 without
                         -2.8 pts, z = 2.7
  REGULARS' kills/round  +1.08% with a stand-in in the lineup, z = 3.2
                         (the substitute absorbs less kill share, so the
                         usual players take more of it)
  STAND-IN's own KPR     +1.1% vs their own career baseline, +/-1.3% --
                         NOT significant. A substitute is not reliably worse
                         than himself; he is just in an unfamiliar system.
                         But his spread is wider (sd 0.350 vs ~0.33), so the
                         honest adjustment is to his VARIANCE, not his mean.

All three effects are small. They are encoded anyway because they are
measured, signed, and free to apply — and because the alternative is
pretending a 17%-frequency roster change has no consequence. What they are
NOT is an edge on their own: the book sees the same announced lineups. The
value here is refusing to price a match whose inputs no longer describe the
teams playing it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from cs2props.model.state_builder import clean_name

log = logging.getLogger(__name__)

REGULAR_WINDOW_DAYS = 60
REGULAR_MIN_MAPS = 5
# A team with too little recent history cannot be judged at all — calling
# every player a "stand-in" because the team is simply new to our data would
# flag most of a tier-C board for no reason.
TEAM_MIN_MAPS = 25
MIN_REGULARS = 4

# --- measured effects (see module docstring) ---
STANDIN_MAP_WIN_DELTA = -0.028  # team wins fewer maps
REGULAR_KPR_FACTOR = 1.0108  # remaining regulars absorb kill share
STANDIN_KPR_SD_RATIO = 0.350 / 0.33  # substitute is more variable, not worse


@dataclass(frozen=True)
class TeamLineup:
    """What is announced for one team, against what it usually fields."""

    team: str
    standins: tuple[str, ...] = ()
    missing_regulars: tuple[str, ...] = ()
    regulars: tuple[str, ...] = ()
    judged: bool = True  # False when history is too thin to conclude

    @property
    def has_standin(self) -> bool:
        return bool(self.standins)

    def describe(self) -> str:
        if not self.judged:
            return f"{self.team}: too little history to judge the lineup"
        if not self.standins:
            return f"{self.team}: usual roster"
        out = f"{self.team}: STAND-IN {', '.join(self.standins)}"
        if self.missing_regulars:
            out += f" (out: {', '.join(self.missing_regulars)})"
        return out


@dataclass
class StandInReport:
    lineups: dict[str, TeamLineup] = field(default_factory=dict)

    def for_team(self, team: str) -> TeamLineup | None:
        return self.lineups.get(team)

    def teams_with_standins(self) -> list[str]:
        return sorted(t for t, v in self.lineups.items() if v.has_standin)


# Board and bo3.gg spell leetspeak nicknames differently — the first live
# run flagged sh1ro/sh1r0 and zont1x/zontix as stand-ins, and kursy/kursyssj4
# as a substitution, when all three are the same player under a different
# rendering. A false stand-in is the WORSE error: a missed one leaves the
# model where it already was, while a phantom one shades a roster that never
# changed. Matching is therefore deliberately generous.
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                       "5": "s", "7": "t", "8": "b", "9": "g"})
MIN_PREFIX_MATCH = 4


def _loose_key(name: str) -> str:
    return clean_name(name).translate(_LEET)


def _same_person(a: str, b: str) -> bool:
    """Are two nicknames plausibly the same player?

    Leet-normalised equality, or one being a prefix of the other (players
    append tags: kursy -> kursyssj4). Deliberately loose in the direction
    that SUPPRESSES stand-in flags.
    """
    ka, kb = _loose_key(a), _loose_key(b)
    if ka == kb:
        return True
    if len(ka) >= MIN_PREFIX_MATCH and len(kb) >= MIN_PREFIX_MATCH:
        return ka.startswith(kb) or kb.startswith(ka)
    return False


def _epoch(ts: str) -> float | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def team_regulars(
    conn: sqlite3.Connection,
    team: str,
    as_of: float,
    window_days: int = REGULAR_WINDOW_DAYS,
    min_maps: int = REGULAR_MIN_MAPS,
) -> tuple[set[str], int]:
    """-> (regular player names, total map-rows the team has in the window).

    Strictly BEFORE ``as_of``: a lineup must be judged on what was known
    before the match, or the "regulars" set is contaminated by the very
    match being assessed.
    """
    lo = as_of - window_days * 86400.0
    rows = conn.execute(
        """
        SELECT player_name, played_at FROM player_maps
        WHERE team = ? AND played_at IS NOT NULL
        """,
        (team,),
    ).fetchall()
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for name, at in rows:
        ts = _epoch(str(at))
        if ts is None or not (lo <= ts < as_of):
            continue
        counts[clean_name(name)] += 1
        total += 1
    return {p for p, c in counts.items() if c >= min_maps}, total


def detect(
    conn: sqlite3.Connection,
    team_lineups: dict[str, list[str]],
    as_of: float,
    window_days: int = REGULAR_WINDOW_DAYS,
    min_maps: int = REGULAR_MIN_MAPS,
) -> StandInReport:
    """Compare each team's announced lineup against its recent regulars.

    ``team_lineups`` maps a team name to the players announced for it. Teams
    whose own history is too thin are returned unjudged rather than guessed
    at — a false stand-in flag is worse than no flag, because it would move
    projections on a roster that never changed.
    """
    report = StandInReport()
    for team, players in team_lineups.items():
        regulars, total = team_regulars(conn, team, as_of, window_days, min_maps)
        if total < TEAM_MIN_MAPS or len(regulars) < MIN_REGULARS:
            report.lineups[team] = TeamLineup(
                team=team, judged=False, regulars=tuple(sorted(regulars))
            )
            continue
        announced = {clean_name(p) for p in players}
        standins = {
            a for a in announced
            if not any(_same_person(a, r) for r in regulars)
        }
        # Only call someone missing when the lineup looks complete enough to
        # tell; a partial board listing is not a benching.
        missing = (
            {r for r in regulars
             if not any(_same_person(a, r) for a in announced)}
            if len(announced) >= MIN_REGULARS else set()
        )
        report.lineups[team] = TeamLineup(
            team=team,
            standins=tuple(sorted(standins)),
            missing_regulars=tuple(sorted(missing)),
            regulars=tuple(sorted(regulars)),
        )
        if standins:
            log.info(
                "stand-in detected for %s: %s (regulars out: %s)",
                team, ", ".join(sorted(standins)),
                ", ".join(sorted(missing)) or "none identified",
            )
    return report


def adjust_map_win(p_win: float, lineup: TeamLineup | None) -> float:
    """Shade a team's map-win probability for a stand-in.

    Applied to the probability directly rather than to the underlying rate:
    the effect was measured as a difference in realised map-win rate, so
    that is the scale it belongs on.
    """
    if lineup is None or not lineup.judged or not lineup.has_standin:
        return p_win
    return min(max(p_win + STANDIN_MAP_WIN_DELTA, 0.02), 0.98)


def kpr_factor(player_name: str, lineup: TeamLineup | None) -> float:
    """Kill-share adjustment for one player given his team's lineup.

    Regulars playing alongside a substitute took +1.08% more kills per round
    in our history. The substitute himself gets NO mean adjustment: measured
    at +1.1% +/- 1.3%, which is indistinguishable from zero, and inventing a
    penalty for him would be modelling a prejudice rather than a finding.
    """
    if lineup is None or not lineup.judged or not lineup.has_standin:
        return 1.0
    if any(_same_person(player_name, s) for s in lineup.standins):
        return 1.0
    return REGULAR_KPR_FACTOR
