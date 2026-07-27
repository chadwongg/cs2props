"""Glue: live board props + replayed history -> per-match simulations.

Groups board props into matches, joins players to history by cleaned name,
builds a :class:`MatchSpec` per match, and runs the correlation engine. The
optimizer (module 4) will consume :class:`MatchSim` objects; until the
calibration gate passes, ``preview`` only reports per-leg marginals and
match diagnostics.

Design decisions:
- Board team abbreviations ("EXR") never match bo3.gg clan names
  ("ex-RUSTEC"), so *players* are the join anchor: each side's history team
  is the majority vote of its matched players' most recent history teams.
- The zero-sum rescale assumes ten players on the server. Board props rarely
  cover all ten, so each side is padded to five with league-average "ghost"
  players who absorb their share of kills but carry no props.
- Series format: Underdog posts maps 1-2 ranges for Bo3 and 1-3 for Bo5 —
  ``bo`` is inferred from the widest range present.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from cs2props.correlation.engine import (
    MatchSpec,
    PlayerSpec,
    PropSpec,
    SimResult,
    simulate_match,
)
from cs2props.ingest.prizepicks import Prop
from cs2props.model.projector import (
    DEFAULT_KPR,
    expected_kpr,
    p_map_win,
    shrunk_hs_rate,
)
from cs2props.model.state_builder import History

log = logging.getLogger(__name__)

SUPPORTED_STATS = {"kills", "headshots"}
DEFAULT_HS_RATE = 0.45
# The book's line is its median estimate and embeds information the model
# cannot see (stand-ins, vetoes, news). Projections are anchored partway
# toward the line; MODEL_WEIGHT is how much we trust ourselves over the
# book. The tracker's live calibration is the instrument for tuning this.
MODEL_WEIGHT = 0.60
EXPECTED_ROUNDS_2MAPS = 43.4


@dataclass
class MatchSim:
    label: str
    start_time: str | None
    props: list[Prop]
    result: SimResult
    p_a_map: float
    side_of: dict[str, str]  # board team name -> "A" | "B"
    matched: int = 0
    unmatched: list[str] = field(default_factory=list)
    benched: list[str] = field(default_factory=list)
    standins: list[str] = field(default_factory=list)


def _start_epoch(start_time: str | None) -> float | None:
    """Board start time -> UTC epoch, for judging a lineup as of kickoff."""
    if not start_time:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(
            str(start_time).replace("Z", "+00:00")
        ).timestamp()
    except (ValueError, TypeError):
        return None


def group_matches(props: list[Prop]) -> dict[tuple[str, ...], list[Prop]]:
    """Group props into matches by unordered team pair + start time."""
    groups: dict[tuple[str, ...], list[Prop]] = defaultdict(list)
    for p in props:
        if not p.team or not p.opponent:
            continue
        key = (*sorted((p.team, p.opponent)), p.start_time or "")
        groups[key].append(p)
    return groups


def _infer_bo(props: list[Prop]) -> int:
    widest = max((p.map_range[1] for p in props if p.map_range), default=2)
    return 5 if widest >= 3 else 3


def build_match_sim(
    props: list[Prop],
    history: History,
    n_iters: int = 50_000,
    seed: int | None = None,
    standins: "object | None" = None,
) -> MatchSim | None:
    """One board match -> simulation. None if neither side matches history.

    ``standins`` (a StandInReport) shades the match for a substitute in
    either lineup: the affected team's map-win probability drops and its
    remaining regulars pick up kill share, both by measured amounts.
    """
    usable = [
        p for p in props
        if p.stat_kind in SUPPORTED_STATS and p.map_range
        and p.board == "standard"  # alt ladders are payout-shaded, not picks
    ]
    if not usable:
        return None
    team_a, team_b = sorted({usable[0].team or "", usable[0].opponent or ""})
    side_of = {team_a: "A", team_b: "B"}

    # join board players to history; collect each side's history-team votes
    joined: dict[str, tuple[str, str | None]] = {}  # board name -> (pid, hist team)
    votes: dict[str, Counter[str]] = {"A": Counter(), "B": Counter()}
    unmatched: list[str] = []
    for name, team in {(p.player_name, p.team or "") for p in usable}:
        hit = history.lookup(name)
        if hit is None:
            unmatched.append(name)
            continue
        pid, hist_team = hit
        joined[name] = (pid, hist_team)
        if hist_team:
            votes[side_of[team]][hist_team] += 1
    if not joined:
        return None
    hist_team_a = votes["A"].most_common(1)[0][0] if votes["A"] else None
    hist_team_b = votes["B"].most_common(1)[0][0] if votes["B"] else None
    opp_state_of = {  # opposing TeamState per side, for expected_kpr
        "A": history.teams.get(hist_team_b or ""),
        "B": history.teams.get(hist_team_a or ""),
    }

    # book lines per player, for line-anchoring
    kills_line: dict[str, float] = {}
    hs_line: dict[str, float] = {}
    for p in usable:
        if p.map_range and p.map_range[0] == 1 and p.map_range[1] == 2:
            if p.stat_kind == "kills":
                kills_line.setdefault(p.player_name, p.line_score)
            elif p.stat_kind == "headshots":
                hs_line.setdefault(p.player_name, p.line_score)

    # player specs for matched board players
    specs: dict[str, PlayerSpec] = {}
    for p in usable:
        name, team = p.player_name, p.team or ""
        if name in specs or name not in joined:
            continue
        pid, _ = joined[name]
        pstate = history.players[pid]
        side = side_of[team]
        kpr = expected_kpr(pstate, opp_state_of[side], history.league)
        hs_rate = shrunk_hs_rate(pstate)  # role-aware, sample-shrunk
        # anchor toward the book's own median estimates
        if name in kills_line:
            model_mean = kpr * EXPECTED_ROUNDS_2MAPS
            blended = (MODEL_WEIGHT * model_mean
                       + (1 - MODEL_WEIGHT) * kills_line[name])
            kpr = blended / EXPECTED_ROUNDS_2MAPS
            if name in hs_line and blended > 1:
                implied_hs = min(hs_line[name] / blended, 0.9)
                hs_rate = (MODEL_WEIGHT * hs_rate
                           + (1 - MODEL_WEIGHT) * implied_hs)
        if standins is not None:
            from cs2props.standins import StandInReport, kpr_factor

            assert isinstance(standins, StandInReport)
            kpr *= kpr_factor(name, standins.for_team(team))
        specs[name] = PlayerSpec(
            player_id=pid,
            name=name,
            side=side,
            kpr=kpr,
            hs_rate=hs_rate,
        )
    # pad each side to five with league-average ghosts (kill-share realism)
    league_kpr = history.league.mean_kpr.value or DEFAULT_KPR
    players = list(specs.values())
    for side in ("A", "B"):
        have = sum(1 for s in players if s.side == side)
        for i in range(max(0, 5 - have)):
            players.append(
                PlayerSpec(f"ghost-{side}{i}", f"ghost-{side}{i}", side, league_kpr)
            )

    sim_props: list[Prop] = []
    prop_specs: list[PropSpec] = []
    for p in usable:
        if p.player_name not in specs or p.map_range is None:
            continue
        sim_props.append(p)
        prop_specs.append(
            PropSpec(
                player_id=specs[p.player_name].player_id,
                stat=p.stat_kind,
                map_lo=p.map_range[0],
                map_hi=p.map_range[1],
                line=p.line_score,
            )
        )
    p_a = p_map_win(
        history.teams.get(hist_team_a or ""), history.teams.get(hist_team_b or "")
    )
    flagged: list[str] = []
    if standins is not None:
        from cs2props.standins import StandInReport, adjust_map_win

        assert isinstance(standins, StandInReport)
        lu_a, lu_b = standins.for_team(team_a), standins.for_team(team_b)
        # Shade each side independently, then renormalise: a stand-in on BOTH
        # teams is close to no relative change, and applying one delta to a
        # shared probability would silently favour whichever team was named
        # first.
        pa = adjust_map_win(p_a, lu_a)
        pb = adjust_map_win(1.0 - p_a, lu_b)
        p_a = pa / (pa + pb) if (pa + pb) > 0 else p_a
        for lu in (lu_a, lu_b):
            if lu is not None and lu.has_standin:
                flagged.append(lu.describe())
    match = MatchSpec(
        bo=_infer_bo(usable),
        p_a_map=p_a,
        players=tuple(players),
        rounds_pool=tuple(history.league.rounds_hist),
    )
    result = simulate_match(match, prop_specs, n_iters=n_iters, seed=seed)
    return MatchSim(
        label=f"{team_a} vs {team_b}",
        start_time=usable[0].start_time,
        props=sim_props,
        result=result,
        p_a_map=p_a,
        side_of=side_of,
        matched=len(specs),
        unmatched=sorted(unmatched),
        standins=flagged,
    )


def simulate_board(
    props: list[Prop],
    history: History,
    n_iters: int = 50_000,
    seed: int | None = None,
    rosters: object | None = None,
    conn: "object | None" = None,
) -> list[MatchSim]:
    """Simulate every joinable match on the board.

    When ``rosters`` (a RosterIndex) is supplied, props on players missing
    from their match's announced lineup are DROPPED: the book voids those
    legs, which silently shrinks a 4-pick to a 3-pick payout.

    When ``conn`` is also supplied, the surviving lineup is checked for
    STAND-INS — a full lineup containing someone who is not one of the
    team's recent regulars. That case is invisible to the benched check and
    distorts every projection in the match, because each state the model
    holds assumes roster continuity.
    """
    import sqlite3

    from cs2props.roster import benched_players

    sims: list[MatchSim] = []
    for key, group in sorted(group_matches(props).items()):
        benched: list[str] = []
        if rosters is not None:
            names = sorted({p.player_name for p in group})
            benched = benched_players(names, rosters)  # type: ignore[arg-type]
            if benched:
                log.info("dropping %d benched player(s) in %s: %s",
                         len(benched), key[:2], ", ".join(benched))
                group = [p for p in group if p.player_name not in benched]
        report = None
        if isinstance(conn, sqlite3.Connection) and group:
            from cs2props.standins import detect

            lineups: dict[str, list[str]] = defaultdict(list)
            for p in group:
                if p.team:
                    lineups[p.team].append(p.player_name)
            as_of = _start_epoch(group[0].start_time)
            if as_of is not None:
                report = detect(conn, {k: sorted(set(v))
                                       for k, v in lineups.items()}, as_of)
        sim = build_match_sim(group, history, n_iters=n_iters, seed=seed,
                              standins=report)
        if sim is None:
            log.info("skipping %s — no players matched in history", key[:2])
            continue
        sim.benched = benched
        sims.append(sim)
    return sims
