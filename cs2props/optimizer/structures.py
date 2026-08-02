"""Structure-comparison search: price every 4-leg slip SHAPE, side by side.

The default optimizer BANS concentrated structures (max one same-match pair)
because the books shift the payout on them. This module answers a different
question: at each structure's REAL multiplier, which shape carries the most
EV on tonight's board? It scores

    1+1+1+1  four matches                      10x   (in-app 2026-07-24)
    2+1+1    one teammate pair                 10x   (in-app 2026-07-26)
    2+2      two teammate pairs                 8x   (EXTRAPOLATED — see
             payouts.json _structures_note; confirm in-app before betting)
    3+1      teammate triple + single         6.5x   (in-app 2026-07-24)
    4        one match, 2+2 across both teams   5x   (in-app 2026-07-24;
             4 from ONE team is illegal — both books hard-reject
             single-team slips, verified live 2026-07-24)

and reports, for every candidate slip, the two edge sources SEPARATELY:

    line edge          product of per-leg marginals vs the structure's
                       implied per-leg break-even — what the model thinks
                       of the lines, correlation ignored
    correlation lift   P(all 4) minus the independence product — what
                       betting the legs TOGETHER adds (or costs)

Correlation is never estimated pairwise from history (roster samples of
50-80 maps carry a ~0.15 standard error — pure noise) and never hardcoded:
P(all 4) is read off the engine's joint hit matrix, where teammate coupling
EMERGES from shared series length and round counts net of the shared team
kill pool. The 4x4 simulated correlation matrix is logged per slip so the
structure of that coupling is inspectable, slip by slip.

This is an ANALYSIS view: it ignores the held-player exclusions the live
scanner applies, and its void handling is the all-or-nothing power
approximation. Numbers here rank structures; the live scanner still owns
what gets suggested.
"""

from __future__ import annotations

import itertools
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from cs2props.config import Payouts
from cs2props.optimizer.search import Leg, _leg_arrays, slip_price_factor
from cs2props.pipeline import MatchSim

log = logging.getLogger(__name__)

# A leg must clear this marginal to enter the pool — same bar as the live
# scanner, so the two searches disagree only about structure, never about
# which legs are worth considering.
MIN_LEG_P = 0.55
# Widest pool per (match, team): pairs/triples are drawn within these.
TOP_PER_TEAM = 4
# Hard filter, as a multiple of each structure's break-even. The spec's
# "P(all 4) >= 0.11" is exactly 1.1x break-even at the assumed 10x; stated
# as a ratio it survives contact with the REAL multipliers (2+2 breaks even
# at 12.5%, not 10%, so a flat 0.11 would admit -EV slips there).
FILTER_RATIO = 1.1
# Fewer than this many maps of history behind any leg -> LOW confidence.
# 20 matches the projector's own min_history for backtest scoring.
LOW_CONFIDENCE_MAPS = 20


@dataclass(frozen=True)
class StructureSlip:
    structure: str
    multiplier: float
    legs: tuple[Leg, ...]
    p_all: float
    p_independent: float
    corr_matrix: tuple[tuple[float, ...], ...]
    confidence: str  # "OK" | "LOW"

    @property
    def corr_lift(self) -> float:
        return self.p_all - self.p_independent

    @property
    def ev(self) -> float:
        return self.multiplier * self.p_all - 1.0

    @property
    def implied_leg_p(self) -> float:
        """Per-leg probability the payout implies at break-even."""
        return float((1.0 / self.multiplier) ** 0.25)

    def to_json(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "multiplier": self.multiplier,
            "p_all": round(self.p_all, 4),
            "p_independent": round(self.p_independent, 4),
            "corr_lift": round(self.corr_lift, 4),
            "ev": round(self.ev, 4),
            "confidence": self.confidence,
            "corr_matrix": [
                [round(x, 3) for x in row] for row in self.corr_matrix
            ],
            "legs": [
                {
                    "player": l.prop.player_name,
                    "team": l.prop.team,
                    "opponent": l.prop.opponent,
                    "side": l.side,
                    "line": l.prop.line_score,
                    "stat": l.prop.stat_kind,
                    "maps": list(l.prop.map_range or (1, 2)),
                    "model_p": round(l.p, 4),
                    "implied_p": round(self.implied_leg_p, 4),
                    "edge_pts": round((l.p - self.implied_leg_p) * 100, 1),
                }
                for l in self.legs
            ],
        }


def collect_structure_legs(
    sims: list[MatchSim],
    stats: frozenset[str] = frozenset({"kills"}),
    maps: tuple[int, int] = (1, 2),
    min_leg_p: float = MIN_LEG_P,
) -> list[Leg]:
    """Candidate legs, filtered to the requested stat kinds and map range.

    Kills-only / maps 1-2 is the spec default, but both are parameters — the
    board's headshot and single-map markets exist and a structure study that
    silently ignored them would be a config surprise later.
    """
    best: dict[str, Leg] = {}
    for si, sim in enumerate(sims):
        for pi, (prop, p_over) in enumerate(zip(sim.props, sim.result.p_over)):
            if prop.stat_kind not in stats:
                continue
            if (prop.map_range or (1, 2)) != maps:
                continue
            for side, p in (("over", p_over), ("under", 1.0 - p_over)):
                if p < min_leg_p:
                    continue
                if prop.side_multipliers and side not in prop.side_multipliers:
                    continue  # the book does not sell this side
                leg = Leg(si, pi, side, p, prop, prop.team)
                cur = best.get(prop.player_name)
                if cur is None or p > cur.p:
                    best[prop.player_name] = leg
    return sorted(best.values(), key=lambda l: -l.p)


def _team_groups(legs: list[Leg]) -> dict[tuple[int, str], list[Leg]]:
    """(sim, team) -> that team's candidate legs, capped at TOP_PER_TEAM."""
    groups: dict[tuple[int, str], list[Leg]] = {}
    for l in legs:
        if l.team is None:
            continue
        groups.setdefault((l.sim_idx, l.team), []).append(l)
    return {k: v[:TOP_PER_TEAM] for k, v in groups.items()}


def _iter_structures(
    legs: list[Leg],
) -> Iterator[tuple[str, tuple[Leg, ...]]]:
    """Yield (structure_name, legs) for every candidate combination.

    Teammate groups are TEAMMATES by construction (the spec's "2 players
    from one team"), matches never repeat across groups, and the
    4-in-one-match case pairs both teams so the two-team book rule holds.
    """
    groups = _team_groups(legs)
    pairs = {
        k: list(itertools.combinations(v, 2))
        for k, v in groups.items() if len(v) >= 2
    }
    triples = {
        k: list(itertools.combinations(v, 3))
        for k, v in groups.items() if len(v) >= 3
    }
    singles = sorted(legs, key=lambda l: -l.p)

    # 2+2 — teammate pair x teammate pair, different matches
    for (ka, pa), (kb, pb) in itertools.combinations(pairs.items(), 2):
        if ka[0] == kb[0]:
            continue  # same match: that is the "4" structure, priced apart
        for a in pa:
            for b in pb:
                yield "2+2", (*a, *b)

    # 3+1 — teammate triple + a single from another match
    for kt, ts in triples.items():
        for t in ts:
            for s in singles:
                if s.sim_idx == kt[0]:
                    continue
                yield "3+1", (*t, s)

    # 4 — both teams of ONE match, 2+2 across them (single-team is illegal)
    by_match: dict[int, list[tuple[int, str]]] = {}
    for k in pairs:
        by_match.setdefault(k[0], []).append(k)
    for _si, keys in by_match.items():
        for ka, kb in itertools.combinations(keys, 2):
            for a in pairs[ka]:
                for b in pairs[kb]:
                    yield "4", (*a, *b)

    # 2+1+1 and 1+1+1+1 — the shapes the live scanner already allows,
    # included so the comparison is across EVERY structure, not just the
    # concentrated ones
    for kp, ps in pairs.items():
        for pr in ps:
            others = [s for s in singles if s.sim_idx != kp[0]]
            for s1, s2 in itertools.combinations(others[:8], 2):
                if s1.sim_idx == s2.sim_idx:
                    continue
                yield "2+1+1", (*pr, s1, s2)
    for combo in itertools.combinations(singles[:12], 4):
        if len({l.sim_idx for l in combo}) == 4:
            yield "1+1+1+1", combo


def _history_maps(
    conn: sqlite3.Connection | None, player: str, days: int = 60
) -> int | None:
    """Maps this player has in the recent archive; None when unknowable."""
    if conn is None:
        return None
    from cs2props.standins import _same_person

    rows = conn.execute(
        "SELECT player_name, COUNT(*) FROM player_maps"
        " WHERE played_at >= datetime('now', ?) GROUP BY player_name",
        (f"-{days} days",),
    ).fetchall()
    total = sum(int(n) for name, n in rows if _same_person(str(name), player))
    return total


def search_structures(
    sims: list[MatchSim],
    payouts: Payouts,
    conn: sqlite3.Connection | None = None,
    stats: frozenset[str] = frozenset({"kills"}),
    maps: tuple[int, int] = (1, 2),
    min_leg_p: float = MIN_LEG_P,
    filter_ratio: float = FILTER_RATIO,
    top: int = 10,
) -> list[StructureSlip]:
    """Score every structure at its real multiplier; return the top slips.

    Structures without a configured multiplier for this book are SKIPPED,
    never guessed — on Underdog only 1+1+1+1 has a trusted price, so the
    concentrated shapes simply do not appear there.
    """
    table = payouts.structures_4pick
    if not table:
        log.warning("no structures_4pick table for this book — nothing to do")
        return []
    legs = collect_structure_legs(sims, stats, maps, min_leg_p)
    if len(legs) < 4:
        return []

    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {
        id(l): _leg_arrays(sims, l) for l in legs
    }
    maps_cache: dict[str, int | None] = {}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    out: list[StructureSlip] = []

    for structure, combo in _iter_structures(legs):
        mult = table.get(structure)
        if mult is None:
            continue
        names = tuple(sorted(l.prop.player_name for l in combo))
        if len(set(names)) < 4:
            continue
        key = (structure, names)
        if key in seen:
            continue
        seen.add(key)

        mult *= slip_price_factor(list(combo))
        hits = [arrays[id(l)][0] for l in combo]
        ok = hits[0]
        for h in hits[1:]:
            ok = ok & h
        p_all = float(ok.mean())
        if p_all < filter_ratio / mult:
            continue  # under (filter_ratio x) break-even for THIS structure
        p_ind = float(np.prod([l.p for l in combo]))

        wonlive = np.vstack([
            (arrays[id(l)][0] & arrays[id(l)][1]).astype(float)
            for l in combo
        ])
        with np.errstate(invalid="ignore"):
            cm = np.corrcoef(wonlive)
        cm = np.nan_to_num(cm, nan=0.0)

        conf = "OK"
        for l in combo:
            n = maps_cache.setdefault(
                l.prop.player_name,
                _history_maps(conn, l.prop.player_name),
            )
            if n is not None and n < LOW_CONFIDENCE_MAPS:
                conf = "LOW"
                break

        out.append(StructureSlip(
            structure=structure, multiplier=round(mult, 3), legs=combo,
            p_all=p_all, p_independent=p_ind,
            corr_matrix=tuple(tuple(float(x) for x in row) for row in cm),
            confidence=conf,
        ))

    out.sort(key=lambda s: -s.ev)
    return out[:top]


def format_report(slips: list[StructureSlip], book: str) -> str:
    if not slips:
        return (f"no structure on the {book} board clears "
                f"{FILTER_RATIO:.1f}x its break-even")
    lines = [f"=== structure comparison — {book} (top {len(slips)}) ===", ""]
    best_by: dict[str, StructureSlip] = {}
    for s in slips:
        best_by.setdefault(s.structure, s)
    lines.append("best EV per structure:")
    for name, s in sorted(best_by.items(), key=lambda kv: -kv[1].ev):
        lines.append(
            f"  {name:<9} {s.multiplier:>5g}x  P(all) {s.p_all:6.1%}  "
            f"corr lift {s.corr_lift * 100:+5.1f}pt  EV {s.ev * 100:+6.1f}%"
        )
    lines.append("")
    for i, s in enumerate(slips, 1):
        lines.append(
            f"#{i} [{s.structure}] {s.multiplier:g}x  "
            f"P(all4) {s.p_all:.1%}  indep {s.p_independent:.1%}  "
            f"corr lift {s.corr_lift * 100:+.1f}pt  EV {s.ev * 100:+.0f}%  "
            f"[{s.confidence}]"
        )
        for l in s.legs:
            edge = (l.p - s.implied_leg_p) * 100
            lines.append(
                f"    {l.side.upper():5} {l.prop.player_name:<14} "
                f"{l.prop.line_score:>5g} {l.prop.stat_kind:<10} "
                f"{l.prop.team or '?':<14} model {l.p:.0%} vs implied "
                f"{s.implied_leg_p:.0%} ({edge:+.0f}pt)"
            )
    lines.append("")
    lines.append(
        "line edge and corr lift are SEPARATE claims: the first says the "
        "model disagrees with the lines, the second says the legs move "
        "together. A slip that needs the lift to clear break-even is a "
        "correlation bet; one that clears on the independence product is a "
        "model bet with the lift as cushion."
    )
    return "\n".join(lines)


def write_json(slips: list[StructureSlip], book: str, path: str) -> None:
    with open(path, "w") as f:
        json.dump(
            {"book": book, "slips": [s.to_json() for s in slips]},
            f, indent=1,
        )
