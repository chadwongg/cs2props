"""Underdog ingestion tests: stat-key normalization + join semantics."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from cs2props.ingest import underdog as ud

PAYLOAD: dict[str, Any] = {
    "players": [
        {"id": "p1", "sport_id": "CS", "first_name": "", "last_name": "donk"},
        {"id": "p2", "sport_id": "CS", "first_name": "", "last_name": "broky"},
        {"id": "p3", "sport_id": "VAL", "first_name": "", "last_name": "TenZ"},
    ],
    "appearances": [
        {"id": "a1", "player_id": "p1", "match_id": 900, "team_id": "t-spirit"},
        {"id": "a2", "player_id": "p2", "match_id": 900, "team_id": "t-faze"},
        {"id": "a3", "player_id": "p3", "match_id": 901, "team_id": "t-sen"},
    ],
    "games": [
        {
            "id": 900,
            "abbreviated_title": "Spirit vs FaZe",  # CS uses "vs", not "@"
            "away_team_id": "t-spirit",
            "home_team_id": "t-faze",
            "scheduled_at": "2026-07-25T18:00:00Z",
        }
    ],
    "over_under_lines": [
        {
            "id": "l1",
            "status": "active",
            "line_type": "balanced",
            "stat_value": "32.5",
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a1",
                    "stat": "kills_on_maps_1_2",
                    "display_stat": "Kills on Maps 1+2",
                }
            },
            "options": [],
        },
        {
            "id": "l2",
            "status": "active",
            "line_type": "balanced",
            "stat_value": "14.5",
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a2",
                    "stat": "headshots_on_maps_1_2_3",
                    "display_stat": "Headshots on Maps 1-3",
                }
            },
            "options": [],
        },
        {
            "id": "l3",
            "status": "active",
            "line_type": "balanced",
            "stat_value": "28.5",
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a3",
                    "stat": "kills_on_maps_1_2",
                    "display_stat": "Kills on Maps 1+2",
                }
            },
            "options": [],
        },
        {
            "id": "l4",
            "status": "suspended",
            "stat_value": "1.5",
            "over_under": {
                "appearance_stat": {"appearance_id": "a1", "stat": "kills_on_maps_1_2"}
            },
            "options": [],
        },
    ],
}


@pytest.mark.parametrize(
    ("key", "kind", "map_range"),
    [
        ("kills_on_maps_1_2", "kills", (1, 2)),
        ("headshots_on_maps_1_2_3", "headshots", (1, 3)),
        ("kills_on_map_1", "kills", (1, 1)),
        ("first_kills_on_maps_1_2", "first kills", (1, 2)),
        ("kills", "kills", None),
    ],
)
def test_normalize_stat_key(
    key: str, kind: str, map_range: tuple[int, int] | None
) -> None:
    assert ud.normalize_stat_key(key) == (kind, map_range)


def test_parse_filters_to_cs_and_active() -> None:
    props = ud.parse_lines(PAYLOAD)
    ids = {p.projection_id for p in props}
    assert ids == {"l1", "l2"}  # l3 is VAL, l4 suspended


def test_team_and_opponent_from_title() -> None:
    props = {p.projection_id: p for p in ud.parse_lines(PAYLOAD)}
    donk = props["l1"]
    assert donk.player_name == "donk"
    assert (donk.team, donk.opponent) == ("Spirit", "FaZe")  # away side
    broky = props["l2"]
    assert (broky.team, broky.opponent) == ("FaZe", "Spirit")  # home side
    assert broky.map_range == (1, 3)


def test_prop_contract_matches_prizepicks_shape() -> None:
    p = ud.parse_lines(PAYLOAD)[0]
    assert p.stat_kind == "kills"
    assert p.map_range == (1, 2)
    assert p.line_score == 32.5
    assert p.board == "standard"
    assert p.start_time == "2026-07-25T18:00:00Z"


def test_client_caches(tmp_path: Any) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=PAYLOAD)

    client = ud.UnderdogClient(
        cache_dir=tmp_path / "ud", transport=httpx.MockTransport(handler)
    )
    assert len(client.fetch_board()) == 2
    assert len(client.fetch_board()) == 2
    assert calls["n"] == 1


def _one_line_payload(line_type: str, options: list[dict]) -> dict:
    import copy

    pl = copy.deepcopy(PAYLOAD)
    pl["over_under_lines"] = [{
        "id": "lx", "status": "active", "stat_value": "32.5",
        "line_type": line_type,
        "over_under": {"appearance_stat": {
            "appearance_id": "a1", "stat": "kills_on_maps_1_2",
            "display_stat": "Kills on Maps 1+2"}},
        "options": options,
    }]
    return pl


def test_alternate_lines_are_alt_regardless_of_multiplier_size() -> None:
    """The feed labels its alternate ladder rungs; a magnitude threshold let
    a 1.49x rung through as "standard" and the optimizer bet the UNDER of an
    over-only line 7 kills above the real balanced market (2026-07-29)."""
    props = ud.parse_lines(_one_line_payload(
        "alternate", [{"choice": "higher", "payout_multiplier": "1.49"}]))
    assert props and all(p.board == "alt" for p in props)


def test_balanced_lines_stay_standard_even_when_shaded() -> None:
    props = ud.parse_lines(_one_line_payload(
        "balanced",
        [{"choice": "higher", "payout_multiplier": "1.03"},
         {"choice": "lower", "payout_multiplier": "0.82"}]))
    assert props and all(p.board == "standard" for p in props)
    assert props[0].side_multipliers == {"over": 1.03, "under": 0.82}


def test_one_sided_line_never_offers_the_missing_side() -> None:
    """Alternate rungs sell only "higher" — the under of such a line is not
    a market that exists. collect_legs must not manufacture it."""
    from cs2props.optimizer.search import Leg

    props = ud.parse_lines(_one_line_payload(
        "balanced", [{"choice": "higher", "payout_multiplier": "1.0"}]))
    prop = props[0]
    assert prop.side_multipliers == {"over": 1.0}
    assert "under" not in prop.side_multipliers
