"""Ingestion tests: parsing semantics + client network policy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from cs2props.ingest import prizepicks as pp

FIXTURE = Path(__file__).parent / "fixtures" / "prizepicks_projections.json"


# ---------------------------------------------------------------- parsing


def fixture_props() -> list[pp.Prop]:
    return pp.load_fixture(FIXTURE)


def test_fixture_parses_expected_count() -> None:
    props = fixture_props()
    # 8 projections, 1 has an unresolvable player -> 7 parsed
    assert len(props) == 7


def test_orphan_projection_skipped() -> None:
    props = fixture_props()
    assert all(p.projection_id != "5301999" for p in props)


def test_player_join() -> None:
    by_id = {p.projection_id: p for p in fixture_props()}
    donk = by_id["5301001"]
    assert donk.player_name == "donk"
    assert donk.team == "Spirit"
    assert donk.opponent == "FaZe"
    assert donk.line_score == 32.5


def test_live_description_maps_suffix_stripped() -> None:
    """Real API descriptions look like 'Keyd Stars MAPS 1-2' — opponent must
    come out clean (regression from first live import 2026-07-24)."""
    payload = {
        "data": [{
            "type": "projection", "id": "x",
            "attributes": {"description": "Keyd Stars MAPS 1-2",
                           "line_score": 25.5, "odds_type": "standard",
                           "stat_type": "MAPS 1-2 Kills"},
            "relationships": {"new_player": {"data": {"id": "p"}},
                              "league": {"data": {"id": "265"}}},
        }],
        "included": [{"type": "new_player", "id": "p",
                      "attributes": {"display_name": "xureba", "team": "KEYD"}}],
    }
    (prop,) = pp.parse_projections(payload)
    assert prop.opponent == "Keyd Stars"


def test_board_types() -> None:
    by_id = {p.projection_id: p for p in fixture_props()}
    assert by_id["5301003"].board == "demon"
    assert by_id["5301004"].board == "goblin"
    assert by_id["5301001"].board == "standard"


def test_demon_goblin_same_player_different_lines() -> None:
    """demon/goblin are alternate lines on the same stat — both must survive."""
    sh1ro = [p for p in fixture_props() if p.player_name == "sh1ro"]
    assert {p.board for p in sh1ro} == {"demon", "goblin"}
    assert {p.line_score for p in sh1ro} == {41.5, 24.5}


@pytest.mark.parametrize(
    ("raw", "kind", "map_range"),
    [
        ("MAPS 1-2 Kills", "kills", (1, 2)),
        ("MAPS 1-2 Headshots", "headshots", (1, 2)),
        ("MAP 3 Kills", "kills", (3, 3)),
        ("Kills", "kills", None),
        ("maps 1-3 Kills", "kills", (1, 3)),
        ("MAPS 1-2  Kills", "kills", (1, 2)),
    ],
)
def test_normalize_stat_type(
    raw: str, kind: str, map_range: tuple[int, int] | None
) -> None:
    assert pp.normalize_stat_type(raw) == (kind, map_range)


def test_map_range_semantics_distinct() -> None:
    """A maps-1-2 prop and a full-series prop must not be conflated."""
    props = fixture_props()
    partial = next(p for p in props if p.projection_id == "5301001")
    full = next(p for p in props if p.projection_id == "5301007")
    assert partial.stat_kind == full.stat_kind == "kills"
    assert partial.map_range == (1, 2)
    assert full.map_range is None


# ---------------------------------------------------------------- client


def _mk_client(
    tmp_path: Path, handler: httpx.MockTransport, ttl: float = 600.0
) -> pp.PrizePicksClient:
    return pp.PrizePicksClient(
        cache_dir=tmp_path / "cache", cache_ttl_s=ttl, transport=handler
    )


def test_leagues_resolution_and_cache(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "data": [
                    {"type": "league", "id": "7", "attributes": {"name": "NBA"}},
                    {"type": "league", "id": "265", "attributes": {"name": "CS2"}},
                ]
            },
        )

    client = _mk_client(tmp_path, httpx.MockTransport(handler))
    assert client.resolve_cs_league_id() == "265"
    # second call served from cache — no new network hit, no rate-limit sleep
    assert client.resolve_cs_league_id() == "265"
    assert calls["n"] == 1


def test_league_not_found(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"type": "league", "id": "7", "attributes": {"name": "NBA"}}]},
        )

    client = _mk_client(tmp_path, httpx.MockTransport(handler))
    with pytest.raises(pp.LeagueNotFound):
        client.resolve_cs_league_id()


def test_cloudflare_403_fails_loudly_no_retry(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="blocked")

    client = _mk_client(tmp_path, httpx.MockTransport(handler))
    with pytest.raises(pp.CloudflareBlocked):
        client.resolve_cs_league_id()
    assert calls["n"] == 1  # exactly one attempt — no retry loop


def test_stale_cache_refetches(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"data": [{"type": "league", "id": "265", "attributes": {"name": "CS2"}}]},
        )

    client = _mk_client(tmp_path, httpx.MockTransport(handler), ttl=0.05)
    client.resolve_cs_league_id()
    time.sleep(0.1)
    # stale cache -> refetch. Zero out the rate-limit stamp so the test
    # doesn't sleep 60s.
    stamp = client.cache_dir / ".last_request"
    import os

    os.utime(stamp, (time.time() - 120, time.time() - 120))
    client.resolve_cs_league_id()
    assert calls["n"] == 2


def test_projections_cache_roundtrip(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _mk_client(tmp_path, httpx.MockTransport(handler))
    raw = client.fetch_projections("265")
    props = pp.parse_projections(raw)
    assert len(props) == 7
