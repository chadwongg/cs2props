"""bo3.gg ingestion tests: pure transforms + pagination cutoff behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from cs2props.ingest import bo3gg

MATCH: dict[str, Any] = {
    "id": 124311,
    "tier": "b",
    "start_date": "2026-07-23T22:00:00.000+00:00",
    "bo_type": 3,
}
GAME: dict[str, Any] = {
    "id": 178092,
    "map_name": "de_dust2",
    "number": 3,
    "begin_at": "2026-07-24T00:09:58.000+00:00",
    "status": "finished",
    "winner_clan_score": 13,
    "loser_clan_score": 6,
}
STAT: dict[str, Any] = {
    "id": 1832085,
    "win": 1,
    "kills": 11,
    "death": 11,
    "assists": 1,
    "headshots": 4,
    "adr": 62.47,
    "player_rating": 5.78,
    "clan_name": "Isurus",
    "enemy_clan_name": "Patins da Ferrari",
    "steam_profile": {"id": 10843, "nickname": "ataraXia", "player_id": 29468},
}


def test_stats_to_rows_maps_fields() -> None:
    rows = bo3gg.stats_to_rows(MATCH, GAME, [STAT])
    assert len(rows) == 1
    r = rows[0]
    assert r.player_id == "29468"
    assert r.player_name == "ataraXia"
    assert r.team == "Isurus"
    assert r.opponent == "Patins da Ferrari"
    assert r.event_tier == "b"
    assert r.map_name == "de_dust2"
    assert r.played_at == "2026-07-24T00:09:58.000+00:00"
    assert (r.kills, r.deaths) == (11, 11)
    assert r.rounds == 19
    assert r.headshots == 4
    assert r.won == 1
    assert r.match_id == "124311"
    assert r.map_number == 3


def test_rounds_none_when_scores_missing() -> None:
    game = dict(GAME, winner_clan_score=None)
    rows = bo3gg.stats_to_rows(MATCH, game, [STAT])
    assert rows[0].rounds is None


def test_stats_to_rows_drops_unattributable() -> None:
    orphan = dict(STAT, steam_profile=None)
    assert bo3gg.stats_to_rows(MATCH, GAME, [orphan]) == []


def test_stats_to_rows_map_number_fallback() -> None:
    game = dict(GAME, number=None, ov_index=1)
    rows = bo3gg.stats_to_rows(MATCH, game, [STAT])
    assert rows[0].map_number == 2


def test_played_at_falls_back_to_match_start() -> None:
    game = dict(GAME, begin_at=None)
    rows = bo3gg.stats_to_rows(MATCH, game, [STAT])
    assert rows[0].played_at == "2026-07-23T22:00:00.000+00:00"


def _match(mid: int, date: str, tier: str = "a") -> dict[str, Any]:
    return {"id": mid, "start_date": date, "tier": tier, "status": "finished"}


def test_iter_matches_stops_at_cutoff_and_filters_tier() -> None:
    pages = [
        [_match(1, "2026-07-20T00:00:00+00:00", "a"),
         _match(2, "2026-07-10T00:00:00+00:00", "c")],   # tier-filtered out
        [_match(3, "2026-06-20T00:00:00+00:00", "s"),
         _match(4, "2025-01-01T00:00:00+00:00", "a")],   # past cutoff -> stop
        [_match(5, "2024-12-01T00:00:00+00:00", "a")],   # never fetched
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("page[offset]", "0"))
        idx = offset // bo3gg.PAGE_SIZE
        calls["n"] += 1
        results = pages[idx] if idx < len(pages) else []
        return httpx.Response(200, json={"results": results})

    client = bo3gg.Bo3Client(delay_s=0.0, transport=httpx.MockTransport(handler))
    since = datetime(2026, 6, 1, tzinfo=timezone.utc)
    got = list(client.iter_finished_matches(since, frozenset({"s", "a", "b"})))
    assert [m["id"] for m in got] == [1, 3]
    assert calls["n"] == 2  # third page never requested


def test_transport_error_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("reset by peer", request=request)
        return httpx.Response(200, json={"results": []})

    client = bo3gg.Bo3Client(delay_s=0.0, transport=httpx.MockTransport(handler))
    client.MAX_TRANSPORT_RETRIES = 3  # type: ignore[misc]
    # patch out the backoff sleep for test speed
    import time as _time

    orig = _time.sleep
    _time.sleep = lambda s: None  # type: ignore[assignment]
    try:
        assert client.fetch_games(1) == []
    finally:
        _time.sleep = orig  # type: ignore[assignment]
    assert calls["n"] == 3


def test_http_error_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    client = bo3gg.Bo3Client(delay_s=0.0, transport=httpx.MockTransport(handler))
    import pytest as _pytest

    with _pytest.raises(httpx.HTTPStatusError):
        client.fetch_games(1)
    assert calls["n"] == 1  # loud, single attempt


def test_fetch_games_keeps_only_finished() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [
                {"id": 1, "status": "finished"},
                {"id": 2, "status": "canceled"},
            ]},
        )

    client = bo3gg.Bo3Client(delay_s=0.0, transport=httpx.MockTransport(handler))
    games = client.fetch_games(124311)
    assert [g["id"] for g in games] == [1]
