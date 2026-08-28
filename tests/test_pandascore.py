"""PandaScore roster confirmation: overlap resolution, strictness, caching."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cs2props.ingest.pandascore import PandaScoreRosters, confirm_flagged_sims


def _cache(tmp_path: Path, team: str, players: list[str],
           active: bool = True) -> None:
    safe = "".join(c if c.isalnum() else "_" for c in team.lower())
    (tmp_path / f"teams_{safe}.json").write_text(json.dumps({
        "fetched_at": time.time(),
        "teams": [{"name": team,
                   "players": [{"name": p, "active": active}
                               for p in players]}],
    }))


def test_full_lineup_match_confirms(tmp_path: Path) -> None:
    _cache(tmp_path, "M80", ["s1n", "reck", "slaxz", "Lake", "malbsMd"])
    r = PandaScoreRosters(tmp_path)
    # board posts a tag variant; loose matching must absorb it
    assert r.confirm_side("M80", ["nlgs1n", "reck", "slaxz"]) is True


def test_one_unknown_player_blocks_confirmation(tmp_path: Path) -> None:
    """A board player absent from the active roster means NOT confirmed —
    that is exactly the real-substitute case the rule must keep excluded."""
    _cache(tmp_path, "M80", ["s1n", "reck", "slaxz"])
    r = PandaScoreRosters(tmp_path)
    assert r.confirm_side("M80", ["s1n", "somebodyelse"]) is False


def test_wrong_org_with_low_overlap_never_confirms(tmp_path: Path) -> None:
    """Name search can return the wrong org; one coincidental player match
    must not count as resolution (MIN_PLAYER_OVERLAP=2)."""
    _cache(tmp_path, "Color", ["s1n"])
    r = PandaScoreRosters(tmp_path)
    assert r.confirm_side("Color", ["s1n"]) is True  # 1-player side: capped
    _cache(tmp_path, "WW", ["Ct0m", "randomguy", "other"])
    assert r.confirm_side("WW", ["Ct0m", "deko"]) is False


def test_inactive_players_do_not_confirm(tmp_path: Path) -> None:
    _cache(tmp_path, "NRG", ["osee", "br0"], active=False)
    r = PandaScoreRosters(tmp_path)
    assert r.confirm_side("NRG", ["osee", "br0"]) is False


def test_no_token_and_no_cache_confirms_nothing(tmp_path: Path,
                                                monkeypatch: Any) -> None:
    monkeypatch.delenv("PANDASCORE_TOKEN", raising=False)
    monkeypatch.setattr(
        "cs2props.ingest.pandascore._token", lambda: None)
    r = PandaScoreRosters(tmp_path)
    r._token = None
    assert r.confirm_side("FURIA", ["yuurih", "KSCERATO"]) is False


class _FakeSim:
    def __init__(self, standins: list[str], props: list[Any]) -> None:
        self.standins = standins
        self.props = props
        self.label = "A vs B"


class _FakeProp:
    def __init__(self, team: str, name: str) -> None:
        self.team = team
        self.player_name = name


def test_confirm_flagged_sims_clears_only_confirmed(tmp_path: Path) -> None:
    _cache(tmp_path, "M80", ["s1n", "reck"])
    _cache(tmp_path, "FURIA", ["yuurih", "KSCERATO"])
    confirmed = _FakeSim(["M80: STAND-IN s1n"], [
        _FakeProp("M80", "nlgs1n"), _FakeProp("M80", "reck"),
        _FakeProp("FURIA", "yuurih"),
    ])
    unconfirmed = _FakeSim(["X: STAND-IN who"], [_FakeProp("X", "who")])
    unflagged = _FakeSim([], [_FakeProp("M80", "reck")])
    n = confirm_flagged_sims([confirmed, unconfirmed, unflagged], tmp_path)
    assert n == 1
    assert confirmed.standins == []
    assert unconfirmed.standins == ["X: STAND-IN who"]
