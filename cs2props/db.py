"""SQLite persistence: parsed props snapshots + (later) historical stats."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from cs2props.ingest.bo3gg import PlayerMapRow
from cs2props.ingest.prizepicks import Prop

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS props (
    scanned_at     REAL NOT NULL,
    projection_id  TEXT NOT NULL,
    player_id      TEXT NOT NULL,
    player_name    TEXT NOT NULL,
    team           TEXT,
    opponent       TEXT,
    stat_type      TEXT NOT NULL,
    stat_kind      TEXT NOT NULL,
    map_lo         INTEGER,
    map_hi         INTEGER,
    line_score     REAL NOT NULL,
    board          TEXT NOT NULL,
    start_time     TEXT,
    league_id      TEXT NOT NULL,
    PRIMARY KEY (scanned_at, projection_id)
);

-- Filled by the bo3.gg history ingester (module 1b).
CREATE TABLE IF NOT EXISTS player_maps (
    player_id   TEXT NOT NULL,
    player_name TEXT NOT NULL,
    team        TEXT,
    opponent    TEXT,
    event_tier  TEXT,           -- bo3.gg tiers: s / a / b / c
    map_name    TEXT,
    played_at   TEXT NOT NULL,
    kills       INTEGER NOT NULL,
    deaths      INTEGER NOT NULL,
    adr         REAL,
    rating      REAL,
    rounds      INTEGER,        -- total rounds on the map (winner+loser score)
    headshots   INTEGER,
    won         INTEGER,        -- 1 if this player's team took the map
    match_id    TEXT NOT NULL,
    map_number  INTEGER NOT NULL,
    PRIMARY KEY (match_id, map_number, player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_maps_player
    ON player_maps (player_name, played_at);

-- Backfill bookkeeping: a match listed here is fully ingested (resume point).
CREATE TABLE IF NOT EXISTS ingested_matches (
    match_id    TEXT PRIMARY KEY,
    ingested_at REAL NOT NULL,
    n_maps      INTEGER NOT NULL,
    tier        TEXT,
    start_date  TEXT
);

-- Module 5: slips the user actually placed, graded against player_maps.
CREATE TABLE IF NOT EXISTS slips (
    slip_id    TEXT PRIMARY KEY,
    book       TEXT NOT NULL,
    placed_at  REAL NOT NULL,
    stake      REAL NOT NULL,
    n_legs     INTEGER NOT NULL,
    claimed_p  REAL,               -- model's P(win) at placement, if known
    multiplier REAL,               -- multiplier the APP showed at placement;
                                   -- Arena prices per-pick and is not derivable
                                   -- from the API, so it must be recorded, not
                                   -- recomputed, or P&L grades at a fake price
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending|won|lost
    payout     REAL
);

CREATE TABLE IF NOT EXISTS slip_legs (
    slip_id     TEXT NOT NULL,
    leg_no      INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    side        TEXT NOT NULL,     -- over|under
    line        REAL NOT NULL,
    stat_kind   TEXT NOT NULL,     -- kills|headshots
    map_lo      INTEGER NOT NULL,
    map_hi      INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|won|lost|void
    observed    REAL,
    PRIMARY KEY (slip_id, leg_no)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    _migrate(conn)
    _add_product_column(conn)
    return conn


def _add_product_column(conn: "sqlite3.Connection") -> None:
    """slips.product — power slips settle on the first lost leg, flex do not."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(slips)")}
    if "product" not in cols:
        conn.execute(
            "ALTER TABLE slips ADD COLUMN product TEXT NOT NULL "
            "DEFAULT 'power'"
        )
        conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Schema drift: if player_maps predates the headshots/won columns, drop
    it and clear the ingest bookmarks so the backfill refetches those
    matches with full stat lines."""
    scols = {r[1] for r in conn.execute("PRAGMA table_info(slips)")}
    if scols and "multiplier" not in scols:
        conn.execute("ALTER TABLE slips ADD COLUMN multiplier REAL")
        conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(player_maps)")}
    if "headshots" not in cols:
        log.warning("migrating player_maps: adding headshots/won via rebuild")
        conn.executescript(
            "DROP TABLE player_maps; DELETE FROM ingested_matches;"
        )
        conn.executescript(SCHEMA)
        conn.commit()


def save_props(conn: sqlite3.Connection, props: list[Prop]) -> float:
    """Persist a board snapshot; returns the snapshot timestamp."""
    ts = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO props VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                ts,
                p.projection_id,
                p.player_id,
                p.player_name,
                p.team,
                p.opponent,
                p.stat_type,
                p.stat_kind,
                p.map_range[0] if p.map_range else None,
                p.map_range[1] if p.map_range else None,
                p.line_score,
                p.board,
                p.start_time,
                p.league_id,
            )
            for p in props
        ],
    )
    conn.commit()
    log.info("saved %d props at snapshot %.0f", len(props), ts)
    return ts


def is_match_ingested(conn: sqlite3.Connection, match_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM ingested_matches WHERE match_id = ?", (match_id,)
    ).fetchone()
    return row is not None


def save_match_maps(
    conn: sqlite3.Connection,
    match_id: str,
    rows: list[PlayerMapRow],
    tier: str | None,
    start_date: str | None,
    n_maps: int,
) -> None:
    """Persist one match's player-map rows and mark it ingested (one txn)."""
    conn.executemany(
        "INSERT OR REPLACE INTO player_maps "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r.player_id,
                r.player_name,
                r.team,
                r.opponent,
                r.event_tier,
                r.map_name,
                r.played_at,
                r.kills,
                r.deaths,
                r.adr,
                r.rating,
                r.rounds,
                r.headshots,
                r.won,
                r.match_id,
                r.map_number,
            )
            for r in rows
        ],
    )
    conn.execute(
        "INSERT OR REPLACE INTO ingested_matches VALUES (?,?,?,?,?)",
        (match_id, time.time(), n_maps, tier, start_date),
    )
    conn.commit()


def history_summary(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """(matches ingested, map-rows, distinct players) for progress reporting."""
    matches = conn.execute("SELECT COUNT(*) FROM ingested_matches").fetchone()[0]
    rows = conn.execute("SELECT COUNT(*) FROM player_maps").fetchone()[0]
    players = conn.execute(
        "SELECT COUNT(DISTINCT player_id) FROM player_maps"
    ).fetchone()[0]
    return int(matches), int(rows), int(players)
