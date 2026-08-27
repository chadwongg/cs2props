#!/bin/zsh
# cs2props daily automation: top-up history, grade open slips, scan, report.
# Installed as a launchd agent (com.cs2props.daily) — runs 8:00 and 15:00.
set -e
cd /Users/chadwong/predictionmodel
mkdir -p logs
{
  echo "=== daily run $(date '+%Y-%m-%d %H:%M') ==="
  # 1. top-up: newest-first walk stops at already-ingested matches
  /Users/chadwong/.local/bin/uv run cs2props backfill \
    --months 1 --tiers s,a,b,c --delay 1.5 --db cs2props.db
  # 2. grade any slips whose matches finished
  /Users/chadwong/.local/bin/uv run cs2props grade --db cs2props.db
  # 2b. WEEKLY recalibration (Sunday morning run only): the archive grows
  #     daily but calibration.json was frozen at whenever calibrate last
  #     ran — every EV number downstream leans on it staying honest. The
  #     real-line backtest is logged alongside as the out-of-sample check.
  if [ "$(date +%u)" = "7" ] && [ "$(date +%H)" -lt 12 ]; then
    echo "--- weekly recalibration ---"
    /Users/chadwong/.local/bin/uv run cs2props calibrate --db cs2props.db
    /Users/chadwong/.local/bin/uv run cs2props reallines --db cs2props.db
    /Users/chadwong/.local/bin/uv run cs2props calmap --db cs2props.db
  fi
  # 3. scan live boards -> terminal log + cs2report.html
  #    (also snapshots both boards, which is what closing lines are built from)
  /Users/chadwong/.local/bin/uv run cs2props scan
  # 4. closing-line value on tracked legs — the fastest read on real edge
  /Users/chadwong/.local/bin/uv run cs2props clv --db cs2props.db
  echo "=== done $(date '+%H:%M') ==="
} >> logs/daily.log 2>&1
