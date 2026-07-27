# cs2props — a pricing & backtesting system for CS2 player props

![ci](https://github.com/chadwongg/cs2props/actions/workflows/ci.yml/badge.svg)

A quantitative research project: can a statistical model price Counter-Strike 2
player props (kills / headshots) well enough to beat the payout structures of
DFS pick'em books? The system ingests live boards from two books, simulates
matches, prices every legal slip at what the book will *actually* pay, tracks
every bet placed, and grades its own accuracy against real results.

**This is a measurement instrument, not a money printer.** Its most important
feature is refusal: most days it prints "no slips today," and its live record
is the experiment, not the product.

## Architecture

```
ingest/          three pipelines: two books' APIs (reverse-engineered payloads,
                 rate-limited, disk-cached) + a stats site backfill
                 (~155k player-map rows across ~7.8k pro matches)
model/           per-player exponentially-weighted state (20-map half-life),
                 opponent adjustment, role-aware headshot priors with
                 empirical-Bayes shrinkage; walk-forward backtesting with
                 reliability curves, log loss, PIT histograms
correlation/     generative Monte Carlo over the full match (series length,
                 map winners, round counts, kill shares) — 50k iterations,
                 joint hit matrices rather than products of marginals
optimizer/       exhaustive slip search under each book's real rules:
                 verified payout ladders, structural payout-shift limits,
                 per-side price shades, push (whole-number line) voiding,
                 Kelly-growth ranking for multi-tier ("flex") products
tracker/adaptive live bet log with automatic grading against ingested
                 results, and a per-leg optimism haircut learned by
                 shrinkage from the model's own graded record
server/report    local dashboard: one-click bet tracking, background
                 rescan/grade jobs, per-book records with an explicit
                 legacy-era cutoff
```

## What the research actually found

- **Calibration ≠ edge.** The projector is well calibrated against held-out
  history (log loss 0.6365 vs 0.6927 base rate; reliability within ±1pt across
  90k observations) — but calibration against *synthetic* lines says nothing
  about beating a book's line. A separate backtest against archived **real**
  lines is the test that matters.
- **Payout structure dominates.** The books' ladders differ enough that
  product choice swings EV by 30+ points at the same hit rate. Several payout
  rules are undocumented and were isolated by controlled in-app experiments
  (e.g. the payout shift is cumulative in same-match *pairs*, team-agnostic).
- **Correlation exists but the books charge more than it's worth.** Measured
  teammate kill correlation ≈ +0.21, cross-team ≈ +0.13; the structural
  payout penalty for stacking exceeds the probability gain, so optimal slips
  are diversified — the joint simulator's main job becomes pricing honesty
  rather than stack-hunting.
- **Measurement bugs are the real enemy.** Whole-number pushes silently
  credited as wins (+6.1pt per leg), a grader matching the wrong match on
  double-header days, and a join that mixed two books' lines each produced
  confident, wrong conclusions until found and pinned with regression tests.

## Honest status

Live record to date is small-sample and unprofitable; the current
configuration (correct products, corrected pricing and grading) started a
fresh tracking era and needs a few hundred graded legs before the hit rate
means anything. The repo exists because the *instrument* is the interesting
part.

## Tooling

Python 3.11+, `uv`, fully type-hinted (`mypy` clean), 229 `pytest` tests,
SQLite, stdlib-only web dashboard.

```bash
uv run cs2props scan          # fetch boards, simulate, search, write report
uv run cs2props serve         # local dashboard
uv run cs2props calibrate     # walk-forward backtest, calibration report
uv run cs2props reallines     # backtest vs archived real book lines
uv run cs2props crossbook     # line-disagreement study between books
uv run pytest && uv run mypy cs2props/
```

## Disclaimer

Personal research project. Not affiliated with, endorsed by, or supported by
any sportsbook or data provider. Nothing here is betting advice; the model's
own dashboard documents it losing money. Scraped data and the author's
betting records are deliberately excluded from this repository.
