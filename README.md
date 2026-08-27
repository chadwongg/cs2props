# cs2props

![ci](https://github.com/chadwongg/cs2props/actions/workflows/ci.yml/badge.svg)

**A model that prices Counter-Strike 2 player props, tracks every bet I
place, and grades its own accuracy.**

Apps like PrizePicks and Underdog let you bet on stat lines for pro CS2
players ("will this player get more or less than 24.5 kills?"). This
project tests one question: can a model find lines those apps priced
wrong?

Every bet is logged and graded against real match results. Every
prediction is saved before the match, so the model can be checked against
what actually happened. Most days it finds no bet worth taking, and it
says so.

## What it looks like

The dashboard shows profit/loss, the record per betting app, how often
picks actually hit, and open bets. One click refreshes the board or grades
finished bets.

![dashboard — stats overview, records, open slips](docs/dashboard.png)

When the model finds a bet, it shows a card: what it pays, the win
probability, and the expected value. The ✓ button logs the bet. The amber
box shows the model refusing to add a 4th pick because none was good
enough.

![slip suggestion card](docs/slipcards.png)

## How it works

1. **Learn the players.** About 170,000 map performances from a year of
   pro matches. Recent games count more.
2. **Simulate each match 50,000 times.** Series length, map winners,
   round counts, and kills are all simulated together. A bet's probability
   is how often it hit across those simulations.
3. **Adjust for what the model can't know.** The model's estimate is
   blended with the app's line (the line contains roster news and sharp
   money). Then every probability is corrected using the model's own
   graded history: "when the model says 65%, what actually happens?"
4. **Price the bet at what the app actually pays.** The apps quietly cut
   payouts for certain pick combinations — four teammates on one slip
   turns "10x" into 5x. These rules aren't documented, so they were
   worked out by building test slips in the apps and reading the quotes.
5. **Keep score.** Bets grade automatically when matches finish. The
   record on the dashboard is the experiment.

## What I found

**The apps price correlation almost perfectly.** Teammates' kills rise
and fall together, so stacking teammates wins more often — and the apps
cut the payout by almost exactly what that boost is worth. One exception:
a single teammate pair boosts the win chance ~16% and is barely charged
for. So every suggested slip is one teammate pair plus two picks from
other matches. That was the only combination worth building.

**Unders beat overs.** Over 5,490 real lines, the model's confident OVER
picks hit 48.9% — worse than a coin flip. Its UNDER picks hit 55.9%. The
apps set lines slightly high because casual bettors like clicking OVER.
The model now only bets unders.

**The model's confidence was empty.** This was the biggest finding. After
a month of tracked bets, legs the model claimed at ~68% were hitting 41%.
Checking 5,490 archived lines showed why:

| model claimed | actually hit |
|---|---|
| 55–60% | 51% |
| 60–65% | 58% |
| 65–70% | 51% |
| 70%+ | 58% |

Whether the model claimed 57% or 72%, the pick hit about 54%. The extra
confidence meant nothing, and the optimizer was selecting bets based on
exactly that. Three changes fixed it:

1. Every probability now goes through a calibration table built from the
   model's own graded history (a raw 65% becomes 54%, a raw 79% becomes
   57%). It refits weekly and only ever lowers a claim, never raises it.
2. Matches with substitute players are no longer bet. The model can shade
   its numbers for a sub, but the app has roster information the model
   doesn't. Those matches lost the most.
3. A side experiment (betting against whichever app posted the higher
   line when the two disagreed) was shut down. The kill rule was written
   in advance: if the edge measured under 5 points at 400 settled cases,
   drop it. The edge went from +15.6 points at 57 cases to +0.0 at 1,037.
   It was noise, and it was dropped on schedule.

After these fixes, most bets that used to look good price out near
breakeven, and the scanner says "no bets today" much more often. That's
the correct output — the earlier version was just wrong about its edge.

**Bugs look like results.** A rounding case counted as a win, the grader
once read the wrong match, a query mixed the two apps' lines. Each
produced a confident wrong conclusion. Each is now a regression test.

## Where it stands

The live record is small and not profitable, and the dashboard says so.
What the project demonstrates is the process: save predictions before the
outcome, grade against reality, measure your own overconfidence, and shut
down ideas by rules written in advance.

## Tech

Python 3.11+ · fully type-hinted (`mypy` clean) · 263 tests · SQLite ·
Monte Carlo simulation · walk-forward backtesting · a stdlib-only web
dashboard · `uv` for everything.

```bash
uv run cs2props scan          # fetch boards, simulate, suggest slips
uv run cs2props serve         # local dashboard
uv run cs2props calibrate     # backtest the model against history
uv run cs2props reallines     # backtest against real archived book lines
uv run cs2props calmap        # refit the probability calibration table
uv run pytest && uv run mypy cs2props/
```

## Disclaimer

Personal research project. Not affiliated with any sportsbook or data
provider. Nothing here is betting advice — the dashboard itself documents
the model losing money. Scraped data and my own betting records are
excluded from this repository.
