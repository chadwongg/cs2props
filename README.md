# cs2props

![ci](https://github.com/chadwongg/cs2props/actions/workflows/ci.yml/badge.svg)

**A stats model for Counter-Strike 2 player props — it prices the bets,
tracks every one I place, and grades its own accuracy.**

Apps like PrizePicks and Underdog let you bet on stat lines for pro CS2
players ("will this player get more or less than 24.5 kills?"). This project
asks one question: **can a model find lines those apps have priced wrong?**

It doesn't just guess. It keeps score on itself — every bet is logged, graded
against real match results, and the model automatically discounts its own
confidence when its live record says it's been too optimistic.

**Most days it says "no bets today."** That's a feature. A tool that always
finds you a bet is selling you action, not an edge.

## What it looks like

The local dashboard: refresh the board and grade finished bets with one
click, see the win/loss record per book, and watch every open bet with its
closing-line value (did the line move in my favor after I bet?).

![dashboard — records, open slips, CLV](docs/dashboard.png)

When the model *does* find something, it shows a card (example data below):
what the bet pays, how often the simulation thinks it hits, and the expected
value — before and after the model's self-applied skepticism discount.

![slip suggestion cards](docs/slipcards.png)

## How it works, in five steps

1. **Learn the players.** A year of pro match history — about 155,000
   individual map performances. Recent games count more than old ones.
2. **Simulate every match 50,000 times.** Not "he averages 25 kills" — it
   plays out the whole game: how long the series goes, who wins each map,
   how many rounds, and how the kills fall. The probability of a bet hitting
   is just how often it hit across 50,000 simulated games.
3. **Respect the bookmaker.** The model blends its own estimate with the
   book's line, because the line contains information the model can't see
   (roster news, sharp money). It also discounts every probability by an
   amount *learned from its own losing bets*.
4. **Price the bet at what the book actually pays.** The apps quietly reduce
   payouts for certain slip structures. Those rules aren't documented — they
   were worked out by building test slips in the app and reading what it
   quoted. The optimizer only suggests slips that get the full payout.
5. **Keep score.** Every bet is graded automatically when matches finish.
   The running hit rate is the whole experiment: the legs need to hit ~54%
   for the math to work, and the record says whether they do.

## What I learned building it

- **A model can be "accurate" and still lose.** The model predicts player
  stats well by standard measures — but the books' lines are built from the
  same public data, so being accurate isn't enough. You have to know
  something the line doesn't. Testing against the books' *real* lines
  (not the model's own practice questions) was the honest test.
- **The payout fine print matters more than the model.** Choosing the wrong
  bet type on the same predictions swings the outcome by 30+ points of
  expected value. The biggest single improvement to this project wasn't a
  smarter model — it was reading the payout rules carefully.
- **Correlation is real but overpriced.** Teammates' kill counts rise and
  fall together (measured: +0.21). The apps know this and cut the payout for
  stacked slips by more than the correlation is worth — so the optimizer
  deliberately spreads bets across different matches.
- **Most bugs look like profits or losses.** A rounding case silently
  counted as a win, a grader once read the wrong match, a query mixed the two
  books' lines. Each produced a confident, wrong conclusion. Each is now a
  regression test.

## Where it stands

The live record is small and not yet profitable. After fixing the bugs above
and switching to the right bet types, the tracking was restarted clean — it
needs a few hundred graded bets before the hit rate means anything. The
interesting part of this repo is the instrument, not the balance.

## Tech

Python 3.11+ · fully type-hinted (`mypy` clean) · 229 tests · SQLite ·
Monte Carlo simulation · walk-forward backtesting · a stdlib-only web
dashboard · `uv` for everything.

```bash
uv run cs2props scan          # fetch boards, simulate, suggest slips
uv run cs2props serve         # local dashboard
uv run cs2props calibrate     # backtest the model against history
uv run cs2props reallines     # backtest against real archived book lines
uv run pytest && uv run mypy cs2props/
```

## Disclaimer

Personal research project. Not affiliated with any sportsbook or data
provider. Nothing here is betting advice — the dashboard itself documents
the model losing money. Scraped data and my own betting records are excluded
from this repository.
