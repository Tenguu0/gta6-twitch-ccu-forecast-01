# GTA VI Launch-Day Twitch CCU Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Tenguu0/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Tenguu0/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [make.fun](https://make.fun) bounty
**gta6-twitch-ccu-forecast-01** — forecast Grand Theft Auto VI's launch-day
Twitch concurrent viewers hour by hour (hours 2–24 of 2026-11-19 UTC),
given only hour 1's print.

## Model

A nearest-neighbour log-ratio forecast over past big game launches:

1. **p1** is hour 1's viewer print (1 when missing or below 1).
2. Each comparable's release-day curve is rebuilt as 24 hourly viewer
   counts from its `hourly` points. Only comparables with a complete
   24-hour curve and viewers ≥ 1 in every hour are kept — the GTA VI
   category's own pre-release history is incomplete on release day, so
   this rule filters it out.
3. The **12 comparables nearest in log hour-1 size**,
   `|ln(h1) − ln(p1)|` (ties broken by comparable name, for determinism),
   anchor the forecast.
4. Per hour h = 2..24, for each neighbour i, `r_i = ln(viewers_h_i /
   viewers_1_i)`. Centre = median(r), spread = max(0.30, sample stdev of
   r). Then `q_tau = p1 · exp(centre + z_tau · spread)` with
   z = [−1.645, −0.674, 0, 0.674, 1.645] for tau [0.05, 0.25, 0.5, 0.75, 0.95].
5. When no comparable survives the filter, it falls back to the house
   baseline: `q_tau = p1 · exp(z_tau · 0.15 · sqrt(h − 1))`.

Quantiles are rounded to 2 decimals (half up), clamped to (0, 1e9], and
forced non-decreasing within each row.

## Why this shape

Big launches do not stay flat after the first hour: viewers dip in the
early-morning hours as the first time zones go to sleep, then climb
through the evening as the US unlocks and prime time arrives. The
neighbours' median log-ratio captures that diurnal shape from real
launches, while the spread floor (0.30) keeps the 90% band honest
instead of collapsing when neighbours happen to agree.

## Run

Python 3.13, standard library only — no network, no clock, no randomness,
fully deterministic.

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory (in name order) and writes
one `<id>.out` per case: `{"hours": [{"hour": h, "q": [q05, q25, q50,
q75, q95]}, ...]}` for hours 2–24.

## License

MIT — see [LICENSE](LICENSE).
