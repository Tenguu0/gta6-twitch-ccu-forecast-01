"""gta6-twitch-ccu-forecast-01: nearest-neighbour log-ratio forecast.

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2..24 of release day, given hour 1's print and the comparables' launch
curves shipped with each case.

Model (calibrated on the practice cases' comparables):
  - p1 is hour 1's print (1 when missing or below 1).
  - Each comparable's release-day curve is rebuilt as 24 hourly viewers:
    hour h = [day_start + (h-1) hours, +1 hour), taking the hourly point
    whose `from` lands exactly on that hour.
  - Only comparables with a complete 24-hour curve and viewers >= 1 in
    every hour are kept. The GTA VI category's own pre-release history is
    incomplete on release day, so it is filtered out by this rule.
  - Neighbours are the 12 comparables closest in log hour-1 size,
    |ln(h1) - ln(p1)|, ties broken by comparable name for determinism.
  - Per hour h = 2..24, for each neighbour i, r_i = ln(viewers_h_i /
    viewers_1_i). centre = median(r), spread = max(0.30, sample stdev of r).
    q_tau = p1 * exp(centre + z_tau * spread), with z = [-1.645, -0.674,
    0, 0.674, 1.645] for tau [0.05, 0.25, 0.5, 0.75, 0.95].
  - When no comparable survives the filter, fall back to the house
    baseline: q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1)).
  - Quantiles are rounded to 2 decimals, half up (whole numbers written
    as ints), clamped to (0, 1e9], and forced non-decreasing per row.

Python 3.13 standard library only. Deterministic: no random, no clock,
no network; neighbour order is stable and inputs are read in name order.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is one file; kept for parity
# with the kit's layout.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# How many nearest neighbours (in log hour-1 size) anchor each hour.
K_NEIGHBOURS = 12
# Floor on the log spread of the neighbours' hour ratios.
MIN_SPREAD = 0.30
# Fallback baseline's log spread per square-root hour.
BASELINE_SIGMA = 0.15
# Output bounds required by the rules.
Q_MIN = 0.01
Q_MAX = 1_000_000_000


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_instant(text: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp such as 2026-11-19T00:00:00Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def release_day_curve(comp: dict) -> list[float] | None:
    """The comparable's 24 hourly viewer counts on its release day.

    Returns None when any hour is missing or has viewers below 1, so the
    comparable cannot anchor a log-ratio curve.
    """
    day_start = parse_instant(comp["day_start"])
    by_from = {}
    for point in comp.get("hourly", []):
        try:
            viewers = point["viewers"]
        except (KeyError, TypeError):
            continue
        by_from[point["from"]] = viewers
    curve = []
    for h in range(1, 25):
        stamp = (day_start + timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        viewers = by_from.get(stamp)
        if viewers is None or viewers < 1:
            return None
        curve.append(float(viewers))
    return curve


def neighbour_curves(case: dict, p1: float) -> list[list[float]]:
    """The K complete comparables nearest to p1 in log hour-1 size."""
    scored = []
    for comp in case.get("comparables", []):
        curve = release_day_curve(comp)
        if curve is None:
            continue
        distance = abs(math.log(curve[0]) - math.log(p1))
        scored.append((distance, str(comp.get("name", "")), curve))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [curve for _, _, curve in scored[:K_NEIGHBOURS]]


def hour_quantiles(p1: float, h: int, neighbours: list[list[float]]) -> list[float | int]:
    """The five quantiles of hour h's viewers, from the neighbours' ratios."""
    if neighbours:
        ratios = [math.log(curve[h - 1] / curve[0]) for curve in neighbours]
        centre = statistics.median(ratios)
        spread = max(MIN_SPREAD, statistics.stdev(ratios)) if len(ratios) > 1 else MIN_SPREAD
        raw = [p1 * math.exp(centre + z * spread) for z in Z]
    else:
        raw = [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in Z]
    quantiles: list[float | int] = []
    running = Q_MIN
    for value in raw:
        clamped = min(max(value, Q_MIN), Q_MAX)
        rounded = round2(clamped)
        rounded = max(rounded, running)
        running = rounded
        quantiles.append(rounded)
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    p1 = max(1, first["viewers"]) if first is not None else 1
    neighbours = neighbour_curves(case, p1)
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], neighbours)}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
