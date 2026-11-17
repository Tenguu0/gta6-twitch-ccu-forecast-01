"""gta6-twitch-ccu-forecast-01: local-linear log-ratio forecast (v2).

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2..24 of release day, given hour 1's print and the comparables' launch
curves shipped with each case.

Model (v2; local-linear log-ratio, validated by leave-one-out on the 37
comparable curves, beating v1 in every hour band):
  - p1 is hour 1's print (1 when missing or below 1).
  - Each comparable's release-day curve is rebuilt as 24 hourly viewers:
    hour h = [day_start + (h-1) hours, +1 hour), taking the hourly point
    whose `from` lands exactly on that hour.
  - Only comparables with a complete 24-hour curve and viewers >= 1 in
    every hour are kept. The GTA VI category's own pre-release history is
    incomplete on release day, so it is filtered out by this rule.
  - Per hour h = 2..24, for each comparable i, x_i = ln(h1_i) - ln(p1)
    and r_i = ln(viewers_h_i / viewers_1_i). Gaussian weight
    w_i = exp(-0.5 * (x_i / 1.0) ** 2) (bandwidth 1.0).
  - Weighted linear regression of r on x: (c, b) minimising
    sum w_i * (r_i - (c + b * x_i)) ** 2, with a small ridge 1e-3 * sw
    added to the determinant (sw = sum w). centre = c (the prediction at
    x = 0, i.e. at p1); spread = max(0.20, sqrt(weighted residual
    variance)). When sw == 0 or the determinant is ~0, centre is the
    weighted mean of r and spread is 0.20.
  - Guard: when the effective sample size sw ** 2 / sum w_i ** 2 is below
    3, fall back to v1 for that hour: the 12 nearest neighbours by
    |ln(h1) - ln(p1)| (ties broken by comparable name), centre = median
    of r, spread = max(0.20, sample stdev of r).
  - When no comparable survives the filter, fall back to the house
    baseline: q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1)).
  - q_tau = p1 * exp(centre + z_tau * spread), with z = [-1.645, -0.674,
    0, 0.674, 1.645] for tau [0.05, 0.25, 0.5, 0.75, 0.95].
  - Quantiles are rounded to 2 decimals, half up (whole numbers written
    as ints), clamped to (0, 1e9], and forced non-decreasing per row.

Python 3.13 standard library only. Deterministic: no random, no clock,
no network; comparables are sorted by name before the regression sums so
floating-point addition is stable, and inputs are read in name order.
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
# Gaussian kernel bandwidth on log hour-1 size.
BANDWIDTH = 1.0
# Ridge added to the regression determinant, as a fraction of sw.
RIDGE = 1e-3
# Minimum effective sample size before the local-linear fit is trusted.
MIN_ESS = 3
# How many nearest neighbours anchor a fallback hour (v1 rule).
K_NEIGHBOURS = 12
# Floor on the log spread of the hour ratios.
MIN_SPREAD = 0.20
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


def filtered_comparables(case: dict) -> list[tuple[str, list[float]]]:
    """The (name, curve) pairs that survive the complete-curve filter,
    sorted by name so every downstream sum is deterministic."""
    kept = []
    for comp in case.get("comparables", []):
        curve = release_day_curve(comp)
        if curve is None:
            continue
        kept.append((str(comp.get("name", "")), curve))
    kept.sort(key=lambda item: item[0])
    return kept


def local_linear(p1: float, h: int, comps: list[tuple[str, list[float]]]) -> tuple[float, float]:
    """centre, spread for hour h from the Gaussian-weighted local-linear fit.

    Assumes comps is non-empty and sorted by name. Falls back to the
    weighted mean of the log ratios (spread 0.20) when the fit is degenerate.
    """
    log_p1 = math.log(p1)
    xs = []
    rs = []
    ws = []
    for _name, curve in comps:
        x = math.log(curve[0]) - log_p1
        r = math.log(curve[h - 1] / curve[0])
        xs.append(x)
        rs.append(r)
        ws.append(math.exp(-0.5 * (x / BANDWIDTH) ** 2))
    sw = sum(ws)
    if sw == 0:
        return sum(rs) / len(rs), MIN_SPREAD
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * r for w, r in zip(ws, rs))
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * r for w, x, r in zip(ws, xs, rs))
    det = sw * sxx - sx * sx
    # Degenerate design (all x ~equal): the slope is unidentified, so the
    # weighted mean is the honest centre.
    if det <= 1e-12 * sw * max(sxx, 1.0):
        return sy / sw, MIN_SPREAD
    det_ridge = det + RIDGE * sw
    centre = (sy * sxx - sx * sxy) / det_ridge
    slope = (sw * sxy - sx * sy) / det_ridge
    resid2 = sum(w * (r - (centre + slope * x)) ** 2 for w, x, r in zip(ws, xs, rs))
    spread = max(MIN_SPREAD, math.sqrt(resid2 / sw))
    return centre, spread


def effective_sample_size(p1: float, comps: list[tuple[str, list[float]]]) -> float:
    """sw ** 2 / sum w_i ** 2 for the Gaussian weights at this p1."""
    log_p1 = math.log(p1)
    ws = [math.exp(-0.5 * ((math.log(curve[0]) - log_p1) / BANDWIDTH) ** 2)
          for _name, curve in comps]
    sw = sum(ws)
    sww = sum(w * w for w in ws)
    return sw * sw / sww if sww > 0 else 0.0


def v1_fallback(p1: float, h: int, comps: list[tuple[str, list[float]]]) -> tuple[float, float]:
    """v1's nearest-neighbour rule: 12 closest in log hour-1 size, ties by
    name; centre = median log ratio, spread = max(0.20, sample stdev)."""
    log_p1 = math.log(p1)
    scored = sorted(
        ((abs(math.log(curve[0]) - log_p1), name, curve) for name, curve in comps),
        key=lambda item: (item[0], item[1]),
    )
    neighbours = [curve for _, _, curve in scored[:K_NEIGHBOURS]]
    ratios = [math.log(curve[h - 1] / curve[0]) for curve in neighbours]
    centre = statistics.median(ratios)
    spread = max(MIN_SPREAD, statistics.stdev(ratios)) if len(ratios) > 1 else MIN_SPREAD
    return centre, spread


def hour_quantiles(p1: float, h: int, comps: list[tuple[str, list[float]]]) -> list[float | int]:
    """The five quantiles of hour h's viewers."""
    if comps:
        if effective_sample_size(p1, comps) < MIN_ESS:
            centre, spread = v1_fallback(p1, h, comps)
        else:
            centre, spread = local_linear(p1, h, comps)
        # Guard: a degenerate fit must never leak NaN/inf into the output;
        # fall through to the house baseline instead of emitting garbage.
        if not (math.isfinite(centre) and math.isfinite(spread)):
            raw = [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in Z]
        else:
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
    comps = filtered_comparables(case)
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], comps)}
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
