"""Transparent market-regime diagnostics for historical research.
All features use prices available at prediction time; no trading signals are issued.
"""
import math
from collections import defaultdict

def classify(momentum14, vol7):
    if vol7 >= 0.055:
        return "high_volatility"
    if momentum14 >= 0.08:
        return "uptrend"
    if momentum14 <= -0.08:
        return "downtrend"
    return "range_or_mixed"

def enrich(candles_by_pair):
    """Return a mapping (pair, day_index) -> regime using only trailing closes."""
    out = {}
    for pair, candles in candles_by_pair.items():
        prices = [price for _, price in candles]
        for i in range(14, len(prices)):
            returns = [math.log(prices[j] / prices[j - 1]) for j in range(i - 6, i + 1)]
            avg = sum(returns) / len(returns)
            vol = math.sqrt(sum((v - avg) ** 2 for v in returns) / len(returns))
            momentum = math.log(prices[i] / prices[i - 14])
            out[(pair, candles[i][0])] = classify(momentum, vol)
    return out

def regime_diagnostics(rows, candles_by_pair):
    regimes = enrich(candles_by_pair)
    counts = defaultdict(int)
    for timestamp, pair, features, outcome in rows:
        counts[regimes.get((pair, timestamp), "unknown")] += 1
    return {
        "method": "Fixed illustrative regimes: 14-day log momentum ±8%; trailing 7-day daily-return volatility 5.5%; volatility takes precedence.",
        "counts": dict(counts),
        "note": "Descriptive research, not learned regime classification or a validated trading signal."
    }

def selective_diagnostics(rows, folds):
    """Evaluate fixed abstention rule on the same walk-forward folds.
    A forecast is accepted only when fixed one-day momentum baseline differs from 50%.
    This is a coverage diagnostic, NOT a selected best strategy.
    """
    if not folds:
        return {"status": "unavailable"}
    by_date = defaultdict(list)
    for timestamp, pair, x, y in rows:
        by_date[timestamp] .append((pair, x, y))
    from datetime import datetime, timezone
    assessed = 0
    selected = 0
    loss = 0.
    for fold in folds:
        lo = int(datetime.fromisoformat(fold["start"]).replace(tzinfo=timezone.utc).timestamp() * 1000)
        hi = int(datetime.fromisoformat(fold["end"]).replace(tzinfo=timezone.utc).timestamp() * 1000)
        for timestamp, items in by_date.items():
            if not lo <= timestamp <= hi:
                continue
            for _, x, y in items:
                assessed += 1
                # Require unusually strong 1-day move: fixed predeclared 3% log return.
                if abs(x[0]) < .03:
                    continue
                selected += 1
                p = .55 if x[0] > 0 else .45
                loss += (p - y) ** 2
    return {
        "status": "evaluated",
        "total": assessed,
        "selected": selected,
        "coverage": round(selected / assessed, 4) if assessed else 0,
        "selectedBrier": round(loss / selected, 6) if selected else None,
        "rule": "Illustrative fixed filter: forecast only if absolute previous-day log return >=3%; fixed momentum probability 55/45.",
        "warning": "Abstention is not evidence of superior predictions; scores on selected examples cannot be directly compared with full-coverage scores."
    }
