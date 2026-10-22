"""Fee-aware, fail-closed paper-trading scorecard for candidate agents.

This module scores timestamped, already-closed paper trades. It cannot place
orders, change model weights, or authorize promotion. Every candidate is
compared with a cash baseline after round-trip costs and rejected when evidence
is too small or malformed.
"""
import math


def score_trades(trades, *, minimum=30, round_trip_cost=0.004,
                 max_drawdown_limit=0.20):
    """Score trades with signed returns after costs; each trade is one closed position.

    Required fields: asset, created, resolved, gross_return, baseline_return.
    gross_return and baseline_return are decimal returns (e.g. 0.02 = +2%).
    Costs are charged on every trade, including losing trades.
    """
    fail = {"status": "INELIGIBLE", "eligible": False, "reward": 0.0}
    if not isinstance(trades, list) or len(trades) < minimum:
        return {**fail, "reason": "insufficient_closed_trades",
                "examples": len(trades) if isinstance(trades, list) else 0}
    if (not isinstance(round_trip_cost, (int, float))
            or not math.isfinite(round_trip_cost)
            or round_trip_cost < 0 or round_trip_cost >= 1):
        return {**fail, "reason": "invalid_cost"}
    seen = set()
    agent_curve = [1.0]
    baseline_curve = [1.0]
    net_returns = []
    base_returns = []
    for trade in sorted(trades, key=lambda x: x.get("resolved", -1) if isinstance(x, dict) else -1):
        if not isinstance(trade, dict):
            return {**fail, "reason": "invalid_record"}
        asset, created, resolved = trade.get("asset"), trade.get("created"), trade.get("resolved")
        key = (asset, created)
        if (not isinstance(asset, str) or not asset or not isinstance(created, int)
                or isinstance(created, bool) or not isinstance(resolved, int)
                or isinstance(resolved, bool) or resolved - created < 86400000):
            return {**fail, "reason": "invalid_identity_or_horizon"}
        if key in seen:
            return {**fail, "reason": "duplicate_trade"}
        seen.add(key)
        gross, baseline = trade.get("gross_return"), trade.get("baseline_return")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               or not math.isfinite(v) or v <= -1 or v > 10
               for v in (gross, baseline)):
            return {**fail, "reason": "invalid_return"}
        net = (1 + gross) * (1 - round_trip_cost) - 1
        base_net = (1 + baseline) * (1 - round_trip_cost) - 1
        net_returns.append(net)
        base_returns.append(base_net)
        agent_curve.append(agent_curve[-1] * (1 + net))
        baseline_curve.append(baseline_curve[-1] * (1 + base_net))
    def max_dd(curve):
        peak = curve[0]
        worst = 0.0
        for value in curve[1:]:
            peak = max(peak, value)
            worst = max(worst, (peak - value) / peak)
        return worst
    n = len(net_returns)
    agent_total = agent_curve[-1] - 1
    baseline_total = baseline_curve[-1] - 1
    drawdown = max_dd(agent_curve)
    # Reward requires positive net compounded return, beating the supplied
    # baseline, and staying inside the predeclared drawdown limit.
    eligible = (agent_total > 0 and agent_total > baseline_total
                and drawdown <= max_drawdown_limit)
    return {
        "status": "ELIGIBLE" if eligible else "NO_REWARD",
        "eligible": eligible,
        "reward": round(min(1.0, max(0.0, agent_total - baseline_total)), 6) if eligible else 0.0,
        "examples": n,
        "roundTripCost": round_trip_cost,
        "netCompoundedReturn": round(agent_total, 6),
        "baselineNetCompoundedReturn": round(baseline_total, 6),
        "maxDrawdown": round(drawdown, 6),
        "meanNetTradeReturn": round(sum(net_returns) / n, 6),
        "winRateAfterCosts": round(sum(r > 0 for r in net_returns) / n, 4),
        "reason": "passed_predeclared_gates" if eligible else "failed_return_baseline_or_drawdown_gate",
        "note": "Research score only. Trades must be independently timestamped; this score does not prove data authenticity or authorize live trading."
    }
