"""Adversarial tests for fee-aware paper-trading reward scorecard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trading_objective import score_trades

def trades(gross=0.03, baseline=0.0, n=40):
    return [{"asset": f"asset-{i}", "created": i * 2 * 86400000,
             "resolved": i * 2 * 86400000 + 86400000,
             "gross_return": gross, "baseline_return": baseline} for i in range(n)]

def test():
    assert score_trades(trades())["eligible"]
    assert not score_trades(trades(gross=0.001))["eligible"]
    assert not score_trades(trades(gross=-0.02))["eligible"]
    assert not score_trades(trades(gross=0.02, baseline=0.03))["eligible"]
    assert score_trades(trades(n=4))["reason"] == "insufficient_closed_trades"
    duplicate = trades(); duplicate[1] = dict(duplicate[0])
    assert score_trades(duplicate)["reason"] == "duplicate_trade"
    too_soon = trades(); too_soon[0]["resolved"] = too_soon[0]["created"]
    assert score_trades(too_soon)["reason"] == "invalid_identity_or_horizon"
    assert score_trades(trades(), round_trip_cost=-0.01)["reason"] == "invalid_cost"
    drawdown = trades()
    for i, t in enumerate(drawdown):
        t["gross_return"] = 0.15 if i < 20 else -0.12
    assert not score_trades(drawdown, max_drawdown_limit=0.1)["eligible"]
    print("Fee-aware trading objective adversarial tests passed")

if __name__ == "__main__":
    test()
