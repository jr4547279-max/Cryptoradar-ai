"""Tests for historical backfill normalization and asset selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from backfill_historical import (
    CORE_ASSETS, DAY_MS, latest_day_samples, merge_observations, select_candidate_ids,
)

NOW_EPOCH = 1_800_000_000
NOW_MS = NOW_EPOCH * 1000


def test_one_latest_valid_point_per_utc_day():
    prices = [[1000, 10.0], [2000, 11.0], [DAY_MS + 1000, 12.0],
              [DAY_MS + 2000, -1.0], [DAY_MS + 3000, float("nan")]]
    rows, invalid = latest_day_samples("bitcoin", prices, 2 * DAY_MS)
    assert len(rows) == 2
    assert rows[0]["price"] == 11.0 and rows[0]["time"] == 2000
    assert rows[1]["price"] == 12.0
    assert invalid == 2


def test_merge_keeps_latest_point_per_asset_day():
    rows = merge_observations(
        [{"id": "bitcoin", "time": DAY_MS + 10, "price": 100.0, "source": "old"}],
        [{"id": "bitcoin", "time": DAY_MS + 20, "price": 101.0, "source": "CoinGecko market_chart"},
         {"id": "ethereum", "time": DAY_MS + 20, "price": 10.0}]
    )
    assert len(rows) == 2
    btc = next(row for row in rows if row["id"] == "bitcoin")
    assert btc["time"] == DAY_MS + 20 and btc["price"] == 101.0


def test_selection_uses_fresh_live_universe_and_skips_stored_history():
    live = {"observations": [
        {"id": "bitcoin", "time": NOW_MS - 1000, "price": 100},
        {"id": "ethereum", "time": NOW_MS - 1000, "price": 10},
        {"id": "solana", "time": NOW_MS - 1000, "price": 5},
        {"id": "tether", "time": NOW_MS - 1000, "price": 1},
    ]}
    store = {"coins": {"bitcoin": {"status": "ok", "pointCount": 365,
                                   "fetchedAtEpoch": NOW_EPOCH}}}
    candidates, qualified = select_candidate_ids(live, store, NOW_EPOCH)
    assert candidates == ["ethereum", "solana"]
    assert qualified == 1


def test_candidate_store_cap_is_enforced():
    qualified_ids = CORE_ASSETS[:30]
    store = {"coins": {cid: {"status": "ok", "pointCount": 365, "fetchedAtEpoch": NOW_EPOCH}
                       for cid in qualified_ids}}
    live = {"observations": [{"id": cid, "time": NOW_MS, "price": 100} for cid in CORE_ASSETS]}
    candidates, qualified = select_candidate_ids(live, store, NOW_EPOCH)
    assert candidates == [] and qualified == 30


if __name__ == "__main__":
    test_one_latest_valid_point_per_utc_day()
    test_merge_keeps_latest_point_per_asset_day()
    test_selection_uses_fresh_live_universe_and_skips_stored_history()
    test_candidate_store_cap_is_enforced()
    print("Historical backfill tests passed (4 cases)")
