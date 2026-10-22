"""Tests for historical backfill normalisation and source-safe merging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from backfill_historical import choose_assets, latest_day_samples, merge_observations

DAY = 86_400_000


def test_one_latest_valid_point_per_utc_day():
    prices = [[1000, 10.0], [2000, 11.0], [DAY + 1000, 12.0],
              [DAY + 2000, -1.0], [DAY + 3000, float("nan")]]
    rows, invalid = latest_day_samples("bitcoin", prices, 2 * DAY)
    assert len(rows) == 2
    assert rows[0]["price"] == 11.0 and rows[0]["time"] == 2000
    assert rows[1]["price"] == 12.0
    assert invalid == 2


def test_merge_keeps_latest_point_per_asset_day():
    rows = merge_observations(
        [{"id": "bitcoin", "time": DAY + 10, "price": 100.0, "source": "old"}],
        [{"id": "bitcoin", "time": DAY + 20, "price": 101.0, "source": "CoinGecko market_chart"},
         {"id": "ethereum", "time": DAY + 20, "price": 10.0}]
    )
    assert len(rows) == 2
    btc = next(row for row in rows if row["id"] == "bitcoin")
    assert btc["time"] == DAY + 20 and btc["price"] == 101.0


def test_candidates_are_ranked_and_stablecoins_excluded():
    candidates = choose_assets([
        {"id": "ethereum", "symbol": "eth", "name": "Ethereum", "market_cap_rank": 2, "current_price": 100},
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "market_cap_rank": 1, "current_price": 100},
        {"id": "tether", "symbol": "usdt", "name": "Tether", "market_cap_rank": 3, "current_price": 1},
        {"id": "pax-gold", "symbol": "paxg", "name": "PAX Gold", "market_cap_rank": 4, "current_price": 2000},
        {"id": "unranked", "symbol": "xyz", "name": "Example", "market_cap_rank": None, "current_price": 1},
    ])
    assert candidates == ["bitcoin", "ethereum"]


if __name__ == "__main__":
    test_one_latest_valid_point_per_utc_day()
    test_merge_keeps_latest_point_per_asset_day()
    test_candidates_are_ranked_and_stablecoins_excluded()
    print("Historical backfill tests passed (3 cases)")
