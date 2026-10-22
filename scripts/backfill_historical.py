#!/usr/bin/env python3
"""Incremental real-history backfill from CoinGecko; no synthetic prices.

Uses the current collector's observed universe, prioritises established crypto
assets, and deliberately stays below public API rate limits. Backfilled data is
used only for chronological training/backtesting, never as pretend live forecasts.
"""
import datetime
import json
import math
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIVE_PATH = ROOT / "research-data.json"
DATA_PATH = ROOT / "historical-data.json"
REPORT_PATH = ROOT / "historical-data-report.json"
BASE = "https://api.coingecko.com/api/v3"
QUOTE = "gbp"
DAYS = 365
MAX_COINS_PER_RUN = 12
MAX_ASSETS_TOTAL = 30
REQUEST_SPACING_SECONDS = 15
HISTORY_REFRESH_DAYS = 180
FAILED_RETRY_DAYS = 7
USER_AGENT = "CryptoRadarAI-Research/1.0"

# Candidate order is a stable priority order, not an asserted ranking or forecast.
CORE_ASSETS = [
    "bitcoin", "ethereum", "binancecoin", "ripple", "solana", "dogecoin",
    "cardano", "tron", "chainlink", "hyperliquid", "bitcoin-cash", "stellar",
    "avalanche-2", "litecoin", "polkadot", "sui", "near", "bittensor",
    "monero", "hedera-hashgraph", "aave", "uniswap", "internet-computer",
    "cosmos", "aptos", "injective-protocol", "algorand", "vechain",
    "render-token", "arbitrum", "optimism", "mantle", "the-open-network",
    "kaspa", "filecoin", "pepe", "shiba-inu", "curve-dao-token",
    "quant-network", "worldcoin-wld", "nexo", "pancakeswap-token",
    "kucoin-shares", "dash", "bitget-token", "jupiter-exchange-solana",
    "morpho", "ether-fi"
]
EXCLUDED_IDS = {
    "tether", "usd-coin", "dai", "ethena-usde", "paypal-usd", "usdd",
    "usds", "usd1-wlfi", "usdgo", "global-dollar", "gho", "bfusd",
    "open-usd", "united-stables", "ripple-usd", "ondo-us-dollar-yield",
    "blackrock-usd-institutional-digital-liquidity-fund",
    "spiko-amundi-overnight-swap-fund-eur",
    "superstate-short-duration-us-government-securities-fund-ustb",
    "pax-gold", "tether-gold", "wrapped-bitcoin", "coinbase-wrapped-btc",
}
DAY_MS = 86_400_000


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso_now():
    return utc_now().isoformat()


def default_store():
    return {"schemaVersion": 1, "source": "CoinGecko public API market_chart",
            "quoteCurrency": QUOTE, "requestedDays": DAYS,
            "observations": [], "coins": {}, "updatedAt": None}


def load_json(path, default):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except (OSError, ValueError):
        return default


def load_store():
    store = load_json(DATA_PATH, default_store())
    for key, value in default_store().items():
        store.setdefault(key, value)
    if not isinstance(store.get("observations"), list):
        store["observations"] = []
    if not isinstance(store.get("coins"), dict):
        store["coins"] = {}
    return store


def fetch_json(url, timeout=35):
    request = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": USER_AGENT
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def live_coin_ids(live_data, now_ms):
    rows = live_data.get("observations", []) if isinstance(live_data, dict) else []
    newest = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        timestamp, price = row.get("time"), row.get("price")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            continue
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            continue
        newest[row["id"]] = max(timestamp, newest.get(row["id"], 0))
    return {cid for cid, timestamp in newest.items() if now_ms - timestamp <= 36 * 60 * 60 * 1000}


def select_candidate_ids(live_data, store, now_epoch):
    available = live_coin_ids(live_data, now_epoch * 1000)
    metadata = store.get("coins", {}) if isinstance(store, dict) else {}
    qualified = {
        cid for cid, record in metadata.items()
        if isinstance(record, dict) and record.get("status") == "ok"
        and isinstance(record.get("pointCount"), int) and record["pointCount"] >= 30
        and cid != "_summary"
    }
    slots = max(0, MAX_ASSETS_TOTAL - len(qualified))
    refresh_candidates, new_candidates = [], []
    for cid in CORE_ASSETS:
        if cid not in available or cid in EXCLUDED_IDS:
            continue
        record = metadata.get(cid, {})
        if not isinstance(record, dict):
            record = {}
        fetched = record.get("fetchedAtEpoch")
        if cid in qualified:
            # Refresh previously successful history after 180 days so the archive
            # keeps a rolling year while live records roll off their 90-day window.
            if not isinstance(fetched, int) or now_epoch - fetched >= HISTORY_REFRESH_DAYS * 86400:
                refresh_candidates.append(cid)
            continue
        last_attempt = record.get("lastAttemptEpoch")
        if (isinstance(last_attempt, int) and now_epoch - last_attempt < FAILED_RETRY_DAYS * 86400
                and record.get("status") != "ok"):
            continue
        if isinstance(fetched, int) and now_epoch - fetched < HISTORY_REFRESH_DAYS * 86400:
            continue
        new_candidates.append(cid)
    return (refresh_candidates + new_candidates[:slots])[:MAX_COINS_PER_RUN], len(qualified)

def latest_day_samples(cid, prices, now_ms):
    """Keep the latest valid provider price sample in each UTC day."""
    by_day = {}
    invalid = 0
    if not isinstance(prices, list):
        return [], 1
    for item in prices:
        if not isinstance(item, list) or len(item) < 2:
            invalid += 1
            continue
        timestamp, price = item[0], item[1]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            invalid += 1
            continue
        timestamp = int(timestamp)
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            invalid += 1
            continue
        if timestamp <= 0 or timestamp > now_ms + 300_000:
            invalid += 1
            continue
        day = timestamp // DAY_MS
        prior = by_day.get(day)
        if prior is None or timestamp > prior["time"]:
            by_day[day] = {"id": cid, "time": timestamp, "price": float(price),
                           "source": "CoinGecko market_chart"}
    return sorted(by_day.values(), key=lambda row: row["time"]), invalid


def merge_observations(existing, incoming):
    """Merge by coin and UTC day, retaining the newest price timestamp without duplicates."""
    by_day = {}
    for row in existing + incoming:
        if not isinstance(row, dict):
            continue
        cid, timestamp, price = row.get("id"), row.get("time"), row.get("price")
        if not isinstance(cid, str) or isinstance(timestamp, bool) or not isinstance(timestamp, int):
            continue
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            continue
        key = (cid, timestamp // DAY_MS)
        previous = by_day.get(key)
        if previous is None or timestamp > previous["time"]:
            by_day[key] = {"id": cid, "time": timestamp, "price": float(price),
                           "source": str(row.get("source", "CoinGecko market_chart"))}
    return sorted(by_day.values(), key=lambda row: (row["time"], row["id"]))


def audit_history(store, failures=None, attempted=0, successful=0, now_ms=None, status=None):
    now_ms = now_ms if now_ms is not None else int(utc_now().timestamp() * 1000)
    observations = store.get("observations", [])
    coin_days, days, coins = set(), set(), set()
    invalid = 0
    for row in observations:
        if not isinstance(row, dict):
            invalid += 1
            continue
        cid, timestamp, price = row.get("id"), row.get("time"), row.get("price")
        if not isinstance(cid, str) or isinstance(timestamp, bool) or not isinstance(timestamp, int):
            invalid += 1
            continue
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            invalid += 1
            continue
        coin_days.add((cid, timestamp // DAY_MS))
        days.add(timestamp // DAY_MS)
        coins.add(cid)
    return {
        "generatedAt": iso_now(), "status": status or ("PASS" if invalid == 0 else "FAIL"),
        "source": "CoinGecko public API /coins/{id}/market_chart",
        "candidateUniverse": "Core-asset priority list intersected with recent live observations",
        "quoteCurrency": QUOTE, "requestedDays": DAYS,
        "assetsAttemptedThisRun": attempted, "assetsFetchedThisRun": successful,
        "assetLimit": MAX_ASSETS_TOTAL, "requestSpacingSeconds": REQUEST_SPACING_SECONDS,
        "assetsWithStoredHistory": len(coins), "uniqueCoinUtcDays": len(coin_days),
        "distinctUtcDays": len(days), "storedObservations": len(observations),
        "invalidStoredRows": invalid, "failedAssets": (failures or [])[:30],
        "limitations": [
            "External-source authenticity is not independently verified.",
            "Historical prices are for chronological training/backtesting, not pretend live forecasts.",
            "Recently listed assets may have less than one year of history.",
            "Backfill is incremental; no synthetic points are created."
        ]
    }


def main():
    store = load_store()
    live = load_json(LIVE_PATH, {})
    start_monotonic = time.monotonic()
    now = utc_now()
    now_ms = int(now.timestamp() * 1000)
    now_epoch = int(now.timestamp())
    failures = []
    attempted = successful = 0
    blocked_by_rate_limit = False
    candidates, qualified_count = select_candidate_ids(live, store, now_epoch)
    incoming = []
    last_request_at = start_monotonic

    for cid in candidates:
        elapsed = time.monotonic() - last_request_at
        if elapsed < REQUEST_SPACING_SECONDS:
            time.sleep(REQUEST_SPACING_SECONDS - elapsed)
        attempted += 1
        record = store["coins"].get(cid, {})
        if not isinstance(record, dict):
            record = {}
        record["lastAttemptAt"] = iso_now()
        record["lastAttemptEpoch"] = int(utc_now().timestamp())
        url = BASE + "/coins/" + urllib.parse.quote(cid, safe="") + "/market_chart?" + urllib.parse.urlencode({
            "vs_currency": QUOTE, "days": str(DAYS)
        })
        last_request_at = time.monotonic()
        try:
            response = fetch_json(url)
            points, invalid = latest_day_samples(cid, response.get("prices") if isinstance(response, dict) else None, now_ms)
            if not points:
                record.update({"status": "no_valid_history", "pointCount": 0, "lastError": "No valid historical price points"})
                store["coins"][cid] = record
                failures.append({"asset": cid, "error": "no valid historical price points"})
                continue
            incoming.extend(points)
            times = [row["time"] for row in points]
            record.update({
                "status": "ok" if len(points) >= 30 else "short_history",
                "fetchedAt": iso_now(), "fetchedAtEpoch": int(utc_now().timestamp()),
                "pointCount": len(points), "firstPointMs": min(times), "lastPointMs": max(times),
                "invalidPointsIgnored": invalid, "lastError": None
            })
            store["coins"][cid] = record
            successful += 1
            print(f"Historical backfill: {cid}: {len(points)} daily samples", flush=True)
        except urllib.error.HTTPError as exc:
            record.update({"status": "http_error", "lastError": f"HTTP {exc.code}"})
            store["coins"][cid] = record
            failures.append({"asset": cid, "error": f"HTTP {exc.code}"})
            print(f"Historical backfill: {cid}: HTTP {exc.code}", flush=True)
            if exc.code == 429:
                blocked_by_rate_limit = True
                break
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            record.update({"status": "fetch_error", "lastError": str(exc)[:250]})
            store["coins"][cid] = record
            failures.append({"asset": cid, "error": str(exc)[:250]})
            print(f"Historical backfill: {cid}: {exc}", flush=True)

    if incoming:
        store["observations"] = merge_observations(store.get("observations", []), incoming)
    # Keep a bounded rolling year plus a small safety margin; older cached points
    # could otherwise silently outlive the provider's public-plan history window.
    cutoff_ms = now_ms - (DAYS + 3) * DAY_MS
    store["observations"] = [
        row for row in store.get("observations", [])
        if isinstance(row, dict) and isinstance(row.get("time"), int) and row["time"] >= cutoff_ms
    ]
    store["coins"]["_summary"] = {
        "coreAssetLimit": MAX_ASSETS_TOTAL,
        "qualifiedCoreAssetsAtStart": qualified_count,
        "backfillAttemptedAt": iso_now()
    }
    store["updatedAt"] = iso_now()
    DATA_PATH.write_text(json.dumps(store, separators=(",", ":")) + "\n", encoding="utf-8")

    status = "RATE_LIMITED" if blocked_by_rate_limit else ("PARTIAL" if failures else "PASS")
    report = audit_history(store, failures, attempted, successful, now_ms, status)
    if not candidates and not failures:
        report["status"] = "CAP_REACHED" if qualified_count >= MAX_ASSETS_TOTAL else "NO_CANDIDATES"
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Historical backfill:", report["status"], "| stored assets:", report["assetsWithStoredHistory"],
          "| stored days:", report["distinctUtcDays"], "| this run:", successful, "/", attempted, flush=True)
    # Rate limiting or provider/network failure is recorded but does not corrupt existing data.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
