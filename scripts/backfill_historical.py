#!/usr/bin/env python3
"""Incremental real-history backfill from CoinGecko. Never synthesizes price data.

This runs before training. It fetches at most 12 missing/aged assets per invocation,
uses a conservative request pace, preserves earlier valid history on failures, and
writes an audit report. These historical points are training data, never forecasts.
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
DATA_PATH = ROOT / "historical-data.json"
REPORT_PATH = ROOT / "historical-data-report.json"
BASE = "https://api.coingecko.com/api/v3"
QUOTE = "gbp"
DAYS = 365
MAX_COINS_PER_RUN = 12
REQUEST_SPACING_SECONDS = 10
HISTORY_REFRESH_DAYS = 180
FAILED_RETRY_DAYS = 7
USER_AGENT = "CryptoRadarAI-Research/1.0 (historical evidence backfill)"

# Exclude stablecoins, tokenized funds/gold, and common wrapped versions from the
# directional-crypto training basket. The list is deliberately conservative.
EXCLUDED_IDS = {
    "tether", "usd-coin", "dai", "ethena-usde", "paypal-usd", "usdd",
    "usds", "usd1-wlfi", "usdgo", "global-dollar", "gho", "bfusd",
    "open-usd", "united-stables", "ripple-usd", "ondo-us-dollar-yield",
    "blackrock-usd-institutional-digital-liquidity-fund",
    "spiko-amundi-overnight-swap-fund-eur",
    "superstate-short-duration-us-government-securities-fund-ustb",
    "pax-gold", "tether-gold", "wrapped-bitcoin", "coinbase-wrapped-btc",
}
EXCLUDED_SYMBOLS = {
    "USDT", "USDC", "DAI", "USDE", "PYUSD", "USDD", "USDS", "USD1",
    "GUSD", "FDUSD", "TUSD", "USDP", "EURC", "EURCV", "XAUT", "PAXG",
}
EXCLUDED_NAME_TERMS = (" stablecoin", "tokenized fund", "digital liquidity fund",
                       "government securities fund", "gold token")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso_now():
    return utc_now().isoformat()


def default_store():
    return {"schemaVersion": 1, "source": "CoinGecko public API market_chart",
            "quoteCurrency": QUOTE, "requestedDays": DAYS,
            "observations": [], "coins": {}, "updatedAt": None}


def load_store():
    try:
        store = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(store, dict):
            raise ValueError("root must be an object")
    except (OSError, ValueError):
        store = default_store()
    store.setdefault("schemaVersion", 1)
    store.setdefault("source", "CoinGecko public API market_chart")
    store.setdefault("quoteCurrency", QUOTE)
    store.setdefault("requestedDays", DAYS)
    store.setdefault("observations", [])
    store.setdefault("coins", {})
    if not isinstance(store["observations"], list):
        store["observations"] = []
    if not isinstance(store["coins"], dict):
        store["coins"] = {}
    return store


def fetch_json(url, timeout=35):
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": USER_AGENT
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def choose_assets(coins):
    selected = []
    for coin in coins if isinstance(coins, list) else []:
        if not isinstance(coin, dict):
            continue
        cid = coin.get("id")
        symbol = str(coin.get("symbol", "")).upper()
        name = str(coin.get("name", "")).lower()
        rank = coin.get("market_cap_rank")
        if not isinstance(cid, str) or not cid or cid in EXCLUDED_IDS:
            continue
        if symbol in EXCLUDED_SYMBOLS or not isinstance(rank, int):
            continue
        if any(term in name for term in EXCLUDED_NAME_TERMS):
            continue
        if not isinstance(coin.get("current_price"), (int, float)):
            continue
        selected.append((rank, cid))
    selected.sort(key=lambda item: item[0])
    return [cid for _, cid in selected]


def latest_day_samples(cid, prices, now_ms):
    """Keep one latest valid observed market point per UTC date."""
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
        day = timestamp // 86_400_000
        previous = by_day.get(day)
        if previous is None or timestamp > previous["time"]:
            by_day[day] = {"id": cid, "time": timestamp, "price": float(price),
                           "source": "CoinGecko market_chart"}
    return sorted(by_day.values(), key=lambda row: row["time"]), invalid


def merge_observations(existing, incoming):
    """Merge by coin and UTC day, retaining the latest timestamp without duplicates."""
    by_day = {}
    for row in existing + incoming:
        if not isinstance(row, dict):
            continue
        cid, timestamp, price = row.get("id"), row.get("time"), row.get("price")
        if not isinstance(cid, str) or isinstance(timestamp, bool) or not isinstance(timestamp, int):
            continue
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            continue
        key = (cid, timestamp // 86_400_000)
        prev = by_day.get(key)
        if prev is None or timestamp > prev["time"]:
            clean = {"id": cid, "time": timestamp, "price": float(price),
                     "source": str(row.get("source", "CoinGecko market_chart"))}
            by_day[key] = clean
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
        coin_days.add((cid, timestamp // 86_400_000))
        days.add(timestamp // 86_400_000)
        coins.add(cid)
    return {
        "generatedAt": iso_now(),
        "status": status or ("PASS" if invalid == 0 else "FAIL"),
        "source": "CoinGecko public API /coins/{id}/market_chart",
        "quoteCurrency": QUOTE, "requestedDays": DAYS,
        "assetsAttemptedThisRun": attempted, "assetsFetchedThisRun": successful,
        "assetsWithStoredHistory": len(coins), "uniqueCoinUtcDays": len(coin_days),
        "distinctUtcDays": len(days), "storedObservations": len(observations),
        "invalidStoredRows": invalid, "failedAssets": (failures or [])[:30],
        "throttleSeconds": REQUEST_SPACING_SECONDS,
        "limitations": [
            "External-source authenticity is not independently verified.",
            "Historical prices are not evidence that a forecast could have been made at the time; this file is only for chronological training/backtesting.",
            "Recently listed assets may have less than one year of history.",
            "Backfill is incremental and may take multiple runs; no synthetic points are created."
        ]
    }


def main():
    store = load_store()
    now = utc_now()
    now_ms = int(now.timestamp() * 1000)
    failures = []
    attempted = successful = 0
    blocked_by_rate_limit = False
    try:
        market_url = BASE + "/coins/markets?" + urllib.parse.urlencode({
            "vs_currency": QUOTE, "order": "market_cap_desc",
            "per_page": 100, "page": 1, "sparkline": "false"
        })
        markets = fetch_json(market_url)
        if not isinstance(markets, list) or not markets:
            raise ValueError("CoinGecko markets endpoint returned no asset list")
        candidates = choose_assets(markets)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        candidates = []
        failures.append({"stage": "market_list", "error": str(exc)[:250]})
        blocked_by_rate_limit = isinstance(exc, urllib.error.HTTPError) and exc.code == 429

    metadata = store["coins"]
    now_epoch = int(now.timestamp())
    to_fetch = []
    for cid in candidates:
        record = metadata.get(cid, {})
        if not isinstance(record, dict):
            record = {}
        fetched_at = record.get("fetchedAtEpoch")
        attempted_at = record.get("lastAttemptEpoch")
        point_count = record.get("pointCount", 0)
        if isinstance(fetched_at, int) and point_count >= 30 and now_epoch - fetched_at < HISTORY_REFRESH_DAYS * 86400:
            continue
        if isinstance(attempted_at, int) and now_epoch - attempted_at < FAILED_RETRY_DAYS * 86400 and record.get("status") != "ok":
            continue
        to_fetch.append(cid)
    to_fetch = to_fetch[:MAX_COINS_PER_RUN]

    incoming = []
    last_request_at = None
    for cid in to_fetch:
        if last_request_at is not None:
            elapsed = time.monotonic() - last_request_at
            if elapsed < REQUEST_SPACING_SECONDS:
                time.sleep(REQUEST_SPACING_SECONDS - elapsed)
        attempted += 1
        record = metadata.get(cid, {}) if isinstance(metadata.get(cid), dict) else {}
        record["lastAttemptAt"] = iso_now()
        record["lastAttemptEpoch"] = int(utc_now().timestamp())
        url = BASE + "/coins/" + urllib.parse.quote(cid, safe="") + "/market_chart?" + urllib.parse.urlencode({
            "vs_currency": QUOTE, "days": str(DAYS)
        })
        last_request_at = time.monotonic()
        try:
            result = fetch_json(url)
            points, invalid = latest_day_samples(cid, result.get("prices") if isinstance(result, dict) else None, now_ms)
            if not points:
                record.update({"status": "no_valid_history", "pointCount": 0, "lastError": "No valid historical price points"})
                metadata[cid] = record
                failures.append({"asset": cid, "error": "no valid historical price points"})
                continue
            incoming.extend(points)
            point_times = [p["time"] for p in points]
            record.update({
                "status": "ok" if len(points) >= 30 else "short_history",
                "fetchedAt": iso_now(), "fetchedAtEpoch": int(utc_now().timestamp()),
                "pointCount": len(points), "firstPointMs": min(point_times),
                "lastPointMs": max(point_times), "invalidPointsIgnored": invalid,
                "lastError": None
            })
            metadata[cid] = record
            successful += 1
            print(f"Historical backfill: {cid}: {len(points)} daily samples", flush=True)
        except urllib.error.HTTPError as exc:
            record.update({"status": "http_error", "lastError": f"HTTP {exc.code}"})
            metadata[cid] = record
            failures.append({"asset": cid, "error": f"HTTP {exc.code}"})
            print(f"Historical backfill: {cid}: HTTP {exc.code}", flush=True)
            if exc.code == 429:
                blocked_by_rate_limit = True
                break
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            record.update({"status": "fetch_error", "lastError": str(exc)[:250]})
            metadata[cid] = record
            failures.append({"asset": cid, "error": str(exc)[:250]})
            print(f"Historical backfill: {cid}: {exc}", flush=True)

    if incoming:
        store["observations"] = merge_observations(store["observations"], incoming)
    store["coins"] = metadata
    store["updatedAt"] = iso_now()
    DATA_PATH.write_text(json.dumps(store, separators=(",", ":")) + "\n", encoding="utf-8")

    status = "RATE_LIMITED" if blocked_by_rate_limit else ("PARTIAL" if failures else "PASS")
    report = audit_history(store, failures, attempted, successful, now_ms, status)
    if not candidates and not failures:
        report["status"] = "NO_CANDIDATES"
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Historical backfill:", report["status"], "| stored assets:", report["assetsWithStoredHistory"],
          "| stored days:", report["distinctUtcDays"], "| this run:", successful, "/", attempted, flush=True)
    # API availability/rate limits do not invalidate existing data or the Data Agent.
    # The report records partial status while the research pipeline continues fail-closed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
