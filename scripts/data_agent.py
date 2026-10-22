#!/usr/bin/env python3
"""Read-only Data Agent: checks the integrity and coverage of CryptoRadar evidence."""
import datetime
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY = 86_400_000


def audit_data(data, now_ms=None, historical=None):
    now_ms = now_ms if now_ms is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    errors, warnings, seen, daily, coins, days = [], [], set(), set(), set(), set()
    observations = data.get("observations", []) if isinstance(data, dict) else []
    forecasts = data.get("forecasts", []) if isinstance(data, dict) else []
    if not isinstance(data, dict):
        return {"status": "FAIL", "errors": ["root_not_object"], "warnings": [],
                "observations": 0, "validObservations": 0, "coins": 0, "utcDays": 0,
                "forecasts": 0, "maturedForecasts": 0, "pendingForecasts": 0}
    if not isinstance(observations, list):
        errors.append("observations_not_list")
        observations = []
    if not isinstance(forecasts, list):
        errors.append("forecasts_not_list")
        forecasts = []
    valid, latest = 0, None
    for i, row in enumerate(observations):
        if not isinstance(row, dict):
            errors.append(f"observation_{i}_not_object")
            continue
        cid, price, timestamp = row.get("id"), row.get("price"), row.get("time")
        if not isinstance(cid, str) or not cid.strip():
            errors.append(f"observation_{i}_invalid_coin_id")
            continue
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            errors.append(f"observation_{i}_invalid_price")
            continue
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
            errors.append(f"observation_{i}_invalid_timestamp")
            continue
        if timestamp > now_ms + 300_000:
            errors.append(f"observation_{i}_timestamp_in_future")
            continue
        key = (cid, timestamp)
        if key in seen:
            errors.append(f"observation_{i}_duplicate_coin_timestamp")
            continue
        seen.add(key)
        day_key = (cid, timestamp // DAY)
        if day_key in daily:
            warnings.append(f"observation_{i}_multiple_samples_same_coin_utc_day")
        daily.add(day_key)
        valid += 1
        coins.add(cid)
        days.add(timestamp // DAY)
        latest = timestamp if latest is None else max(latest, timestamp)

    matured = pending = 0
    forecast_seen = set()
    for i, row in enumerate(forecasts):
        if not isinstance(row, dict):
            errors.append(f"forecast_{i}_not_object")
            continue
        cid, created, price, probability = row.get("id"), row.get("created"), row.get("price"), row.get("pUp")
        if not isinstance(cid, str) or not cid.strip() or isinstance(created, bool) or not isinstance(created, int) or created <= 0:
            errors.append(f"forecast_{i}_invalid_identity_or_timestamp")
            continue
        key = (cid, created)
        if key in forecast_seen:
            errors.append(f"forecast_{i}_duplicate_coin_timestamp")
        forecast_seen.add(key)
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            errors.append(f"forecast_{i}_invalid_price")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            errors.append(f"forecast_{i}_invalid_probability")
        outcome, scored = row.get("outcome"), row.get("scoredAt")
        if outcome is None:
            pending += 1
            if scored is not None:
                errors.append(f"forecast_{i}_pending_but_has_scored_at")
        elif outcome in (0, 1) and not isinstance(outcome, bool) and isinstance(scored, int) and scored >= created + DAY:
            closing = row.get("scoredPrice")
            if isinstance(closing, bool) or not isinstance(closing, (int, float)) or not math.isfinite(closing) or closing <= 0:
                errors.append(f"forecast_{i}_invalid_scored_price")
            elif isinstance(price, (int, float)) and outcome != int(closing > price):
                errors.append(f"forecast_{i}_outcome_disagrees_with_prices")
            else:
                matured += 1
        else:
            errors.append(f"forecast_{i}_invalid_resolution")

    if valid == 0:
        warnings.append("no_valid_observations")
    if len(days) < 21:
        warnings.append("fewer_than_21_distinct_utc_days")
    if len(coins) < 10:
        warnings.append("fewer_than_10_distinct_coins")
    if not forecasts:
        warnings.append("no_forecasts_yet")
    if latest is not None and now_ms - latest > 36 * 60 * 60 * 1000:
        warnings.append("latest_observation_older_than_36_hours")

    historical_report = None
    if historical is not None:
        hist_rows = historical.get("observations", []) if isinstance(historical, dict) else None
        hist_audit = audit_data({"observations": hist_rows, "forecasts": []}, now_ms) if isinstance(hist_rows, list) else {
            "status": "FAIL", "errors": ["historical_observations_not_list"], "warnings": [],
            "validObservations": 0, "coins": 0, "utcDays": 0
        }
        hist_valid = hist_audit.get("validObservations", 0)
        hist_structural_status = hist_audit["status"]
        historical_report = {
            "status": ("INSUFFICIENT_DATA" if hist_valid == 0 and hist_structural_status == "PASS" else hist_structural_status),
            "observations": hist_audit.get("observations", len(hist_rows) if isinstance(hist_rows, list) else 0),
            "validObservations": hist_valid, "coins": hist_audit.get("coins", 0),
            "utcDays": hist_audit.get("utcDays", 0), "warnings": [w for w in hist_audit.get("warnings", []) if w != "no_forecasts_yet"],
            "sourceDeclaredByBackfill": historical.get("source") if isinstance(historical, dict) else None,
            "quoteCurrency": historical.get("quoteCurrency") if isinstance(historical, dict) else None,
            "externalAuthenticityVerified": False
        }
        if hist_valid == 0:
            warnings.append("historical_data_not_available_yet")
        for item in hist_audit.get("errors", []):
            errors.append("historical_" + item)
        if isinstance(historical, dict) and historical.get("quoteCurrency") not in (None, "gbp"):
            errors.append("historical_quote_currency_not_gbp")
        if isinstance(hist_rows, list):
            for i, row in enumerate(hist_rows):
                if isinstance(row, dict) and (not isinstance(row.get("source"), str) or not row.get("source", "").strip()):
                    errors.append(f"historical_observation_{i}_missing_source_attribution")

    return {
        "status": "PASS" if not errors else "FAIL",
        "checkedAt": datetime.datetime.fromtimestamp(now_ms / 1000, datetime.timezone.utc).isoformat(),
        "sourceDeclaredByCollector": "CoinGecko public markets endpoint; source authenticity is not independently verified",
        "observations": len(observations), "validObservations": valid, "coins": len(coins),
        "utcDays": len(days), "latestObservationMs": latest, "forecasts": len(forecasts),
        "maturedForecasts": matured, "pendingForecasts": pending,
        "errorCount": len(errors), "errors": errors[:100], "warningCount": len(warnings),
        "warnings": warnings[:100], "historicalDataQuality": historical_report,
        "limitations": [
            "PASS checks structural consistency only, not external-source authenticity.",
            "Sparse history is warned about, never filled with synthetic data.",
            "The Data Agent does not train models, edit observations, score strategy performance, or place orders."
        ]
    }


def main():
    data = json.loads((ROOT / "research-data.json").read_text(encoding="utf-8"))
    try:
        historical = json.loads((ROOT / "historical-data.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        historical = {"observations": [], "source": "No historical backfill available yet"}
    report = audit_data(data, historical=historical)
    (ROOT / "data-agent-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hist = report.get("historicalDataQuality") or {}
    print(f"Data Agent: {report['status']} | {report['validObservations']}/{report['observations']} live valid | "
          f"{report['coins']} live coins | {report['utcDays']} live days | {report['errorCount']} errors | "
          f"historical points={hist.get('validObservations', 0)}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
