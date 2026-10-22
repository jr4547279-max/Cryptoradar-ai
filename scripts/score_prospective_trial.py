#!/usr/bin/env python3
"""Publish the prospective scorecard for the one frozen Trend model."""
import json
import pathlib
import datetime
from prospective_trial import score_trial

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRIAL_PATH = ROOT / "prospective-trial.json"
DATA_PATH = ROOT / "research-data.json"
REPORT_PATH = ROOT / "prospective-trial-report.json"


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def main():
    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    report = score_trial(read_json(TRIAL_PATH), read_json(DATA_PATH), now_ms)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Prospective trial scorecard:", report.get("status"),
          "| matured forecasts:", report.get("maturedForecasts", 0),
          "| decision:", report.get("decision", "WAITING"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
