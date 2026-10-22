#!/usr/bin/env python3
"""Freeze one qualified Trend candidate for independent prospective paper evaluation."""
import datetime
import json
import pathlib

from prospective_trial import retain_frozen_trial

ROOT = pathlib.Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "candidate-model.json"
TRAINING_PATH = ROOT / "training-report.json"
TRIAL_PATH = ROOT / "prospective-trial.json"
STATUS_PATH = ROOT / "prospective-trial-status.json"


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def activate():
    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    existing = read_json(TRIAL_PATH)
    candidate = read_json(CANDIDATE_PATH)
    report = read_json(TRAINING_PATH)
    trial, status = retain_frozen_trial(existing or None, candidate, report, now_ms)
    if trial is not None and existing.get("status") != "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL":
        TRIAL_PATH.write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print("Prospective trial:", status.get("status"), "| trial:", status.get("trialId", "none"),
          "| model hash:", status.get("modelHash", "none"))
    if status.get("status") == "TRIAL_INTEGRITY_FAIL":
        raise SystemExit(1)
    return status


if __name__ == "__main__":
    activate()
