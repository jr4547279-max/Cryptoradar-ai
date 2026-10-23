#!/usr/bin/env python3
"""Freeze a development-selected Trend challenger for future prospective comparison.

This does not inspect the historical holdout, change the incumbent trial, or
promote a model. It only fits the already selected feature group on development
observations and saves a reproducible paper-only challenger.
"""
import datetime
import hashlib
import json
from train_research import (
    ROOT, read_json, merge_observations, build_examples, chronological_split,
    fit_logistic, dataset_fingerprint, FEATURE_NAMES
)

EXPERIMENT_PATH = ROOT / "trend-experiment-report.json"
OUTPUT_PATH = ROOT / "trend-challenger.json"


def mask_features(rows, indices):
    keep = set(indices)
    return [dict(row, x=[value if j in keep else 0.0 for j, value in enumerate(row["x"])])
            for row in rows]


def main():
    experiments = read_json(EXPERIMENT_PATH)
    best = experiments.get("bestDevelopmentCandidate") or {}
    if experiments.get("status") != "EVALUATED" or not best.get("beatsConstantBaselineOnDevelopment"):
        print("No development-qualified challenger; leaving any existing challenger unchanged.")
        return
    observations = merge_observations(
        read_json(ROOT / "research-data.json"),
        read_json(ROOT / "historical-data.json"))
    examples = build_examples(observations)
    development, _, start_day = chronological_split(examples)
    indices = best.get("featureIndices", [])
    if (not development or start_day is None or not isinstance(indices, list)
            or not indices or any(not isinstance(i, int) or i < 0 or i >= len(FEATURE_NAMES) for i in indices)):
        raise ValueError("Invalid challenger selection or training examples")
    model = fit_logistic(mask_features(development, indices), l2=float(best["l2"]))
    model.update({
        "status": "FROZEN_DEVELOPMENT_CHALLENGER",
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "datasetSha256": dataset_fingerprint(observations),
        "featureIndices": indices,
        "featureGroup": best["featureGroup"],
        "selectedDevelopmentBrier": best["developmentBrier"],
        "developmentBaselineBrier": best["bestConstantBaselineBrier"],
        "developmentExamples": len(development),
        "outerHoldoutUsedForSelection": False,
        "outerHoldoutUsedForEvaluation": False,
        "prospectiveEvidenceAvailable": False,
        "promotionEligible": False,
        "realTradingEnabled": False,
        "mode": "SHADOW_RESEARCH_ONLY",
        "note": "Development-selected challenger. No independent holdout claim. Not deployed into the existing paper trial."
    })
    OUTPUT_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print("Frozen development challenger:", model["featureGroup"], "examples:", len(development))


if __name__ == "__main__":
    main()
