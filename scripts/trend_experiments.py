#!/usr/bin/env python3
"""Fast, development-only Brier experiments. Never touches the final holdout.

Each candidate is selected on chronological expanding-window folds, compared
with a fitted training-rate constant and a neutral 0.5 baseline. The output is
an experiment report, not a promoted model or evidence of trading profitability.
"""
import datetime
import json
import pathlib
from train_research import (
    ROOT, DAY_MS, MIN_TRAIN_EXAMPLES, MIN_TEST_EXAMPLES,
    read_json, merge_observations, build_examples, chronological_split,
    fit_logistic, predict_probability, brier_score
)

REPORT = ROOT / "trend-experiment-report.json"
L2_VALUES = [0.02, 0.1, 0.5, 2.0]
FEATURE_GROUPS = {
    "short_momentum": [0, 1, 2, 5, 7],
    "medium_momentum": [1, 2, 3, 4, 6, 8],
    "trend_distance": [0, 2, 9, 10, 11],
    "all_features": list(range(12))
}


def fold_slices(development):
    days = sorted({row["time"] // DAY_MS for row in development})
    folds = []
    for frac in (0.5, 0.65, 0.8):
        start = int(len(days) * frac)
        if start >= len(days) - 8:
            continue
        val_days = set(days[start:min(len(days), start + max(10, int(len(days) * .12)))])
        cutoff = days[start]
        train = [r for r in development if r["time"] // DAY_MS < cutoff - 1]
        validation = [r for r in development if r["time"] // DAY_MS in val_days]
        if len(train) >= MIN_TRAIN_EXAMPLES and len(validation) >= MIN_TEST_EXAMPLES:
            folds.append((train, validation))
    return folds


def subset(rows, indices):
    return [{"time": r["time"], "coin": r["coin"], "x": [r["x"][j] for j in indices], "y": r["y"]} for r in rows]


def evaluate_groups(development):
    folds = fold_slices(development)
    if len(folds) < 2:
        return {"status": "INSUFFICIENT_DEVELOPMENT_FOLDS", "candidates": []}
    candidates = []
    for group, indices in FEATURE_GROUPS.items():
        for l2 in L2_VALUES:
            scores, constant_scores, neutral_scores, counts = [], [], [], []
            for train, validation in folds:
                # The trainer's standardizer uses global feature count, so temporarily
                # retain zeroed inactive features rather than changing model dimensions.
                train_masked = [dict(r, x=[v if j in indices else 0.0 for j, v in enumerate(r["x"])]) for r in train]
                val_masked = [dict(r, x=[v if j in indices else 0.0 for j, v in enumerate(r["x"])]) for r in validation]
                model = fit_logistic(train_masked, l2=l2)
                labels = [r["y"] for r in validation]
                probs = [predict_probability(r["x"], model) for r in val_masked]
                rate = sum(r["y"] for r in train) / len(train)
                scores.append(brier_score(probs, labels))
                constant_scores.append(brier_score([rate] * len(labels), labels))
                neutral_scores.append(brier_score([0.5] * len(labels), labels))
                counts.append(len(labels))
            total = sum(counts)
            avg = lambda values: sum(value * n for value, n in zip(values, counts)) / total
            score = avg(scores)
            constant, neutral = avg(constant_scores), avg(neutral_scores)
            candidates.append({
                "featureGroup": group, "featureIndices": indices, "l2": l2,
                "developmentBrier": round(score, 6),
                "trainingRateBaselineBrier": round(constant, 6),
                "neutralBaselineBrier": round(neutral, 6),
                "bestConstantBaselineBrier": round(min(constant, neutral), 6),
                "beatsConstantBaselineOnDevelopment": score < min(constant, neutral),
                "validationExamples": total, "folds": len(folds)
            })
    candidates.sort(key=lambda row: (row["developmentBrier"], len(row["featureIndices"]), -row["l2"]))
    return {"status": "EVALUATED", "candidates": candidates,
            "bestDevelopmentCandidate": candidates[0],
            "candidateCount": len(candidates),
            "limitations": [
                "This is development-only model selection, not an independent holdout.",
                "The historical holdout has previously been inspected and is not retested here.",
                "Selecting among many candidates increases overfitting risk; prospective shadow testing is required.",
                "No model is promoted or connected to real-money trading."
            ]}


def main():
    live = read_json(ROOT / "research-data.json")
    hist = read_json(ROOT / "historical-data.json")
    examples = build_examples(merge_observations(live, hist))
    development, _, _ = chronological_split(examples)
    result = evaluate_groups(development)
    result["generatedAt"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["developmentExamples"] = len(development)
    result["outerHoldoutUsedForSelection"] = False
    result["realTradingEnabled"] = False
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Trend experiments:", result["status"], "| candidates:", result.get("candidateCount", 0))
    if result.get("bestDevelopmentCandidate"):
        print("Best development candidate:", result["bestDevelopmentCandidate"])


if __name__ == "__main__":
    main()
