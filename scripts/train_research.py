#!/usr/bin/env python3
"""Chronological research trainer. Uses genuine recorded CoinGecko prices only.

Historical downloads and prospective samples are merged per asset/UTC day.
Synthetic scenario fixtures are deliberately not used here. Trained outputs remain
unpromoted shadow candidates; they are not connected to live trading.
"""
import datetime
import hashlib
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIVE_PATH = ROOT / "research-data.json"
HIST_PATH = ROOT / "historical-data.json"
REPORT_PATH = ROOT / "training-report.json"
MODEL_PATH = ROOT / "candidate-model.json"
DAY_MS = 86_400_000
MIN_EXAMPLES = 300
MIN_DAYS = 120
MIN_COINS = 10
MIN_TRAIN_EXAMPLES = 150
MIN_TEST_EXAMPLES = 50
MIN_TEST_DAYS = 21
FEATURE_NAMES = ["log_return_1d", "log_return_previous_day", "log_return_two_days_ago"]
MODEL_VERSION = "logistic-regression-v2"
LEARNING_RATE = 0.08
L2 = 0.02
MAX_EPOCHS = 800
GRADIENT_TOLERANCE = 1e-6
L2_GRID = [0.005, 0.02, 0.1, 0.5]


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def valid_price_row(row):
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        return False
    timestamp, price = row.get("time"), row.get("price")
    return (not isinstance(timestamp, bool) and isinstance(timestamp, int) and timestamp > 0
            and not isinstance(price, bool) and isinstance(price, (int, float))
            and math.isfinite(price) and price > 0)


def merge_observations(live_data, historical_data):
    """Deduplicate by asset/UTC day, keeping the latest valid source timestamp."""
    candidates = []
    for label, dataset in (("live", live_data), ("historical", historical_data)):
        rows = dataset.get("observations", []) if isinstance(dataset, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not valid_price_row(row):
                continue
            candidates.append((row["id"], row["time"], float(row["price"]), label))
    daily = {}
    for cid, timestamp, price, label in candidates:
        key = (cid, timestamp // DAY_MS)
        current = daily.get(key)
        # A later timestamp wins; a live sample wins ties over the historical copy.
        if current is None or timestamp > current["time"] or (
                timestamp == current["time"] and label == "live" and current["label"] != "live"):
            daily[key] = {"id": cid, "time": timestamp, "price": price, "label": label}
    return sorted(daily.values(), key=lambda row: (row["time"], row["id"]))


def build_examples(observations):
    by_coin = {}
    for row in observations:
        if valid_price_row(row):
            by_coin.setdefault(row["id"], []).append((row["time"], float(row["price"])))
    examples = []
    for coin, points in by_coin.items():
        points.sort()
        daily = {}
        for timestamp, price in points:
            daily[timestamp // DAY_MS] = (timestamp, price)
        ordered = [daily[key] for key in sorted(daily)]
        for index, (timestamp, price) in enumerate(ordered):
            def prior(days_ago):
                eligible = [p for p in ordered[:index] if p[0] <= timestamp - days_ago * DAY_MS]
                return eligible[-1][1] if eligible else None

            previous = prior(1)
            previous2 = prior(2)
            previous3 = prior(3)
            if not (previous and previous2 and previous3):
                continue
            future = next((p for p in ordered[index + 1:] if p[0] >= timestamp + DAY_MS), None)
            if future is None:
                continue
            features = [
                max(-0.5, min(0.5, math.log(price / previous))),
                max(-0.5, min(0.5, math.log(previous / previous2))),
                max(-0.5, min(0.5, math.log(previous2 / previous3))),
            ]
            if not all(math.isfinite(value) for value in features):
                continue
            examples.append({"time": timestamp, "coin": coin, "x": features,
                             "y": int(future[1] > price)})
    examples.sort(key=lambda row: (row["time"], row["coin"]))
    return examples


def chronological_split(examples):
    """Split by UTC date, not individual rows; keep a full UTC-day embargo."""
    days = sorted({row["time"] // DAY_MS for row in examples})
    if len(days) < 3:
        return [], [], None
    index = max(1, min(len(days) - 1, int(len(days) * 0.70)))
    test_start_day = days[index]
    # Do not train on the UTC day immediately preceding the test period.
    train = [row for row in examples if row["time"] // DAY_MS < test_start_day - 1]
    test = [row for row in examples if row["time"] // DAY_MS >= test_start_day]
    return train, test, test_start_day


def sigmoid(value):
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def standardizer(train):
    means, scales = [], []
    for j in range(len(FEATURE_NAMES)):
        values = [row["x"][j] for row in train]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-9 else 1.0)
    return means, scales


def transformed(features, means, scales):
    return [(features[j] - means[j]) / scales[j] for j in range(len(FEATURE_NAMES))]


def fit_logistic(train, l2=L2):
    """Full-batch, deterministic L2-regularized logistic regression."""
    means, scales = standardizer(train)
    normalized = [(transformed(row["x"], means, scales), row["y"]) for row in train]
    weights = [0.0] * (len(FEATURE_NAMES) + 1)
    epochs_used = 0
    for epoch in range(MAX_EPOCHS):
        gradients = [0.0] * len(weights)
        for features, label in normalized:
            probability = sigmoid(weights[0] + sum(weights[j + 1] * features[j]
                                                  for j in range(len(features))))
            error = probability - label
            gradients[0] += error
            for j, value in enumerate(features):
                gradients[j + 1] += error * value
        gradients = [gradient / len(normalized) for gradient in gradients]
        for j in range(1, len(gradients)):
            gradients[j] += l2 * weights[j]
        max_gradient = max(abs(value) for value in gradients)
        for j, gradient in enumerate(gradients):
            weights[j] -= LEARNING_RATE * gradient
        epochs_used = epoch + 1
        if max_gradient < GRADIENT_TOLERANCE:
            break
    model = {
        "modelVersion": MODEL_VERSION,
        "features": FEATURE_NAMES,
        "featureMeans": means,
        "featureScales": scales,
        "weights": weights,
        "optimizer": {"name": "full_batch_gradient_descent", "learningRate": LEARNING_RATE,
                      "l2": l2, "maxEpochs": MAX_EPOCHS, "epochsRun": epochs_used,
                      "convergedByGradientTolerance": epochs_used < MAX_EPOCHS},
        "promotionEligible": False,
        "mode": "SHADOW_RESEARCH_ONLY"
    }
    return model


def walk_forward_select_l2(development):
    """Select regularization from expanding-window validation, never the outer holdout."""
    days = sorted({row["time"] // DAY_MS for row in development})
    if len(days) < 60:
        return {"available": False, "reason": "too_few_development_days", "selectedL2": L2,
                "candidates": [], "folds": []}
    validation_length = max(10, int(len(days) * 0.12))
    starts = [int(len(days) * fraction) for fraction in (0.50, 0.65, 0.80)]
    folds = []
    for fold_index, start_index in enumerate(starts, 1):
        if start_index >= len(days) - 5:
            continue
        end_index = min(len(days), start_index + validation_length)
        start_day, end_day = days[start_index], days[end_index - 1]
        fit_rows = [row for row in development if row["time"] // DAY_MS < start_day - 1]
        validation_rows = [row for row in development
                           if start_day <= row["time"] // DAY_MS <= end_day]
        if len(fit_rows) < MIN_TRAIN_EXAMPLES or len(validation_rows) < MIN_TEST_EXAMPLES:
            continue
        folds.append({"fold": fold_index, "train": fit_rows, "validation": validation_rows,
                      "startDay": start_day, "endDay": end_day})
    if len(folds) < 2:
        return {"available": False, "reason": "insufficient_walk_forward_folds", "selectedL2": L2,
                "candidates": [], "folds": []}

    scores = []
    for l2 in L2_GRID:
        fold_results = []
        weighted_candidate, weighted_constant, weighted_neutral, total = 0.0, 0.0, 0.0, 0
        for fold in folds:
            fit_rows, validation_rows = fold["train"], fold["validation"]
            model = fit_logistic(fit_rows, l2=l2)
            labels = [row["y"] for row in validation_rows]
            probs = [predict_probability(row["x"], model) for row in validation_rows]
            train_rate = sum(row["y"] for row in fit_rows) / len(fit_rows)
            candidate_brier = brier_score(probs, labels)
            constant_brier = brier_score([train_rate] * len(labels), labels)
            neutral_brier = brier_score([0.5] * len(labels), labels)
            count = len(labels)
            weighted_candidate += candidate_brier * count
            weighted_constant += constant_brier * count
            weighted_neutral += neutral_brier * count
            total += count
            fold_results.append({
                "fold": fold["fold"], "validationExamples": count,
                "validationStartUtcDay": fold["startDay"], "validationEndUtcDay": fold["endDay"],
                "candidateBrier": round(candidate_brier, 6),
                "trainRateBaselineBrier": round(constant_brier, 6),
                "neutral50BaselineBrier": round(neutral_brier, 6)
            })
        if total:
            mean_brier = weighted_candidate / total
            train_rate_brier = weighted_constant / total
            neutral_brier = weighted_neutral / total
            best_baseline = min(train_rate_brier, neutral_brier)
            scores.append({
                "l2": l2, "meanValidationBrier": round(mean_brier, 6),
                "trainRateBaselineBrier": round(train_rate_brier, 6),
                "neutral50BaselineBrier": round(neutral_brier, 6),
                "bestConstantBaselineBrier": round(best_baseline, 6),
                "beatsBestConstantBaseline": mean_brier < best_baseline,
                "validationExamples": total, "folds": fold_results
            })
    if not scores:
        return {"available": False, "reason": "no_valid_scores", "selectedL2": L2,
                "candidates": [], "folds": []}
    # Hyperparameters are selected on development-only walk-forward scores.
    # On ties within 1e-6, prefer the more regularized (simpler) candidate.
    best = min(scores, key=lambda row: (round(row["meanValidationBrier"], 6), -row["l2"]))
    return {
        "available": True, "method": "3-fold expanding-window chronological validation",
        "selectionMetric": "weighted mean validation Brier score",
        "selectionData": "development dates only; final chronological holdout excluded",
        "selectedL2": best["l2"], "selectedMeanValidationBrier": best["meanValidationBrier"],
        "selectedBeatsBestConstantBaseline": best["beatsBestConstantBaseline"],
        "candidateGrid": L2_GRID, "candidates": scores,
        "warning": "Validation windows are correlated market periods and do not establish future trading profitability."
    }


def predict_probability(features, model):
    scaled = transformed(features, model["featureMeans"], model["featureScales"])
    return sigmoid(model["weights"][0] + sum(model["weights"][j + 1] * scaled[j]
                                             for j in range(len(scaled))))


def brier_score(probabilities, labels):
    if not labels or len(probabilities) != len(labels):
        return None
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)


def log_loss(probabilities, labels):
    if not labels or len(probabilities) != len(labels):
        return None
    epsilon = 1e-15
    return -sum(label * math.log(max(epsilon, min(1 - epsilon, p))) +
                (1 - label) * math.log(max(epsilon, min(1 - epsilon, 1 - p)))
                for p, label in zip(probabilities, labels)) / len(labels)


def calibration_bins(probabilities, labels, bins=10):
    output = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        pairs = [(p, y) for p, y in zip(probabilities, labels)
                 if low <= p < high or (index == bins - 1 and p == 1)]
        if pairs:
            output.append({"lowerBound": round(low, 2), "upperBound": round(high, 2),
                           "count": len(pairs),
                           "meanPredictedProbability": round(sum(p for p, _ in pairs) / len(pairs), 5),
                           "observedUpRate": round(sum(y for _, y in pairs) / len(pairs), 5)})
    return output


def utc_date(day_number):
    return datetime.datetime.fromtimestamp(day_number * DAY_MS / 1000, datetime.timezone.utc).date().isoformat()


def dataset_fingerprint(observations):
    canonical = "\n".join(
        f"{row['id']}|{row['time']}|{float(row['price']):.12g}"
        for row in observations
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def insufficient_report(examples, observations, fingerprint, reason):
    unique_coins = sorted({row["coin"] for row in examples})
    unique_days = sorted({row["time"] // DAY_MS for row in examples})
    report = {
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "insufficient_data", "model": MODEL_VERSION,
        "trainingData": "Real CoinGecko historical market_chart prices and prospective recorded prices only; synthetic scenarios excluded",
        "validation": "Chronological 70% UTC-day train / later holdout with a full UTC-day embargo; no random shuffling",
        "examples": len(examples), "uniqueCoins": len(unique_coins), "distinctExampleUtcDays": len(unique_days),
        "validMergedObservations": len(observations), "datasetSha256": fingerprint,
        "betterThanConstantBaseline": None, "warning": "Research only; not a trading recommendation or proof of future predictive skill.",
        "note": reason
    }
    candidate = {
        "modelVersion": MODEL_VERSION, "status": "NOT_FITTED",
        "reason": reason, "datasetSha256": fingerprint, "promotionEligible": False,
        "mode": "SHADOW_RESEARCH_ONLY"
    }
    return report, candidate


def evaluate_training(live_data, historical_data):
    observations = merge_observations(live_data, historical_data)
    examples = build_examples(observations)
    fingerprint = dataset_fingerprint(observations)
    train, test, test_start_day = chronological_split(examples)
    coin_count = len({row["coin"] for row in examples})
    all_days = sorted({row["time"] // DAY_MS for row in examples})
    test_days = sorted({row["time"] // DAY_MS for row in test})
    if (len(examples) < MIN_EXAMPLES or len(all_days) < MIN_DAYS or coin_count < MIN_COINS
            or len(train) < MIN_TRAIN_EXAMPLES or len(test) < MIN_TEST_EXAMPLES
            or len(test_days) < MIN_TEST_DAYS):
        reason = (f"Requires at least {MIN_EXAMPLES} examples, {MIN_DAYS} distinct example UTC days, "
                  f"{MIN_COINS} coins, {MIN_TRAIN_EXAMPLES} training examples, {MIN_TEST_EXAMPLES} holdout examples "
                  f"and {MIN_TEST_DAYS} holdout UTC days; observed {len(examples)} examples, {len(all_days)} days, "
                  f"{coin_count} coins, {len(train)} train examples, {len(test)} holdout examples and {len(test_days)} holdout days.")
        report, candidate = insufficient_report(examples, observations, fingerprint, reason)
        return report, candidate

    cv = walk_forward_select_l2(train)
    selected_l2 = cv.get("selectedL2", L2)
    model = fit_logistic(train, l2=selected_l2)
    labels = [row["y"] for row in test]
    probabilities = [predict_probability(row["x"], model) for row in test]
    train_rate = sum(row["y"] for row in train) / len(train)
    constant_probabilities = [train_rate] * len(test)
    neutral_probabilities = [0.5] * len(test)
    holdout_brier = brier_score(probabilities, labels)
    constant_brier = brier_score(constant_probabilities, labels)
    neutral_brier = brier_score(neutral_probabilities, labels)
    accuracy = sum(int((p >= 0.5) == bool(y)) for p, y in zip(probabilities, labels)) / len(labels)
    majority_accuracy = sum(int((train_rate >= 0.5) == bool(y)) for y in labels) / len(labels)
    train_dates = [row["time"] // DAY_MS for row in train]
    test_dates = [row["time"] // DAY_MS for row in test]
    best_constant_brier = min(constant_brier, neutral_brier)
    better = holdout_brier < best_constant_brier
    report = {
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "evaluated", "model": MODEL_VERSION,
        "trainingData": "Real CoinGecko historical market_chart prices and prospective recorded prices only; synthetic scenarios excluded",
        "validation": "Chronological UTC-day split; 70% earlier dates for training; one-day embargo; later dates held out; no random shuffling",
        "examples": len(examples), "uniqueCoins": coin_count, "distinctExampleUtcDays": len(all_days),
        "trainExamples": len(train), "testExamples": len(test), "testDays": len(set(test_dates)),
        "trainStartUtcDay": min(train_dates), "trainEndUtcDay": max(train_dates),
        "holdoutStartUtcDay": min(test_dates), "holdoutEndUtcDay": max(test_dates),
        "trainStartDateUtc": utc_date(min(train_dates)),
        "trainEndDateUtc": utc_date(max(train_dates)),
        "holdoutStartDateUtc": utc_date(min(test_dates)),
        "holdoutEndDateUtc": utc_date(max(test_dates)),
        "holdoutBrier": round(holdout_brier, 6),
        "constantTrainRateBrier": round(constant_brier, 6),
        "neutral50Brier": round(neutral_brier, 6),
        "bestConstantBaselineBrier": round(best_constant_brier, 6),
        "bestConstantBaselineName": "training_rate" if constant_brier <= neutral_brier else "neutral_0.5",
        "betterThanConstantBaseline": better,
        "baselineImprovementBrier": round(best_constant_brier - holdout_brier, 6),
        "holdoutLogLoss": round(log_loss(probabilities, labels), 6),
        "holdoutAccuracy": round(accuracy, 6),
        "constantMajorityAccuracy": round(majority_accuracy, 6),
        "calibrationBins": calibration_bins(probabilities, labels),
        "walkForwardCV": cv,
        "optimizer": model["optimizer"],
        "datasetSha256": fingerprint,
        "holdoutLimitations": [
            "A single chronological holdout is not proof of durable predictive skill.",
            "Crypto assets share market-wide shocks, so asset rows are not independent trials.",
            "This is a direction-probability model, not a tested net-return or trading strategy.",
            "No transaction costs, execution slippage or market impact are included in this prediction metric."
        ],
        "warning": "Research only. The model remains an unpromoted paper/research candidate."
    }
    candidate = dict(model)
    candidate.update({
        "status": "FITTED_UNPROMOTED_CANDIDATE",
        "datasetSha256": fingerprint, "generatedAt": report["generatedAt"],
        "trainingExamples": len(train), "holdoutExamples": len(test),
        "trainStartUtcDay": min(train_dates), "trainEndUtcDay": max(train_dates),
        "holdoutStartUtcDay": min(test_dates), "holdoutEndUtcDay": max(test_dates),
        "trainStartDateUtc": utc_date(min(train_dates)), "trainEndDateUtc": utc_date(max(train_dates)),
        "holdoutStartDateUtc": utc_date(min(test_dates)), "holdoutEndDateUtc": utc_date(max(test_dates)),
        "holdoutBrier": round(holdout_brier, 6), "constantTrainRateBrier": round(constant_brier, 6),
        "betterThanConstantBaseline": better,
        "walkForwardCV": cv,
        "note": "Weights are fitted for research reproducibility only. Automatic promotion and real trading are disabled."
    })
    return report, candidate


def main():
    live = read_json(LIVE_PATH)
    historical = read_json(HIST_PATH)
    report, candidate = evaluate_training(live, historical)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    MODEL_PATH.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    print(f"Research training: {report['status']} | examples: {report['examples']} | "
          f"coins: {report['uniqueCoins']} | days: {report['distinctExampleUtcDays']}")
    if report.get("status") == "evaluated":
        print(f"Holdout Brier {report['holdoutBrier']} vs constant baseline {report['constantTrainRateBrier']} | "
              f"better baseline: {report['betterThanConstantBaseline']}")
    # Insufficient history is an honest research state, not a workflow crash.
    if report.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
