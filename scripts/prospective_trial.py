"""Frozen, paper-only forward trial utilities for the first learned Trend model.

The trial model is copied once and then immutable. Prospective scores include only
forecasts created after its activation and carry its exact model hash. This module
never places orders or promotes models.
"""
import datetime
import hashlib
import json
import math
import random

from train_research import DAY_MS, FEATURE_NAMES, feature_vector_for_day, predict_probability

MODEL_KEY = "trend_logistic_v1"
TRIAL_ID = "trend-logistic-v1-forward-001"
MIN_MATURED_FORECASTS = 500
MIN_DISTINCT_DAYS = 30
MIN_DISTINCT_COINS = 10
BOOTSTRAP_REPS = 1000


def _iso(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).isoformat()


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def frozen_model_payload(candidate):
    keys = ("modelVersion", "features", "featureMeans", "featureScales", "weights",
            "optimizer", "datasetSha256", "trainingPositiveRate")
    return {key: candidate[key] for key in keys if key in candidate}


def model_fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def candidate_validation_error(candidate, training_report):
    if not isinstance(candidate, dict) or candidate.get("status") != "FITTED_UNPROMOTED_CANDIDATE":
        return "No fitted unpromoted candidate is available."
    if candidate.get("modelVersion") != "trend-logistic-v1":
        return "Candidate is not the expected Trend model version."
    if candidate.get("features") != FEATURE_NAMES:
        return "Candidate feature schema does not match the current inference builder."
    if not isinstance(candidate.get("weights"), list) or len(candidate["weights"]) != len(FEATURE_NAMES) + 1:
        return "Candidate weight count does not match its feature schema."
    if not isinstance(candidate.get("featureMeans"), list) or len(candidate["featureMeans"]) != len(FEATURE_NAMES):
        return "Candidate feature means do not match its feature schema."
    if not isinstance(candidate.get("featureScales"), list) or len(candidate["featureScales"]) != len(FEATURE_NAMES):
        return "Candidate feature scales do not match its feature schema."
    if not all(_finite_number(value) for value in candidate["weights"] + candidate["featureMeans"] + candidate["featureScales"]):
        return "Candidate contains a non-finite model parameter."
    if any(value <= 0 for value in candidate["featureScales"]):
        return "Candidate contains a non-positive feature scale."
    if not isinstance(candidate.get("datasetSha256"), str) or len(candidate["datasetSha256"]) != 64:
        return "Candidate dataset fingerprint is missing."
    if not _finite_number(candidate.get("trainingPositiveRate")) or not 0 <= candidate["trainingPositiveRate"] <= 1:
        return "Candidate training label prevalence is missing or invalid."
    cv = training_report.get("walkForwardCV", {}) if isinstance(training_report, dict) else {}
    if not isinstance(cv, dict) or cv.get("available") is not True:
        return "Development-only walk-forward validation is not available."
    if cv.get("selectedBeatsBestConstantBaseline") is not True:
        return "Candidate did not beat the constant baseline in development-only walk-forward validation."
    return None


def create_frozen_trial(candidate, training_report, active_from_ms):
    """Freeze a single experimental candidate only after the development CV gate."""
    error = candidate_validation_error(candidate, training_report)
    if error:
        return None, {"status": "WAITING_FOR_QUALIFIED_CANDIDATE", "checkedAt": _iso(active_from_ms),
                      "reason": error, "realTradingEnabled": False, "automaticPromotion": False}
    payload = frozen_model_payload(candidate)
    fingerprint = model_fingerprint(payload)
    trial = {
        "schemaVersion": 1,
        "trialId": TRIAL_ID,
        "status": "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL",
        "activatedAt": _iso(active_from_ms),
        "activeFromMs": active_from_ms,
        "modelVersion": candidate["modelVersion"],
        "modelHash": fingerprint,
        "model": payload,
        "trainingProvenance": {
            "datasetSha256": candidate["datasetSha256"],
            "trainingExamples": candidate.get("trainingExamples"),
            "trainingPositiveRate": candidate["trainingPositiveRate"],
            "trainingCoins": candidate.get("trainingCoins", []),
            "trainingStartDateUtc": candidate.get("trainStartDateUtc"),
            "trainingEndDateUtc": candidate.get("trainEndDateUtc"),
            "walkForwardSelectedL2": (training_report.get("walkForwardCV") or {}).get("selectedL2"),
            "walkForwardMeanValidationBrier": (training_report.get("walkForwardCV") or {}).get("selectedMeanValidationBrier"),
            "walkForwardBeatsConstantBaseline": (training_report.get("walkForwardCV") or {}).get("selectedBeatsBestConstantBaseline"),
            "historicalHoldoutStatus": training_report.get("holdoutStatus", "UNSPECIFIED"),
            "historicalHoldoutBrier": training_report.get("holdoutBrier"),
            "historicalHoldoutBestConstantBaselineBrier": training_report.get("bestConstantBaselineBrier"),
            "historicalHoldoutBeatsConstantBaseline": training_report.get("betterThanConstantBaseline")
        },
        "evaluationProtocol": {
            "minimumMaturedForecasts": MIN_MATURED_FORECASTS,
            "minimumDistinctUtcDays": MIN_DISTINCT_DAYS,
            "minimumDistinctCoins": MIN_DISTINCT_COINS,
            "bootstrapReplicates": BOOTSTRAP_REPS,
            "confidenceLevel": 0.95,
            "clusterUnit": "forecast creation UTC day",
            "mustBeatBoth": ["neutral_0.5", "frozen_training_up_rate"],
            "frozenAfterActivation": True
        },
        "promotionEligible": False,
        "automaticPromotion": False,
        "realTradingEnabled": False,
        "purpose": "Independent prospective probability-quality evaluation only; historical holdout is exploratory and reused."
    }
    status = {
        "status": "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL",
        "checkedAt": _iso(active_from_ms),
        "createdNow": True,
        "trialId": TRIAL_ID,
        "modelVersion": trial["modelVersion"],
        "modelHash": fingerprint,
        "historicalHoldoutBeatsBaseline": training_report.get("betterThanConstantBaseline"),
        "warning": "This is a frozen research experiment, not an endorsement or a trading signal.",
        "realTradingEnabled": False,
        "automaticPromotion": False
    }
    return trial, status


def retain_frozen_trial(existing_trial, candidate=None, training_report=None, now_ms=None):
    """Never overwrite an already active trial when a newer candidate is trained."""
    now_ms = now_ms if now_ms is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    if isinstance(existing_trial, dict) and existing_trial.get("status") == "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL":
        payload = existing_trial.get("model", {})
        expected = model_fingerprint(payload) if isinstance(payload, dict) else None
        if expected != existing_trial.get("modelHash"):
            return existing_trial, {
                "status": "TRIAL_INTEGRITY_FAIL", "checkedAt": _iso(now_ms),
                "reason": "Frozen model parameters no longer match the stored model hash.",
                "trialId": existing_trial.get("trialId"), "modelHash": existing_trial.get("modelHash"),
                "realTradingEnabled": False, "automaticPromotion": False
            }
        return existing_trial, {
            "status": "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL", "checkedAt": _iso(now_ms),
            "createdNow": False, "trialId": existing_trial.get("trialId"),
            "modelVersion": existing_trial.get("modelVersion"), "modelHash": existing_trial.get("modelHash"),
            "activatedAt": existing_trial.get("activatedAt"),
            "warning": "Frozen parameters retained; newly trained candidates do not change this trial.",
            "realTradingEnabled": False, "automaticPromotion": False
        }
    return create_frozen_trial(candidate or {}, training_report or {}, now_ms)


def price_series_for_coin(coin_id, historical_observations, live_observations, now_ms):
    """Return one latest-known price per UTC day, never including future-timestamped rows."""
    per_day = {}
    for source_priority, rows in ((0, historical_observations), (1, live_observations)):
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") != coin_id:
                continue
            timestamp, price = row.get("time"), row.get("price")
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0 or timestamp > now_ms:
                continue
            if not _finite_number(price) or price <= 0:
                continue
            day = timestamp // DAY_MS
            current = per_day.get(day)
            if current is None or timestamp > current[0] or (timestamp == current[0] and source_priority > current[2]):
                per_day[day] = (timestamp, float(price), source_priority)
    return {day: row[1] for day, row in per_day.items()}


def predict_trial_probability(trial, coin_id, historical_observations, live_observations, now_ms):
    if not isinstance(trial, dict) or trial.get("status") != "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL":
        return None
    if not isinstance(trial.get("activeFromMs"), int) or now_ms < trial["activeFromMs"]:
        return None
    model = trial.get("model")
    if not isinstance(model, dict) or model.get("features") != FEATURE_NAMES:
        return None
    trained_coins = (trial.get("trainingProvenance") or {}).get("trainingCoins")
    if isinstance(trained_coins, list) and trained_coins and coin_id not in trained_coins:
        return None
    if model_fingerprint(model) != trial.get("modelHash"):
        return None
    prices = price_series_for_coin(coin_id, historical_observations, live_observations, now_ms)
    today = now_ms // DAY_MS
    features = feature_vector_for_day(prices, today)
    if features is None:
        return None
    probability = predict_probability(features, model)
    if not _finite_number(probability) or not 0 <= probability <= 1:
        return None
    return {
        "pUp": probability,
        "trialId": trial["trialId"],
        "modelVersion": trial["modelVersion"],
        "modelHash": trial["modelHash"],
        "trainingDatasetSha256": (trial.get("trainingProvenance") or {}).get("datasetSha256"),
    }


def brier_score(probabilities, labels):
    if not labels or len(probabilities) != len(labels):
        return None
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)


def _log_loss(probabilities, labels):
    if not labels or len(probabilities) != len(labels):
        return None
    eps = 1e-15
    return -sum(y * math.log(max(eps, min(1 - eps, p))) +
                (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
                for p, y in zip(probabilities, labels)) / len(labels)


def _cluster_bootstrap(rows, baseline_probability, seed, replicates=BOOTSTRAP_REPS):
    by_day = {}
    for row in rows:
        by_day.setdefault(row["created"] // DAY_MS, []).append(row)
    days = sorted(by_day)
    if len(days) < 2:
        return {"lower95": None, "upper95": None, "replicates": 0, "clusters": len(days)}
    rng = random.Random(seed)
    differences = []
    for _ in range(replicates):
        sample_days = [rng.choice(days) for _ in days]
        model_losses, baseline_losses = [], []
        for day in sample_days:
            for row in by_day[day]:
                model_losses.append((row["pUp"] - row["outcome"]) ** 2)
                baseline_losses.append((baseline_probability - row["outcome"]) ** 2)
        if model_losses:
            differences.append(sum(model_losses) / len(model_losses) -
                               sum(baseline_losses) / len(baseline_losses))
    differences.sort()
    if not differences:
        lower = upper = None
    else:
        lower = differences[int(0.025 * (len(differences) - 1))]
        upper = differences[int(0.975 * (len(differences) - 1))]
    return {"lower95": round(lower, 6) if lower is not None else None,
            "upper95": round(upper, 6) if upper is not None else None,
            "replicates": len(differences), "clusters": len(days)}


def score_trial(trial, research_data, now_ms=None):
    now_ms = now_ms if now_ms is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    if not isinstance(trial, dict) or trial.get("status") != "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL":
        return {"status": "NOT_STARTED", "checkedAt": _iso(now_ms), "maturedForecasts": 0,
                "note": "No frozen prospective trial has been activated.", "realTradingEnabled": False}
    forecasts = research_data.get("forecasts", []) if isinstance(research_data, dict) else []
    rows, pending, excluded = [], 0, 0
    start = trial.get("activeFromMs", 0)
    for forecast in forecasts if isinstance(forecasts, list) else []:
        if not isinstance(forecast, dict):
            continue
        created = forecast.get("created")
        if isinstance(created, bool) or not isinstance(created, int) or created < start:
            continue
        challenge = (forecast.get("challengers") or {}).get(MODEL_KEY)
        if not isinstance(challenge, dict):
            continue
        expected_data_hash = (trial.get("trainingProvenance") or {}).get("datasetSha256")
        if (challenge.get("trialId") != trial.get("trialId")
                or challenge.get("modelVersion") != trial.get("modelVersion")
                or challenge.get("modelHash") != trial.get("modelHash")
                or challenge.get("trainingDatasetSha256") != expected_data_hash):
            excluded += 1
            continue
        probability = challenge.get("pUp")
        if not _finite_number(probability) or not 0 <= probability <= 1:
            excluded += 1
            continue
        outcome, scored = forecast.get("outcome"), forecast.get("scoredAt")
        if outcome is None:
            pending += 1
            continue
        if outcome not in (0, 1) or isinstance(outcome, bool) or not isinstance(scored, int) or scored < created + DAY_MS:
            excluded += 1
            continue
        rows.append({"id": forecast.get("id"), "created": created,
                     "scoredAt": scored, "pUp": float(probability), "outcome": int(outcome)})

    coin_ids = {row["id"] for row in rows if isinstance(row.get("id"), str)}
    days = sorted({row["created"] // DAY_MS for row in rows})
    generated_at = _iso(now_ms)
    report = {
        "checkedAt": generated_at,
        "trialId": trial.get("trialId"), "modelVersion": trial.get("modelVersion"),
        "modelHash": trial.get("modelHash"), "activatedAt": trial.get("activatedAt"),
        "activeFromMs": start, "frozenModel": True,
        "maturedForecasts": len(rows), "pendingForecasts": pending,
        "excludedOrInvalidForecasts": excluded,
        "uniqueCoins": len(coin_ids), "distinctForecastCreationUtcDays": len(days),
        "minimumEvidence": {
            "maturedForecasts": MIN_MATURED_FORECASTS,
            "distinctForecastCreationUtcDays": MIN_DISTINCT_DAYS,
            "uniqueCoins": MIN_DISTINCT_COINS
        },
        "realTradingEnabled": False, "automaticPromotion": False,
        "note": "Prospective forecast-probability evaluation only; not evidence of net trading profitability."
    }
    if not rows:
        report.update({"status": "WAITING_FOR_MATURED_EVIDENCE", "brier": None,
                       "neutral50Brier": None, "frozenTrainingRateBrier": None,
                       "passesProbabilityBenchmark": False})
        return report

    probs = [row["pUp"] for row in rows]
    labels = [row["outcome"] for row in rows]
    training_rate = trial.get("trainingProvenance", {}).get("trainingPositiveRate", 0.5)
    if not _finite_number(training_rate) or not 0 <= training_rate <= 1:
        training_rate = 0.5
    candidate_brier = brier_score(probs, labels)
    neutral_brier = brier_score([0.5] * len(rows), labels)
    train_rate_brier = brier_score([training_rate] * len(rows), labels)
    best_baseline = min(neutral_brier, train_rate_brier)
    neutral_delta = candidate_brier - neutral_brier
    train_rate_delta = candidate_brier - train_rate_brier
    seed = int(trial["modelHash"][:8], 16)
    bootstrap_neutral = _cluster_bootstrap(rows, 0.5, seed)
    bootstrap_train_rate = _cluster_bootstrap(rows, training_rate, seed + 1)
    accuracy = sum(int((row["pUp"] >= 0.5) == bool(row["outcome"])) for row in rows) / len(rows)
    enough = (len(rows) >= MIN_MATURED_FORECASTS and len(days) >= MIN_DISTINCT_DAYS
              and len(coin_ids) >= MIN_DISTINCT_COINS)
    beats_both = candidate_brier < neutral_brier and candidate_brier < train_rate_brier
    ci_beats_both = (
        bootstrap_neutral["upper95"] is not None and bootstrap_neutral["upper95"] < 0
        and bootstrap_train_rate["upper95"] is not None and bootstrap_train_rate["upper95"] < 0
    )
    report.update({
        "status": "EVALUATED" if enough else "INSUFFICIENT_EVIDENCE",
        "firstForecastUtc": _iso(min(row["created"] for row in rows)),
        "lastForecastUtc": _iso(max(row["created"] for row in rows)),
        "lastResolutionUtc": _iso(max(row["scoredAt"] for row in rows)),
        "brier": round(candidate_brier, 6), "neutral50Brier": round(neutral_brier, 6),
        "frozenTrainingRate": round(training_rate, 6),
        "frozenTrainingRateBrier": round(train_rate_brier, 6),
        "bestConstantBaselineBrier": round(best_baseline, 6),
        "brierImprovementVsNeutral50": round(neutral_brier - candidate_brier, 6),
        "brierImprovementVsTrainingRate": round(train_rate_brier - candidate_brier, 6),
        "logLoss": round(_log_loss(probs, labels), 6),
        "accuracyAt50Percent": round(accuracy, 6),
        "observedUpRate": round(sum(labels) / len(labels), 6),
        "bootstrapBrierDifferenceVsNeutral50": bootstrap_neutral,
        "bootstrapBrierDifferenceVsTrainingRate": bootstrap_train_rate,
        "beatsBothConstantBaselines": beats_both,
        "clusterBootstrapUpper95BelowZeroVsBoth": ci_beats_both,
        "passesProbabilityBenchmark": bool(enough and beats_both and ci_beats_both),
        "decision": ("PASSES_PROBABILITY_BENCHMARK_ONLY" if enough and beats_both and ci_beats_both
                     else "NOT_DEMONSTRATED" if enough else "WAIT_FOR_MORE_FORWARD_EVIDENCE")
    })
    return report
