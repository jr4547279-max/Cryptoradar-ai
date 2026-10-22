"""Unit tests for frozen prospective model trials; fixtures never enter production training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_research import DAY_MS, FEATURE_NAMES
from prospective_trial import (
    MODEL_KEY, create_frozen_trial, predict_trial_probability, retain_frozen_trial, score_trial,
)

NOW_MS = 1_800_000_000_000
DAY = DAY_MS


def candidate_fixture():
    return {
        "status": "FITTED_UNPROMOTED_CANDIDATE",
        "modelVersion": "trend-logistic-v1",
        "features": FEATURE_NAMES,
        "featureMeans": [0.0] * len(FEATURE_NAMES),
        "featureScales": [1.0] * len(FEATURE_NAMES),
        "weights": [0.0] * (len(FEATURE_NAMES) + 1),
        "optimizer": {"name": "test-fixture"},
        "datasetSha256": "a" * 64,
        "trainingPositiveRate": 0.54,
        "trainingExamples": 1000,
        "trainStartDateUtc": "2025-01-01",
        "trainEndDateUtc": "2025-09-01",
    }


def report_fixture(cv_pass=True):
    return {
        "status": "evaluated",
        "holdoutStatus": "REUSED_HISTORICAL_EXPLORATORY_BENCHMARK",
        "holdoutBrier": 0.26,
        "bestConstantBaselineBrier": 0.25,
        "betterThanConstantBaseline": False,
        "walkForwardCV": {
            "available": True,
            "selectedL2": 0.5,
            "selectedMeanValidationBrier": 0.249,
            "selectedBeatsBestConstantBaseline": cv_pass,
        }
    }


def test_trial_freezes_only_a_qualified_candidate():
    candidate = candidate_fixture()
    trial, status = create_frozen_trial(candidate, report_fixture(True), NOW_MS)
    assert trial["status"] == "ACTIVE_EXPERIMENTAL_SHADOW_TRIAL"
    assert trial["modelHash"] and trial["model"]["weights"] == candidate["weights"]
    assert trial["promotionEligible"] is False and trial["realTradingEnabled"] is False
    assert status["createdNow"] is True
    missing, status = create_frozen_trial(candidate, report_fixture(False), NOW_MS)
    assert missing is None and status["status"] == "WAITING_FOR_QUALIFIED_CANDIDATE"


def test_existing_trial_is_never_retrained_or_overwritten():
    candidate = candidate_fixture()
    trial, _ = create_frozen_trial(candidate, report_fixture(True), NOW_MS)
    changed_candidate = dict(candidate)
    changed_candidate["weights"] = [0.1] + [0.0] * len(FEATURE_NAMES)
    changed_candidate["datasetSha256"] = "b" * 64
    kept, status = retain_frozen_trial(trial, changed_candidate, report_fixture(True), NOW_MS + DAY)
    assert kept["modelHash"] == trial["modelHash"]
    assert kept["model"]["weights"] == trial["model"]["weights"]
    assert status["createdNow"] is False


def test_trial_hash_detects_changed_frozen_weights():
    trial, _ = create_frozen_trial(candidate_fixture(), report_fixture(True), NOW_MS)
    trial["model"]["weights"][0] += 0.01
    kept, status = retain_frozen_trial(trial, candidate_fixture(), report_fixture(True), NOW_MS + DAY)
    assert kept["model"]["weights"][0] != candidate_fixture()["weights"][0]
    assert status["status"] == "TRIAL_INTEGRITY_FAIL"


def test_live_inference_uses_only_available_past_daily_prices():
    trial, _ = create_frozen_trial(candidate_fixture(), report_fixture(True), NOW_MS)
    today = NOW_MS // DAY
    now_ms = today * DAY + DAY - 1
    history = [{"id": "bitcoin", "time": day * DAY, "price": 100.0 + day,
                "source": "fixture-only"} for day in range(today - 30, today + 1)]
    probability = predict_trial_probability(trial, "bitcoin", history, [], now_ms)
    assert probability is not None and abs(probability["pUp"] - 0.5) < 1e-12
    future = history + [{"id": "bitcoin", "time": now_ms + 60_000, "price": 1.0e9}]
    with_future = predict_trial_probability(trial, "bitcoin", future, [], now_ms)
    assert with_future == probability


def test_prospective_scorer_requires_mature_frozen_model_forecasts():
    trial, _ = create_frozen_trial(candidate_fixture(), report_fixture(True), NOW_MS)
    forecasts = []
    first_day = NOW_MS // DAY + 1
    for day in range(first_day, first_day + 30):
        for coin_index in range(30):
            created = day * DAY + coin_index * 100
            forecasts.append({
                "id": f"coin-{coin_index}", "created": created, "price": 100,
                "outcome": 1, "scoredAt": created + DAY, "scoredPrice": 101,
                "challengers": {MODEL_KEY: {
                    "pUp": 0.8, "trialId": trial["trialId"],
                    "modelVersion": trial["modelVersion"], "modelHash": trial["modelHash"],
                    "trainingDatasetSha256": trial["trainingProvenance"]["datasetSha256"]
                }}
            })
    report = score_trial(trial, {"forecasts": forecasts}, now_ms=(first_day + 100) * DAY)
    assert report["status"] == "EVALUATED"
    assert report["maturedForecasts"] == 900
    assert report["distinctForecastCreationUtcDays"] == 30
    assert report["passesProbabilityBenchmark"] is True
    assert report["bootstrapBrierDifferenceVsNeutral50"]["upper95"] < 0


if __name__ == "__main__":
    test_trial_freezes_only_a_qualified_candidate()
    test_existing_trial_is_never_retrained_or_overwritten()
    test_trial_hash_detects_changed_frozen_weights()
    test_live_inference_uses_only_available_past_daily_prices()
    test_prospective_scorer_requires_mature_frozen_model_forecasts()
    print("Prospective trial tests passed (5 cases)")
