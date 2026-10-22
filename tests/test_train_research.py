"""Tests for training math and temporal safeguards. Fixtures are test-only, never production evidence."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_research import (
    DAY_MS, FEATURE_NAMES, brier_score, build_examples, chronological_split, evaluate_training,
    fit_logistic, predict_probability,
)


def synthetic_fixture_only_for_test():
    observations = []
    for coin_index in range(10):
        coin = f"fixture-coin-{coin_index}"
        for day in range(220):
            # Deterministic fixtures exercise code paths; production loads only persisted provider data.
            price = 100.0 * math.exp(0.002 * day + 0.025 * math.sin(day * 0.27 + coin_index * 0.61))
            observations.append({"id": coin, "time": day * DAY_MS + coin_index * 1000,
                                 "price": price, "source": "unit-test-fixture"})
    return {"observations": observations}


def test_brier_score_uses_each_label():
    assert abs(brier_score([0.9, 0.1], [1, 0]) - 0.01) < 1e-12
    assert abs(brier_score([0.5, 0.5], [1, 0]) - 0.25) < 1e-12


def test_builder_and_split_are_chronological_with_day_embargo():
    data = synthetic_fixture_only_for_test()
    examples = build_examples(data["observations"])
    train, test, start_day = chronological_split(examples)
    assert len(examples) > 500
    assert {row["coin"] for row in examples} == {f"fixture-coin-{i}" for i in range(10)}
    train_days = {row["time"] // DAY_MS for row in train}
    test_days = {row["time"] // DAY_MS for row in test}
    assert train_days and test_days
    assert max(train_days) < start_day - 1
    assert min(test_days) == start_day
    assert not (train_days & test_days)


def test_model_fit_is_finite_and_reproducible():
    examples = build_examples(synthetic_fixture_only_for_test()["observations"])
    train, test, _ = chronological_split(examples)
    model_a = fit_logistic(train)
    model_b = fit_logistic(train)
    assert model_a["weights"] == model_b["weights"]
    assert len(model_a["weights"]) == len(FEATURE_NAMES) + 1
    assert all(math.isfinite(value) for value in model_a["weights"])
    assert all(0 <= predict_probability(row["x"], model_a) <= 1 for row in test[:100])


def test_evaluation_emits_shadow_candidate_not_promoted_model():
    live = synthetic_fixture_only_for_test()
    report, candidate = evaluate_training(live, {"observations": []})
    assert report["status"] == "evaluated"
    assert report["testExamples"] > 0
    assert report["walkForwardCV"]["available"] is True
    assert report["walkForwardCV"]["selectedL2"] in report["walkForwardCV"]["candidateGrid"]
    assert report["testDays"] >= 21
    assert "holdoutBrier" in report and "constantTrainRateBrier" in report
    assert candidate["status"] == "FITTED_UNPROMOTED_CANDIDATE"
    assert candidate["promotionEligible"] is False
    assert candidate["mode"] == "SHADOW_RESEARCH_ONLY"


def test_insufficient_data_does_not_fit_or_promote():
    observations = [{"id": "only-one", "time": day * DAY_MS, "price": 100 + day}
                    for day in range(10)]
    report, candidate = evaluate_training({"observations": observations}, {"observations": []})
    assert report["status"] == "insufficient_data"
    assert candidate["status"] == "NOT_FITTED"
    assert candidate["promotionEligible"] is False


if __name__ == "__main__":
    test_brier_score_uses_each_label()
    test_builder_and_split_are_chronological_with_day_embargo()
    test_model_fit_is_finite_and_reproducible()
    test_evaluation_emits_shadow_candidate_not_promoted_model()
    test_insufficient_data_does_not_fit_or_promote()
    print("Research trainer tests passed (5 cases)")
