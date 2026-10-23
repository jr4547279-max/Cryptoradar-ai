"""Challenger inference integrity checks. All fixtures are synthetic tests only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from challenger_shadow import predict_challenger
from train_research import FEATURE_NAMES, DAY_MS


def test_rejects_missing_or_unfrozen():
    assert predict_challenger({}, "coin", [], [], 100 * DAY_MS) is None
    assert predict_challenger({"status": "EVALUATED"}, "coin", [], [], 100 * DAY_MS) is None


def test_rejects_prediction_before_freeze():
    candidate = {
        "status": "FROZEN_DEVELOPMENT_CHALLENGER",
        "generatedAt": "2026-10-23T10:00:00+00:00",
        "features": FEATURE_NAMES,
        "weights": [0.0] * (len(FEATURE_NAMES) + 1),
        "featureMeans": [0.0] * len(FEATURE_NAMES),
        "featureScales": [1.0] * len(FEATURE_NAMES),
        "featureIndices": [1, 2, 3],
        "featureGroup": "medium_momentum",
    }
    assert predict_challenger(candidate, "coin", [], [], 100 * DAY_MS) is None


if __name__ == "__main__":
    test_rejects_missing_or_unfrozen()
    test_rejects_prediction_before_freeze()
    print("Challenger inference integrity tests passed")
