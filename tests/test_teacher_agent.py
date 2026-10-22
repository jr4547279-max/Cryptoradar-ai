"""Teacher tests: schema-aware grading, persistent memory, hard safety boundaries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from teacher_agent import teach


def test():
    training = {"status": "evaluated", "holdoutBrier": 0.25,
                "constantTrainRateBrier": 0.24, "examples": 420}
    rewards = {"agents": {"modelA": {"eligible": False}}}
    integrity = {"status": "PASS", "forecasts": 0}
    result, memory = teach(training, rewards, integrity, {}, {})
    assert result["grade"] == "NEEDS_MORE_EVIDENCE"
    assert any(x["topic"] == "predictive_skill" and "did not beat the best available constant baseline" in x["finding"] for x in result["lessons"])
    assert "place_orders" in result["actionsForbidden"]
    assert "edit_model_weights" in result["actionsForbidden"]
    assert memory["history"]

    report2, memory2 = teach({"status": "evaluated", "holdoutBrier": 0.20,
                  "constantTrainRateBrier": 0.25, "examples": 420},
                 {"agents": {"modelA": {"eligible": True}}},
                 {"status": "PASS", "forecasts": 30}, {}, memory)
    assert report2["grade"] == "SHADOW_REVIEW_ONLY"
    assert report2["eligibleForReward"] == ["modelA"]
    assert "promote_model" in report2["actionsForbidden"]
    assert any(x["topic"] == "predictive_skill" and "promising observation" in x["lesson"] for x in report2["lessons"])
    assert len(memory2["history"]) == 2

    repeated, _ = teach(training, rewards, integrity, {}, memory2)
    assert repeated["sameLessonPatternSeenBefore"] is True

    bad_integrity, _ = teach({}, {"agents": {}}, {"status": "FAIL", "forecasts": 2}, {})
    assert any(x["topic"] == "integrity" for x in bad_integrity["lessons"])
    assert bad_integrity["grade"] == "NEEDS_MORE_EVIDENCE"
    print("Teacher agent adversarial tests passed")


if __name__ == "__main__":
    test()
