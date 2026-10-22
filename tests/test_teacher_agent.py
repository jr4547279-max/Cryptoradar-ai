"""Teacher agent tests: no self-promotion, useful feedback, fail-closed behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from teacher_agent import teach


def test():
    training = {"scores": {"baseline": {"brier": 0.24}, "modelA": {"brier": 0.25}}}
    rewards = {"agents": {"modelA": {"eligible": False}}}
    integrity = {"status": "PASS"}
    result = teach(training, rewards, integrity, {})
    assert result["grade"] == "NEEDS_MORE_EVIDENCE"
    assert any(x["topic"] == "predictive_skill" for x in result["lessons"])
    assert "place_orders" in result["actionsForbidden"]
    assert "edit_model_weights" in result["actionsForbidden"]

    good = teach({"scores": {"baseline": {"brier": 0.25}, "modelA": {"brier": 0.20}}},
                 {"agents": {"modelA": {"eligible": True}}}, {"status": "PASS"}, {})
    assert good["grade"] == "SHADOW_REVIEW_ONLY"
    assert good["eligibleForReward"] == ["modelA"]
    assert "promote_model" in good["actionsForbidden"]

    bad_integrity = teach({}, {"agents": {}}, {"status": "FAIL"}, {})
    assert any(x["topic"] == "integrity" for x in bad_integrity["lessons"])
    assert bad_integrity["grade"] == "NEEDS_MORE_EVIDENCE"
    print("Teacher agent adversarial tests passed")


if __name__ == "__main__":
    test()
