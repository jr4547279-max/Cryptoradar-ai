"""Teacher agent: grades research agents and keeps a bounded lesson history.

Deterministic supervisor, not an LLM. It can recommend experiments and remember
past lessons, but cannot change model weights, scoring rules, promotion state, or trade.
"""
import json
import pathlib
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEMORY_PATH = ROOT / "teacher-memory.json"


def _brier_score(report):
    # Current trainer schema, plus backward-compatible schemas used by earlier versions.
    for key in ("holdoutBrier", "aggregateBrier"):
        value = report.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    scores = report.get("scores") or {}
    if not isinstance(scores, dict):
        return None, None
    baseline = scores.get("constant", scores.get("baseline", {}))
    base = baseline.get("brier") if isinstance(baseline, dict) else baseline
    candidates = [(name, item.get("brier")) for name, item in scores.items()
                  if name not in ("constant", "baseline") and isinstance(item, dict)
                  and isinstance(item.get("brier"), (int, float))]
    best = min(candidates, key=lambda x: x[1]) if candidates else None
    return (best[0], best[1], base) if best else (None, None, base)


def teach(training, rewards, integrity, scenario, memory=None):
    lessons = []
    status = training.get("status")
    if status == "evaluated" and isinstance(training.get("holdoutBrier"), (int, float)) and isinstance(training.get("constantTrainRateBrier"), (int, float)):
        model = training["holdoutBrier"]
        baseline = training["constantTrainRateBrier"]
        if model < baseline:
            lessons.append({
                "topic": "predictive_skill",
                "finding": f"Chronological holdout Brier {model:.5f} beat the training-rate baseline {baseline:.5f}.",
                "lesson": "One holdout win is a promising observation, not proof of a durable edge.",
                "nextExperiment": "Freeze the model and test on a new, untouched forward window with fees and calibration reported."
            })
        else:
            lessons.append({
                "topic": "predictive_skill",
                "finding": f"Chronological holdout Brier {model:.5f} did not beat baseline {baseline:.5f}.",
                "lesson": "The current model has not demonstrated predictive skill against its simple baseline.",
                "nextExperiment": "Keep the candidate shadow-only; test simpler features and probability calibration without touching the holdout."
            })
    else:
        lessons.append({
            "topic": "evidence_quality",
            "finding": f"Training status is {status or 'MISSING'}; examples={training.get('examples', 'unknown')}.",
            "lesson": "No grade for predictive skill is justified until enough real timestamped observations exist.",
            "nextExperiment": "Continue collecting real observations; require chronological holdout evaluation before comparing candidates."
        })

    if integrity.get("status") != "PASS" or integrity.get("forecasts", 0) == 0:
        lessons.append({
            "topic": "integrity",
            "finding": f"Integrity status is {integrity.get('status', 'MISSING')} with {integrity.get('forecasts', 0)} forecasts.",
            "lesson": "A clean empty file is not evidence of predictive performance; malformed or duplicated records must receive no reward.",
            "nextExperiment": "Verify that valid specialist predictions are being recorded and rerun adversarial integrity tests."
        })

    agent_results = rewards.get("agents", {})
    eligible = sorted(name for name, result in agent_results.items()
                      if isinstance(result, dict) and result.get("eligible") is True)
    if not eligible:
        lessons.append({
            "topic": "reward_discipline",
            "finding": "No shadow agent currently qualifies for reward.",
            "lesson": "Zero reward is correct when evidence is insufficient or an agent fails its baseline.",
            "nextExperiment": "Keep every agent in shadow mode until enough mature, valid, independent outcomes exist."
        })
    if scenario.get("status") in ("FAIL", "failed"):
        lessons.append({
            "topic": "robustness",
            "finding": "The scenario report indicates a failure.",
            "lesson": "A strategy that breaks under stress should not be promoted based on average results.",
            "nextExperiment": "Isolate the failing regime and test risk limits without tuning on the final holdout."
        })

    previous = memory if isinstance(memory, dict) else {}
    history = previous.get("history", [])
    if not isinstance(history, list):
        history = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fingerprint = "|".join(sorted(item["topic"] for item in lessons))
    history.append({"at": now, "grade": "NEEDS_MORE_EVIDENCE" if not eligible else "SHADOW_REVIEW_ONLY",
                    "trainingStatus": status or "missing", "lessonTopics": sorted(item["topic"] for item in lessons),
                    "fingerprint": fingerprint})
    history = history[-50:]
    repeated = sum(1 for item in history[:-1] if item.get("fingerprint") == fingerprint)
    report = {
        "generatedAt": now,
        "teacherVersion": "teacher-v2-rule-based-memory",
        "mode": "RESEARCH_SUPERVISOR_ONLY",
        "grade": "NEEDS_MORE_EVIDENCE" if not eligible else "SHADOW_REVIEW_ONLY",
        "trainingStatus": status or "missing",
        "trainingExamples": training.get("examples", 0),
        "eligibleForReward": eligible,
        "lessons": lessons,
        "lessonHistoryCount": len(history),
        "sameLessonPatternSeenBefore": repeated > 0,
        "repeatPatternCount": repeated,
        "actionsPermitted": ["publish_lessons", "recommend_next_experiment", "remember_lesson_history"],
        "actionsForbidden": ["edit_model_weights", "change_scoring_rules", "promote_model", "place_orders"],
        "warning": "Deterministic teaching and memory only. It does not update model weights; reward and trading authorization remain separate."
    }
    return report, {"version": 1, "updatedAt": now, "history": history}


def read(name):
    try:
        return json.loads((ROOT / name).read_text())
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    report, memory = teach(read("training-report.json"), read("reward-report.json"),
                           read("forecast-integrity-report.json"), read("scenario-report.json"),
                           read("teacher-memory.json"))
    (ROOT / "teacher-report.json").write_text(json.dumps(report, indent=2) + "\n")
    MEMORY_PATH.write_text(json.dumps(memory, indent=2) + "\n")
    print("Teacher:", report["grade"], "| lessons:", len(report["lessons"]),
          "| history:", report["lessonHistoryCount"], "| repeated pattern:", report["sameLessonPatternSeenBefore"])
