"""Teacher agent: grades research agents and emits bounded lessons.

This is a deterministic research supervisor, not a language model. It audits
published scorecards, identifies specific failure modes, and writes lessons for
the next experiment. It cannot change weights, promote a model, or trade.
"""
import json
import pathlib
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]


def teach(training, rewards, integrity, scenario):
    lessons = []
    scores = training.get("aggregateBrier") or (training.get("scores") or {})
    if scores:
        baseline = scores.get("constant", scores.get("baseline", {}))
        base_brier = baseline.get("brier") if isinstance(baseline, dict) else baseline
        if base_brier is not None:
            candidates = []
            for name, item in scores.items():
                if name in ("constant", "baseline") or not isinstance(item, dict):
                    continue
                brier = item.get("brier")
                if isinstance(brier, (int, float)):
                    candidates.append((name, brier))
            if candidates:
                best, value = min(candidates, key=lambda x: x[1])
                if value < base_brier:
                    lessons.append({
                        "topic": "predictive_skill",
                        "finding": f"{best} has the lowest reported Brier score in this evaluation.",
                        "lesson": "Repeat on a fresh chronological window before trusting the apparent edge.",
                        "nextExperiment": "Freeze this candidate and compare it with the constant baseline on future unseen forecasts."
                    })
                else:
                    lessons.append({
                        "topic": "predictive_skill",
                        "finding": "No candidate beat the constant baseline in the available evaluation.",
                        "lesson": "Complexity is not skill; do not reward agents for predictions that fail to beat a simple baseline.",
                        "nextExperiment": "Test simpler features and calibrate probabilities using training data only."
                    })
    if not lessons:
        lessons.append({
            "topic": "evidence_quality",
            "finding": "There is not yet a usable comparable candidate scorecard.",
            "lesson": "Do not infer skill from simulation volume or incomplete samples.",
            "nextExperiment": "Collect timestamped predictions and wait for their outcomes to mature before grading."
        })
    if integrity.get("status") != "PASS":
        lessons.append({
            "topic": "integrity",
            "finding": f"Forecast integrity status is {integrity.get('status', 'MISSING')}.",
            "lesson": "Invalid, duplicate, or inconsistent records must receive zero reward.",
            "nextExperiment": "Repair the record schema and rerun adversarial integrity tests before scoring agents."
        })
    agent_results = rewards.get("agents", {})
    eligible = [name for name, result in agent_results.items()
                if isinstance(result, dict) and result.get("eligible") is True]
    if not eligible:
        lessons.append({
            "topic": "reward_discipline",
            "finding": "No shadow agent currently qualifies for reward.",
            "lesson": "Zero reward is the correct result when evidence is insufficient or an agent fails its baseline.",
            "nextExperiment": "Keep all agents in shadow mode until there are enough mature, valid, independent observations."
        })
    if scenario.get("status") in ("FAIL", "failed"):
        lessons.append({
            "topic": "robustness",
            "finding": "The scenario report indicates a failure.",
            "lesson": "A strategy that breaks under stress should not be promoted based on average results.",
            "nextExperiment": "Isolate the failing regime and test risk limits without tuning on the final holdout."
        })
    return {
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "teacherVersion": "teacher-v1-rule-based",
        "mode": "RESEARCH_SUPERVISOR_ONLY",
        "grade": "NEEDS_MORE_EVIDENCE" if not eligible else "SHADOW_REVIEW_ONLY",
        "eligibleForReward": eligible,
        "lessons": lessons,
        "actionsPermitted": ["publish_lessons", "recommend_next_experiment"],
        "actionsForbidden": ["edit_model_weights", "change_scoring_rules", "promote_model", "place_orders"],
        "warning": "Teacher conclusions are deterministic checks over current reports, not proof of trading skill. Reward decisions require valid prospective evidence."
    }


def read(name):
    try:
        return json.loads((ROOT / name).read_text())
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    result = teach(read("training-report.json"), read("reward-report.json"),
                   read("forecast-integrity-report.json"), read("scenario-report.json"))
    (ROOT / "teacher-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Teacher:", result["grade"], "| lessons:", len(result["lessons"]))
