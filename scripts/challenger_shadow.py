"""Paper-only frozen Trend challenger inference; no orders and no auto-promotion."""
import hashlib
import json
import math
from train_research import FEATURE_NAMES, DAY_MS, feature_vector_for_day, predict_probability
from prospective_trial import price_series_for_coin


def challenger_fingerprint(challenger):
    payload = {k: challenger.get(k) for k in ("weights", "featureMeans", "featureScales",
                                              "featureIndices", "datasetSha256", "featureGroup")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def predict_challenger(challenger, coin_id, historical, live, now_ms):
    if not isinstance(challenger, dict) or challenger.get("status") != "FROZEN_DEVELOPMENT_CHALLENGER":
        return None
    if not isinstance(challenger.get("generatedAt"), str):
        return None
    from datetime import datetime
    try:
        frozen_ms = int(datetime.fromisoformat(challenger["generatedAt"]).timestamp() * 1000)
    except (ValueError, TypeError):
        return None
    if now_ms <= frozen_ms:
        return None
    if challenger.get("features") != FEATURE_NAMES:
        return None
    if len(challenger.get("weights", [])) != len(FEATURE_NAMES) + 1:
        return None
    if len(challenger.get("featureMeans", [])) != len(FEATURE_NAMES):
        return None
    if len(challenger.get("featureScales", [])) != len(FEATURE_NAMES):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in challenger["weights"] + challenger["featureMeans"] + challenger["featureScales"]):
        return None
    if any(s <= 0 for s in challenger["featureScales"]):
        return None
    prices = price_series_for_coin(coin_id, historical, live, now_ms)
    features = feature_vector_for_day(prices, now_ms // DAY_MS)
    if features is None:
        return None
    indices = challenger.get("featureIndices", [])
    if not isinstance(indices, list) or not indices or any(not isinstance(i, int) or i not in range(len(FEATURE_NAMES)) for i in indices):
        return None
    features = [value if i in indices else 0.0 for i, value in enumerate(features)]
    probability = predict_probability(features, challenger)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return None
    return {"pUp": probability, "modelHash": challenger_fingerprint(challenger),
            "modelVersion": "trend-medium-momentum-shadow-v1",
            "featureGroup": challenger["featureGroup"],
            "frozenAt": challenger["generatedAt"],
            "trainingDatasetSha256": challenger.get("datasetSha256"),
            "researchOnly": True}
