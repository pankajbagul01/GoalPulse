import math

from .ml_service import INTENT_LABELS, predict_intent, preprocess_text, vectorize_text


def tokenize(text: str) -> list[str]:
    return preprocess_text(text).split()


def vectorize(text: str) -> list[float]:
    return vectorize_text(text)


def detect_intent_llm(text: str) -> dict[str, object]:
    # The name is kept for compatibility with older imports.
    return predict_intent(text)


def _intent_payload(text: str) -> tuple[str, float]:
    result = predict_intent(text)
    return str(result.get("intent", "other")), float(result.get("confidence", 0.0))


def extract_intent(text: str) -> dict[str, object]:
    category, confidence = _intent_payload(text)

    return {
        "intent_label": INTENT_LABELS.get(category, INTENT_LABELS["other"]),
        "category": category,
        "confidence": confidence,
        "embedding": vectorize(text),
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
