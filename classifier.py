# classifier.py
from transformers import pipeline

_sentiment_pipe = None
_emotion_pipe = None


def get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            framework="pt"
        )
    return _sentiment_pipe


def get_stress_score(text: str):
    """
    Approximate stress detection using sentiment.
    Negative sentiment => high stress
    Neutral => medium
    Positive => low stress
    """
    pipe = get_sentiment_pipe()
    results = pipe(text)

    if not results or not isinstance(results, list):
        return {"stress_label": "unknown", "stress_score": 0.0}

    best = max(results, key=lambda x: x["score"])
    label = best["label"].lower()

    # Map sentiment -> stress
    if "negative" in label:
        stress_label, stress_score = "high", float(best["score"])
    elif "neutral" in label:
        stress_label, stress_score = "medium", float(best["score"])
    else:
        stress_label, stress_score = "low", float(best["score"])

    return {"stress_label": stress_label, "stress_score": stress_score}

# -----------------------------
# Emotion Detection Pipeline
# -----------------------------
def get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is None:
        _emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            framework="pt"
        )
    return _emotion_pipe


def get_emotion_probs(text: str):
    """
    Run emotion classification on text.
    Returns: dict mapping emotion -> score
    """
    pipe = get_emotion_pipe()
    results = pipe(text)

    if not results or not isinstance(results, list):
        return {}

    return {r["label"].lower(): float(r["score"]) for r in results}

# -----------------------------
# Combined Stress + Emotion
# -----------------------------
def detect_stress(text: str):
    """
    Combined stress + emotion classification.
    Returns:
      {
        "stress_label": str,
        "stress_score": float,
        "emotions": dict
      }
    """
    stress = get_stress_score(text)
    emotions = get_emotion_probs(text)

    return {
        "stress_label": stress["stress_label"],
        "stress_score": stress["stress_score"],
        "emotions": emotions,
    }
