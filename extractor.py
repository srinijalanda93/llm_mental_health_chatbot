# extractor.py
# NER-based extraction of triggers/symptoms/coping hints using local HF pipeline + Groq fallback

import os
import threading
from typing import Dict, List
from transformers import pipeline
from config import NER_MODEL, DEVICE
from groq import Groq

# --- Local NER setup ---
_ner_pipe = None
_ner_lock = threading.Lock()

def get_ner_pipe():
    global _ner_pipe
    if _ner_pipe is None:
        with _ner_lock:
            if _ner_pipe is None:
                _ner_pipe = pipeline(
                    "ner",
                    model=NER_MODEL,
                    aggregation_strategy="simple",
                    device=DEVICE
                )
    return _ner_pipe

# --- Keyword dictionaries ---
COPING_HINTS = [
    "walk", "walking", "run", "running", "exercise", "yoga", "meditate",
    "breathing", "breathe", "journal", "journaling", "talk", "therapy",
    "counselor", "counsellor", "music", "nap", "rest"
]

SYMPTOMS = [
    "tired", "fatigue", "can't sleep", "insomnia", "panic", "heart racing",
    "headache", "nausea", "crying", "shaking", "stressed", "anxious",
    "overwhelmed", "burnout", "dizzy", "hopeless", "worthless"
]

REDFLAGS = [
    "suicide", "suicidal", "self-harm", "kill myself", "end it all",
    "hopeless", "worthless", "overdose", "cutting"
]

# --- Groq setup (optional fallback) ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_signals(text: str) -> Dict[str, List[str] or bool]:
    """Extract triggers, symptoms, coping, red_flags, urgent from text."""

    t = (text or "").strip()
    if not t:
        return {"triggers": [], "symptoms": [], "coping": [], "red_flags": [], "urgent": False}

    # ---- Local NER pass ----
    ner = get_ner_pipe()(t[:512])
    triggers = set()
    for ent in ner:
        word = ent.get("word", "").strip()
        group = ent.get("entity_group", "")
        if word and group in ("ORG", "MISC", "LOC", "PER", "DATE"):
            triggers.add(word)

    coping = sorted([w for w in COPING_HINTS if w.lower() in t.lower()])
    symptoms = sorted([s for s in SYMPTOMS if s.lower() in t.lower()])
    red_flags = sorted([r for r in REDFLAGS if r.lower() in t.lower()])
    urgent = len(red_flags) > 0

    # ---- If everything is empty → fallback to Groq ----
    if not triggers and not symptoms and not coping:
        try:
            prompt = f"""
            Extract structured mental health signals from this user text.
            Return valid JSON only with these keys: triggers, symptoms, coping.

            User text: "{t}"
            JSON:
            """
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            groq_out = resp.choices[0].message.content.strip()
            import json
            data = json.loads(groq_out)

            triggers = data.get("triggers", [])
            symptoms = data.get("symptoms", [])
            coping = data.get("coping", [])
        except Exception as e:
            print("Groq fallback failed:", e)

    return {
        "triggers": sorted(set(triggers)),
        "symptoms": sorted(set(symptoms)),
        "coping": sorted(set(coping)),
        "red_flags": red_flags,
        "urgent": urgent
    }
