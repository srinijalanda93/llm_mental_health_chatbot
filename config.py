# config.py
# small central place for settings

# device: -1 -> CPU, >=0 -> GPU device id
DEVICE = -1

# stress threshold for binary evaluation
STRESS_THRESHOLD = 0.5

# Models (change if you want lighter/heavier)
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
NER_MODEL = "dslim/bert-base-NER"
FLAN_MODEL = "google/flan-t5-small"  # faster than base


