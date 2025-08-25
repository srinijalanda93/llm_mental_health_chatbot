# # app.py
# import streamlit as st
# import matplotlib.pyplot as plt

# from classifier import detect_stress, get_emotion_probs
# from extractor import extract_signals
# from generate_response import empathetic_reply

# st.set_page_config(page_title="Local HF Mental Health Chatbot", page_icon="🫶", layout="centered")
# st.title("🫶 Mental Health Chatbot — Local Hugging Face Pipelines")
# st.caption("Educational demo (non-clinical).")

# with st.expander("Privacy & Safety", expanded=False):
#     st.write("- Runs locally using downloaded models (no external API).")
#     st.write("- Not a substitute for professional care. If in immediate danger, contact local emergency services.")

# user_text = st.text_area("What's on your mind?", height=170, placeholder="e.g., I'm overwhelmed with deadlines and can't sleep...")

# col1, col2 = st.columns(2)
# with col1:
#     run = st.button("Analyze")
# with col2:
#     demo = st.button("Use demo text")
#     if demo:
#         user_text = "Exams are this week; I barely sleep and my heart races when I think about them."
#         st.session_state["demo"] = True
#         run = True

# if run:
#     if not user_text.strip():
#         st.warning("Please enter some text.")
#     else:
#         with st.spinner("Running models (may take ~10s first time)..."):
#             res = detect_stress(user_text)
#             score = res["stress_score"]
#             label = res["stress_label"]
#             emotions = res.get("emotion_probs", {})
#             signals = extract_signals(user_text)

#         st.subheader(f"Detected stress: **{label.capitalize()}** ({score:.2f})")

#         if emotions:
#             fig = plt.figure(figsize=(6,3))
#             plt.bar(list(emotions.keys()), list(emotions.values()))
#             plt.title("Emotion Probabilities")
#             plt.xlabel("Emotion")
#             plt.ylabel("Probability")
#             st.pyplot(fig)

#         with st.expander("Extracted signals (NER)", expanded=False):
#             st.write({"triggers": signals["triggers"], "symptoms": signals["symptoms"], "coping": signals["coping"]})

#         if signals["urgent"]:
#             st.error("⚠️ Red-flag phrases detected. If you’re in immediate danger or thinking about self-harm, please seek help now.")

#         st.markdown("---")
#         st.subheader("Supportive response")
#         st.markdown(empathetic_reply(user_text, label, score))

#         st.markdown("---")
#         st.caption("If you can, consider talking with a trusted friend or mental health professional.")


# # app.py
# import streamlit as st
# import matplotlib.pyplot as plt

# from classifier import detect_stress, get_emotion_probs
# from extractor import extract_signals
# from generate_response import empathetic_reply

# # --- Streamlit page setup ---
# st.set_page_config(page_title="Local HF + Groq Mental Health Chatbot", page_icon="🫶", layout="centered")
# st.title("🫶 Mental Health Chatbot — Local HF + Groq")
# st.caption("Educational demo (non-clinical).")

# with st.expander("Privacy & Safety", expanded=False):
#     st.write("- Runs locally with Hugging Face pipelines; optionally enhanced with Groq.")
#     st.write("- Not a substitute for professional care. If in immediate danger, contact local emergency services.")

# # --- User input ---
# user_text = st.text_area(
#     "What's on your mind?",
#     height=170,
#     placeholder="e.g., I'm overwhelmed with deadlines and can't sleep..."
# )

# col1, col2 = st.columns(2)
# with col1:
#     run = st.button("Analyze")
# with col2:
#     demo = st.button("Use demo text")
#     if demo:
#         user_text = "Exams are this week; I barely sleep and my heart races when I think about them."
#         st.session_state["demo"] = True
#         run = True

# # --- Run analysis ---
# if run:
#     if not user_text.strip():
#         st.warning("Please enter some text.")
#     else:
#         with st.spinner("Running models (may take ~10s first time)..."):
#             res = detect_stress(user_text)
#             score = res["stress_score"]
#             label = res["stress_label"]
#             emotions = res.get("emotion_probs", {})
#             signals = extract_signals(user_text)

#         # --- Stress result ---
#         st.subheader(f"Detected stress: **{label.capitalize()}** ({score:.2f})")

#         # --- Emotions chart ---
#         if emotions:
#             fig = plt.figure(figsize=(6, 3))
#             plt.bar(list(emotions.keys()), list(emotions.values()), color="skyblue")
#             plt.title("Emotion Probabilities")
#             plt.xlabel("Emotion")
#             plt.ylabel("Probability")
#             st.pyplot(fig)

#         # --- Extracted signals ---
#         with st.expander("Extracted signals (NER)", expanded=False):
#             st.json({
#                 "triggers": signals["triggers"],
#                 "symptoms": signals["symptoms"],
#                 "coping": signals["coping"]
#             })

#         # --- Red flag warning ---
#         if signals["urgent"]:
#             st.error("⚠️ Red-flag phrases detected. If you’re in immediate danger or thinking about self-harm, please seek help now.")

#         reply = empathetic_reply(user_text, label, score, signals)
#         st.markdown(reply)

#         st.markdown("---")
#         st.caption("If you can, consider talking with a trusted friend or mental health professional.")

#         # app.py snippet
#         st.subheader("Supportive responses")
#         responses = empathetic_reply(user_text, label, score, signals)
#         st.markdown("### 🤖 FLAN (local)")
#         st.markdown(responses["flan"])
#         st.markdown("---")
#         st.markdown("### ⚡ Groq (cloud)")
#         st.markdown(responses["groq"])


# app.py
import streamlit as st
import matplotlib.pyplot as plt

from classifier import detect_stress
from extractor import extract_signals
from generate_response import empathetic_reply
from evaluation import evaluate_classifier, evaluate_responses

# --- Streamlit setup ---
st.set_page_config(page_title="Local HF + Groq Mental Health Chatbot", page_icon="🫶", layout="centered")
st.title("🫶 Mental Health Chatbot — Local HF + Groq")
st.caption("Educational demo (non-clinical).")

with st.expander("Privacy & Safety", expanded=False):
    st.write("- Runs locally with Hugging Face pipelines; optionally enhanced with Groq.")
    st.write("- Not a substitute for professional care. If in immediate danger, contact local emergency services.")

# --- Input ---
user_text = st.text_area(
    "What's on your mind?",
    height=170,
    placeholder="e.g., I'm overwhelmed with deadlines and can't sleep..."
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Analyze")
with col2:
    demo = st.button("Use demo text")
    if demo:
        user_text = "Exams are this week; I barely sleep and my heart races when I think about them."
        st.session_state["demo"] = True
        run = True

# --- Analysis ---
if run:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running models (may take ~10s first time)..."):
            res = detect_stress(user_text)
            score = res["stress_score"]
            label = res["stress_label"]
            emotions = res.get("emotion_probs", {})
            signals = extract_signals(user_text)

        # Stress classification
        st.subheader(f"Detected stress: **{label.capitalize()}** ({score:.2f})")

        # Emotion chart
        if emotions:
            fig = plt.figure(figsize=(6, 3))
            plt.bar(list(emotions.keys()), list(emotions.values()), color="skyblue")
            plt.title("Emotion Probabilities")
            plt.xlabel("Emotion")
            plt.ylabel("Probability")
            st.pyplot(fig)

        # Extracted signals
        # with st.expander("Extracted signals (NER)", expanded=False):
        #     st.json({
        #         "triggers": signals["triggers"],
        #         "symptoms": signals["symptoms"],
        #         "coping": signals["coping"]
        #     })

        # Red flag warning
        if signals["urgent"]:
            st.error("⚠️ Red-flag phrases detected. If you’re in immediate danger or thinking about self-harm, please seek help now.")

        # Supportive responses
        responses = empathetic_reply(user_text, label, score, signals)
        st.subheader("Supportive responses")
        st.markdown("### 🤖 FLAN (local)")
        st.markdown(responses["flan"])
        st.markdown("---")
        st.markdown("### ⚡ Groq (cloud)")
        st.markdown(responses["groq"])

        st.markdown("---")
        st.caption("If you can, consider talking with a trusted friend or mental health professional.")

               # --- Evaluation Demo ---
        st.subheader("📊 Model Evaluation (Demo)")

        # classifier evaluation (fake demo for now)
        y_true = [1, 0, 1, 1, 0]   # true labels (stress vs no stress)
        y_pred = [1, 0, 1, 0, 0]   # predicted labels
        clf_metrics = evaluate_classifier(y_true, y_pred)

        # --- Classifier metrics bar chart ---
        st.markdown("### 🔍 Stress Classifier Metrics")
        fig1 = plt.figure(figsize=(5, 3))
        plt.bar(clf_metrics.keys(), clf_metrics.values(), color="skyblue")
        plt.title("Classifier Evaluation")
        plt.ylabel("Score")
        plt.ylim(0, 1)  # keep metrics normalized
        st.pyplot(fig1)

        # response evaluation (demo with dummy references)
        preds = [responses["flan"], responses["groq"]]
        refs = [
            "I understand this is tough. Try small breaks, breathing exercises, and talking to a friend.",
            "It sounds stressful. Maybe take a walk, journal your thoughts, and get some rest."
        ]
        resp_metrics = evaluate_responses(preds, refs)

        # --- Response metrics bar chart ---
        st.markdown("### 💬 Response Quality Metrics")
        fig2 = plt.figure(figsize=(5, 3))
        plt.bar(resp_metrics.keys(), resp_metrics.values(), color="lightgreen")
        plt.title("Response Evaluation (BLEU / ROUGE)")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        st.pyplot(fig2)

