

# import os
# import threading
# from transformers import pipeline
# from groq import Groq

# from config import GEN_MODEL, DEVICE

# # ------------------------------
# # FLAN pipeline (local)
# # ------------------------------
# _gen_pipe = None
# _gen_lock = threading.Lock()

# def get_gen_pipe():
#     global _gen_pipe
#     if _gen_pipe is None:
#         with _gen_lock:
#             if _gen_pipe is None:
#                 _gen_pipe = pipeline(
#                     "text2text-generation",
#                     model=GEN_MODEL,   # e.g. "google/flan-t5-base"
#                     device=DEVICE
#                 )
#     return _gen_pipe

# # ------------------------------
# # Groq client
# # ------------------------------
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # set in .env

# # ------------------------------
# # Helpers
# # ------------------------------
# SAFETY_NOTE = (
#     "I'm not a medical professional. If you feel unsafe or have thoughts of self-harm, "
#     "please contact local emergency services or a trusted person."
# )

# def clean_response(text: str) -> str:
#     """Remove repetition and instruction-like phrases from FLAN output"""
#     bad_phrases = ["validate feelings", "suggest 3", "Response:"]
#     for b in bad_phrases:
#         text = text.replace(b, "")
#     # deduplicate sentences
#     sentences = text.split(". ")
#     unique = []
#     for s in sentences:
#         s = s.strip()
#         if s and s not in unique:
#             unique.append(s)
#     return ". ".join(unique).strip()

# # ------------------------------
# # Generate empathetic response
# # ------------------------------
# def empathetic_reply(user_text: str, stress_label: str, stress_score: float, signals: dict):
#     """
#     Returns both FLAN and Groq responses.
#     """
#     # ----------------- Prompt -----------------
#     flan_prompt = (
#         f"You are a supportive friend. A student shares: \"{user_text}\". "
#         f"(Detected stress: {stress_label}, score {stress_score:.2f}).\n\n"
#         "Write a warm, empathetic reply (50-120 words). "
#         "Validate their feelings and suggest 3 small, realistic coping steps. "
#         "Do not repeat instructions."
#     )

#     # ----------------- FLAN -----------------
#     try:
#         gen = get_gen_pipe()
#         flan_out = gen(flan_prompt, max_length=160, num_beams=4)[0]["generated_text"]
#         flan_out = clean_response(flan_out)
#         flan_out = f"{flan_out}\n\n_{SAFETY_NOTE}_"
#     except Exception as e:
#         flan_out = (
#             "Thanks for sharing — that sounds tough. Try: (1) 5 deep breaths, "
#             "(2) step outside for a short walk, (3) write down one next step. "
#             "If you feel unsafe, seek help.\n\n_" + SAFETY_NOTE + "_"
#         )

#     # ----------------- Groq -----------------
#     try:
#         groq_prompt = (
#             f"A student is feeling stressed.\n\n"
#             f"User: {user_text}\n"
#             f"Stress: {stress_label} (score {stress_score:.2f})\n"
#             f"Signals: {signals}\n\n"
#             "Write a compassionate, natural reply in 80-120 words. "
#             "Be supportive, mention their situation, and give 3 gentle coping suggestions."
#         )

#         groq_out = groq_client.chat.completions.create(
#             model="llama3-8b-8192",
#             messages=[{"role": "user", "content": groq_prompt}],
#             temperature=0.7,
#             max_tokens=180
#         )
#         groq_text = groq_out.choices[0].message.content.strip()
#         groq_out = f"{groq_text}\n\n_{SAFETY_NOTE}_"
#     except Exception:
#         groq_out = "Groq service unavailable. Please try again later."

#     return {"flan": flan_out, "groq": groq_out}


# generate_response.py
import os
import re
from groq import Groq
from transformers import pipeline
from config import FLAN_MODEL, DEVICE

# ---- Local FLAN pipeline ----
_flan_pipe = pipeline("text2text-generation", model=FLAN_MODEL, device=DEVICE)

# ---- Groq client ----
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def clean_text(text: str) -> str:
    """Remove repeated words/sentences and clean up spacing."""
    # Remove consecutive duplicates
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # Remove excessive "I'm sorry" etc.
    text = re.sub(r"(I'm sorry.*?)( \1)+", r"\1", text)
    return text.strip()

def flan_reply(user_text: str, stress_label: str, stress_score: float, signals: dict) -> str:
    prompt = f"""
You are a supportive mental health companion.
User text: {user_text}
Stress level: {stress_label} ({stress_score:.2f})
Extracted signals: {signals}

Write a short, empathetic response. Validate feelings and suggest 2–3 practical coping steps.
"""
    raw = _flan_pipe(prompt, max_new_tokens=120)[0]["generated_text"]
    return clean_text(raw)

def groq_reply(user_text: str, stress_label: str, stress_score: float, signals: dict) -> str:
    prompt = f"""
You are a supportive mental health companion.
User text: {user_text}
Stress level: {stress_label} ({stress_score:.2f})
Extracted signals: {signals}

Write a short, empathetic response. Validate feelings and suggest 2–3 practical coping steps.
"""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    raw = chat_completion.choices[0].message.content
    return clean_text(raw)

def empathetic_reply(user_text: str, stress_label: str, stress_score: float, signals: dict) -> dict:
    """Return both FLAN and Groq responses for comparison."""
    return {
        "flan": flan_reply(user_text, stress_label, stress_score, signals),
        "groq": groq_reply(user_text, stress_label, stress_score, signals)
    }
