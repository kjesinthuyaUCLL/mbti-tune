import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("⚠️ GOOGLE_API_KEY is not set in .env – Gemini features will fail.")
else:
    genai.configure(api_key=API_KEY)

GEMINI_MODEL_NAME = "gemini-2.5-flash"


def _get_model():
    try:
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        print("Gemini model init error:", e)
        return None


def generate_personality_breakdown(mbti_type, result, top_artists, summaries):
    """
    summaries = list of 1–3 song summaries
    """

    model = _get_model()
    if model is None:
        return "⚠️ Gemini model could not be initialized."

    combined_summaries = "\n\n---\n\n".join(summaries)

    prompt = f"""
You are an AI psychologist analyzing personality from music.

MBTI Prediction:
- Type: {mbti_type}
- E/I: {result['E/I']:.1f}%
- S/N: {result['S/N']:.1f}%
- T/F: {result['T/F']:.1f}%
- J/P: {result['J/P']:.1f}%

Top Artists: {', '.join(top_artists) if top_artists else 'Unknown'}

Song Summaries (3 songs max):
{combined_summaries}

Write a fun, insightful, slightly roasted psychological breakdown (3–4 paragraphs).
Reference their artists and the themes in these summaries.
Avoid generic MBTI boilerplate; tie your reasoning to the music.
"""

    try:
        resp = model.generate_content(prompt)
        return resp.text or "⚠️ Gemini returned an empty response."
    except Exception as e:
        print("Gemini error:", e)
        return "⚠️ Gemini API error while generating the breakdown."
