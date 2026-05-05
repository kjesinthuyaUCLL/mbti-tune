# src/gemini_utils.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables once
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("⚠️ GOOGLE_API_KEY is not set in .env – Gemini features will fail.")
else:
    genai.configure(api_key=API_KEY)

# Use a fast, text-capable model from your allowed list
# (you confirmed models/gemini-2.5-flash is available)
GEMINI_MODEL_NAME = "gemini-2.5-flash"

def _get_model():
    """
    Lazily construct the GenerativeModel so import-time errors don't kill Streamlit.
    """
    try:
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        print("Gemini model init error:", e)
        return None


def generate_personality_breakdown(mbti_type, percentages, top_artists, lyrics_context):
    """
    Call Gemini to generate a psychological breakdown based on:
    - MBTI prediction
    - dimension percentages
    - top artists
    - lyrics context (summaries)
    """
    model = _get_model()
    if model is None:
        return "⚠️ Gemini model could not be initialized. Check API key and model name."

    prompt = f"""
You are an AI psychologist analyzing personality from music.

MBTI Prediction:
- Type: {mbti_type}
- Extraversion: {percentages['E']*100:.1f}%
- Intuition: {percentages['N']*100:.1f}%
- Thinking: {percentages['T']*100:.1f}%
- Judging: {percentages['J']*100:.1f}%

Top Artists: {', '.join(top_artists) if top_artists else 'Unknown'}

Song Summaries:
{lyrics_context}

Write a fun, insightful, slightly roasted psychological breakdown (3–4 paragraphs).
Reference their artists and the song themes (NOT exact lyrics).
Avoid generic MBTI boilerplate; tie your reasoning to the music and summaries.
    """.strip()

    try:
        resp = model.generate_content(prompt)
        return resp.text or "⚠️ Gemini returned an empty response."
    except Exception as e:
        print("Gemini error:", e)
        return "⚠️ Gemini API error while generating the breakdown."
