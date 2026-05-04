# gemini_utils.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")


def generate_personality_breakdown(mbti_type, percentages, top_artists, lyrics_context, model=gemini_model):
    prompt = f"""
    You are an AI psychologist analyzing personality from music.

    MBTI Prediction:
    - Type: {mbti_type}
    - Extraversion: {percentages['E']*100:.1f}%
    - Intuition: {percentages['N']*100:.1f}%
    - Thinking: {percentages['T']*100:.1f}%
    - Judging: {percentages['J']*100:.1f}%

    Top Artists: {', '.join(top_artists)}

    Song Summaries:
    {lyrics_context}

    Write a fun, insightful, slightly roasted psychological breakdown (3–4 paragraphs).
    Reference their artists and the song themes (NOT exact lyrics).
    """

    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print("Gemini error:", e)
        return "⚠️ Gemini API error."
