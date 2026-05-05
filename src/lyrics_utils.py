# src/lyrics_utils.py

import requests
from langdetect import detect
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = genai.GenerativeModel("gemini-2.5-flash")


def get_lyrics_lrclib(track_name, artist_name, retries=3, timeout=15):
    url = "https://lrclib.net/api/search"
    params = {"track_name": track_name, "artist_name": artist_name}

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            for track in data:
                if track.get("plainLyrics"):
                    return track["plainLyrics"]

            return None

        except Exception as e:
            print(f"Lyrics API error (attempt {attempt}/{retries}):", e)
            if attempt == retries:
                return None


def translate_and_summarize(text):
    prompt = f"""
Translate the following lyrics to English (if needed), then summarize the meaning
and emotional themes in 3–4 sentences. Do NOT quote the lyrics.

Lyrics:
{text}
"""

    try:
        resp = MODEL.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print("Gemini translation/summarization error:", e)
        return None


def build_lyrics_context(tracks, limit=3):
    context = []

    for name, artist in tracks[:limit]:
        lyrics = get_lyrics_lrclib(name, artist)

        if not lyrics:
            continue

        try:
            lang = detect(lyrics)
        except:
            lang = "unknown"

        summary = translate_and_summarize(lyrics)

        if summary:
            context.append(f"{name} - {artist} ({lang})\n{summary}\n")

    return "\n".join(context) if context else "No lyrics found."
