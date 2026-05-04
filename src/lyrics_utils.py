import requests
import google.generativeai as genai
from dotenv import load_dotenv
from langdetect import detect

load_dotenv()

# ---------------- LYRICS API ----------------
def get_lyrics_lrclib(track_name, artist_name):
    url = "https://lrclib.net/api/search"
    params = {
        "track_name": track_name,
        "artist_name": artist_name
    }

    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()

        for track in data:
            if track.get("plainLyrics"):
                return track["plainLyrics"]

        return None

    except Exception as e:
        print("Lyrics API error:", e)
        return None


# ---------------- LYRICS PROCESSING ----------------
def build_lyrics_context(tracks, limit=3):
    context = []

    for name, artist in tracks[:limit]:
        lyrics = get_lyrics_lrclib(name, artist)

        if lyrics:
            try:
                lang = detect(lyrics)
            except:
                lang = "unknown"

            summary = lyrics[:300].replace("\n", " ")
            context.append(f"{name} - {artist} ({lang})\n{summary}\n")

    return "\n".join(context) if context else "No lyrics found."