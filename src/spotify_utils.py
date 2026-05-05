import os
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "raw", "pretrain", "spotify_tracks.csv"
)

KEYS = list(range(12))
MODES = [0, 1]

def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501"),
        scope="user-top-read"
    )

def get_artist_genres(sp, artist_id):
    try:
        artist = sp.artist(artist_id)
        return artist.get("genres", [])
    except:
        return []

def load_fallback_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        df = df.dropna()
        return df
    except Exception as e:
        print("Fallback dataset load failed:", e)
        return None

NUMERIC_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

def _aggregate_numeric(df):
    agg = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            agg[f"{col}_mean"] = 0.0
            agg[f"{col}_stdev"] = 0.0
            continue
        agg[f"{col}_mean"] = float(df[col].mean())
        agg[f"{col}_stdev"] = float(df[col].std() or 0.0)
    return agg

def _aggregate_key_mode(df):
    agg = {}

    if "key" not in df.columns or "mode" not in df.columns:
        for k in KEYS:
            for m in MODES:
                agg[f"key_mode_{k}_{m}"] = 0
        return agg

    df = df.copy()
    df["key"] = df["key"].astype(int)
    df["mode"] = df["mode"].astype(int)

    df["key_mode_idx"] = df.apply(
        lambda r: f"key_mode_{int(r['key'])}_{int(r['mode'])}",
        axis=1
    )

    counts = df["key_mode_idx"].value_counts().to_dict()

    for k in KEYS:
        for m in MODES:
            agg[f"key_mode_{k}_{m}"] = 0

    for k, v in counts.items():
        agg[k] = int(v)

    return agg

def build_features(df):
    agg = {}
    agg.update(_aggregate_numeric(df))
    agg.update(_aggregate_key_mode(df))
    agg["track_count"] = int(len(df))
    return agg

def fetch_user_data(token_info, feature_cols, limit=20):
    sp = spotipy.Spotify(auth=token_info["access_token"])

    try:
        top_tracks = sp.current_user_top_tracks(limit=limit, time_range="medium_term")
    except Exception as e:
        print("Error fetching top tracks:", e)
        return None, None, None, None

    if not top_tracks or not top_tracks.get("items"):
        return None, None, None, None

    track_ids, track_names, artists, genres = [], [], [], []

    for item in top_tracks["items"]:
        track_ids.append(item["id"])
        track_names.append(item["name"])

        artist_obj = item["artists"][0]
        artist_name = artist_obj["name"]
        artist_id = artist_obj["id"]

        artists.append(artist_name)
        genres.append(get_artist_genres(sp, artist_id))

    audio_features = []
    try:
        for i in range(0, len(track_ids), 100):
            batch = sp.audio_features(track_ids[i:i+100])
            if batch:
                audio_features.extend([f for f in batch if f])
    except Exception as e:
        print("Spotify audio_features failed:", e)

    agg = None

    if audio_features:
        df = pd.DataFrame(audio_features)
        if len(df) > 0:
            agg = build_features(df)

    if agg is None:
        df = load_fallback_dataset()
        if df is not None and len(df) > 0:
            df = df.sample(min(len(df), 2000), random_state=42)
            agg = build_features(df)

    if agg is None:
        print("No valid audio features from Spotify or fallback dataset.")
        return None, None, None, None

    final_vector = np.array(
        [[agg.get(col, 0.0) for col in feature_cols]],
        dtype=np.float32
    )

    top_artists = list(pd.Series(artists).value_counts().head(3).index)
    tracks = list(zip(track_names, artists))

    return final_vector, tracks, top_artists, genres
