# src/spotify_utils.py

import os
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Fallback dataset (pretrain data)
DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "pretrain",
    "spotify_tracks.csv"
)

# Spotify key/mode are numeric: key ∈ [0..11], mode ∈ {0,1}
KEYS = list(range(12))   # 0..11
MODES = [0, 1]           # 0=minor, 1=major

# ---------------- AUTH ----------------
def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501"),
        scope="user-top-read"
    )

# ---------------- FALLBACK DATASET ----------------
def load_fallback_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna()
    return df

# ---------------- SHARED FEATURE AGGREGATION ----------------
NUMERIC_COLS_LIVE = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

NUMERIC_COLS_DATASET = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

def _aggregate_numeric(df, numeric_cols):
    agg = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        col_mean = float(df[col].mean())
        col_std = float(df[col].std() or 0.0)
        agg[f"{col}_mean"] = col_mean
        agg[f"{col}_stdev"] = col_std
    return agg

def _aggregate_key_mode(df):
    """
    Build key_mode_<key>_<mode> counts, where:
    - key in [0..11]
    - mode in {0,1}
    """
    agg = {}

    # Ensure key/mode exist and are ints
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

    # Initialize all to 0
    for k in KEYS:
        for m in MODES:
            agg[f"key_mode_{k}_{m}"] = 0

    # Fill observed counts
    for k, v in counts.items():
        agg[k] = int(v)

    return agg

# ---------------- FEATURE BUILDER (DATASET MODE) ----------------
def build_features_from_dataset(df):
    """
    Build aggregated features from the fallback dataset.
    Schema matches pretrain_features.json:
    - <numeric>_mean, <numeric>_stdev
    - key_mode_<key>_<mode>
    - track_count
    """
    agg = {}

    # numeric stats
    agg.update(_aggregate_numeric(df, NUMERIC_COLS_DATASET))

    # key/mode counts
    agg.update(_aggregate_key_mode(df))

    # track count
    agg["track_count"] = int(len(df))

    return agg

# ---------------- FEATURE BUILDER (LIVE SPOTIFY AUDIO FEATURES) ----------------
def build_features_from_audio_features(df):
    """
    Build aggregated features from Spotify audio_features.
    Same schema as dataset mode.
    """
    agg = {}

    # numeric stats
    agg.update(_aggregate_numeric(df, NUMERIC_COLS_LIVE))

    # key/mode counts
    agg.update(_aggregate_key_mode(df))

    # track count
    agg["track_count"] = int(len(df))

    return agg

# ---------------- MAIN PIPELINE ----------------
def fetch_user_data(token_info, feature_cols, limit=20):
    """
    Returns:
      - final_vector: np.array shape (1, len(feature_cols))
      - tracks: list[(track_name, artist_name)]
      - top_artists: list[str]
    """
    sp = spotipy.Spotify(auth=token_info["access_token"])

    top_tracks = sp.current_user_top_tracks(limit=limit, time_range="medium_term")

    if not top_tracks or not top_tracks.get("items"):
        return None, None, None

    track_ids, track_names, artists = [], [], []

    for item in top_tracks["items"]:
        track_ids.append(item["id"])
        track_names.append(item["name"])
        artists.append(item["artists"][0]["name"])

    # ---------------- TRY SPOTIFY AUDIO FEATURES ----------------
    audio_features = []
    try:
        for i in range(0, len(track_ids), 100):
            batch = sp.audio_features(track_ids[i:i+100])
            if batch:
                audio_features.extend([f for f in batch if f])
    except Exception as e:
        print("Spotify API failed → switching to dataset fallback:", e)

    # ---------------- FALLBACK TO DATASET ----------------
    if not audio_features:
        df = load_fallback_dataset()
        # sample a reasonable subset
        df = df.sample(min(len(df), 2000), random_state=42)
        agg = build_features_from_dataset(df)
    else:
        df = pd.DataFrame(audio_features)
        agg = build_features_from_audio_features(df)

    # ---------------- BUILD FEATURE VECTOR ----------------
    final_vector = np.array(
        [[agg.get(col, 0.0) for col in feature_cols]],
        dtype=np.float32
    )

    top_artists = list(pd.Series(artists).value_counts().head(3).index)
    tracks = list(zip(track_names, artists))

    return final_vector, tracks, top_artists
