import os
import json
import spotipy
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/raw/pretrain/spotify_tracks.csv"
)

KEY_MAPPING = {
    0: "C", 1: "C#/Db", 2: "D", 3: "D#/Eb", 4: "E", 5: "F",
    6: "F#/Gb", 7: "G", 8: "G#/Ab", 9: "A", 10: "A#/Bb", 11: "B"
}

MODE_MAPPING = {
    0: "minor",
    1: "major"
}

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

    # clean missing values
    df = df.dropna()

    return df

# ---------------- FEATURE BUILDER (DATASET MODE) ----------------
def build_features_from_dataset(df):
    agg = {}

    numeric_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo"
    ]

    for col in numeric_cols:
        agg[f"{col}_mean"] = float(df[col].mean())
        agg[f"{col}_stdev"] = float(df[col].std() or 0)

    # key-mode encoding
    df["key_mode"] = df.apply(
        lambda r: f"{KEY_MAPPING.get(int(r['key']), 'C')}{MODE_MAPPING.get(int(r['mode']), 'major')}_count",
        axis=1
    )

    key_counts = df["key_mode"].value_counts().to_dict()

    for k in KEY_MAPPING.values():
        for m in MODE_MAPPING.values():
            agg[f"{k}{m}_count"] = 0

    for k, v in key_counts.items():
        agg[k] = v

    agg["track_count"] = len(df)

    return agg

# ---------------- MAIN PIPELINE ----------------
def fetch_user_data(token_info, feature_cols, limit=20):
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

        # sample similar size to user tracks
        df = df.sample(min(len(df), 2000))

        agg = build_features_from_dataset(df)

    else:
        df = pd.DataFrame(audio_features)

        numeric_cols = [
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "mode"
        ]

        agg = {}

        for col in numeric_cols:
            agg[f"{col}_mean"] = float(df[col].mean())
            agg[f"{col}_stdev"] = float(df[col].std() or 0)

        df["key_mode"] = df.apply(
            lambda r: f"{KEY_MAPPING[r['key']]}{MODE_MAPPING[r['mode']]}_count",
            axis=1
        )

        key_counts = df["key_mode"].value_counts().to_dict()

        for k in KEY_MAPPING.values():
            for m in MODE_MAPPING.values():
                agg[f"{k}{m}_count"] = 0

        for k, v in key_counts.items():
            agg[k] = v

        agg["track_count"] = len(df)

    # ---------------- BUILD FEATURE VECTOR ----------------
    final_vector = np.array(
        [[agg.get(col, 0.0) for col in feature_cols]],
        dtype=np.float32
    )

    top_artists = list(pd.Series(artists).value_counts().head(3).index)
    tracks = list(zip(track_names, artists))

    return final_vector, tracks, top_artists