import os
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

BACKUP_DATASET_PATH = os.path.join(
    BASE_DIR, "data", "raw", "spotify_data.csv"   # 1M-song backup
)

NUMERIC_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

KEYS = list(range(12))
MODES = [0, 1]


def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501"),
        scope="user-top-read"
    )


def load_backup_dataset():
    try:
        df = pd.read_csv(BACKUP_DATASET_PATH)
        df = df.dropna()
        return df
    except Exception as e:
        print("Backup dataset load failed:", e)
        return None


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
        genres.append([])  # optional

    # Try Spotify audio_features
    audio_features = []
    try:
        for i in range(0, len(track_ids), 100):
            batch = sp.audio_features(track_ids[i:i+100])
            if batch:
                audio_features.extend([f for f in batch if f])
    except Exception:
        audio_features = []

    # If Spotify API blocked → use backup dataset
    if not audio_features:
        backup_df = load_backup_dataset()
        if backup_df is None:
            print("No backup dataset available.")
            return None, None, None, None

        # Match tracks by name + artist
        matched = backup_df[
            backup_df["track_name"].isin(track_names)
        ]

        if matched.empty:
            print("Tracks not found in backup dataset.")
            return None, None, None, None

        agg = build_features(matched)

    else:
        df = pd.DataFrame(audio_features)
        agg = build_features(df)

    final_vector = np.array(
        [[agg.get(col, 0.0) for col in feature_cols]],
        dtype=np.float32
    )

    top_artists = list(pd.Series(artists).value_counts().head(3).index)
    tracks = list(zip(track_names, artists))

    return final_vector, tracks, top_artists, genres
