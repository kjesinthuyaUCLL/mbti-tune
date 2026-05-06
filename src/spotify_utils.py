import os
import pandas as pd
import numpy as np
import spotipy
import joblib
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def get_spotify_oauth():
    """Returns the OAuth object used by app.py for user login."""
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=redirect_uri,
        scope="user-top-read",
        cache_path=None  # Don't cache to avoid issues with multiple users
    )


def build_features(df):
    """
    Aggregates individual track data into the 45-feature format 
    trained in Notebook 3.
    
    IMPORTANT: The feature order MUST match mbti_features.json exactly!
    """
    # Order must match Notebook 3's feature_cols - this is the critical part
    audio_cols = [
        "danceability", "energy", "valence", "acousticness", 
        "instrumentalness", "speechiness", "loudness", "tempo", "liveness"
    ]
    
    res = {}
    
    # 1. Calculate Means and Standard Deviations (18 features)
    for col in audio_cols:
        if col in df.columns:
            res[f"{col}_mean"] = float(df[col].mean())
            res[f"{col}_stdev"] = float(df[col].std() if len(df) > 1 else 0.0)
        else:
            # Handle missing columns gracefully
            print(f"Warning: Column '{col}' not found in audio features")
            res[f"{col}_mean"] = 0.0
            res[f"{col}_stdev"] = 0.0

    # 2. Calculate Key and Mode counts (24 features)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Ensure key and mode columns exist
    if 'key' in df.columns and 'mode' in df.columns:
        for val, name in enumerate(key_names):
            res[f"{name}Major_count"] = float(len(df[(df['key'] == val) & (df['mode'] == 1)]))
            res[f"{name}minor_count"] = float(len(df[(df['key'] == val) & (df['mode'] == 0)]))
    else:
        # Fallback if key/mode missing
        for name in key_names:
            res[f"{name}Major_count"] = 0.0
            res[f"{name}minor_count"] = 0.0

    # 3. Add track count (1 feature)
    res["track_count"] = float(len(df))
    
    # Verify we have 45 features
    if len(res) != 45:
        print(f"Warning: Expected 45 features, got {len(res)}")
    
    return res


def fetch_user_data(token_info, feature_cols):
    """
    Fetches user top tracks and processes them into a single 45-feature vector.
    
    Args:
        token_info: Spotify OAuth token info
        feature_cols: List of 45 feature names from mbti_features.json
    
    Returns:
        tuple: (scaled_features_vector, track_info, top_artists, genres)
    """
    sp = spotipy.Spotify(auth=token_info["access_token"])
    
    try:
        # Get top 20 tracks
        top = sp.current_user_top_tracks(limit=20, time_range="medium_term")
        if not top or not top['items']:
            print("No top tracks found")
            return None, None, None, None

        track_ids = [t['id'] for t in top['items']]
        track_info = [(t['name'], t['artists'][0]['name']) for t in top['items']]
        
        # Get audio features for these tracks
        feats = sp.audio_features(track_ids)
        feats = [f for f in feats if f is not None]
        
        if len(feats) < 3:
            print(f"Only {len(feats)} tracks have audio features - need at least 3")
            return None, None, None, None
            
        df = pd.DataFrame(feats)
        
        # Aggregate features to 45 columns (raw values)
        agg = build_features(df)
        
        # Create raw feature vector in EXACT order from feature_cols
        raw_vector = np.zeros((1, len(feature_cols)), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            raw_vector[0, i] = agg.get(col, 0.0)
        
        # Load and apply the 45-feature scaler
        base_dir = Path(__file__).parent.parent
        scaler_path = base_dir / "data" / "processed" / "mbti_scaler.pkl"
        
        if not scaler_path.exists():
            print(f"ERROR: Scaler not found at {scaler_path}")
            print("Please run Notebook 3 and save mbti_scaler.pkl")
            return None, None, None, None
        
        scaler = joblib.load(scaler_path)
        scaled_vector = scaler.transform(raw_vector).astype(np.float32)
        
        # Get simplified artist list for Gemini
        artists_list = [t[1] for t in track_info]
        top_artists = list(pd.Series(artists_list).value_counts().head(5).index)
        
        # Get artist genres (optional - for richer analysis)
        genres = []
        try:
            # Get unique artists
            unique_artists = list(set(artists_list))
            for artist_name in unique_artists[:5]:  # Limit to first 5
                results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
                if results['artists']['items']:
                    artist_genres = results['artists']['items'][0].get('genres', [])
                    if artist_genres:
                        genres.extend(artist_genres[:2])
            genres = list(set(genres))[:5]  # Dedupe and limit
        except Exception as e:
            print(f"Could not fetch genres: {e}")
            genres = []

        print(f"✅ Processed {len(track_info)} tracks into {scaled_vector.shape[1]}-feature vector")
        return scaled_vector, track_info, top_artists, genres
        
    except Exception as e:
        print(f"Error in fetch_user_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None