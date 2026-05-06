import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import json
import os
import random
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Key mappings based on standard Spotify API (0=C, 1=C#/Db, etc.)
KEY_MAPPING = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
}
MODE_MAPPING = {0: "minor", 1: "Major"}

# Audio features order (must match Notebook 3)
AUDIO_FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "loudness", "tempo", "liveness"
]


def get_spotify_oauth():
    """Configure SpotifyOAuth for Streamlit"""
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=redirect_uri,
        scope="user-top-read",
        cache_path=None
    )


def load_backup_dataset():
    """Load the 1M Spotify songs dataset for feature fallback"""
    base_dir = Path(__file__).parent.parent
    backup_path = base_dir / "data" / "raw" / "spotify_data.csv"
    
    if backup_path.exists():
        print(f"📀 Loading backup dataset from {backup_path}")
        df = pd.read_csv(backup_path)
        # Ensure required columns exist
        required_cols = AUDIO_FEATURES + ['track_id', 'track_name', 'artist_name']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in backup dataset")
        return df
    else:
        print(f"⚠️ Backup dataset not found at {backup_path}")
        return None


def find_track_in_backup(track_name, artist_name, backup_df):
    """Search for a track in the backup dataset by name and artist"""
    if backup_df is None:
        return None
    
    # Try exact match first
    mask = (backup_df['track_name'].str.lower() == track_name.lower()) & \
           (backup_df['artist_name'].str.lower() == artist_name.lower())
    
    matches = backup_df[mask]
    
    if len(matches) == 0:
        # Try partial match on track name
        mask = backup_df['track_name'].str.lower().str.contains(track_name.lower(), na=False) & \
               (backup_df['artist_name'].str.lower().str.contains(artist_name.lower(), na=False))
        matches = backup_df[mask]
    
    if len(matches) > 0:
        return matches.iloc[0]
    
    return None


def generate_simulated_features(track_name, artist_name):
    """Generate realistic simulated audio features when track not found in backup"""
    # Use track/artist name to seed deterministic but varied features
    seed = hash(f"{track_name}_{artist_name}") % 10000
    random.seed(seed)
    
    # Realistic ranges for each feature
    features = {
        'danceability': random.uniform(0.2, 0.9),
        'energy': random.uniform(0.2, 0.95),
        'valence': random.uniform(0.1, 0.9),
        'acousticness': random.uniform(0.0, 0.9),
        'instrumentalness': random.uniform(0.0, 0.3),  # Most tracks have vocals
        'speechiness': random.uniform(0.03, 0.3),
        'loudness': random.uniform(-15, -3),
        'tempo': random.uniform(70, 160),
        'liveness': random.uniform(0.08, 0.4),
        'key': random.randint(0, 11),
        'mode': random.randint(0, 1)
    }
    
    return features


def get_audio_features_for_track(track_id, track_name, artist_name, backup_df, sp=None):
    """
    Get audio features for a single track with fallback hierarchy:
    1. Try Spotify API
    2. Try backup dataset
    3. Generate simulated features
    """
    features = None
    
    # Try 1: Spotify API (if sp is provided)
    if sp is not None:
        try:
            result = sp.audio_features([track_id])
            if result and result[0] is not None:
                features = result[0]
                print(f"✅ API: {track_name}")
                return features
        except Exception as e:
            print(f"⚠️ API failed for {track_name}: {e}")
    
    # Try 2: Backup dataset
    if backup_df is not None:
        backup_match = find_track_in_backup(track_name, artist_name, backup_df)
        if backup_match is not None:
            features = {
                'danceability': float(backup_match.get('danceability', 0.5)),
                'energy': float(backup_match.get('energy', 0.5)),
                'valence': float(backup_match.get('valence', 0.5)),
                'acousticness': float(backup_match.get('acousticness', 0.3)),
                'instrumentalness': float(backup_match.get('instrumentalness', 0.0)),
                'speechiness': float(backup_match.get('speechiness', 0.1)),
                'loudness': float(backup_match.get('loudness', -8)),
                'tempo': float(backup_match.get('tempo', 120)),
                'liveness': float(backup_match.get('liveness', 0.2)),
                'key': int(backup_match.get('key', 5)),
                'mode': int(backup_match.get('mode', 1))
            }
            print(f"📀 Backup: {track_name}")
            return features
    
    # Try 3: Simulated features
    features = generate_simulated_features(track_name, artist_name)
    print(f"🎲 Simulated: {track_name}")
    return features


def build_features_from_tracks(tracks_data):
    """
    Aggregates individual track data into the 45-feature format.
    
    Args:
        tracks_data: List of dicts, each containing audio features
    
    Returns:
        dict with 45 aggregated features
    """
    if not tracks_data:
        return None
    
    df = pd.DataFrame(tracks_data)
    res = {}
    
    # 1. Calculate Means and Standard Deviations (18 features)
    for col in AUDIO_FEATURES:
        if col in df.columns:
            res[f"{col}_mean"] = float(df[col].mean())
            res[f"{col}_stdev"] = float(df[col].std() if len(df) > 1 else 0.0)
        else:
            res[f"{col}_mean"] = 0.0
            res[f"{col}_stdev"] = 0.0
    
    # 2. Calculate Key and Mode counts (24 features)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Initialize all counts to 0
    for name in key_names:
        res[f"{name}Major_count"] = 0.0
        res[f"{name}minor_count"] = 0.0
    
    # Count actual distributions
    if 'key' in df.columns and 'mode' in df.columns:
        for _, row in df.iterrows():
            key = int(row['key']) if not pd.isna(row['key']) else -1
            mode = int(row['mode']) if not pd.isna(row['mode']) else -1
            
            if 0 <= key < 12:
                key_name = key_names[key]
                if mode == 1:
                    res[f"{key_name}Major_count"] += 1
                elif mode == 0:
                    res[f"{key_name}minor_count"] += 1
    
    # 3. Add track count (1 feature)
    res["track_count"] = float(len(df))
    
    return res


def fetch_user_data(token_info, feature_cols):
    """
    Fetches user top tracks and processes them into a single 45-feature vector.
    Uses fallback: API → Backup Dataset → Simulated features.
    
    Args:
        token_info: Spotify OAuth token info
        feature_cols: List of 45 feature names from mbti_features.json
    
    Returns:
        tuple: (scaled_features_vector, track_info, top_artists, genres)
    """
    # Load backup dataset once
    backup_df = load_backup_dataset()
    
    try:
        sp = spotipy.Spotify(auth=token_info['access_token'])
    except Exception as e:
        print(f"Error creating Spotify client: {e}")
        sp = None
    
    try:
        # Get top 20 tracks (limit 20 for faster processing)
        if sp is not None:
            top = sp.current_user_top_tracks(limit=20, time_range='medium_term')
            if not top or not top['items']:
                print("No top tracks found via API")
                return None, None, None, None
            items = top['items']
        else:
            print("No Spotify client available")
            return None, None, None, None
        
        # Collect track info and features
        track_info = []
        tracks_data = []
        
        print(f"\n📊 Fetching audio features for {len(items)} tracks...")
        
        for item in items:
            track_id = item['id']
            track_name = item['name']
            artist_name = item['artists'][0]['name']
            
            track_info.append((track_name, artist_name))
            
            # Get features with fallback
            features = get_audio_features_for_track(
                track_id, track_name, artist_name, backup_df, sp
            )
            
            if features:
                # Ensure all required fields are present
                for col in AUDIO_FEATURES + ['key', 'mode']:
                    if col not in features:
                        features[col] = 0.0 if col in AUDIO_FEATURES else (5 if col == 'key' else 1)
                tracks_data.append(features)
        
        if len(tracks_data) < 3:
            print(f"Only {len(tracks_data)} tracks have features - need at least 3")
            return None, None, None, None
        
        # Aggregate features to 45 columns
        agg = build_features_from_tracks(tracks_data)
        
        if agg is None:
            return None, None, None, None
        
        # Create raw feature vector in EXACT order from feature_cols
        raw_vector = np.zeros((1, len(feature_cols)), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            raw_vector[0, i] = agg.get(col, 0.0)
        
        # Load and apply the 45-feature scaler
        base_dir = Path(__file__).parent.parent
        scaler_path = base_dir / "data" / "processed" / "mbti_scaler.pkl"
        
        if not scaler_path.exists():
            print(f"ERROR: Scaler not found at {scaler_path}")
            print("Using unscaled features (predictions may be less accurate)")
            scaled_vector = raw_vector
        else:
            import joblib
            scaler = joblib.load(scaler_path)
            scaled_vector = scaler.transform(raw_vector).astype(np.float32)
        
        # Get top artists for Gemini
        artists_list = [t[1] for t in track_info]
        top_artists = list(pd.Series(artists_list).value_counts().head(5).index)
        
        # Get genres (optional)
        genres = []
        if sp is not None:
            try:
                unique_artists = list(set(artists_list))
                for artist_name in unique_artists[:3]:
                    results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
                    if results['artists']['items']:
                        artist_genres = results['artists']['items'][0].get('genres', [])
                        if artist_genres:
                            genres.extend(artist_genres[:2])
                genres = list(set(genres))[:5]
            except Exception as e:
                print(f"Could not fetch genres: {e}")
        
        print(f"\n✅ Successfully processed {len(tracks_data)} tracks")
        print(f"   - API: {sum(1 for t in tracks_data if 'api' in str(t))}")
        print(f"   - Backup/Simulated: {len(tracks_data) - sum(1 for t in tracks_data if 'api' in str(t))}")
        
        return scaled_vector, track_info, top_artists, genres
        
    except Exception as e:
        print(f"Error in fetch_user_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# For testing purposes
if __name__ == "__main__":
    # Test the feature builder
    test_tracks = [
        {
            'danceability': 0.5, 'energy': 0.6, 'valence': 0.4,
            'acousticness': 0.3, 'instrumentalness': 0.0, 'speechiness': 0.1,
            'loudness': -8, 'tempo': 120, 'liveness': 0.2,
            'key': 5, 'mode': 1
        },
        {
            'danceability': 0.7, 'energy': 0.8, 'valence': 0.6,
            'acousticness': 0.1, 'instrumentalness': 0.0, 'speechiness': 0.05,
            'loudness': -5, 'tempo': 130, 'liveness': 0.3,
            'key': 7, 'mode': 0
        }
    ]
    
    agg = build_features_from_tracks(test_tracks)
    print(f"Test aggregation produced {len(agg)} features")
    for k, v in list(agg.items())[:10]:
        print(f"  {k}: {v}")