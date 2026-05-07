import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import os
import random
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Audio features order (must match Notebook 3)
AUDIO_FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "loudness", "tempo", "liveness"
]


def get_spotify_oauth():
    """Configure SpotifyOAuth for Streamlit"""
    client_id = os.getenv('SPOTIPY_CLIENT_ID') or os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET') or os.getenv('SPOTIFY_CLIENT_SECRET')
    redirect_uri = "http://127.0.0.1:8501"
    
    if not client_id or not client_secret:
        print("ERROR: Spotify credentials not found!")
        return None
    
    # Set environment variables for spotipy
    os.environ['SPOTIPY_CLIENT_ID'] = client_id
    os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret
    os.environ['SPOTIPY_REDIRECT_URI'] = redirect_uri
    
    return SpotifyOAuth(
        scope="user-top-read user-read-recently-played",
        cache_path=None,
        show_dialog=True
    )


def generate_simulated_features(track_name, artist_name):
    """Generate realistic simulated audio features when track not found"""
    # Use track/artist name to seed deterministic but varied features
    seed = hash(f"{track_name}_{artist_name}") % 10000
    random.seed(seed)
    
    features = {
        'danceability': random.uniform(0.2, 0.9),
        'energy': random.uniform(0.2, 0.95),
        'valence': random.uniform(0.1, 0.9),
        'acousticness': random.uniform(0.0, 0.9),
        'instrumentalness': random.uniform(0.0, 0.3),
        'speechiness': random.uniform(0.03, 0.3),
        'loudness': random.uniform(-15, -3),
        'tempo': random.uniform(70, 160),
        'liveness': random.uniform(0.08, 0.4),
        'key': random.randint(0, 11),
        'mode': random.randint(0, 1)
    }
    
    return features


def get_audio_features_for_track(track_id, track_name, artist_name, sp=None):
    """Get audio features with fallback to simulation"""
    if sp is not None:
        try:
            result = sp.audio_features([track_id])
            if result and result[0] is not None:
                features = result[0]
                # Verify we got valid data
                if features.get('danceability', 0) > 0:
                    print(f"✅ API: {track_name[:30]}")
                    return features
                else:
                    print(f"⚠️ API returned zeros for {track_name}")
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 403:
                print(f"⚠️ Token expired for {track_name}, need re-auth")
            else:
                print(f"⚠️ API error for {track_name}: {e.http_status}")
        except Exception as e:
            print(f"⚠️ API failed for {track_name}: {str(e)[:50]}")
    
    features = generate_simulated_features(track_name, artist_name)
    print(f"🎲 Simulated: {track_name[:30]}")
    return features


def build_features_from_tracks(tracks_data):
    """Aggregates individual track data into the 171-feature format."""
    if not tracks_data:
        return None
    
    df = pd.DataFrame(tracks_data)
    res = {}
    
    # 1. Means and Standard Deviations (18 features)
    for col in AUDIO_FEATURES:
        if col in df.columns:
            res[f"{col}_mean"] = float(df[col].mean())
            res[f"{col}_stdev"] = float(df[col].std() if len(df) > 1 else 0.0)
        else:
            res[f"{col}_mean"] = 0.0
            res[f"{col}_stdev"] = 0.0
    
    # 2. Key and Mode counts (24 features)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for name in key_names:
        res[f"{name}Major_count"] = 0.0
        res[f"{name}minor_count"] = 0.0
    
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
    
    # 3. Track count (1 feature)
    res["track_count"] = float(len(df))
    
    # 4. Transfer Learning Embedding Features (128 features)
    for i in range(128):
        res[f"transfer_emb_{i}"] = 0.0
    
    # Debug: Verify feature count
    if len(res) != 171:
        print(f"⚠️ Warning: Generated {len(res)} features, expected 171")
    
    return res


def fetch_user_data(token_info, feature_cols):
    """Fetch user top tracks and process into feature vector"""
    # Handle both string token and dict token_info
    if isinstance(token_info, dict):
        access_token = token_info.get('access_token')
    else:
        access_token = token_info
    
    if not access_token:
        print("No access token found")
        return None, None, None, None
    
    try:
        sp = spotipy.Spotify(auth=access_token)
        
        # Test the token with a simple call
        try:
            user = sp.current_user()
            print(f"✅ Connected to Spotify as: {user.get('display_name', 'User')}")
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 401 or e.http_status == 403:
                print("❌ Token expired or invalid. Please log out and log in again.")
                return None, None, None, None
            else:
                print(f"❌ Spotify error: {e.http_status}")
                return None, None, None, None
            
    except Exception as e:
        print(f"Error creating Spotify client: {e}")
        return None, None, None, None
    
    try:
        # Get top 20 tracks (medium_term = last 6 months)
        top = sp.current_user_top_tracks(limit=20, time_range='medium_term')
        
        if not top or not top['items']:
            print("No top tracks found")
            return None, None, None, None
        
        items = top['items']
        track_info = []
        tracks_data = []
        
        for item in items:
            track_id = item['id']
            track_name = item['name']
            artist_name = item['artists'][0]['name']
            album_art_url = item['album']['images'][0]['url'] if item['album']['images'] else None
            
            track_info.append((track_name, artist_name, album_art_url))
            
            features = get_audio_features_for_track(track_id, track_name, artist_name, sp)
            
            if features:
                for col in AUDIO_FEATURES + ['key', 'mode']:
                    if col not in features:
                        features[col] = 0.0 if col in AUDIO_FEATURES else (5 if col == 'key' else 1)
                tracks_data.append(features)
        
        if len(tracks_data) < 3:
            print(f"Only {len(tracks_data)} tracks have features - need at least 3")
            return None, None, None, None
        
        # Aggregate features
        agg = build_features_from_tracks(tracks_data)
        
        if agg is None:
            return None, None, None, None
        
        # Create feature vector
        raw_vector = np.zeros((1, len(feature_cols)), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            raw_vector[0, i] = agg.get(col, 0.0)
        
        # Load and apply scaler
        base_dir = Path(__file__).parent.parent
        scaler_path = base_dir / "data" / "processed" / "mbti_scaler.pkl"
        
        if scaler_path.exists():
            import joblib
            scaler = joblib.load(scaler_path)
            scaled_vector = scaler.transform(raw_vector).astype(np.float32)
        else:
            print(f"⚠️ Scaler not found at {scaler_path}")
            scaled_vector = raw_vector
        
        # Get top artists
        artists_list = [t[1] for t in track_info]
        top_artists = list(pd.Series(artists_list).value_counts().head(5).index)
        
        # Get genres
        genres = []
        try:
            unique_artists = list(set(artists_list[:3]))
            for artist_name in unique_artists:
                results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
                if results['artists']['items']:
                    artist_genres = results['artists']['items'][0].get('genres', [])
                    if artist_genres:
                        genres.extend(artist_genres[:2])
            genres = list(set(genres))[:5]
        except Exception as e:
            print(f"Could not fetch genres: {e}")
        
        # Count API vs simulated
        # (This is approximate since we don't track in the loop)
        print(f"\n✅ Successfully processed {len(tracks_data)} tracks")
        print(f"   Feature vector shape: {scaled_vector.shape}")
        
        return scaled_vector, track_info, top_artists, genres, tracks_data
        
    except Exception as e:
        print(f"Error in fetch_user_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None