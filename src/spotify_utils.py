import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import os
import random
import joblib
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Module-level cache for the song encoder (loaded once)
_SONG_ENCODER = None
_SONG_SCALER = None

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


def load_song_encoder():
    """
    Load the pretrained SongAutoencoder encoder (from Notebook 1/2) and its
    song-level StandardScaler. These are used to generate the 128-dim transfer
    embeddings required by the MBTIClassifier.

    Since the PlaylistLSTMEncoder weights were never saved, we approximate
    the 128 transfer features as:
        [mean(32-dim), std(32-dim), min(32-dim), max(32-dim)]
    of all per-song latent encodings. This exactly fills the 128 slots and
    preserves distributional information.

    Returns: (encoder_net, song_scaler) or (None, None) on failure.
    """
    global _SONG_ENCODER, _SONG_SCALER

    if _SONG_ENCODER is not None:
        return _SONG_ENCODER, _SONG_SCALER

    import torch
    import torch.nn as nn
    from src.model import SongAutoencoder

    base_dir = Path(__file__).parent.parent
    autoencoder_path = base_dir / "data" / "processed" / "autoencoder_model.pth"
    song_scaler_path = base_dir / "data" / "processed" / "song_scaler.pkl"

    if not autoencoder_path.exists():
        print(f"WARNING: Autoencoder not found at {autoencoder_path}. Transfer embeddings will be zeros.")
        return None, None

    try:
        device = torch.device("cpu")  # Always CPU for inference
        state_dict = torch.load(autoencoder_path, map_location=device, weights_only=False)

        # SongAutoencoder was trained with input_dim=9, latent_dim=32
        autoencoder = SongAutoencoder(input_dim=9, latent_dim=32)
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()

        # We only need the encoder half
        encoder = autoencoder.encoder

        # Load song-level scaler if it exists
        song_scaler = None
        if song_scaler_path.exists():
            song_scaler = joblib.load(song_scaler_path)
            print(f"Loaded song scaler (expects {song_scaler.mean_.shape[0]} features)")
        else:
            print("Song scaler not found - will normalise audio features manually.")

        _SONG_ENCODER = encoder
        _SONG_SCALER = song_scaler
        print("Loaded SongAutoencoder encoder for transfer embeddings.")
        return encoder, song_scaler

    except Exception as e:
        print(f"WARNING: Could not load SongAutoencoder: {e}. Transfer embeddings will be zeros.")
        return None, None


def encode_songs_to_transfer_emb(tracks_data, encoder, song_scaler):
    """
    Encodes a list of track feature dicts using the SongAutoencoder encoder.
    Produces a 128-dim transfer embedding vector by computing
    [mean, std, min, max] of the per-song 32-dim latent vectors.

    Falls back to zeros if encoder is None or if encoding fails.
    """
    import torch

    TRANSFER_DIM = 128
    fallback = {f"transfer_emb_{i}": 0.0 for i in range(TRANSFER_DIM)}

    if encoder is None or not tracks_data:
        return fallback

    try:
        # Build matrix of 9 audio features per song
        feat_matrix = []
        for t in tracks_data:
            row = [float(t.get(f, 0.0)) for f in AUDIO_FEATURES]
            feat_matrix.append(row)

        feat_np = np.array(feat_matrix, dtype=np.float32)  # (N, 9)

        # Normalise with song_scaler if available
        if song_scaler is not None and song_scaler.mean_.shape[0] == 9:
            feat_np = song_scaler.transform(feat_np).astype(np.float32)

        # BatchNorm1d needs batch_size >= 2 — pad if necessary
        needs_pad = feat_np.shape[0] < 2
        if needs_pad:
            feat_np = np.vstack([feat_np, feat_np])  # duplicate

        x = torch.tensor(feat_np, dtype=torch.float32)

        with torch.no_grad():
            latents = encoder(x).cpu().numpy()  # (N, 32)

        if needs_pad:
            latents = latents[:1]  # keep only the original

        # Compute distributional statistics: mean, std, min, max  (4 × 32 = 128)
        emb_mean = latents.mean(axis=0)          # (32,)
        emb_std  = latents.std(axis=0)           # (32,)
        emb_min  = latents.min(axis=0)           # (32,)
        emb_max  = latents.max(axis=0)           # (32,)

        transfer_128 = np.concatenate([emb_mean, emb_std, emb_min, emb_max])  # (128,)

        return {f"transfer_emb_{i}": float(v) for i, v in enumerate(transfer_128)}

    except Exception as e:
        print(f"WARNING: Could not generate transfer embeddings: {e}. Using zeros.")
        return fallback


def generate_simulated_features(track_name, artist_name):
    """Generate realistic simulated audio features when track not found"""
    import hashlib
    # Use hashlib to create a deterministic seed across Python runs (built-in hash() is randomized)
    seed_str = f"{track_name}_{artist_name}".encode('utf-8')
    seed = int(hashlib.md5(seed_str).hexdigest()[:8], 16)
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
    # Filled by the caller via encode_songs_to_transfer_emb() if an encoder is available.
    # Defaulting to zeros here; callers should override with real embeddings.
    for i in range(128):
        res[f"transfer_emb_{i}"] = 0.0

    # Debug: Verify feature count
    if len(res) != 171:
        print(f"WARNING: Generated {len(res)} features, expected 171")

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

        # Aggregate statistical features (43 features)
        agg = build_features_from_tracks(tracks_data)

        if agg is None:
            return None, None, None, None

        # Generate the 128-dim transfer embeddings using the SongAutoencoder
        encoder, song_scaler = load_song_encoder()
        transfer_emb_dict = encode_songs_to_transfer_emb(tracks_data, encoder, song_scaler)
        agg.update(transfer_emb_dict)  # Override the zero placeholders with real embeddings

        print(f"Transfer embeddings non-zero count: {sum(1 for v in transfer_emb_dict.values() if v != 0.0)}/128")

        # Create feature vector in exact order of feature_cols (171 total)
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