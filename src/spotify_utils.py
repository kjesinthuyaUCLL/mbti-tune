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

_SONG_ENCODER = None
_SONG_SCALER = None

_SONG_DATABASE = None
_SONG_DATABASE_METADATA = None

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "loudness", "tempo", "liveness"
]


def get_spotify_oauth():
    client_id = os.getenv('SPOTIPY_CLIENT_ID') or os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET') or os.getenv('SPOTIFY_CLIENT_SECRET')
    redirect_uri = "http://127.0.0.1:8501"
    
    if not client_id or not client_secret:
        print("ERROR: Spotify credentials not found!")
        return None
    

    os.environ['SPOTIPY_CLIENT_ID'] = client_id
    os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret
    os.environ['SPOTIPY_REDIRECT_URI'] = redirect_uri
    
    return SpotifyOAuth(
        scope="user-top-read user-read-recently-played",
        cache_handler=spotipy.cache_handler.MemoryCacheHandler(),
        show_dialog=True
    )


def load_song_encoder():
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
        device = torch.device("cpu")
        state_dict = torch.load(autoencoder_path, map_location=device, weights_only=False)


        autoencoder = SongAutoencoder(input_dim=9, latent_dim=32)
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()

        encoder = autoencoder.encoder


        song_scaler = None
        if song_scaler_path.exists():
            song_scaler = joblib.load(song_scaler_path)
            print(f"✅ Loaded song scaler (expects {song_scaler.mean_.shape[0]} features)")
        else:
            print("⚠️ Song scaler not found - will normalise audio features manually.")

        _SONG_ENCODER = encoder
        _SONG_SCALER = song_scaler
        print("✅ Loaded SongAutoencoder encoder for transfer embeddings.")
        return encoder, song_scaler

    except Exception as e:
        print(f"⚠️ Could not load SongAutoencoder: {e}. Transfer embeddings will be zeros.")
        return None, None


def load_song_database():
    global _SONG_DATABASE, _SONG_DATABASE_METADATA
    
    if _SONG_DATABASE is not None:
        return _SONG_DATABASE, _SONG_DATABASE_METADATA
    
    base_dir = Path(__file__).parent.parent
    db_dir = base_dir / "data" / "processed_song_database"
    
    sample_features = db_dir / "sample_features_100k.npy"
    sample_metadata = db_dir / "sample_metadata_100k.parquet"
    
    if not sample_features.exists() or not sample_metadata.exists():
        print(f"⚠️ Song database not found at {db_dir}. Using beta distribution fallback.")
        return None, None
    
    try:
        _SONG_DATABASE = np.load(sample_features)
        _SONG_DATABASE_METADATA = pd.read_parquet(sample_metadata)
        print(f"✅ Loaded {len(_SONG_DATABASE):,} real songs from database")
        return _SONG_DATABASE, _SONG_DATABASE_METADATA
    except Exception as e:
        print(f"⚠️ Could not load song database: {e}")
        return None, None


def encode_songs_to_transfer_emb(tracks_data, encoder, song_scaler):
    import torch

    TRANSFER_DIM = 128
    fallback = {f"transfer_emb_{i}": 0.0 for i in range(TRANSFER_DIM)}

    if encoder is None or not tracks_data:
        return fallback

    try:
        feat_matrix = []
        for t in tracks_data:
            row = [float(t.get(f, 0.0)) for f in AUDIO_FEATURES]
            feat_matrix.append(row)

        feat_np = np.array(feat_matrix, dtype=np.float32)

        if song_scaler is not None and song_scaler.mean_.shape[0] == 9:
            feat_np = song_scaler.transform(feat_np).astype(np.float32)

        needs_pad = feat_np.shape[0] < 2
        if needs_pad:
            feat_np = np.vstack([feat_np, feat_np])

        x = torch.tensor(feat_np, dtype=torch.float32)

        with torch.no_grad():
            latents = encoder(x).cpu().numpy()

        if needs_pad:
            latents = latents[:1]


        emb_mean = latents.mean(axis=0)
        emb_std = latents.std(axis=0)
        emb_min = latents.min(axis=0)
        emb_max = latents.max(axis=0)

        transfer_128 = np.concatenate([emb_mean, emb_std, emb_min, emb_max])  # (128,)

        return {f"transfer_emb_{i}": float(v) for i, v in enumerate(transfer_128)}

    except Exception as e:
        print(f"⚠️ Could not generate transfer embeddings: {e}. Using zeros.")
        return fallback


def generate_simulated_features_from_database(track_name, artist_name):
    import hashlib
    import numpy as np
    
    song_features, song_metadata = load_song_database()
    
    if song_features is not None:
        seed_str = f"{track_name}_{artist_name}".encode('utf-8')
        seed = int(hashlib.md5(seed_str).hexdigest()[:8], 16)
        idx = seed % len(song_features)
        
        real_features = song_features[idx]
        
        noise = np.random.normal(0, 0.05, size=real_features.shape)
        final_features = real_features + noise
        
        final_features = np.clip(final_features, -3, 3)
        
        genre = "unknown"
        if song_metadata is not None and idx < len(song_metadata):
            genre = song_metadata.iloc[idx].get('genre', 'unknown')
        
        print(f"🎵 Database: {track_name[:30]} (matched with {genre})")
        
        return {
            'danceability': np.clip(final_features[0] * 0.2 + 0.5, 0.1, 0.95),
            'energy': np.clip(final_features[1] * 0.25 + 0.5, 0.1, 0.98),
            'valence': np.clip(final_features[2] * 0.25 + 0.5, 0.1, 0.95),
            'acousticness': np.clip(final_features[3] * 0.3 + 0.3, 0.0, 1.0),
            'instrumentalness': np.clip(final_features[4] * 0.2 + 0.05, 0.0, 0.9),
            'speechiness': np.clip(final_features[5] * 0.15 + 0.08, 0.02, 0.5),
            'loudness': np.clip(final_features[6] * 6 - 8, -25, -2),
            'tempo': np.clip(final_features[7] * 35 + 105, 60, 180),
            'liveness': np.clip(final_features[8] * 0.2 + 0.15, 0.05, 0.6),
            'key': np.random.randint(0, 12),
            'mode': 1 if np.random.random() < 0.7 else 0
        }
    
    return None


def generate_simulated_features_beta(track_name, artist_name):
    import hashlib
    import numpy as np
    
    seed_str = f"{track_name}_{artist_name}".encode('utf-8')
    seed = int(hashlib.md5(seed_str).hexdigest()[:8], 16)
    np.random.seed(seed)
    
    def bounded_beta(a, b, low, high):
        return low + (high - low) * np.random.beta(a, b)
    
    features = {
        'danceability': bounded_beta(3, 3, 0.25, 0.85),
        'energy': bounded_beta(2.5, 3, 0.2, 0.92),
        'valence': bounded_beta(3, 3, 0.15, 0.85),
        'acousticness': bounded_beta(1.5, 4, 0, 0.5) if np.random.random() < 0.4 else bounded_beta(4, 1.5, 0.5, 0.95),
        'instrumentalness': bounded_beta(0.8, 5, 0, 0.15) if np.random.random() < 0.9 else bounded_beta(3, 3, 0.15, 0.8),
        'speechiness': bounded_beta(1.2, 6, 0.025, 0.12) if np.random.random() < 0.7 else bounded_beta(3, 2, 0.12, 0.35),
        'loudness': bounded_beta(2.5, 2.5, -14, -4),
        'tempo': bounded_beta(2, 2, 65, 155),
        'liveness': bounded_beta(1.5, 5, 0.04, 0.35),
        'key': np.random.randint(0, 12),
        'mode': 1 if np.random.random() < 0.7 else 0
    }
    
    return features


def generate_simulated_features(track_name, artist_name):

    db_features = generate_simulated_features_from_database(track_name, artist_name)
    if db_features is not None:
        return db_features
    
    print(f"🎲 Beta sim: {track_name[:30]}")
    return generate_simulated_features_beta(track_name, artist_name)


def get_audio_features_for_track(track_id, track_name, artist_name, sp=None):
    if sp is not None:
        try:
            result = sp.audio_features([track_id])
            if result and result[0] is not None:
                features = result[0]

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
    return features


def build_features_from_tracks(tracks_data):
    if not tracks_data:
        return None
    
    df = pd.DataFrame(tracks_data)
    res = {}
    
    # Means and Standard Deviations
    for col in AUDIO_FEATURES:
        if col in df.columns:
            res[f"{col}_mean"] = float(df[col].mean())
            res[f"{col}_stdev"] = float(df[col].std() if len(df) > 1 else 0.0)
        else:
            res[f"{col}_mean"] = 0.0
            res[f"{col}_stdev"] = 0.0
    
    # Key and Mode counts
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
    
    # Track count
    res["track_count"] = float(len(df))
    
    # Transfer Learning
    for i in range(128):
        res[f"transfer_emb_{i}"] = 0.0


    if len(res) != 171:
        print(f"⚠️ Generated {len(res)} features, expected 171")

    return res


def fetch_user_data(token_info, feature_cols):
    
    if isinstance(token_info, dict):
        access_token = token_info.get('access_token')
    else:
        access_token = token_info
    
    if not access_token:
        print("No access token found")
        return None, None, None, None, None
    
    try:
        sp = spotipy.Spotify(auth=access_token)
        
        try:
            user = sp.current_user()
            print(f"✅ Connected to Spotify as: {user.get('display_name', 'User')}")
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 401 or e.http_status == 403:
                print("❌ Token expired or invalid. Please log out and log in again.")
                return None, None, None, None, None
            else:
                print(f"❌ Spotify error: {e.http_status}")
                return None, None, None, None, None
            
    except Exception as e:
        print(f"Error creating Spotify client: {e}")
        return None, None, None, None, None
    
    try:

        top = sp.current_user_top_tracks(limit=20, time_range='medium_term')
        
        if not top or not top['items']:
            print("No top tracks found")
            return None, None, None, None, None
        
        items = top['items']
        track_info = []
        tracks_data = []
        api_count = 0
        sim_count = 0
        
        for item in items:
            track_id = item['id']
            track_name = item['name']
            artist_name = item['artists'][0]['name']
            album_art_url = item['album']['images'][0]['url'] if item['album']['images'] else None
            
            track_info.append((track_name, artist_name, album_art_url))
            
            features = get_audio_features_for_track(track_id, track_name, artist_name, sp)
            
            if features:
                if features.get('danceability', 0) > 0.1 and features.get('energy', 0) > 0:
                    api_count += 1
                else:
                    sim_count += 1
                    
                for col in AUDIO_FEATURES + ['key', 'mode']:
                    if col not in features:
                        features[col] = 0.0 if col in AUDIO_FEATURES else (5 if col == 'key' else 1)
                tracks_data.append(features)
        
        if len(tracks_data) < 3:
            print(f"Only {len(tracks_data)} tracks have features - need at least 3")
            return None, None, None, None, None

        print(f"\n📊 Data sources: {api_count} API, {sim_count} simulated")

        agg = build_features_from_tracks(tracks_data)

        if agg is None:
            return None, None, None, None, None

        encoder, song_scaler = load_song_encoder()
        transfer_emb_dict = encode_songs_to_transfer_emb(tracks_data, encoder, song_scaler)
        agg.update(transfer_emb_dict)

        transfer_nonzero = sum(1 for v in transfer_emb_dict.values() if v != 0.0)
        print(f"Transfer embeddings non-zero count: {transfer_nonzero}/128")

        raw_vector = np.zeros((1, len(feature_cols)), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            raw_vector[0, i] = agg.get(col, 0.0)
            
        base_dir = Path(__file__).parent.parent
        scaler_path = base_dir / "data" / "processed" / "mbti_scaler.pkl"
        
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            scaled_vector = scaler.transform(raw_vector).astype(np.float32)
            
            print("\n" + "="*50)
            print("🔧 APPLYING STABILIZATION FIX")
            print("="*50)
            
            # Clip range
            old_min, old_max = np.min(scaled_vector), np.max(scaled_vector)
            scaled_vector = np.clip(scaled_vector, -3, 3)
            print(f"   Clipping: [{old_min:.2f}, {old_max:.2f}] → [{np.min(scaled_vector):.2f}, {np.max(scaled_vector):.2f}]")
            

            small_std_threshold = 0.15
            downweighted_count = 0
            for i, col in enumerate(feature_cols):
                if scaler.scale_[i] < small_std_threshold:
                    scaled_vector[0, i] = scaled_vector[0, i] * 0.3
                    downweighted_count += 1
            
            if downweighted_count > 0:
                print(f"   Down-weighted {downweighted_count} low-variance features")
            
            print("="*50)
            
        else:
            print(f"⚠️ Scaler not found at {scaler_path}")
            scaled_vector = raw_vector
        

        artists_list = [t[1] for t in track_info]
        top_artists = list(pd.Series(artists_list).value_counts().head(5).index)
        

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
        
        print(f"\n✅ Successfully processed {len(tracks_data)} tracks")
        print(f"   Feature vector shape: {scaled_vector.shape}")
        
        return scaled_vector, track_info, top_artists, genres, tracks_data
        
    except Exception as e:
        print(f"Error in fetch_user_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None