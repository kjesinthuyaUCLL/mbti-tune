# app/app.py
import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
from google import genai
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MBTI Tune",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .big-font {
        font-size: 20px !important;
    }
    .result-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model and Scaler (cached)
# ============================================
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    
    class MBTIModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)
    
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Load model
    scaler = joblib.load(os.path.join(models_dir, 'scaler_aggregated.pkl'))
    feature_cols = joblib.load(os.path.join(models_dir, 'agg_feature_cols.pkl'))
    
    model = MBTIModel(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(os.path.join(models_dir, 'mbti_model_aggregated.pth'), map_location='cpu'))
    model.eval()
    
    return model, scaler, feature_cols

# ============================================
# Gemini API Setup
# ============================================
@st.cache_resource
def setup_gemini():
    """Initialize Gemini client"""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return None

def generate_description(mbti_type, top_artists, top_genres=None):
    """Generate funny personality description using Gemini"""
    client = setup_gemini()
    if not client:
        return get_fallback_description(mbti_type, top_artists)
    
    artists_str = ', '.join(top_artists[:3]) if top_artists else "various artists"
    
    prompt = f"""Write a funny, short personality description for someone who is {mbti_type}.
Their top artists are {artists_str}.
Make it humorous, slightly roasting, and under 100 words.
Reference their music taste in the description.
Keep it positive and fun.
"""
    
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return get_fallback_description(mbti_type, top_artists)

def get_fallback_description(mbti_type, top_artists):
    """Fallback description if Gemini fails"""
    artists_str = ', '.join(top_artists[:2]) if top_artists else "your favorite artists"
    
    fallbacks = {
        "INTJ": f"As an INTJ who loves {artists_str}, you probably analyze music like a chess game - strategic and calculated.",
        "INTP": f"With your INTP mind and love for {artists_str}, you don't just hear music - you deconstruct it.",
        "ENTJ": f"ENTJ and {artists_str}? You command your playlist like a CEO commands a boardroom.",
        "ENTP": f"As an ENTP listening to {artists_str}, you argue with yourself about which song is better.",
        "INFJ": f"INFJ with {artists_str}? You feel music deeply - sometimes too deeply for your own good.",
        "INFP": f"INFP and {artists_str}? Your playlist is basically your diary set to music.",
        "ENFJ": f"As an ENFJ who loves {artists_str}, you're probably curating playlists for all your friends.",
        "ENFP": f"ENFP with {artists_str}? You have 27 playlists for 27 different moods.",
        "ISTJ": f"As an ISTJ listening to {artists_str}, you have one perfect playlist and you never change it.",
        "ISFJ": f"ISFJ and {artists_str}? You remember every lyric from every song you've ever heard.",
        "ESTJ": f"ESTJ with {artists_str}? Your playlist is organized by genre, then year, then artist.",
        "ESFJ": f"As an ESFJ who loves {artists_str}, you've probably shared this music with everyone you know.",
        "ISTP": f"ISTP and {artists_str}? You listen while fixing things. Lots of things.",
        "ISFP": f"ISFP with {artists_str}? Your taste is as unique as your art.",
        "ESTP": f"As an ESTP listening to {artists_str}, you're the life of every party.",
        "ESFP": f"ESFP and {artists_str}? You ARE the party. The playlist just follows.",
    }
    return fallbacks.get(mbti_type, f"You're a {mbti_type} who loves {artists_str}. Your playlist is as unique as you are!")

# ============================================
# Spotify Integration
# ============================================
def get_spotify_client():
    """Initialize Spotify client with user authentication"""
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri="https://localhost:8501",
            scope="user-top-read user-read-private",
            cache_path=".spotify_cache"
        ))
        return sp
    except Exception as e:
        st.error(f"Spotify connection error: {e}")
        return None

def get_user_top_tracks(sp, limit=20):
    """Fetch user's top tracks and their audio features"""
    try:
        # Get top tracks
        top_tracks = sp.current_user_top_tracks(limit=limit, time_range='medium_term')
        
        tracks_data = []
        track_ids = []
        
        for item in top_tracks['items']:
            tracks_data.append({
                'name': item['name'],
                'artist': item['artists'][0]['name'],
                'id': item['id']
            })
            track_ids.append(item['id'])
        
        # Get audio features for all tracks
        if track_ids:
            features = sp.audio_features(track_ids)
            for i, feat in enumerate(features):
                if feat:
                    tracks_data[i].update({
                        'danceability': feat.get('danceability', 0),
                        'energy': feat.get('energy', 0),
                        'valence': feat.get('valence', 0),
                        'acousticness': feat.get('acousticness', 0),
                        'instrumentalness': feat.get('instrumentalness', 0),
                        'liveness': feat.get('liveness', 0),
                        'speechiness': feat.get('speechiness', 0),
                        'tempo': feat.get('tempo', 120),
                        'loudness': feat.get('loudness', -10),
                        'key': feat.get('key', 0),
                        'mode': feat.get('mode', 0)
                    })
        
        return tracks_data
    except Exception as e:
        st.error(f"Error fetching Spotify data: {e}")
        return None

def calculate_user_profile(tracks):
    """Calculate average audio features from user's tracks"""
    if not tracks:
        return None
    
    features = ['danceability', 'energy', 'valence', 'acousticness', 
                'instrumentalness', 'liveness', 'speechiness', 'tempo', 
                'loudness', 'key', 'mode']
    
    profile = {}
    for feature in features:
        values = [t.get(feature, 0) for t in tracks if feature in t]
        profile[feature] = np.mean(values) if values else 0
    
    # Get top artists for description
    artist_counts = {}
    for t in tracks:
        artist = t.get('artist', 'Unknown')
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
    
    top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_artists = [a[0] for a in top_artists]
    
    return profile, top_artists

# ============================================
# Prediction Functions
# ============================================
def mbti_to_dimensions(mbti_type):
    """Convert MBTI type to 4 dimensions (not used for prediction, just display)"""
    return [
        1 if 'E' in mbti_type else 0,
        1 if 'N' in mbti_type else 0,
        1 if 'T' in mbti_type else 0,
        1 if 'J' in mbti_type else 0
    ]

def predict_mbti(model, scaler, feature_cols, user_profile):
    """Predict MBTI dimensions from user profile"""
    # Create feature vector in correct order
    feature_vector = []
    for col in feature_cols:
        # Map column names to user profile keys
        col_base = col.replace('_mean', '').replace('_stdev', '')
        if col_base in user_profile:
            feature_vector.append(user_profile[col_base])
        else:
            feature_vector.append(0)
    
    # Scale features
    feature_array = np.array(feature_vector).reshape(1, -1)
    scaled = scaler.transform(feature_array)
    
    # Predict
    tensor = torch.FloatTensor(scaled)
    with torch.no_grad():
        percentages = model(tensor).numpy()[0] * 100
    
    # Determine MBTI type from percentages
    e, n, t, j = percentages
    mbti_type = ('E' if e > 50 else 'I') + \
                 ('N' if n > 50 else 'S') + \
                 ('T' if t > 50 else 'F') + \
                 ('J' if j > 50 else 'P')
    
    return percentages, mbti_type

# ============================================
# Main App
# ============================================
def main():
    st.title("🎵 MBTI Tune")
    st.subheader("Discover your music personality")
    st.markdown("---")
    
    # Check if logged in
    if 'spotify_logged_in' not in st.session_state:
        st.session_state.spotify_logged_in = False
        st.session_state.user_profile = None
        st.session_state.top_artists = None
        st.session_state.predicted_mbti = None
        st.session_state.dimensions = None
    
    # Sidebar for login
    with st.sidebar:
        st.header("🎧 Connect Spotify")
        
        if not st.session_state.spotify_logged_in:
            if st.button("🔗 Login with Spotify", type="primary"):
                sp = get_spotify_client()
                if sp:
                    st.session_state.spotify_client = sp
                    st.session_state.spotify_logged_in = True
                    st.rerun()
        else:
            st.success("✅ Connected to Spotify")
            st.caption("Click Logout to disconnect")
            if st.button("🚪 Logout"):
                for key in ['spotify_logged_in', 'spotify_client', 'user_profile', 'top_artists', 'predicted_mbti', 'dimensions']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Main content
    if not st.session_state.spotify_logged_in:
        st.info("👈 Please login with Spotify to discover your music personality!")
        
        st.markdown("""
        ### How it works:
        1. **Login with Spotify** - Connect your Spotify account
        2. **We analyze** your top 20 most listened tracks
        3. **Our AI predicts** your MBTI personality dimensions
        4. **Get a fun description** personalized to your music taste
        
        Your data stays private. We only access your top tracks.
        """)
        
        # Sample output preview
        with st.expander("See example output"):
            st.markdown("""
            ### Your Music Personality: ENFJ
            
            | Dimension | Score |
            |-----------|-------|
            | Extraversion (E) | 78% ████████░░ |
            | Intuition (N) | 65% ██████░░░░ |
            | Thinking (T) | 32% ███░░░░░░░ |
            | Judging (J) | 82% ████████░░ |
            
            **Gemini says:** You're the kind of person who listens to Hozier when you need to feel deep emotions, but also blasts Lizzo when you need to hype yourself up...
            """)
    
    else:
        # User is logged in
        sp = st.session_state.spotify_client
        
        if st.session_state.user_profile is None:
            # Fetch and analyze
            with st.spinner("🎵 Fetching your top tracks..."):
                tracks = get_user_top_tracks(sp, limit=20)
                
                if tracks:
                    profile, top_artists = calculate_user_profile(tracks)
                    st.session_state.user_profile = profile
                    st.session_state.top_artists = top_artists
                    st.session_state.tracks = tracks
                    st.rerun()
                else:
                    st.error("Could not fetch your Spotify data. Make sure you have listening history.")
        
        else:
            # Show results
            profile = st.session_state.user_profile
            top_artists = st.session_state.top_artists
            
            # Load model and predict
            if st.session_state.predicted_mbti is None:
                with st.spinner("🤖 Analyzing your music personality..."):
                    model, scaler, feature_cols = load_model_and_scaler()
                    percentages, mbti_type = predict_mbti(model, scaler, feature_cols, profile)
                    st.session_state.predicted_mbti = mbti_type
                    st.session_state.dimensions = percentages
                    st.session_state.model_loaded = True
                    st.rerun()
            
            else:
                # Display results
                percentages = st.session_state.dimensions
                mbti_type = st.session_state.predicted_mbti
                
                # Progress bars for dimensions
                st.subheader(f"🎯 Your Music Personality: **{mbti_type}**")
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### E vs I")
                    e = percentages[0]
                    st.progress(e / 100)
                    st.caption(f"Extraversion (E): {e:.0f}%")
                    st.caption(f"Introversion (I): {100-e:.0f}%")
                    
                    st.markdown("#### N vs S")
                    n = percentages[1]
                    st.progress(n / 100)
                    st.caption(f"Intuition (N): {n:.0f}%")
                    st.caption(f"Sensing (S): {100-n:.0f}%")
                
                with col2:
                    st.markdown("#### T vs F")
                    t = percentages[2]
                    st.progress(t / 100)
                    st.caption(f"Thinking (T): {t:.0f}%")
                    st.caption(f"Feeling (F): {100-t:.0f}%")
                    
                    st.markdown("#### J vs P")
                    j = percentages[3]
                    st.progress(j / 100)
                    st.caption(f"Judging (J): {j:.0f}%")
                    st.caption(f"Perceiving (P): {100-j:.0f}%")
                
                st.markdown("---")
                
                # Gemini description
                st.subheader("🤖 Gemini Says")
                with st.spinner("Crafting your personality description..."):
                    description = generate_description(mbti_type, top_artists)
                    st.info(description)
                
                st.markdown("---")
                
                # Top artists
                st.subheader("🎤 Your Top Artists")
                artist_cols = st.columns(min(5, len(top_artists)))
                for i, artist in enumerate(top_artists[:5]):
                    with artist_cols[i]:
                        st.markdown(f"**{i+1}**")
                        st.caption(artist)
                
                # Share button
                st.markdown("---")
                if st.button("📤 Share Your Result"):
                    st.success("Copy your result and share with friends!")
                
                # Refresh button
                if st.button("🔄 Refresh Analysis", type="secondary"):
                    for key in ['user_profile', 'top_artists', 'predicted_mbti', 'dimensions', 'tracks']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

if __name__ == "__main__":
    main()