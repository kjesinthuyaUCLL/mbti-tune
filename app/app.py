import os
import sys
import time
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import matplotlib.pyplot as plt

# Must be the first Streamlit command
st.set_page_config(page_title="MBTI Tune", page_icon="🎵", layout="wide")

# Custom CSS Styling
st.markdown("""
<style>
.gradient-container {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    position: relative;
}

.gradient-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 20px;
    padding: 2px;
    background: linear-gradient(135deg, #1DB954, #8A2BE2);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

/* Center the title and subtitle */
.title-gradient {
    text-align: center !important;
    font-size: 3rem !important;
    font-weight: bold !important;
    background: linear-gradient(135deg, #1DB954, #8A2BE2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
}

.subtitle {
    text-align: center !important;
    font-size: 1.1rem !important;
    color: #aaa !important;
    margin-bottom: 3rem !important;
}

/* Gradient green button - centered and shorter */
.stButton {
    display: flex !important;
    justify-content: center !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1DB954, #0d8c3f) !important;
    color: white !important;
    border: none !important;
    padding: 12px 28px !important;
    font-size: 1.1rem !important;
    font-weight: bold !important;
    border-radius: 30px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    width: auto !important;
    min-width: 200px !important;
    max-width: 280px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3) !important;
    background: linear-gradient(135deg, #1ed760, #0d8c3f) !important;
}

/* MBTI Type Styling */
.mbti-highlight {
    font-size: 3.5rem !important;
    font-weight: bold !important;
    text-align: center !important;
    background: linear-gradient(135deg, #1DB954, #8A2BE2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 20px 0;
}

/* MBTI percentage labels */
.dominant-label {
    font-size: 1.2rem !important;
    font-weight: 500 !important;
}

/* Audio Feature Metrics */
[data-testid="stMetricLabel"] {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: bold !important;
}

/* Progress bar styling */
.stProgress > div > div {
    background-color: #1DB954 !important;
}

/* Track styling */
.track-number {
    font-size: 0.8rem;
    color: #1DB954;
    margin-top: 0.5rem;
    font-weight: bold;
}

.track-title {
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0.3rem 0;
}

.track-artist {
    font-size: 0.8rem;
    color: #aaa;
}

/* Sidebar styling */
.css-1d391kg, .css-12oz5g7 {
    background: rgba(0, 0, 0, 0.5);
}

/* Spotify login button */
.spotify-login-btn {
    cursor: pointer;
    background: linear-gradient(135deg, #1DB954, #0d8c3f);
    color: white;
    border: none;
    padding: 12px 28px;
    border-radius: 30px;
    font-weight: bold;
    width: auto;
    min-width: 200px;
    transition: all 0.3s ease;
    display: inline-block;
    text-align: center;
    text-decoration: none;
}

.spotify-login-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
    background: linear-gradient(135deg, #1ed760, #0d8c3f);
}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure src is in path for custom imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.spotify_utils import get_spotify_oauth, fetch_user_data, AUDIO_FEATURES
from src.lyrics_utils import build_lyrics_context
from src.inference import load_model_and_scaler, predict_mbti
from src.gemini_utils import generate_personality_breakdown

# Title and Subtitle
st.markdown('<div class="title-gradient">🎵 MBTI Tune</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover your psychological type through your Spotify listening habits</div>', unsafe_allow_html=True)

# Load AI assets
@st.cache_resource
def load_assets():
    """Load model and scaler with caching"""
    try:
        return load_model_and_scaler()
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info("Please ensure all models are trained and saved in `data/processed/`")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

# Load models
model, scaler, device, feature_cols, idx_to_type = load_assets()

# Stop if models failed to load
if model is None:
    st.stop()

# Spotify OAuth Setup
oauth = get_spotify_oauth()

# Handle OAuth redirect
if 'code' in st.query_params:
    code = st.query_params['code']
    try:
        token_info = oauth.get_access_token(code)
        st.session_state['token_info'] = token_info
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Authentication failed: {e}")

# Check for token in session
token_info = st.session_state.get('token_info', None)

# If we have a token, check if it's expired and refresh
if token_info:
    expires_at = token_info.get('expires_at', 0)
    if expires_at < time.time():
        st.info("🔄 Refreshing Spotify connection...")
        try:
            refresh_token = token_info.get('refresh_token')
            if refresh_token:
                new_token = oauth.refresh_access_token(refresh_token)
                st.session_state['token_info'] = new_token
                token_info = new_token
                st.rerun()
            else:
                st.warning("Session expired. Please log in again.")
                st.session_state.clear()
                token_info = None
        except Exception as e:
            st.warning(f"Could not refresh token: {e}")
            st.session_state.clear()
            token_info = None

# Helper function to display audio features
def display_audio_features(tracks_data):
    """Display audio features in a nice format"""
    if not tracks_data:
        return
    
    st.markdown("#### 🎵 Audio Feature Analysis")
    st.caption("These audio features from your top tracks influenced your MBTI prediction")
    
    # Create DataFrame for display
    df = pd.DataFrame(tracks_data)
    
    # Normalize feature names for display
    display_names = {
        'danceability': '💃 Danceability',
        'energy': '⚡ Energy',
        'valence': '😊 Positivity (Valence)',
        'acousticness': '🎸 Acousticness',
        'instrumentalness': '🎹 Instrumentalness',
        'speechiness': '🗣️ Speechiness',
        'loudness': '🔊 Loudness (dB)',
        'tempo': '⏱️ Tempo (BPM)',
        'liveness': '🎤 Liveness'
    }
    
    # Calculate average features
    avg_features = {}
    for feat in AUDIO_FEATURES:
        if feat in df.columns:
            avg_features[feat] = df[feat].mean()
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Display metrics in columns
    for i, (feat, value) in enumerate(avg_features.items()):
        with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
            display_name = display_names.get(feat, feat.title())
            # Scale values for better display (loudness is negative, tempo is high)
            if feat == 'loudness':
                formatted_value = f"{value:.1f} dB"
                progress_value = (value + 60) / 60 if value < 0 else 0.5
            elif feat == 'tempo':
                formatted_value = f"{value:.0f} BPM"
                progress_value = min(value / 200, 1.0)
            else:
                formatted_value = f"{value:.2%}"
                progress_value = value
            
            st.metric(display_name, formatted_value)
            st.progress(min(progress_value, 1.0))
    
    # Add a bar chart of all features
    st.markdown("\n")
    st.markdown("#### 📊 Audio Profile Summary")
    
    # Prepare data for bar chart
    chart_data = []
    for feat in AUDIO_FEATURES:
        if feat in df.columns:
            value = df[feat].mean()
            if feat == 'loudness':
                # Normalize loudness (-60 to 0) to 0-1 scale for chart
                chart_value = (value + 60) / 60
                label = f"{display_names.get(feat, feat)} ({value:.1f} dB)"
            elif feat == 'tempo':
                chart_value = min(value / 200, 1.0)
                label = f"{display_names.get(feat, feat)} ({value:.0f} BPM)"
            else:
                chart_value = value
                label = f"{display_names.get(feat, feat)} ({value:.2f})"
            chart_data.append({'Feature': label, 'Value': chart_value})
    
    chart_df = pd.DataFrame(chart_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(chart_df['Feature'], chart_df['Value'], color='#1DB954')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Intensity (0-1 scale)', fontsize=12)
    ax.set_title('Your Audio Profile', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333')
    ax.spines['bottom'].set_color('#333')
    ax.tick_params(colors='white')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    st.pyplot(fig)

# App UI Logic
if not token_info:
    # Center the login card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="gradient-container" style="text-align:center;">', unsafe_allow_html=True)
        st.write("🎧 Connect your Spotify account to let our AI analyze your music taste and reveal your personality.")
        st.write("We analyze your top 20 tracks using a PyTorch neural network trained on 4,000+ playlists.")
        auth_url = oauth.get_authorize_url()
        
        # Center the login button
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            st.markdown(f'<a href="{auth_url}" target="_self"><button class="spotify-login-btn" style="width: 100%;">🔗 Log in with Spotify</button></a>', unsafe_allow_html=True)
        
        st.caption("We only access your top tracks - no playlist modification or sharing.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.sidebar.success("✅ Connected to Spotify")
    st.sidebar.markdown("---")
    st.sidebar.caption("🎵 Your data is processed locally and not stored.")
    st.sidebar.caption("ℹ️ Audio features use Spotify API")
    
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # Main analysis button - centered with limited width
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Start AI Analysis", use_container_width=True):
            
            # Step 1: Fetch Spotify data
            with st.spinner("🎵 Fetching your top tracks from Spotify..."):
                features_vector, tracks, top_artists, genres, tracks_data_raw = fetch_user_data(token_info, feature_cols)

            if features_vector is None or len(tracks) == 0:
                st.error("❌ Not enough Spotify data found. Please listen to more music and try again.")
            else:
                # Step 2: Display Top Tracks with Album Art
                with st.container():
                    st.markdown('<div class="gradient-container">', unsafe_allow_html=True)
                    st.subheader("🎧 Your Top 5 Tracks")
                    
                    cols = st.columns(5, gap="small")
                    
                    for i, track_data in enumerate(tracks[:5]):
                        if len(track_data) == 3:
                            name, artist, album_art_url = track_data
                        else:
                            name, artist = track_data
                            album_art_url = None
                        
                        with cols[i]:
                            if album_art_url:
                                st.image(album_art_url, use_container_width=True)
                            else:
                                colors = ["#1DB954", "#8A2BE2", "#FF6B6B", "#4ECDC4", "#45B7D1"]
                                color = colors[i % len(colors)]
                                st.markdown(f'<div style="background: linear-gradient(135deg, {color}, {color}88); border-radius: 8px; aspect-ratio: 1; display: flex; align-items: center; justify-content: center;"><span style="font-size: 2rem;">🎵</span></div>', unsafe_allow_html=True)
                            
                            st.markdown(f'<div class="track-number">#{i+1}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="track-title" title="{name}">{name[:25]}{"..." if len(name) > 25 else ""}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="track-artist" title="{artist}">{artist[:20]}{"..." if len(artist) > 20 else ""}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                # Step 3: MBTI Prediction
                st.markdown('<div class="gradient-container">', unsafe_allow_html=True)
                st.subheader("🧠 Neural Network Analysis")
                
                with st.spinner("🤖 Analyzing your musical fingerprint..."):
                    try:
                        # Use temperature scaling for more realistic confidence
                        temperature = 1
                        result = predict_mbti(features_vector, model, scaler, device, feature_cols, idx_to_type, temperature=temperature)
                        mbti_type = result["mbti"]

                        st.markdown(f'<div class="mbti-highlight">{mbti_type}</div>', unsafe_allow_html=True)
                        st.write("---")

                        # Display percentages for each axis
                        axes = ["E/I", "S/N", "T/F", "J/P"]
                        
                        col1, col2 = st.columns(2)
                        
                        axis_descriptions = {
                            "E/I": ("Extraversion", "Introversion"),
                            "S/N": ("Sensing", "Intuition"),
                            "T/F": ("Thinking", "Feeling"),
                            "J/P": ("Judging", "Perceiving")
                        }
                        
                        for i, axis in enumerate(axes):
                            if axis in result:
                                letter, prob = result[axis]
                                percentage = prob * 100
                                desc1, desc2 = axis_descriptions.get(axis, (letter, letter))
                                
                                with (col1 if i % 2 == 0 else col2):
                                    st.markdown(f'<span class="dominant-label">{letter} ({desc1}): {percentage:.1f}%</span>', unsafe_allow_html=True)
                                    st.progress(prob)
                                    st.caption(f"vs {desc2}")
                        
                        # Show top 3 most likely MBTI types
                        if "all_probs" in result:
                            sorted_probs = sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)
                            st.write("---")
                            st.caption("🎯 Other possible types you might relate to:")
                            cols = st.columns(3)
                            for i, (mbti, prob) in enumerate(sorted_probs[1:4]):
                                with cols[i]:
                                    st.caption(f"{mbti}: {prob*100:.1f}%")
                                    
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.stop()
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Step 4: Audio Features Analysis
                if tracks_data_raw:
                    with st.container():
                        st.markdown('<div class="gradient-container">', unsafe_allow_html=True)
                        display_audio_features(tracks_data_raw)
                        st.markdown('</div>', unsafe_allow_html=True)

                # Step 5: Lyrics Analysis
                st.markdown('<div class="gradient-container">', unsafe_allow_html=True)
                st.subheader("📝 Lyrics Theme Analysis")
                with st.spinner("🔍 Searching for lyrics in your top tracks..."):
                    summaries = build_lyrics_context(tracks[:20])
                    if summaries:
                        if len(summaries) == 1 and "No lyrics could be found" in summaries[0]:
                            st.warning(summaries[0])
                        else:
                            for i, summary in enumerate(summaries, start=1):
                                with st.expander(f"Track {i}", expanded=(i==1)):
                                    st.info(summary)
                    else:
                        st.warning("No lyrics could be found for any of your top tracks.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Step 6: AI Psychological Breakdown
                st.markdown('<div class="gradient-container">', unsafe_allow_html=True)
                st.subheader("✨ AI Psychological Breakdown")
                with st.spinner("🧠 Generating personality insights..."):
                    try:
                        full_analysis = generate_personality_breakdown(
                            mbti_type, result, top_artists, summaries
                        )
                        st.markdown(full_analysis)
                    except Exception as e:
                        st.warning(f"Could not generate analysis: {e}")
                        st.info(f"Based on your music, you appear to be an **{mbti_type}**. This type is characterized by {mbti_type[0] if mbti_type else 'introverted/extroverted'} tendencies.")
                st.markdown('</div>', unsafe_allow_html=True)