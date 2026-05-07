import os
import sys
import time
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import matplotlib.pyplot as plt

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

# Must be the first Streamlit command
st.set_page_config(page_title="MBTI Tune", page_icon="🎵", layout="wide")

# [Keep all your existing CSS styles - they are fine]

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
    
    # Create two columns for metrics
    col1, col2 = st.columns(2)
    
    # Display metrics in columns
    for i, (feat, value) in enumerate(avg_features.items()):
        with col1 if i % 2 == 0 else col2:
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
    ax.set_xlabel('Intensity (0-1 scale)')
    ax.set_title('Your Audio Profile')
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
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.write("🎧 Connect your Spotify account to let our AI analyze your music taste and reveal your personality.")
        st.write("We analyze your top 20 tracks using a PyTorch neural network trained on 4,000+ playlists.")
        auth_url = oauth.get_authorize_url()
        st.markdown(f'<a href="{auth_url}" target="_self"><button style="cursor:pointer; background: #1DB954; color:white; border:none; padding:12px 24px; border-radius:30px; font-weight:bold; width:100%;">🔗 Log in with Spotify</button></a>', unsafe_allow_html=True)
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

    # Main analysis button
    if st.button("🎯 Start AI Analysis", use_container_width=True):
        
        # Step 1: Fetch Spotify data
        with st.spinner("🎵 Fetching your top tracks from Spotify..."):
            features_vector, tracks, top_artists, genres, tracks_data_raw = fetch_user_data(token_info, feature_cols)

        if features_vector is None or len(tracks) == 0:
            st.error("❌ Not enough Spotify data found. Please listen to more music and try again.")
        else:
            # Step 2: Display Top Tracks with Album Art
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
                        
                        st.markdown(f'<div class="track-title" title="{name}">{name[:25]}{"..." if len(name) > 25 else ""}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="track-artist" title="{artist}">{artist[:20]}{"..." if len(artist) > 20 else ""}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="track-number">#{i+1}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Step 3: Audio Features Analysis (NEW - before MBTI prediction)
            if tracks_data_raw:
                with st.container():
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    display_audio_features(tracks_data_raw)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Step 4: MBTI Prediction
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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

            # Step 5: Lyrics Analysis
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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