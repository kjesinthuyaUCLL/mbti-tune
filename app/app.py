import streamlit as st
import sys
import os

# Ensure src is in path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.spotify_utils import get_spotify_oauth, fetch_user_data
from src.lyrics_utils import fetch_top_lyrics
from src.inference import load_model_and_scaler, predict_mbti, get_mbti_type
from src.gemini_utils import generate_personality_breakdown

# Must be the first Streamlit command
st.set_page_config(page_title="MBTI Tune", page_icon="🎵", layout="wide")

# Premium Custom CSS
st.markdown("""
<style>
    /* Dark Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* Target the main container safely */
    [data-testid="stAppViewContainer"] {
        background-color: #0d0d12;
    }
    
    /* Vibrant Gradient Text */
    .title-gradient {
        background: linear-gradient(90deg, #1DB954 0%, #8A2BE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0b0;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Login Button */
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 30px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(29, 185, 84, 0.4);
    }
    
    /* Progress bars styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8A2BE2 0%, #1DB954 100%);
    }
    
    /* Personality Type Highlight */
    .mbti-highlight {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #1DB954;
        text-shadow: 0 0 20px rgba(29, 185, 84, 0.5);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-gradient">MBTI Tune</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover your psychological type through your Spotify listening habits.</div>', unsafe_allow_html=True)

# 1. Spotify OAuth Setup
oauth = get_spotify_oauth()

# Handle OAuth redirect
if 'code' in st.query_params:
    code = st.query_params['code']
    token_info = oauth.get_access_token(code)
    st.session_state['token_info'] = token_info
    # Clear URL params to prevent re-triggering
    st.query_params.clear()

token_info = st.session_state.get('token_info', None)
if not token_info:
    token_info = oauth.get_cached_token()

if not token_info:
    # Not logged in
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.write("Connect your Spotify account to let our PyTorch AI analyze your audio features and lyrics.")
        auth_url = oauth.get_authorize_url()
        st.markdown(f'<a href="{auth_url}" target="_self"><button style="background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%); color: white; font-weight: 600; border: none; border-radius: 30px; padding: 0.75rem 2rem; cursor: pointer; margin-top: 1rem;">Log in with Spotify</button></a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Logged in
    st.success("Successfully connected to Spotify!")
    
    if st.button("Start AI Analysis"):
        with st.spinner("Fetching your top tracks and audio features..."):
            features_vector, tracks, top_artists = fetch_user_data(token_info)
            
        if features_vector is None:
            st.error("Not enough Spotify data found. Listen to more music!")
        else:
            # Layout
            col_res, col_lyrics = st.columns([1, 1])
            
            with col_lyrics:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🎵 LRCLIB Lyrics Extraction")
                with st.spinner("Extracting semantic meaning from your lyrics using LRCLIB API..."):
                    lyrics_context = fetch_top_lyrics(tracks, limit=3)
                    st.text_area("Analyzed Snippets", lyrics_context, height=200, disabled=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_res:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🧠 PyTorch Prediction")
                with st.spinner("Running Multi-Modal Deep Neural Network..."):
                    model, scaler, device = load_model_and_scaler()
                    percentages = predict_mbti(features_vector, model, scaler, device)
                    mbti_type = get_mbti_type(percentages)
                    
                    st.markdown(f'<div class="mbti-highlight">{mbti_type}</div>', unsafe_allow_html=True)
                    
                    st.write("---")
                    st.write(f"**Extraversion (E)**: {percentages['E']*100:.1f}%")
                    st.progress(percentages['E'])
                    
                    st.write(f"**Intuition (N)**: {percentages['N']*100:.1f}%")
                    st.progress(percentages['N'])
                    
                    st.write(f"**Thinking (T)**: {percentages['T']*100:.1f}%")
                    st.progress(percentages['T'])
                    
                    st.write(f"**Judging (J)**: {percentages['J']*100:.1f}%")
                    st.progress(percentages['J'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Gemini Section
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("✨ Gemini Psychological Breakdown")
            with st.spinner("Generating personalized insights..."):
                breakdown = generate_personality_breakdown(mbti_type, percentages, top_artists, lyrics_context)
                st.write(breakdown)
            st.markdown('</div>', unsafe_allow_html=True)
