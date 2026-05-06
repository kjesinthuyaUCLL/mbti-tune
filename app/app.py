import os
import sys
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure src is in path for custom imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.spotify_utils import get_spotify_oauth, fetch_user_data
from src.lyrics_utils import build_lyrics_context
from src.inference import load_model_and_scaler, predict_mbti
from src.gemini_utils import generate_personality_breakdown

# Must be the first Streamlit command
st.set_page_config(page_title="MBTI Tune", page_icon="🎵", layout="wide")

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    [data-testid="stAppViewContainer"] { 
        background: linear-gradient(135deg, #0d0d12 0%, #1a1a2e 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #1DB954 0%, #8A2BE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0b0;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(29, 185, 84, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        font-weight: 600;
        border-radius: 30px;
        padding: 0.75rem 2rem;
        width: 100%;
        border: none;
        transition: transform 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #1ed760 0%, #1DB954 100%);
    }
    
    .mbti-highlight {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1DB954 0%, #8A2BE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
        margin: 10px 0;
    }
    
    .dominant-label {
        font-weight: 600;
        color: #1DB954;
        font-size: 1.1rem;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1DB954, #8A2BE2);
    }
    
    hr {
        margin: 1rem 0;
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #1DB954;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #1ed760;
    }
</style>
""", unsafe_allow_html=True)

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

token_info = st.session_state.get('token_info', None)
if not token_info:
    token_info = oauth.get_cached_token()

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
    
    # Check token expiry
    import time
    if token_info.get('expires_at', 0) < time.time():
        st.sidebar.warning("Session expired. Please log in again.")
        if st.sidebar.button("Log In Again"):
            st.session_state.clear()
            st.rerun()
    else:
        st.sidebar.markdown("---")
        st.sidebar.caption("Your data is processed locally and not stored.")
    
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # Main analysis button
    if st.button("🎯 Start AI Analysis", use_container_width=True):
        with st.spinner("🎵 Fetching your top tracks from Spotify..."):
            features_vector, tracks, top_artists, genres = fetch_user_data(token_info, feature_cols)

        if features_vector is None or len(tracks) == 0:
            st.error("❌ Not enough Spotify data found. Please listen to more music and try again.")
        else:
            # 1. Top Songs Display
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🎧 Your Top 5 Tracks")
                cols = st.columns(5)
                for i, (name, artist) in enumerate(tracks[:5]):
                    with cols[i]:
                        st.write(f"**{name[:25]}**" + ("..." if len(name) > 25 else ""))
                        st.caption(f"*{artist[:20]}*")
                st.markdown('</div>', unsafe_allow_html=True)

            col_lyrics, col_res = st.columns([1, 1])

            # 2. Lyrics Analysis Column
            with col_lyrics:
                st.markdown('<div class="glass-card" style="height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
                st.subheader("📝 Lyrics Theme Analysis")
                with st.spinner("🔍 Finding and analyzing lyrics..."):
                    summaries = build_lyrics_context(tracks[:3])
                    if summaries:
                        for i, summary in enumerate(summaries, start=1):
                            with st.expander(f"Track {i}", expanded=(i==1)):
                                st.info(summary)
                    else:
                        st.warning("Could not fetch lyrics for your top tracks.")
                st.markdown('</div>', unsafe_allow_html=True)

            # 3. MBTI Prediction Column
            with col_res:
                st.markdown('<div class="glass-card" style="height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
                st.subheader("🧠 Neural Network Analysis")
                
                with st.spinner("🤖 Running PyTorch classifier..."):
                    try:
                        result = predict_mbti(features_vector, model, scaler, device, feature_cols, idx_to_type)
                        mbti_type = result["mbti"]

                        st.markdown(f'<div class="mbti-highlight">{mbti_type}</div>', unsafe_allow_html=True)
                        st.write("---")

                        # Display percentages for each axis
                        axes = ["E/I", "S/N", "T/F", "J/P"]
                        for axis in axes:
                            if axis in result:
                                letter, prob = result[axis]
                                percentage = prob * 100
                                st.markdown(f'<span class="dominant-label">{letter}: {percentage:.1f}%</span>', unsafe_allow_html=True)
                                st.progress(prob)
                            else:
                                # Fallback to percentages dict
                                if axis == "E/I":
                                    st.markdown(f'E: {result["percentages"]["E"]*100:.1f}% | I: {result["percentages"]["I"]*100:.1f}%')
                                elif axis == "S/N":
                                    st.markdown(f'S: {result["percentages"]["S"]*100:.1f}% | N: {result["percentages"]["N"]*100:.1f}%')
                                elif axis == "T/F":
                                    st.markdown(f'T: {result["percentages"]["T"]*100:.1f}% | F: {result["percentages"]["F"]*100:.1f}%')
                                elif axis == "J/P":
                                    st.markdown(f'J: {result["percentages"]["J"]*100:.1f}% | P: {result["percentages"]["P"]*100:.1f}%')
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.stop()
                st.markdown('</div>', unsafe_allow_html=True)

            # 4. Gemini Breakdown
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