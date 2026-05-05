import os
import sys
import json
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# Ensure src is in path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.spotify_utils import get_spotify_oauth, fetch_user_data
from src.lyrics_utils import build_lyrics_context
from src.inference import load_model_and_scaler, predict_mbti, get_mbti_type
from src.gemini_utils import generate_personality_breakdown

# Must be the first Streamlit command
st.set_page_config(page_title="MBTI Tune", page_icon="🎵", layout="wide")

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    [data-testid="stAppViewContainer"] { background-color: #0d0d12; }
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
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        font-weight: 600;
        border-radius: 30px;
        padding: 0.75rem 2rem;
    }
    .mbti-highlight {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #1DB954;
        text-shadow: 0 0 20px rgba(29,185,84,0.5);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-gradient">MBTI Tune</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover your psychological type through your Spotify listening habits.</div>', unsafe_allow_html=True)

# Load feature names (42 features)
try:
    with open(os.path.join("models", "pretrain_features.json"), 'r') as f:
        feature_cols = json.load(f)
except FileNotFoundError:
    st.error("⚠️ Missing 'pretrain_features.json' in models/.")
    st.stop()

# Spotify OAuth Setup
oauth = get_spotify_oauth()

# Handle OAuth redirect
if 'code' in st.query_params:
    code = st.query_params['code']
    token_info = oauth.get_access_token(code)
    st.session_state['token_info'] = token_info
    st.query_params.clear()

token_info = st.session_state.get('token_info', None)
if not token_info:
    token_info = oauth.get_cached_token()

if not token_info:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.write("Connect your Spotify account to let our PyTorch AI analyze your audio features and lyrics.")
        auth_url = oauth.get_authorize_url()
        st.markdown(f'<a href="{auth_url}" target="_self"><button>Log in with Spotify</button></a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.success("Successfully connected to Spotify!")

    if st.button("Start AI Analysis"):
        with st.spinner("Fetching your top tracks and audio features..."):
            features_vector, tracks, top_artists = fetch_user_data(token_info, feature_cols)

        if features_vector is None:
            st.error("Not enough Spotify data found. Listen to more music!")
        else:
            col_res, col_lyrics = st.columns([1, 1])

            # LYRICS SECTION
            with col_lyrics:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🎵 Lyrics Meaning Analysis (Translated + Summarized)")
                with st.spinner("Analyzing your top songs..."):
                    lyrics_context = build_lyrics_context(tracks, limit=3)
                    st.text_area("Song Summaries", lyrics_context, height=250, disabled=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # MBTI SECTION
            with col_res:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🧠 PyTorch Prediction")
                with st.spinner("Running Deep Neural Network..."):
                    model, scaler, device, _ = load_model_and_scaler()   # FIXED: do NOT overwrite feature_cols
                    percentages = predict_mbti(features_vector, model, scaler, device)
                    mbti_type = get_mbti_type(percentages)

                    st.markdown(f'<div class="mbti-highlight">{mbti_type}</div>', unsafe_allow_html=True)

                    st.write("---")
                    for dim in ["E", "N", "T", "J"]:
                        st.write(f"**{dim}**: {percentages[dim]*100:.1f}%")
                        st.progress(percentages[dim])
                st.markdown('</div>', unsafe_allow_html=True)

            # GEMINI SECTION
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("✨ Gemini Psychological Breakdown")
            with st.spinner("Generating personalized insights..."):
                breakdown = generate_personality_breakdown(
                    mbti_type, percentages, top_artists, lyrics_context
                )
                st.write(breakdown)
            st.markdown('</div>', unsafe_allow_html=True)
