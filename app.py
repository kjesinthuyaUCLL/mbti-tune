import os
import sys
import time
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="MBTI Tune", layout="centered", initial_sidebar_state="expanded")


st.markdown("""
<style>
/* Background and Base */
.stApp {
    background-color: #fcfafc;
    color: #4a4a4a;
    font-family: 'Inter', -apple-system, sans-serif;
    /* Soft decorative background blobs */
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(255, 182, 193, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(161, 140, 209, 0.15) 0%, transparent 40%);
    background-attachment: fixed;
}

/* Typography */
h1, h2, h3 {
    color: #333 !important;
    font-weight: 800 !important;
}

/* Main Title Area */
.header-container {
    text-align: center;
    padding: 3rem 0 2rem 0;
}

.main-title {
    font-size: 4rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px !important;
    letter-spacing: -1px;
}

.sub-title {
    font-size: 1.2rem !important;
    color: #888 !important;
    margin-top: 0.5rem !important;
}

/* Login Card Decorations */
.login-card {
    background: white;
    border-radius: 20px;
    padding: 3rem 2rem;
    box-shadow: 0 10px 40px rgba(255, 154, 158, 0.15);
    border: 1px solid #fff0f5;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 2rem;
}
.login-card::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 100px; height: 100px;
    background: linear-gradient(135deg, #ff9a9e, #fbc2eb);
    border-radius: 50%;
    opacity: 0.2;
}
.login-card::after {
    content: '';
    position: absolute;
    bottom: -30px; left: -30px;
    width: 80px; height: 80px;
    background: linear-gradient(135deg, #a18cd1, #fbc2eb);
    border-radius: 50%;
    opacity: 0.2;
}

/* Buttons */
.spotify-login-btn {
    background: #1DB954;
    color: white !important;
    border: none;
    padding: 16px 45px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 1.2rem;
    text-decoration: none !important;
    display: inline-block;
    box-shadow: 0 8px 25px rgba(29, 185, 84, 0.3);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-top: 1.5rem;
    z-index: 2;
    position: relative;
}
.spotify-login-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(29, 185, 84, 0.4);
    text-decoration: none !important;
}

/* Sections */
.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #4a4a4a;
    display: flex;
    align-items: center;
    margin: 3rem 0 1.5rem 0;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 24px;
    background: linear-gradient(to bottom, #ff9a9e, #fbc2eb);
    border-radius: 4px;
    margin-right: 12px;
}

/* Top Tracks grid */
.track-card {
    background: white;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    border: 1px solid #f8f8f8;
    text-align: center;
    transition: transform 0.2s;
}
.track-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(255, 154, 158, 0.15);
}
.track-img {
    width: 100%;
    border-radius: 8px;
    margin-bottom: 8px;
}
.track-title {
    font-weight: 800;
    font-size: 0.9rem;
    color: #333;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.track-artist {
    font-size: 0.8rem;
    color: #888;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* MBTI Beautiful Custom Bars (No Overlap!) */
.mbti-result-box {
    text-align: center;
    margin-bottom: 3rem;
}
.mbti-highlight {
    font-size: 6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a18cd1, #ff9a9e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    text-shadow: 2px 2px 20px rgba(255, 154, 158, 0.2);
}

.mbti-bar-container {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    border: 1px solid #fef0f2;
    margin-bottom: 12px;
}
.mbti-labels {
    display: flex;
    justify-content: space-between;
    font-weight: 800;
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 8px;
}
.mbti-track {
    width: 100%;
    height: 16px;
    background: #f0f0f0;
    border-radius: 10px;
    display: flex;
    overflow: hidden;
}

/* Custom Audio Badges (Fixes truncated text) */
.audio-badge {
    background: white;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    border: 1px solid #f8f8f8;
    margin-bottom: 10px;
}
.badge-value {
    font-size: 1.5rem;
    font-weight: 900;
    color: #ff9a9e;
}
.badge-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Lyrics Theme Cards */
.lyrics-theme-card {
    background: linear-gradient(145deg, #ffffff, #fdfbfd);
    border-left: 4px solid #fbc2eb;
    padding: 16px 40px 16px 20px; /* Added right padding to prevent quote overlap */
    border-radius: 0 12px 12px 0;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(251, 194, 235, 0.1);
    font-size: 0.95rem;
    color: #444;
    line-height: 1.5;
    position: relative;
}
.lyrics-theme-card::before {
    content: '❝';
    position: absolute;
    top: -10px; right: 10px;
    font-size: 3rem;
    color: rgba(251, 194, 235, 0.3);
    font-family: serif;
    z-index: 0;
}

.nlp-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 30px rgba(161, 140, 209, 0.1);
    border: 1px solid #f3f0f8;
    color: #444;
    line-height: 1.7;
}

/* Sidebar styling */
.css-1d391kg, .css-12oz5g7 {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.spotify_utils import get_spotify_oauth, fetch_user_data, AUDIO_FEATURES
from src.lyrics_utils import build_lyrics_context
from src.inference import load_model_and_scaler, predict_mbti
from src.gemini_utils import generate_personality_breakdown

@st.cache_resource
def load_assets():
    try:
        return load_model_and_scaler()
    except Exception as e:
        return None, None, None, None, None, None

model, poly, scaler, device, feature_cols, idx_to_type = load_assets()

if model is None:
    st.error("Model files not found.")
    st.stop()

oauth = get_spotify_oauth()

if 'code' in st.query_params:
    code = st.query_params['code']
    try:
        token_info = oauth.get_access_token(code)
        st.session_state['token_info'] = token_info
        st.query_params.clear()
        st.rerun()
    except Exception:
        pass

token_info = st.session_state.get('token_info', None)

if token_info:
    expires_at = token_info.get('expires_at', 0)
    if expires_at < time.time():
        try:
            refresh_token = token_info.get('refresh_token')
            if refresh_token:
                new_token = oauth.refresh_access_token(refresh_token)
                st.session_state['token_info'] = new_token
                token_info = new_token
                st.rerun()
            else:
                st.session_state.clear()
                token_info = None
        except Exception:
            st.session_state.clear()
            token_info = None


st.markdown("""
<div class="header-container">
    <div class="main-title">MBTI Tune</div>
    <div class="sub-title">Discover your psychological archetype through music</div>
</div>
""", unsafe_allow_html=True)

if not token_info:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-card">
            <h3 style="color:#555; margin-bottom: 10px;">Ready to explore?</h3>
            <p style="color:#888; margin-bottom: 20px;">Connect your Spotify account to extract your audio DNA and lyric semantics.</p>
        """, unsafe_allow_html=True)
        
        auth_url = oauth.get_authorize_url()
        # Wrapped in a center div to ensure explicit centering
        st.markdown(f'<div style="text-align: center;"><a href="{auth_url}" target="_self" class="spotify-login-btn">Connect Spotify</a></div></div>', unsafe_allow_html=True)

else:
    with st.sidebar:
        st.markdown("### Profile")
        st.success("Connected to Spotify")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    # Automatically fetch data
    with st.spinner("Extracting audio features and running neural models..."):
        features_vector, tracks, top_artists, genres, tracks_data_raw = fetch_user_data(token_info, feature_cols)

    if features_vector is None or tracks is None:
        st.error("Data loading error. Ensure your account is registered in the Spotify Developer Dashboard and you have listened to at least 3 tracks.")
        if st.button("Force Logout & Try Again"):
            st.session_state.clear()
            st.rerun()
    else:
        
        st.markdown('<div class="section-title">Your Neural Prediction</div>', unsafe_allow_html=True)
        
        result = predict_mbti(features_vector, model, poly, scaler, device, feature_cols, idx_to_type, temperature=4.0)
        mbti_type = result["mbti"]
        
        st.markdown(f'''
        <div class="mbti-result-box">
            <div class="mbti-highlight">{mbti_type}</div>
            <div class="mbti-subtext">Dominant Psychological Archetype</div>
        </div>
        ''', unsafe_allow_html=True)
        
        axes_data = [
            ("Introversion", "Extraversion", "I", "E", result.get("E/I", ("E", 0.5))),
            ("Sensing", "Intuition", "S", "N", result.get("S/N", ("S", 0.5))),
            ("Thinking", "Feeling", "T", "F", result.get("T/F", ("T", 0.5))),
            ("Judging", "Perceiving", "J", "P", result.get("J/P", ("J", 0.5)))
        ]
        
        for left_label, right_label, left_char, right_char, axis_res in axes_data:
            dom_letter, dom_prob = axis_res
            if dom_letter == right_char:
                right_val = dom_prob * 100
                left_val = 100 - right_val
            else:
                left_val = dom_prob * 100
                right_val = 100 - left_val
            
            left_color = "linear-gradient(90deg, #ff9a9e, #fbc2eb)" if left_val > 50 else "transparent"
            right_color = "linear-gradient(90deg, #a18cd1, #fbc2eb)" if right_val > 50 else "transparent"
            
            st.markdown(f'''
            <div class="mbti-bar-container">
                <div class="mbti-labels">
                    <span style="color:{'#ff9a9e' if left_val > 50 else '#888'}">{left_label} ({left_val:.0f}%)</span>
                    <span style="color:{'#a18cd1' if right_val > 50 else '#888'}">({right_val:.0f}%) {right_label}</span>
                </div>
                <div class="mbti-track">
                    <div style="width: {left_val}%; background: {left_color}; border-right: 2px solid white;"></div>
                    <div style="width: {right_val}%; background: {right_color};"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)


        st.markdown('<div class="section-title">Your Audio DNA</div>', unsafe_allow_html=True)
        
        col_radar, col_metrics = st.columns([1.5, 1])
        
        with col_radar:
            if tracks_data_raw:
                df = pd.DataFrame(tracks_data_raw)
                # Radar Chart
                avg_features = {}
                for feat in ['danceability', 'energy', 'valence', 'acousticness', 'liveness']:
                    if feat in df.columns:
                        avg_features[feat.capitalize()] = df[feat].mean()
                
                categories = list(avg_features.keys())
                values = list(avg_features.values())
                categories.append(categories[0])
                values.append(values[0])
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values, theta=categories, fill='toself',
                    fillcolor='rgba(255, 154, 158, 0.4)',
                    line=dict(color='#ff9a9e', width=3),
                    marker=dict(size=6, color='#a18cd1')
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor='#eee', gridcolor='#eee'),
                        angularaxis=dict(tickfont=dict(size=12, color='#666'), linecolor='#ddd', gridcolor='#ddd')
                    ),
                    showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=30, r=30, t=20, b=20), height=320
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_metrics:
            # Custom Badges so text never truncates
            st.write("")
            loudness = df.get('loudness', pd.Series([-5])).mean()
            tempo = df.get('tempo', pd.Series([120])).mean()
            speech = df.get('speechiness', pd.Series([0.1])).mean() * 100
            
            st.markdown(f'''
            <div class="audio-badge">
                <div class="badge-value">{loudness:.1f} dB</div>
                <div class="badge-label">Avg Loudness</div>
            </div>
            <div class="audio-badge">
                <div class="badge-value">{tempo:.0f} BPM</div>
                <div class="badge-label">Avg Tempo</div>
            </div>
            <div class="audio-badge">
                <div class="badge-value">{speech:.1f}%</div>
                <div class="badge-label">Speechiness</div>
            </div>
            ''', unsafe_allow_html=True)


        st.markdown('<div class="section-title">Source Tracks</div>', unsafe_allow_html=True)
        cols = st.columns(5, gap="small")
        for i, track_data in enumerate(tracks[:5]):
            name = track_data[0]
            artist = track_data[1]
            img = track_data[2] if len(track_data) == 3 else "https://via.placeholder.com/150"
            with cols[i]:
                st.markdown(f'''
                <div class="track-card">
                    <img src="{img}" class="track-img">
                    <div class="track-title">{name}</div>
                    <div class="track-artist">{artist}</div>
                </div>
                ''', unsafe_allow_html=True)


        st.markdown('<div class="section-title">Lyrical Semantics & Synthesis</div>', unsafe_allow_html=True)
        
        with st.spinner("Fetching lyrics and executing LLM synthesis..."):
            summaries = build_lyrics_context(tracks[:20])
            
        col_lyr, col_nlp = st.columns([1, 1.2])
        
        with col_lyr:
            st.markdown("<h4 style='color:#a18cd1;'>Extracted Themes</h4>", unsafe_allow_html=True)
            valid_summaries = [s for s in summaries if "no lyrics could be found" not in s.lower()]
            
            if not valid_summaries:
                st.info("No lyrics found to extract semantics.")
            else:
                for summary in valid_summaries:
                    st.markdown(f'<div class="lyrics-theme-card">{summary}</div>', unsafe_allow_html=True)

        with col_nlp:
            st.markdown("<h4 style='color:#a18cd1;'>Psychological Profile</h4>", unsafe_allow_html=True)
            with st.spinner("Generating synthesis..."):
                try:
                    full_analysis = generate_personality_breakdown(
                        mbti_type, result, top_artists, summaries
                    )
                    st.markdown(f'<div class="nlp-card">{full_analysis}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error("Failed to generate psychological synthesis.")