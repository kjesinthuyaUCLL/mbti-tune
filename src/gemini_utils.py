import streamlit as st
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from src.groq_utils import generate_with_groq, is_groq_available

load_dotenv()

def get_secret(key):
    """Get secret from st.secrets (HF) or environment variable (local)"""
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError, AttributeError):
        return os.getenv(key)

API_KEY = get_secret("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def get_gemini_model():
    try:
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        try:
            return genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
            return None

def generate_personality_breakdown(mbti_type, result, top_artists, summaries):
    axis_stats = []
    for axis in ["E/I", "S/N", "T/F", "J/P"]:
        if axis in result:
            letter, prob = result[axis]
            axis_stats.append(f"{axis}: {letter} ({prob*100:.1f}%)")
    
    stats_text = "\n".join(axis_stats) if axis_stats else "No axis data available"
    artists_text = ", ".join(top_artists[:5]) if top_artists else "various artists"
    summaries_text = "\n".join(summaries) if summaries else "No lyrics data available"
    personality_desc = _get_mbti_description(mbti_type)
    
    prompt = f"""You are "MBTI Tune", an AI music psychologist. Analyze this user's personality based on their Spotify listening habits.

MBTI Type: {mbti_type} - {personality_desc}
Axis Preferences:
{stats_text}

Top Artists: {artists_text}

Lyrics Themes:
{summaries_text}

Write a structured HTML psychological profile that looks like a clean UI dashboard component. Use EXACTLY this structure:

<div style="margin-bottom: 20px;">
  <p>Based on your Spotify listening habits and lyrical themes, our neural networks have identified your primary psychological archetype as <b>{mbti_type}</b> ({personality_desc}).</p>
</div>

<h4 style="color: #9b59b6; font-size: 14px; text-transform: uppercase;">1. Your Cognitive Breakdown</h4>
<ul>
  <li><b>[Dominant Axis 1]</b>: (Mention the percentage here) - Short explanation of what this means generally.</li>
  <li><b>[Dominant Axis 2]</b>: (Mention the percentage here) - Short explanation of what this means generally.</li>
  <li><b>[Dominant Axis 3]</b>: (Mention the percentage here) - Short explanation of what this means generally.</li>
  <li><b>[Dominant Axis 4]</b>: (Mention the percentage here) - Short explanation of what this means generally.</li>
</ul>

<h4 style="color: #9b59b6; font-size: 14px; text-transform: uppercase; margin-top: 20px;">2. The Musical Connection</h4>
<p>Explain in 2-3 sentences how their specific top artists and specific lyrical themes perfectly align with the personality breakdown above.</p>

<h4 style="color: #9b59b6; font-size: 14px; text-transform: uppercase; margin-top: 20px;">3. The Playful Roast</h4>
<p>Lightheartedly call out their listening patterns with humor in 1-2 sentences.</p>

CRITICAL INSTRUCTIONS:
- DO NOT use ANY emojis.
- DO NOT use markdown format (like ** or #). ONLY use the exact HTML tags provided above.
- Make the text flow nicely but keep the structure intact.
"""
    
    if is_groq_available():
        print("📡 Using Groq API...")
        response = generate_with_groq(prompt)
        if response:
            return response
        print("Groq failed, trying Gemini...")
    
    gemini_model = get_gemini_model()
    if gemini_model:
        print("📡 Trying Gemini API...")
        try:
            response = gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
    
    print("⚠️ All APIs failed, using fallback analysis")
    return _generate_fallback_analysis(mbti_type, result, top_artists, summaries)

def _get_mbti_description(mbti_type):
    descriptions = {
        "INTJ": "The Architect - strategic, analytical, and innovative",
        "INTP": "The Logician - curious, theoretical, and abstract thinker",
        "ENTJ": "The Commander - bold, charismatic, and natural leader",
        "ENTP": "The Debater - creative, clever, and visionary",
        "INFJ": "The Advocate - creative, insightful, and principled",
        "INFP": "The Mediator - idealistic, empathetic, and curious",
        "ENFJ": "The Protagonist - charismatic, inspiring, and altruistic",
        "ENFP": "The Campaigner - enthusiastic, creative, and social",
        "ISTJ": "The Logistician - practical, fact-minded, and reliable",
        "ISFJ": "The Defender - dedicated, warm, and protective",
        "ESTJ": "The Executive - efficient, outgoing, and organized",
        "ESFJ": "The Consul - caring, social, and community-oriented",
        "ISTP": "The Virtuoso - bold, practical, and experimental",
        "ISFP": "The Adventurer - charming, artistic, and sensitive",
        "ESTP": "The Entrepreneur - energetic, perceptive, and spontaneous",
        "ESFP": "The Entertainer - spontaneous, energetic, and enthusiastic"
    }
    return descriptions.get(mbti_type, "unique personality type")

def _generate_fallback_analysis(mbti_type, result, top_artists, summaries):
    
    axis_stats = []
    for axis in ["E/I", "S/N", "T/F", "J/P"]:
        if axis in result:
            letter, prob = result[axis]
            axis_stats.append(f"{axis}: {letter} ({prob*100:.1f}%)")
    stats_text = "\n".join(axis_stats) if axis_stats else "No axis data available"
    
    personality_desc = _get_mbti_description(mbti_type)
    artists_text = ", ".join(top_artists[:5]) if top_artists else "various artists"
    
    music_desc = _get_mbti_music_description(mbti_type)
    value_desc = _get_mbti_value_description(mbti_type)
    
    return f"""<div style="margin-bottom: 20px;">
  <p>Based on your Spotify listening habits, our offline fallback model has identified your primary psychological archetype as <b>{mbti_type}</b> ({personality_desc}).</p>
</div>

<h4 style="color: #9b59b6; font-size: 14px; text-transform: uppercase;">1. Your Cognitive Breakdown</h4>
<ul>
  <li>{stats_text.replace(chr(10), '</li><li>')}</li>
</ul>

<h4 style="color: #9b59b6; font-size: 14px; text-transform: uppercase; margin-top: 20px;">2. The Musical Connection</h4>
<p>People with your personality type tend to gravitate towards music that {music_desc}. Your top artists include {artists_text}, which perfectly aligns with these preferences.</p>

<h4 style="color: #9b59b6; font-size: 14px; text-transform: uppercase; margin-top: 20px;">3. The Insight</h4>
<p>Your music taste suggests you deeply value {value_desc}. Consider exploring artists outside your comfort zone to discover new dimensions of your personality!</p>"""

def _get_mbti_music_description(mbti_type):
    if not mbti_type or len(mbti_type) < 2:
        return "matches your unique personality"
    first_letter = mbti_type[0]
    second_letter = mbti_type[1]
    if first_letter == "E":
        energy = "high-energy, upbeat tracks with strong beats"
    else:
        energy = "more introspective, chill, and atmospheric sounds"
    if second_letter == "N":
        creativity = "creative, complex, and thought-provoking lyrics"
    else:
        creativity = "grounded, relatable, and emotionally direct content"
    return f"{energy} with {creativity}"

def _get_mbti_value_description(mbti_type):
    if not mbti_type or len(mbti_type) < 4:
        return "authentic expression and musical discovery"
    third_letter = mbti_type[2]
    fourth_letter = mbti_type[3]
    if third_letter == "T":
        thinking = "logic, analysis, and intellectual stimulation"
    else:
        thinking = "emotional connection and authentic expression"
    if fourth_letter == "J":
        structure = "structure, closure, and curated experiences"
    else:
        structure = "spontaneity, discovery, and open-ended exploration"
    return f"{thinking} while also embracing {structure}"