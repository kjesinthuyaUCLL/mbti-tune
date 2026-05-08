import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from src.groq_utils import generate_with_groq, is_groq_available

load_dotenv()

# Configure Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


def get_gemini_model():
    """Get available Gemini model"""
    try:
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        try:
            return genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
            return None


def generate_personality_breakdown(mbti_type, result, top_artists, summaries):
    """
    Generate personality analysis - tries Groq first (more reliable),
    then falls back to Gemini, then built-in fallback.
    """
    # Extract axis preferences safely
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

Write a 3-paragraph analysis with this structure:
1. <b>Musical Fingerprint</b> (2 sentences): What their artist choices and lyrics preferences reveal about their {mbti_type} personality
2. <b>The Playful Roast</b> (2 sentences): Lightheartedly call out their listening patterns with humor
3. <b>The Insight</b> (2 sentences): One actionable observation about how their music taste reflects their MBTI type

CRITICAL INSTRUCTIONS:
- DO NOT use ANY emojis.
- DO NOT use asterisks (**) for bolding. Use HTML <b> tags instead.
- Do not use markdown headers. Just write paragraphs separated by blank lines.
"""
    
    # Try Groq FIRST (more reliable, higher limits)
    if is_groq_available():
        print("📡 Using Groq API...")
        response = generate_with_groq(prompt)
        if response:
            return response
        print("Groq failed, trying Gemini...")
    
    # Try Gemini as fallback
    gemini_model = get_gemini_model()
    if gemini_model:
        print("📡 Trying Gemini API...")
        try:
            response = gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
    
    # Final fallback
    print("⚠️ All APIs failed, using fallback analysis")
    return _generate_fallback_analysis(mbti_type, result, top_artists, summaries)


# Rest of the helper functions remain the same...
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
    """Generate fallback analysis when both APIs fail"""
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
    
    return f"""<b>Your Musical Personality: {mbti_type}</b>

Based on your listening habits, our AI has identified you as an <b>{mbti_type}</b> - {personality_desc}.

<b>Your Listening Profile:</b><br>{stats_text.replace(chr(10), '<br>')}

<b>What This Means:</b>
People with your personality type tend to gravitate towards music that {music_desc}. Your top artists include {artists_text}, which aligns with these preferences.

<b>The Insight:</b>
Your music taste suggests you value {value_desc}. Consider exploring artists outside your comfort zone - you might discover new dimensions of your personality!"""


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