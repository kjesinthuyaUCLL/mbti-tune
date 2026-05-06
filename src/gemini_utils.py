import google.generativeai as genai


def generate_personality_breakdown(mbti_type, result, top_artists, summaries):
    """
    Generate a fun, engaging personality analysis using Gemini.
    Includes fallback when API fails.
    
    Args:
        mbti_type: The predicted MBTI type (e.g., "INTJ")
        result: The prediction result dict from predict_mbti()
        top_artists: List of top artists
        summaries: List of lyrics summaries
    
    Returns:
        str: Gemini-generated analysis or fallback text
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Extract axis preferences safely
    axis_stats = []
    for axis in ["E/I", "S/N", "T/F", "J/P"]:
        if axis in result:
            letter, prob = result[axis]
            axis_stats.append(f"{axis}: {letter} ({prob*100:.1f}%)")
    
    stats_text = "\n".join(axis_stats) if axis_stats else "No axis data available"
    
    # Handle empty artists list
    artists_text = ", ".join(top_artists[:5]) if top_artists else "various artists"
    
    # Handle empty summaries
    summaries_text = "\n".join(summaries) if summaries else "No lyrics data available"
    
    # Get personality description based on MBTI type
    mbti_descriptions = {
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
    
    personality_desc = mbti_descriptions.get(mbti_type, "unique personality type")
    
    prompt = f"""You are "MBTI Tune", an AI music psychologist. Analyze this user's personality based on their Spotify listening habits.

MBTI Type: {mbti_type} - {personality_desc}
Axis Preferences:
{stats_text}

Top Artists: {artists_text}

Lyrics Themes:
{summaries_text}

Write a 3-paragraph analysis with this structure:
1. **Musical Fingerprint** (2 sentences): What their artist choices and lyrics preferences reveal about their {mbti_type} personality
2. **The Playful Roast** (2 sentences): Lightheartedly call out their listening patterns with humor
3. **The Insight** (2 sentences): One actionable observation about how their music taste reflects their MBTI type

Keep the tone fun, engaging, and slightly witty. Be conversational. Do not use markdown headers - just write paragraphs separated by blank lines.
"""
    
    try:
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text
        else:
            raise Exception("Empty response from Gemini")
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback analysis
        return f"""🎵 **Your Musical Personality: {mbti_type}**

Based on your listening habits, our AI has identified you as an **{mbti_type}** - {personality_desc}.

**Your Listening Profile:**
{stats_text}

**What This Means:**
People with your personality type tend to gravitate towards music that {get_mbti_music_description(mbti_type)} Your top artists include {artists_text}, which aligns with these preferences.

**The Insight:**
Your music taste suggests you value {get_mbti_value_description(mbti_type)}. Consider exploring artists outside your comfort zone - you might discover new dimensions of your personality!

*Note: Full AI analysis temporarily unavailable. This is a fallback analysis.*"""


def get_mbti_music_description(mbti_type):
    """Get music preference description for MBTI type fallback"""
    first_letter = mbti_type[0] if mbti_type else "E"
    second_letter = mbti_type[1] if len(mbti_type) > 1 else "N"
    
    if first_letter == "E":
        energy = "high-energy, upbeat tracks with strong beats"
    else:
        energy = "more introspective, chill, and atmospheric sounds"
    
    if second_letter == "N":
        creativity = "creative, complex, and thought-provoking lyrics"
    else:
        creativity = "grounded, relatable, and emotionally direct content"
    
    return f"{energy} with {creativity}"


def get_mbti_value_description(mbti_type):
    """Get value description for MBTI type fallback"""
    third_letter = mbti_type[2] if len(mbti_type) > 2 else "T"
    fourth_letter = mbti_type[3] if len(mbti_type) > 3 else "J"
    
    if third_letter == "T":
        thinking = "logic, analysis, and intellectual stimulation"
    else:
        thinking = "emotional connection and authentic expression"
    
    if fourth_letter == "J":
        structure = "structure, closure, and curated experiences"
    else:
        structure = "spontaneity, discovery, and open-ended exploration"
    
    return f"{thinking} while also embracing {structure}"