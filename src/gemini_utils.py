import google.generativeai as genai


def generate_personality_breakdown(mbti_type, result, top_artists, summaries):
    """
    Generate a fun, engaging personality analysis using Gemini.
    
    Args:
        mbti_type: The predicted MBTI type (e.g., "INTJ")
        result: The prediction result dict from predict_mbti()
        top_artists: List of top artists
        summaries: List of lyrics summaries
    
    Returns:
        str: Gemini-generated analysis
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
    
    prompt = f"""You are "MBTI Tune", an AI music psychologist. Analyze this user's personality based on their Spotify listening habits.

MBTI Type: {mbti_type}
Axis Preferences:
{stats_text}

Top Artists: {artists_text}

Lyrics Themes:
{summaries_text}

Write a 3-paragraph analysis with this structure:
1. **Musical Fingerprint** (2 sentences): What their artist choices reveal about their personality
2. **The Roast** (2 sentences): Playfully call out their listening patterns with humor
3. **The Insight** (2 sentences): One actionable observation about how their music taste reflects their MBTI type

Keep the tone fun, engaging, and slightly witty. Use emojis sparingly (max 3 total). Do not use markdown headers - just write paragraphs.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🎵 *Analysis temporarily unavailable.* Based on your music, you appear to be an {mbti_type} type. Your listening preferences show a preference for {stats_text}. Try again later for a full analysis!"