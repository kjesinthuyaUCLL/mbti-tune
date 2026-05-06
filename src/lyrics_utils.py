import requests
import time
import urllib.parse
import google.generativeai as genai


def generate_lyrics_summary(track_name, artist_name, lyrics, model):
    """Generate summary for a single track's lyrics"""
    if not lyrics or len(lyrics.strip()) < 50:
        return f"*Limited lyrics available* - Could not generate detailed theme analysis."
    
    # Truncate extremely long lyrics
    if len(lyrics) > 3000:
        lyrics = lyrics[:3000]
    
    prompt = f"""You are a music psychologist. Analyze these lyrics and provide ONLY a 2-sentence summary of the main themes and emotional tone.

Song: {track_name} by {artist_name}
Lyrics: {lyrics}

Summary:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini summary error for {track_name}: {e}")
        return f"*Could not analyze lyrics for {track_name}*"


def build_lyrics_context(tracks, needed=3):
    """
    Fetch lyrics from LRCLIB API and summarize with Gemini.
    Uses multiple fallback strategies when lyrics not found.
    
    Args:
        tracks: List of (track_name, artist_name) tuples
        needed: Number of tracks to process
    
    Returns:
        List of summary strings
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    summaries = []
    
    # Alternative lyric APIs (in case LRCLIB fails)
    alt_apis = [
        "https://api.lyrics.ovh/v1/{artist}/{title}",
    ]
    
    for track_name, artist_name in tracks[:needed]:
        if len(summaries) >= needed:
            break
        
        lyrics = None
        source = None
        
        # Try LRCLIB first
        try:
            encoded_name = urllib.parse.quote(track_name)
            encoded_artist = urllib.parse.quote(artist_name)
            
            url = f"https://lrclib.net/api/search?track_name={encoded_name}&artist_name={encoded_artist}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0 and "plainLyrics" in data[0]:
                    lyrics = data[0]['plainLyrics']
                    source = "LRCLIB"
                    print(f"✅ Lyrics found for {track_name} via LRCLIB")
        except Exception as e:
            print(f"LRCLIB error for {track_name}: {e}")
        
        # Try alternative API if LRCLIB failed
        if not lyrics:
            try:
                alt_url = alt_apis[0].format(artist=urllib.parse.quote(artist_name), title=urllib.parse.quote(track_name))
                response = requests.get(alt_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'lyrics' in data and data['lyrics']:
                        lyrics = data['lyrics']
                        source = "Lyrics.ovh"
                        print(f"✅ Lyrics found for {track_name} via Lyrics.ovh")
            except Exception as e:
                print(f"Alternative API error for {track_name}: {e}")
        
        # Generate summary or fallback message
        if lyrics:
            summary = generate_lyrics_summary(track_name, artist_name, lyrics, model)
            summaries.append(f"**{track_name}**: {summary}")
        else:
            # Fallback: Generate a personality-based guess about the song
            fallback_prompt = f"""Based on the song title "{track_name}" by {artist_name}, what might be the likely emotional tone or theme of this song? Give one sentence only, starting with "Likely theme:"."""
            try:
                response = model.generate_content(fallback_prompt)
                summaries.append(f"**{track_name}**: {response.text.strip()} (lyrics unavailable)")
            except:
                summaries.append(f"**{track_name}**: Lyrics not found in any database")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # If no summaries at all, return a default message
    if not summaries:
        return ["No lyrics could be fetched for your top tracks. The personality analysis is based solely on audio features."]
    
    return summaries