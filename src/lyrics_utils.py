import requests
import time
import urllib.parse
import google.generativeai as genai


def build_lyrics_context(tracks, needed=3):
    """
    Fetch lyrics from LRCLIB API and summarize with Gemini.
    
    Args:
        tracks: List of (track_name, artist_name) tuples
        needed: Number of tracks to process
    
    Returns:
        List of summary strings
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    summaries = []
    
    for name, artist in tracks[:needed]:  # Only process up to needed
        if len(summaries) >= needed:
            break
            
        try:
            # URL encode parameters
            encoded_name = urllib.parse.quote(name)
            encoded_artist = urllib.parse.quote(artist)
            
            url = f"https://lrclib.net/api/search?track_name={encoded_name}&artist_name={encoded_artist}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0 and "plainLyrics" in data[0]:
                    lyrics = data[0]['plainLyrics']
                    # Truncate extremely long lyrics (Gemini has context limits)
                    if len(lyrics) > 3000:
                        lyrics = lyrics[:3000]
                    
                    prompt = f"""You are a music psychologist. Analyze these lyrics and provide ONLY a 2-sentence summary of the main themes and emotional tone.

Song: {name} by {artist}
Lyrics: {lyrics}

Summary:"""
                    
                    summary = model.generate_content(prompt).text
                    summaries.append(f"{name}: {summary}")
                else:
                    summaries.append(f"{name}: Lyrics not found in database")
            else:
                summaries.append(f"{name}: Could not fetch lyrics (HTTP {response.status_code})")
                
        except requests.exceptions.Timeout:
            summaries.append(f"{name}: Request timeout")
        except Exception as e:
            summaries.append(f"{name}: Error - {str(e)}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    return summaries