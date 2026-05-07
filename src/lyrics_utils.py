import requests
import time
import urllib.parse
import os
from dotenv import load_dotenv
from src.groq_utils import generate_with_groq, is_groq_available

load_dotenv()


def generate_lyrics_summary(track_name, artist_name, lyrics):
    """Generate summary using Groq (fast) or fallback"""
    if not lyrics or len(lyrics.strip()) < 50:
        return None
    
    if len(lyrics) > 3000:
        lyrics = lyrics[:3000]
    
    prompt = f"""Analyze these lyrics and provide ONLY a 2-sentence summary of the main themes and emotional tone.

Song: {track_name} by {artist_name}
Lyrics: {lyrics}

Summary:"""
    
    # Try Groq first (fast and reliable)
    if is_groq_available():
        response = generate_with_groq(prompt, max_retries=1, model_name="llama-3.1-8b-instant")
        if response:
            return response
    
    return None


def fetch_lyrics(track_name, artist_name):
    """Fetch lyrics from APIs, return lyrics string or None"""
    lyrics = None
    
    # Try LRCLIB first
    try:
        encoded_name = urllib.parse.quote(track_name)
        encoded_artist = urllib.parse.quote(artist_name)
        url = f"https://lrclib.net/api/search?track_name={encoded_name}&artist_name={encoded_artist}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0 and data[0].get('plainLyrics'):
                lyrics = data[0]['plainLyrics']
                print(f"✅ Lyrics found for {track_name}")
                return lyrics
    except Exception as e:
        print(f"LRCLIB error for {track_name}: {e}")
    
    # Try Lyrics.ovh as fallback
    if not lyrics:
        try:
            url = f"https://api.lyrics.ovh/v1/{urllib.parse.quote(artist_name)}/{urllib.parse.quote(track_name)}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('lyrics'):
                    lyrics = data['lyrics']
                    print(f"✅ Lyrics found for {track_name} via Lyrics.ovh")
                    return lyrics
        except Exception as e:
            print(f"Lyrics.ovh error for {track_name}: {e}")
    
    return None


def build_lyrics_context(tracks, needed=3):
    """
    Fetch lyrics from top tracks, searching through all until we find 'needed' songs with lyrics.
    Only show error if no lyrics found after checking all tracks.
    
    Args:
        tracks: List of (track_name, artist_name, album_art_url) tuples
        needed: Number of tracks with lyrics needed (default 3)
    
    Returns:
        List of summary strings
    """
    summaries = []
    
    # Search through all tracks until we find enough with lyrics
    for track_name, artist_name, _ in tracks:
        if len(summaries) >= needed:
            break
        
        print(f"🔍 Searching for lyrics: {track_name} by {artist_name}")
        
        # Fetch lyrics
        lyrics = fetch_lyrics(track_name, artist_name)
        
        if lyrics:
            summary = generate_lyrics_summary(track_name, artist_name, lyrics)
            if summary:
                summaries.append(f"**{track_name}**: {summary}")
            else:
                summaries.append(f"**{track_name}**: Lyrics found but analysis failed")
        else:
            print(f"❌ No lyrics found for {track_name}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Only show error if we searched all tracks and found nothing
    if not summaries:
        return ["No lyrics could be found for any of your top 20 tracks. The personality analysis is based solely on audio features."]
    
    return summaries