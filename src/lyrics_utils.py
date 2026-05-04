import requests

def get_lyrics_lrclib(track_name, artist_name):
    """
    Fetch lyrics from LRCLIB API using track and artist name.
    LRCLIB does not require an API key and is open source.
    """
    base_url = "https://lrclib.net/api/search"
    params = {
        "track_name": track_name,
        "artist_name": artist_name
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            # We take the first match that has plainLyrics
            for track in data:
                if track.get('plainLyrics'):
                    return track['plainLyrics']
        return None
    except Exception as e:
        print(f"Error fetching lyrics for {track_name}: {e}")
        return None

def fetch_top_lyrics(tracks, limit=3):
    """
    Fetch lyrics for the top N tracks.
    tracks: list of tuples (track_name, artist_name)
    """
    lyrics_context = ""
    fetched_count = 0
    
    for track_name, artist_name in tracks:
        if fetched_count >= limit:
            break
            
        lyrics = get_lyrics_lrclib(track_name, artist_name)
        if lyrics:
            # We only take the first 500 characters of each song to avoid overloading Gemini
            snippet = lyrics[:500].replace('\\n', ' ') + "..."
            lyrics_context += f"\\nSong: {track_name} by {artist_name}\\nLyrics snippet: {snippet}\\n"
            fetched_count += 1
            
    return lyrics_context
