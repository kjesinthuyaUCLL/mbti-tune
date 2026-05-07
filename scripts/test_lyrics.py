import requests
import urllib.parse

print("=" * 50)
print("LYRICS API DIAGNOSTIC")
print("=" * 50)

# Test tracks from your list
test_tracks = [
    ("满天星辰不及你", "ycccc"),
    ("Crown", "EXO"),
    ("Teeth", "5 Seconds of Summer")
]

for track_name, artist_name in test_tracks:
    print(f"\nTesting: {track_name} by {artist_name}")
    print("-" * 40)
    
    # Try LRCLIB
    try:
        encoded_name = urllib.parse.quote(track_name)
        encoded_artist = urllib.parse.quote(artist_name)
        
        url = f"https://lrclib.net/api/search?track_name={encoded_name}&artist_name={encoded_artist}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0 and data[0].get('plainLyrics'):
                lyrics = data[0]['plainLyrics']
                print(f"   ✅ LRCLIB: Found lyrics! Length: {len(lyrics)} chars")
                print(f"   Preview: {lyrics[:100]}...")
            else:
                print(f"   ❌ LRCLIB: No lyrics found")
        else:
            print(f"   ❌ LRCLIB: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ LRCLIB error: {e}")
    
    # Try Lyrics.ovh as fallback
    try:
        url = f"https://api.lyrics.ovh/v1/{urllib.parse.quote(artist_name)}/{urllib.parse.quote(track_name)}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('lyrics'):
                lyrics = data['lyrics']
                print(f"   ✅ Lyrics.ovh: Found lyrics! Length: {len(lyrics)} chars")
            else:
                print(f"   ❌ Lyrics.ovh: No lyrics found")
        else:
            print(f"   ❌ Lyrics.ovh: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Lyrics.ovh error: {e}")

print("\n" + "=" * 50)
print("NOTE: Many songs don't have lyrics available in free APIs")
print("=" * 50)