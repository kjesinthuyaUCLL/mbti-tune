import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("SPOTIFY TOKEN DIAGNOSTIC")
print("=" * 50)

# Get client credentials
client_id = os.getenv('SPOTIPY_CLIENT_ID') or os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET') or os.getenv('SPOTIFY_CLIENT_SECRET')
redirect_uri = "http://127.0.0.1:8501"

print(f"Client ID: {'✅' if client_id else '❌'}")
print(f"Client Secret: {'✅' if client_secret else '❌'}")
print(f"Redirect URI: {redirect_uri}")

# Create OAuth with full permissions
sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-top-read user-read-recently-played user-read-private user-read-email",
    cache_path=None,
    show_dialog=True
)

# Check cached token
token_info = sp_oauth.get_cached_token()

if token_info:
    print(f"\n✅ Cached token found!")
    print(f"   Token type: {token_info.get('token_type')}")
    print(f"   Expires in: {token_info.get('expires_in')} seconds")
    print(f"   Scopes: {token_info.get('scope')}")
    
    # Test the token
    sp = spotipy.Spotify(auth=token_info['access_token'])
    
    try:
        # Test user profile (requires user-read-private)
        me = sp.current_user()
        print(f"\n✅ User profile accessible: {me.get('display_name', 'Unknown')}")
        
        # Test top tracks (requires user-top-read)
        top = sp.current_user_top_tracks(limit=5, time_range='medium_term')
        print(f"✅ Top tracks accessible: {len(top['items'])} tracks")
        
        # Test audio features
        if top['items']:
            track_id = top['items'][0]['id']
            features = sp.audio_features([track_id])
            print(f"✅ Audio features accessible for: {top['items'][0]['name']}")
            if features and features[0]:
                print(f"   Danceability: {features[0].get('danceability', 'N/A')}")
        
        print("\n✅ ALL SPOTIFY API TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        print("Token might have expired or missing permissions")
        
else:
    print("\n🔑 No cached token found. Need to authenticate.")
    auth_url = sp_oauth.get_authorize_url()
    print(f"\nOpen this URL in your browser:")
    print(auth_url)
    
    # Wait for user to paste the redirect URL
    print("\nAfter authorizing, you'll be redirected to a URL.")
    print("Copy the ENTIRE redirect URL and paste it below:")
    
    response_url = input("Paste the full redirect URL here: ")
    
    # Extract code from URL
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(response_url)
    code = parse_qs(parsed.query).get('code', [None])[0]
    
    if code:
        token_info = sp_oauth.get_access_token(code)
        print(f"\n✅ Got token! Scopes: {token_info.get('scope')}")
    else:
        print("\n❌ No code found in URL")