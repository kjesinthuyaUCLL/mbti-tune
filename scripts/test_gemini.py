# scripts/test_gemini_rest.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

data = {
    "contents": [{
        "parts": [{"text": "Say 'Hello from Gemini REST API!' in one sentence."}]
    }]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    text = result['candidates'][0]['content']['parts'][0]['text']
    print(f"✅ Success! {text}")
else:
    print(f"❌ Error {response.status_code}: {response.text}")