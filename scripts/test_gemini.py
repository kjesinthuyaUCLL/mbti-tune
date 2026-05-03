import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL_NAME = "models/gemini-2.5-flash"

prompt = "Say hello in one sentence."
response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt
)

print(response.text)