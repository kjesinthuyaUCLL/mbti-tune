# src/groq_utils.py
from groq import Groq
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_groq_client = None


def get_groq_client():
    """Get Groq client instance (singleton)"""
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        try:
            _groq_client = Groq(api_key=GROQ_API_KEY)
            print("✅ Groq client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Groq client: {e}")
            return None
    return _groq_client


def generate_with_groq(prompt, model_name="llama-3.3-70b-versatile", max_retries=2):
    """
    Generate text using Groq API
    
    Args:
        prompt: The prompt text
        model_name: Groq model (options: "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
        max_retries: Number of retries on failure
    
    Returns:
        Generated text or None if failed
    """
    client = get_groq_client()
    if not client:
        print("⚠️ Groq client not available")
        return None
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful music psychologist assistant. Keep responses concise and engaging."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800,
                top_p=0.9
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait_time = 2 ** (attempt + 1)
                print(f"Groq rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Groq API error: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
    
    return None


def generate_with_groq_short(prompt, max_retries=2):
    """Generate using faster model for simple tasks"""
    return generate_with_groq(prompt, model_name="llama-3.1-8b-instant", max_retries=max_retries)


def is_groq_available():
    """Check if Groq API is configured and working"""
    if not GROQ_API_KEY:
        return False
    client = get_groq_client()
    return client is not None


def test_groq_connection():
    """Test if Groq API is working"""
    try:
        result = generate_with_groq("Say 'OK' in one word.", max_retries=1)
        return result is not None and "OK" in str(result).upper()
    except:
        return False


# Quick test when module loads
if __name__ == "__main__":
    print("Testing Groq connection...")
    if test_groq_connection():
        print("✅ Groq API is working!")
    else:
        print("❌ Groq API test failed. Check your API key.")