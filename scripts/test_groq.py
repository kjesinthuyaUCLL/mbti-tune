# scripts/test_groq.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.groq_utils import generate_with_groq, test_groq_connection

print("Testing Groq API...")
print("=" * 40)

# Test connection
if test_groq_connection():
    print("✅ Groq connection successful!")
    
    # Test actual generation
    response = generate_with_groq("Write a one-sentence music recommendation for an ENFJ personality.")
    if response:
        print(f"\n📝 Response: {response}")
    else:
        print("❌ Generation failed")
else:
    print("❌ Groq connection failed. Check your API key in .env file")