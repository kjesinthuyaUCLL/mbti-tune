import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def generate_personality_breakdown(mbti_type, percentages, top_artists, lyrics_context):
    """
    Uses Google Gemini API to generate a personalized, funny psychological breakdown
    based on the user's music taste and PyTorch MBTI predictions.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ Gemini API key not found. Please add GOOGLE_API_KEY to your .env file."
        
    genai.configure(api_key=api_key)
    
    # Use the latest stable model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an AI psychologist that predicts personality traits from music. 
    I just analyzed a user's Spotify listening habits using a PyTorch neural network.
    
    Here are the results:
    - Predicted MBTI Type: {mbti_type}
    - Specific Dimension Percentages: 
      Extraversion: {percentages['E']*100:.1f}%
      Intuition: {percentages['N']*100:.1f}%
      Thinking: {percentages['T']*100:.1f}%
      Judging: {percentages['J']*100:.1f}%
      
    - Their Top 3 Artists: {', '.join(top_artists)}
    
    - Here are some snippets of the lyrics they listen to most (from LRCLIB API):
    {lyrics_context}
    
    Write a fun, insightful, and slightly roasted psychological breakdown (about 3-4 paragraphs) explaining exactly WHY their music taste reflects this specific MBTI type and percentages. Mention their specific artists and quote a bit of the lyrics to prove your point. Keep it entertaining and modern!
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return "⚠️ Sorry, there was an error communicating with the Gemini API."
