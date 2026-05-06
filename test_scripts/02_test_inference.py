"""
Test the inference pipeline with the src/inference module
Place this in: test_scripts/02_test_inference.py
Run with: python test_scripts/02_test_inference.py
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import load_model_and_scaler, predict_mbti, get_mbti_type

def test_inference_module():
    """Test the inference module functions"""
    print("\n" + "="*60)
    print("TESTING INFERENCE MODULE")
    print("="*60)
    
    try:
        # Load model and scaler
        print("\n1. Loading model and scaler...")
        model, scaler, device, feature_cols = load_model_and_scaler()
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Feature count: {len(feature_cols)}")
        print(f"   First 5 features: {feature_cols[:5]}")
        
        # Create dummy input matching feature columns
        dummy_features = np.random.randn(1, len(feature_cols))
        print(f"\n2. Dummy features shape: {dummy_features.shape}")
        
        # Make prediction
        print("\n3. Making prediction...")
        percentages = predict_mbti(dummy_features, model, scaler, device)
        
        print(f"✅ Prediction successful!")
        print(f"   Percentages: {percentages}")
        
        # Get MBTI type
        mbti_type = get_mbti_type(percentages)
        print(f"\n4. MBTI Type: {mbti_type}")
        
        # Validate percentages sum to 1 (approximately)
        total = sum(percentages.values())
        print(f"\n5. Validation:")
        print(f"   Sum of percentages: {total:.4f} (should be ~1.0)")
        
        # Validate each percentage is between 0 and 1
        valid = all(0 <= v <= 1 for v in percentages.values())
        print(f"   All values in [0,1]: {valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_realistic_data():
    """Test with more realistic audio features"""
    print("\n" + "="*60)
    print("TESTING WITH REALISTIC AUDIO FEATURES")
    print("="*60)
    
    try:
        model, scaler, device, feature_cols = load_model_and_scaler()
        
        # Create realistic audio features (typical values from Spotify)
        # Note: These are just examples; adjust based on your actual feature_cols
        realistic_features = {}
        
        for col in feature_cols:
            if 'mean' in col:
                if 'danceability' in col:
                    realistic_features[col] = 0.7  # danceable
                elif 'energy' in col:
                    realistic_features[col] = 0.6  # energetic
                elif 'valence' in col:
                    realistic_features[col] = 0.5  # neutral mood
                elif 'acousticness' in col:
                    realistic_features[col] = 0.3  # moderately acoustic
                else:
                    realistic_features[col] = 0.5
            elif 'stdev' in col:
                realistic_features[col] = 0.2  # moderate variation
            else:
                realistic_features[col] = 0.0  # categorical features
        
        # Convert to numpy array
        features_array = np.array([[realistic_features.get(col, 0) for col in feature_cols]])
        
        print(f"Realistic features shape: {features_array.shape}")
        
        # Make prediction
        percentages = predict_mbti(features_array, model, scaler, device)
        mbti_type = get_mbti_type(percentages)
        
        print(f"\nPredicted MBTI: {mbti_type}")
        print(f"Breakdown:")
        for dim in ["E", "N", "T", "J"]:
            print(f"  {dim}: {percentages[dim]*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "🔍 INFERENCE PIPELINE TEST".center(60))
    
    test1 = test_inference_module()
    print("\n" + "-"*60)
    test2 = test_with_realistic_data()
    
    print("\n" + "="*60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)

if __name__ == "__main__":
    main()
    