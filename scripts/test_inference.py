import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.inference import load_model_and_scaler, predict_mbti

print("="*60)
print("🧪 Testing Inference Module")
print("="*60)

# Load model
model, scaler, device, feature_cols, idx_to_type = load_model_and_scaler()

print(f"\n📊 Model Info:")
print(f"   Feature count: {len(feature_cols)}")
print(f"   First 5 features: {feature_cols[:5]}")
print(f"   Last 5 features: {feature_cols[-5:]}")
print(f"   Device: {device}")

# Test with random input
print(f"\n🎲 Testing with random input...")
random_features = np.random.randn(1, len(feature_cols)).astype(np.float32)
result = predict_mbti(random_features, model, scaler, device, feature_cols, idx_to_type)

print(f"\n✅ Prediction successful!")
print(f"   Predicted MBTI: {result['mbti']}")
print(f"   E/I: {result['E/I'][0]} ({result['E/I'][1]*100:.1f}%)")
print(f"   S/N: {result['S/N'][0]} ({result['S/N'][1]*100:.1f}%)")
print(f"   T/F: {result['T/F'][0]} ({result['T/F'][1]*100:.1f}%)")
print(f"   J/P: {result['J/P'][0]} ({result['J/P'][1]*100:.1f}%)")

# Test with zeros (simulating no data)
print(f"\n🎲 Testing with zero features...")
zero_features = np.zeros((1, len(feature_cols))).astype(np.float32)
result2 = predict_mbti(zero_features, model, scaler, device, feature_cols, idx_to_type)

print(f"   Predicted MBTI (zero input): {result2['mbti']}")

print("\n✅ Test complete!")