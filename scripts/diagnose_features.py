# scripts/diagnose_features.py
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import json
from src.inference import load_model_and_scaler

print("="*60)
print("🔍 DIAGNOSING FEATURE ISSUES")
print("="*60)

# Load model and scaler
model, scaler, device, feature_cols, idx_to_type = load_model_and_scaler()

print(f"\n📊 Model expects {len(feature_cols)} features")
print(f"   First 10 features: {feature_cols[:10]}")
print(f"   Last 10 features: {feature_cols[-10:]}")

# Check for transfer embedding features in the model
transfer_features = [f for f in feature_cols if 'transfer_emb' in f]
stats_features = [f for f in feature_cols if 'transfer_emb' not in f]
print(f"\n   Statistics features: {len(stats_features)}")
print(f"   Transfer embedding features: {len(transfer_features)}")

# Check scaler
print(f"\n📊 Scaler has {scaler.mean_.shape[0]} features")

# Create a test input with all zeros
zero_input = np.zeros((1, len(feature_cols))).astype(np.float32)
scaled_zero = scaler.transform(zero_input)

# Get prediction for zero input
import torch
from src.model import MBTIClassifier

model.eval()
with torch.no_grad():
    x = torch.tensor(scaled_zero, dtype=torch.float32)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    
print(f"\n📊 Prediction with ALL ZERO features:")
print(f"   Predicted class: {torch.argmax(probs).item()}")
print(f"   Confidence: {torch.max(probs).item():.2%}")
print(f"   Top 3 probabilities:")
top3 = torch.topk(probs[0], 3)
for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
    print(f"      {idx_to_type[idx.item()]}: {prob.item():.2%}")

# Create a test input with random values
random_input = np.random.randn(1, len(feature_cols)).astype(np.float32)
scaled_random = scaler.transform(random_input)

with torch.no_grad():
    x = torch.tensor(scaled_random, dtype=torch.float32)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)

print(f"\n📊 Prediction with RANDOM features:")
print(f"   Predicted class: {torch.argmax(probs).item()}")
print(f"   Confidence: {torch.max(probs).item():.2%}")
print(f"   Top 3 probabilities:")
top3 = torch.topk(probs[0], 3)
for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
    print(f"      {idx_to_type[idx.item()]}: {prob.item():.2%}")

print("\n" + "="*60)
print("💡 INTERPRETATION:")
print("="*60)

if torch.max(probs).item() > 0.95:
    print("⚠️ Model is OVERCONFIDENT - predictions >95% are suspicious")
    print("   Possible causes:")
    print("   1. Feature extraction in spotify_utils.py is wrong")
    print("   2. Scaler doesn't match model features")
    print("   3. Transfer embedding features are all zeros causing bias")
else:
    print("✅ Model confidence looks reasonable")