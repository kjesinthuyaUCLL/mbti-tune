# scripts/debug_full_prediction_pipeline.py
"""
Full debugging of the prediction pipeline
Shows every step from raw features to final percentages
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import json
from src.inference import load_model_and_scaler
from src.spotify_utils import build_features_from_tracks, AUDIO_FEATURES

print("="*80)
print("🔍 FULL PREDICTION PIPELINE DEBUG")
print("="*80)

# 1. Load model
model, scaler, device, feature_cols, idx_to_type = load_model_and_scaler()

print(f"\n📊 Model expects {len(feature_cols)} features")
print(f"   Transfer features: {len([f for f in feature_cols if 'transfer_emb' in f])}")

# 2. Create sample tracks (simulating Spotify data)
print("\n" + "="*80)
print("📊 STEP 1: Raw Spotify Track Data")
print("="*80)

sample_tracks = [
    {
        'danceability': 0.85, 'energy': 0.90, 'valence': 0.80,
        'acousticness': 0.05, 'instrumentalness': 0.00, 'speechiness': 0.03,
        'loudness': -3.5, 'tempo': 128.0, 'liveness': 0.15,
        'key': 5, 'mode': 1
    },
    {
        'danceability': 0.75, 'energy': 0.85, 'valence': 0.70,
        'acousticness': 0.10, 'instrumentalness': 0.00, 'speechiness': 0.05,
        'loudness': -5.0, 'tempo': 120.0, 'liveness': 0.12,
        'key': 7, 'mode': 0
    },
    {
        'danceability': 0.65, 'energy': 0.70, 'valence': 0.50,
        'acousticness': 0.30, 'instrumentalness': 0.00, 'speechiness': 0.08,
        'loudness': -7.0, 'tempo': 110.0, 'liveness': 0.10,
        'key': 2, 'mode': 1
    }
]

print(f"Sample tracks: {len(sample_tracks)} songs")
for i, track in enumerate(sample_tracks):
    print(f"  Track {i+1}: dance={track['danceability']:.2f}, energy={track['energy']:.2f}, tempo={track['tempo']:.0f}")

# 3. Build features
print("\n" + "="*80)
print("📊 STEP 2: Feature Extraction (build_features_from_tracks)")
print("="*80)

features_dict = build_features_from_tracks(sample_tracks)
print(f"Generated {len(features_dict)} features")

# Show key features
print("\n📊 Key statistical features:")
key_stats = ['danceability_mean', 'energy_mean', 'valence_mean', 'tempo_mean', 'track_count']
for stat in key_stats:
    if stat in features_dict:
        print(f"   {stat}: {features_dict[stat]:.4f}")

# Show transfer embeddings (first 5)
transfer_keys = [k for k in features_dict.keys() if 'transfer_emb' in k]
if transfer_keys:
    print(f"\n📊 Transfer embeddings (first 5):")
    for i in range(min(5, len(transfer_keys))):
        print(f"   {transfer_keys[i]}: {features_dict[transfer_keys[i]]:.4f}")

# 4. Create feature vector in correct order
print("\n" + "="*80)
print("📊 STEP 3: Creating Feature Vector (Order matters!)")
print("="*80)

feature_vector = np.zeros((1, len(feature_cols)), dtype=np.float32)
for i, col in enumerate(feature_cols):
    feature_vector[0, i] = features_dict.get(col, 0.0)

print(f"Feature vector shape: {feature_vector.shape}")
print(f"Min: {feature_vector.min():.6f}, Max: {feature_vector.max():.6f}")
print(f"Mean: {feature_vector.mean():.6f}, Std: {feature_vector.std():.6f}")

# Check which features are non-zero
non_zero_count = np.count_nonzero(feature_vector)
print(f"Non-zero features: {non_zero_count}/{len(feature_cols)}")

# 5. Apply scaler
print("\n" + "="*80)
print("📊 STEP 4: Applying StandardScaler")
print("="*80)

scaled_vector = scaler.transform(feature_vector)
print(f"Scaled - Min: {scaled_vector.min():.6f}, Max: {scaled_vector.max():.6f}")
print(f"Scaled - Mean: {scaled_vector.mean():.6f}, Std: {scaled_vector.std():.6f}")

# 6. Get raw logits from model
print("\n" + "="*80)
print("📊 STEP 5: Model Forward Pass (Raw Logits)")
print("="*80)

x = torch.tensor(scaled_vector, dtype=torch.float32)
with torch.no_grad():
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Raw logits values:")
    for i in range(16):
        mbti = idx_to_type.get(i, f"Class_{i}")
        print(f"   {i:2d} {mbti:<6}: {logits[0][i].item():+.4f}")

# 7. Apply softmax (probabilities)
print("\n" + "="*80)
print("📊 STEP 6: Softmax (Class Probabilities)")
print("="*80)

probs = torch.softmax(logits, dim=1)[0].numpy()
print(f"Class probabilities (sum = {probs.sum():.4f}):")
sorted_indices = np.argsort(probs)[::-1]
for idx in sorted_indices[:5]:
    mbti = idx_to_type.get(idx, f"Class_{idx}")
    print(f"   {mbti}: {probs[idx]:.2%}")

# 8. Aggregate to axis probabilities
print("\n" + "="*80)
print("📊 STEP 7: Axis Aggregation")
print("="*80)

idx_to_type_int = {int(k): v for k, v in idx_to_type.items()}
lp = {"E": 0.0, "I": 0.0, "S": 0.0, "N": 0.0,
      "T": 0.0, "F": 0.0, "J": 0.0, "P": 0.0}

print("\n📊 How each MBTI type contributes to axes:")
for i, prob in enumerate(probs):
    if i not in idx_to_type_int:
        continue
    mbti_type = idx_to_type_int[i]
    contribution = f"  {mbti_type}: {prob:.2%} →"
    
    if len(mbti_type) >= 1:
        lp[mbti_type[0]] += prob
        contribution += f" {mbti_type[0]}+={prob:.2%}"
    if len(mbti_type) >= 2:
        lp[mbti_type[1]] += prob
        contribution += f", {mbti_type[1]}+={prob:.2%}"
    if len(mbti_type) >= 3:
        lp[mbti_type[2]] += prob
        contribution += f", {mbti_type[2]}+={prob:.2%}"
    if len(mbti_type) >= 4:
        lp[mbti_type[3]] += prob
        contribution += f", {mbti_type[3]}+={prob:.2%}"
    
    if prob > 0.01:  # Only show significant contributions
        print(contribution)

print("\n📊 Raw aggregated axis probabilities (before normalization):")
for letter in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']:
    print(f"   {letter}: {lp[letter]:.4f}")

# 9. Normalize axes
print("\n" + "="*80)
print("📊 STEP 8: Axis Normalization (Each axis sums to 100%)")
print("="*80)

axes = [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]
for axis in axes:
    total = lp[axis[0]] + lp[axis[1]]
    if total > 0:
        old_e = lp[axis[0]]
        old_i = lp[axis[1]]
        lp[axis[0]] /= total
        lp[axis[1]] /= total
        print(f"\n  {axis[0]}/{axis[1]} axis:")
        print(f"    Before: {axis[0]}={old_e:.4f}, {axis[1]}={old_i:.4f}, total={total:.4f}")
        print(f"    After:  {axis[0]}={lp[axis[0]]:.2%}, {axis[1]}={lp[axis[1]]:.2%}")

# 10. Final result
print("\n" + "="*80)
print("📊 STEP 9: Final Result")
print("="*80)

result = {"mbti": "", "percentages": {}}
for (letter1, letter2), _ in zip(axes, range(4)):
    if lp[letter1] >= lp[letter2]:
        dominant, percentage = letter1, lp[letter1]
    else:
        dominant, percentage = letter2, lp[letter2]
    
    result["percentages"][letter1] = float(lp[letter1])
    result["percentages"][letter2] = float(lp[letter2])
    result[f"{letter1}/{letter2}"] = (dominant, float(percentage))
    result["mbti"] += dominant

print(f"\n🎯 Predicted MBTI: {result['mbti']}")
print(f"\n📊 Final Axis Percentages:")
print(f"   E/I: {result['E/I'][0]} = {result['E/I'][1]*100:.1f}%")
print(f"   S/N: {result['S/N'][0]} = {result['S/N'][1]*100:.1f}%")
print(f"   T/F: {result['T/F'][0]} = {result['T/F'][1]*100:.1f}%")
print(f"   J/P: {result['J/P'][0]} = {result['J/P'][1]*100:.1f}%")

print("\n" + "="*80)
print("💡 INTERPRETATION:")
print("="*80)

# Check for issues
if max(probs) > 0.95:
    print("⚠️ Model is OVERCONFIDENT (max probability >95%)")
    print("   → Solution: Add temperature scaling (temperature=1.3 to 1.5)")
    
if len(set([result['E/I'][0], result['S/N'][0], result['T/F'][0], result['J/P'][0]])) < 4:
    print("⚠️ Model is producing the same letter across multiple axes")
    print("   → This may indicate feature extraction issues")

if all(v == 0 for v in [features_dict.get(f'transfer_emb_{i}', 1) for i in range(5)]):
    print("⚠️ Transfer embeddings are all ZERO")
    print("   → This causes the model to rely only on statistical features")
    print("   → May lead to biased predictions")

print("\n✅ To fix overconfidence, add temperature scaling to predict_mbti()")
print("   Example: logits = logits / 1.5 before softmax")
print("="*80)