# scripts/show_model_data.py
"""
Display all model data in terminal (no file creation)
Run: python scripts/show_model_data.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import joblib
from src.inference import load_model_and_scaler

print("="*80)
print("📊 MBTI TUNE - MODEL DATA OVERVIEW")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# 1. LOAD MODEL AND METADATA
# ============================================================================
print("\n📁 1. MODEL OVERVIEW")
print("-" * 50)

model, scaler, device, feature_cols, idx_to_type = load_model_and_scaler()

print(f"✅ Model Features: {len(feature_cols)}")
print(f"✅ MBTI Classes: {len(idx_to_type)}")
print(f"✅ Device: {device}")

# Load checkpoint for additional data
model_path = project_root / "data" / "processed" / "mbti_classifier.pth"
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# ============================================================================
# 2. FEATURE LIST
# ============================================================================
print("\n📋 2. FEATURE COMPOSITION")
print("-" * 50)

stat_features = [f for f in feature_cols if not f.startswith('transfer_emb')]
transfer_features = [f for f in feature_cols if f.startswith('transfer_emb')]
mean_features = [f for f in stat_features if '_mean' in f]
std_features = [f for f in stat_features if '_stdev' in f]
key_features = [f for f in stat_features if 'count' in f]

print(f"Total Features: {len(feature_cols)}")
print(f"  ├─ Statistical features: {len(stat_features)}")
print(f"  │   ├─ Means: {len(mean_features)}")
print(f"  │   ├─ Standard Deviations: {len(std_features)}")
print(f"  │   └─ Key/Mode counts: {len(key_features)}")
print(f"  └─ Transfer Learning embeddings: {len(transfer_features)}")

print("\n📝 First 10 features:")
for i, f in enumerate(feature_cols[:10]):
    print(f"   {i:2d}. {f}")

print("\n📝 Last 10 features:")
for i, f in enumerate(feature_cols[-10:]):
    print(f"   {i+len(feature_cols)-10:2d}. {f}")

# ============================================================================
# 3. MBTI TYPE MAPPING
# ============================================================================
print("\n🏷️ 3. MBTI TYPE MAPPING")
print("-" * 50)

for idx, mbti in idx_to_type.items():
    print(f"   {idx:2d} → {mbti}")

# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================
print("\n🏗️ 4. MODEL ARCHITECTURE")
print("-" * 50)

print(f"   Input Layer:     {len(feature_cols)} neurons")
print(f"   Hidden Layer 1:  64 neurons + BatchNorm + ReLU + Dropout(0.3)")
print(f"   Hidden Layer 2:  32 neurons + BatchNorm + ReLU + Dropout(0.3)")
print(f"   Hidden Layer 3:  16 neurons + BatchNorm + ReLU + Dropout(0.15)")
print(f"   Output Layer:    16 neurons (Softmax)")

# ============================================================================
# 5. MODEL WEIGHTS SUMMARY
# ============================================================================
print("\n🔢 5. MODEL WEIGHTS SUMMARY")
print("-" * 50)

state_dict = checkpoint.get('model_state_dict', {})

total_params = 0
for key, tensor in state_dict.items():
    if 'weight' in key and 'running' not in key:
        params = tensor.numel()
        total_params += params
        print(f"   {key}: shape {tuple(tensor.shape)} ({params:,} params)")

print(f"\n   Total trainable parameters: {total_params:,}")

# ============================================================================
# 6. OUTPUT LAYER ANALYSIS (Per-class weights)
# ============================================================================
print("\n🎯 6. OUTPUT LAYER ANALYSIS (Per MBTI Type)")
print("-" * 50)

# Find output layer
output_weights = None
output_bias = None
for key, tensor in state_dict.items():
    if 'net.12.weight' in key or 'classifier.4.weight' in key:
        output_weights = tensor
    if 'net.12.bias' in key or 'classifier.4.bias' in key:
        output_bias = tensor

if output_weights is not None:
    output_data = []
    for i in range(output_weights.shape[0]):
        mbti = idx_to_type.get(i, f"Class_{i}")
        weight_mag = torch.abs(output_weights[i]).sum().item()
        bias_val = output_bias[i].item() if output_bias is not None else 0
        output_data.append((i, mbti, weight_mag, bias_val))
    
    # Sort by weight magnitude
    output_data.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n   {'Idx':<4} {'MBTI':<6} {'Weight Magnitude':<18} {'Bias':<10}")
    print("   " + "-" * 45)
    for idx, mbti, mag, bias in output_data:
        arrow = "↑" if bias > 0 else "↓" if bias < 0 else "→"
        print(f"   {idx:<4} {mbti:<6} {mag:<18.4f} {bias:>+8.4f} {arrow}")

# ============================================================================
# 7. SCALER STATISTICS
# ============================================================================
print("\n📐 7. SCALER STATISTICS")
print("-" * 50)

print(f"   Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
print(f"   Scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
print(f"   Variance range: [{scaler.var_.min():.4f}, {scaler.var_.max():.4f}]")

# ============================================================================
# 8. MODEL PERFORMANCE
# ============================================================================
print("\n📈 8. MODEL PERFORMANCE")
print("-" * 50)

test_acc = checkpoint.get('test_accuracy', 0)
best_loss = checkpoint.get('best_val_loss', 0)
transfer_used = checkpoint.get('transfer_learning_used', False)

print(f"   Test Accuracy:      {test_acc:.2%} ({test_acc*100:.2f}%)")
print(f"   Best Val Loss:      {best_loss:.4f}")
print(f"   Transfer Learning:  {'✅ Yes (128 embeddings)' if transfer_used else '❌ No'}")
print(f"   Random Baseline:    6.25%")

# Compare to random
improvement = (test_acc - 0.0625) / 0.0625 * 100
print(f"   Improvement over random: +{improvement:.0f}%")

# ============================================================================
# 9. TOP FEATURES (Based on first layer weights)
# ============================================================================
print("\n🔍 9. TOP INFLUENTIAL FEATURES")
print("-" * 50)

# Get first layer weights
first_layer = None
for key, tensor in state_dict.items():
    if 'net.0.weight' in key:
        first_layer = tensor
        break

if first_layer is not None:
    # Calculate importance as mean absolute weight per input feature
    importance = torch.abs(first_layer).mean(dim=0).numpy()
    
    # Get top 10 features
    top_indices = np.argsort(importance)[-10:][::-1]
    print("\n   Top 10 Most Important Features:")
    for i, idx in enumerate(top_indices):
        if idx < len(feature_cols):
            print(f"   {i+1:2d}. {feature_cols[idx]:<30} (importance: {importance[idx]:.4f})")
    
    # Get bottom 5 features
    bottom_indices = np.argsort(importance)[:5]
    print("\n   Bottom 5 Least Important Features:")
    for i, idx in enumerate(bottom_indices):
        if idx < len(feature_cols):
            print(f"   {i+1:2d}. {feature_cols[idx]:<30} (importance: {importance[idx]:.4f})")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 MODEL SUMMARY")
print("="*80)

print(f"""
   Metric                    | Value
   --------------------------|-------------------------
   Model Type                | Neural Network Classifier
   Input Features            | {len(feature_cols)}
   Hidden Layers             | 3 (64→32→16)
   Output Classes            | 16 (All MBTI Types)
   Total Parameters          | {total_params:,}
   Transfer Learning         | {'Yes' if transfer_used else 'No'}
   Test Accuracy             | {test_acc:.2%}
   Best Validation Loss      | {best_loss:.4f}
   Improvement vs Random     | +{improvement:.0f}%
   Feature Composition       | {len(stat_features)} stats + {len(transfer_features)} embeddings
""")

print("="*80)
print("✅ MODEL DATA DISPLAY COMPLETE")
print("="*80)