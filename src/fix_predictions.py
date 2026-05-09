"""
Fix for extreme MBTI predictions
Adds feature clipping and variance-based feature stabilization
"""

import numpy as np
import joblib
from pathlib import Path

def load_and_fix_scaler(scaler_path, feature_cols):
    """
    Load scaler and identify problematic features (near-zero variance)
    Returns: scaler, problematic_feature_indices, feature_weights
    """
    scaler = joblib.load(scaler_path)
    
    # Identify features with very small standard deviation
    small_std_threshold = 0.15
    problematic_indices = []
    
    print("\n🔍 Analyzing scaler for problematic features:")
    for i, col in enumerate(feature_cols):
        if scaler.scale_[i] < small_std_threshold:
            problematic_indices.append(i)
            print(f"   ⚠️ {col:30} std={scaler.scale_[i]:.6f} (NEAR-ZERO VARIANCE)")
    
    print(f"\n📊 Found {len(problematic_indices)} problematic features out of {len(feature_cols)}")
    
    # Create feature weights (down-weight problematic features)
    feature_weights = np.ones(len(feature_cols))
    for i in problematic_indices:
        feature_weights[i] = 0.1  # Reduce influence of problematic features
    
    return scaler, problematic_indices, feature_weights


def stabilize_features(raw_vector, scaler, feature_weights=None, clip_range=(-3, 3)):
    """
    Apply stabilization to feature vector:
    1. Clip extreme values
    2. Down-weight problematic features
    """
    # Apply scaling
    scaled = (raw_vector - scaler.mean_) / scaler.scale_
    
    # Clip extreme values (prevents single feature domination)
    scaled = np.clip(scaled, clip_range[0], clip_range[1])
    
    # Apply feature weights if provided
    if feature_weights is not None:
        scaled = scaled * feature_weights
    
    return scaled.astype(np.float32)


def get_prediction_breakdown(probs, idx_to_type):
    """
    Get detailed prediction breakdown for debugging
    """
    result = {
        "top_3_types": [],
        "entropy": -np.sum(probs * np.log(probs + 1e-8)),
        "max_confidence": np.max(probs),
        "distribution": {}
    }
    
    top_3_idx = np.argsort(probs)[-3:][::-1]
    for i, idx in enumerate(top_3_idx):
        mbti = idx_to_type[idx]
        prob = probs[idx]
        result["top_3_types"].append((mbti, prob))
        result["distribution"][mbti] = prob
    
    return result