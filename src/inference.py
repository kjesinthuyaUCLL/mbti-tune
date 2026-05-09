import torch
import numpy as np
import json
import joblib
from pathlib import Path
from src.model import MBTIClassifier


def load_model_and_scaler():
    """Load MBTI classifier, scaler, and metadata from processed directory"""
    base_dir = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    features_path = base_dir / "data" / "processed" / "mbti_features.json"
    scaler_path = base_dir / "data" / "processed" / "mbti_scaler.pkl"
    model_path = base_dir / "data" / "processed" / "mbti_classifier.pth"
    
    for path in [features_path, scaler_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    with open(features_path, "r") as f:
        feature_cols = json.load(f)
    
    scaler = joblib.load(scaler_path)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            idx_to_type = checkpoint.get('idx_to_type')
        else:
            state_dict = checkpoint
            idx_to_type = None
    else:
        state_dict = checkpoint
        idx_to_type = None
    
    if idx_to_type is not None:
        if all(isinstance(k, str) for k in idx_to_type.keys()):
            idx_to_type = {int(k): v for k, v in idx_to_type.items()}
    else:
        idx_to_type = {
            0: "ENFJ", 1: "ENFP", 2: "ENTJ", 3: "ENTP",
            4: "ESFJ", 5: "ESFP", 6: "ESTJ", 7: "ESTP",
            8: "INFJ", 9: "INFP", 10: "INTJ", 11: "INTP",
            12: "ISFJ", 13: "ISFP", 14: "ISTJ", 15: "ISTP"
        }
    
    model = MBTIClassifier(input_dim=len(feature_cols))
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model with {len(feature_cols)} features on {device}")
    print(f"✅ Loaded scaler with {scaler.mean_.shape[0]} features")
    print(f"✅ MBTI types: {len(idx_to_type)}")
    
    return model, scaler, device, feature_cols, idx_to_type


def stabilize_features(features_vector, scaler, feature_cols, clip_range=(-3, 3)):
    """
    Apply stabilization to feature vector:
    1. Identify problematic features (near-zero variance in training)
    2. Clip extreme values
    3. Down-weight problematic features
    """
    
    stabilized = features_vector.copy()
    
    # Clip extreme values
    old_min, old_max = np.min(stabilized), np.max(stabilized)
    stabilized = np.clip(stabilized, clip_range[0], clip_range[1])
    
    # Down-weight problematic features
    small_std_threshold = 0.15
    downweighted_count = 0
    
    # Get scaler parameters
    if hasattr(scaler, 'scale_'):
        for i in range(len(feature_cols)):
            if i < len(scaler.scale_):
                if scaler.scale_[i] < small_std_threshold:
                    stabilized[i] = stabilized[i] * 0.3
                    downweighted_count += 1
    
    # Print diagnostic info
    if not hasattr(stabilize_features, '_printed'):
        print(f"🔧 Stabilization active: clipping to {clip_range}, down-weighting {downweighted_count} low-variance features")
        stabilize_features._printed = True
    
    return stabilized.astype(np.float32)


def predict_mbti(features_vector, model, scaler, device, feature_cols, idx_to_type, temperature=4.0):
    """
    Predict MBTI type with stabilization and temperature scaling.
    """
    if features_vector.ndim == 1:
        features_vector = features_vector.reshape(1, -1)
    
    if features_vector.shape[1] != len(feature_cols):
        raise ValueError(f"Expected {len(feature_cols)} features, got {features_vector.shape[1]}")
    
    # Apply stabilization
    raw_vector = features_vector[0].copy()
    stabilized_vector = stabilize_features(raw_vector, scaler, feature_cols, clip_range=(-3, 3))
    
    x = torch.tensor(stabilized_vector.reshape(1, -1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(x)
        # Apply temperature scaling
        logits = logits / temperature
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Add small smoothing to prevent extreme percentages
    smoothing = 0.005
    probs = (1 - smoothing) * probs + smoothing / len(probs)
    probs = probs / probs.sum()
    
    idx_to_type_int = {int(k): v for k, v in idx_to_type.items()}
    
    # Aggregate 16 classes into 4 MBTI axes
    lp = {"E": 0.0, "I": 0.0, "S": 0.0, "N": 0.0,
          "T": 0.0, "F": 0.0, "J": 0.0, "P": 0.0}
    
    for i, prob in enumerate(probs):
        if i not in idx_to_type_int:
            continue
        mbti_type = idx_to_type_int[i]
        if len(mbti_type) >= 1:
            lp[mbti_type[0]] += prob
        if len(mbti_type) >= 2:
            lp[mbti_type[1]] += prob
        if len(mbti_type) >= 3:
            lp[mbti_type[2]] += prob
        if len(mbti_type) >= 4:
            lp[mbti_type[3]] += prob
    
    # Normalize each axis
    for axis in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
        total = lp[axis[0]] + lp[axis[1]]
        if total > 0:
            lp[axis[0]] /= total
            lp[axis[1]] /= total
        else:
            lp[axis[0]] = 0.5
            lp[axis[1]] = 0.5
    
    # Build result
    result = {"mbti": "", "percentages": {}}
    
    axes = [(("E", "I"), 0), (("S", "N"), 1), (("T", "F"), 2), (("J", "P"), 3)]
    
    for (letter1, letter2), pos in axes:
        if lp[letter1] >= lp[letter2]:
            dominant, percentage = letter1, lp[letter1]
        else:
            dominant, percentage = letter2, lp[letter2]
        
        result["percentages"][letter1] = float(lp[letter1])
        result["percentages"][letter2] = float(lp[letter2])
        result[f"{letter1}/{letter2}"] = (dominant, float(percentage))
        result["mbti"] += dominant
    
    result["all_probs"] = {idx_to_type_int[i]: float(p) for i, p in enumerate(probs) if i in idx_to_type_int}
    
    return result