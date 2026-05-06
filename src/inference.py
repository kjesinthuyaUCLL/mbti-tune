import torch
import numpy as np
import json
import joblib
import os
from pathlib import Path
from src.model import MBTIClassifier


def load_model_and_scaler():
    """Load MBTI classifier, scaler, and metadata from processed directory"""
    # Get project root (parent of src directory)
    base_dir = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    features_path = base_dir / "data" / "processed" / "mbti_features.json"
    scaler_path = base_dir / "data" / "processed" / "mbti_scaler.pkl"
    model_path = base_dir / "data" / "processed" / "mbti_classifier.pth"
    idx_to_type_path = base_dir / "data" / "processed" / "idx_to_type.json"
    
    # Check if all files exist
    for path in [features_path, scaler_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Load feature columns
    with open(features_path, "r") as f:
        feature_cols = json.load(f)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load idx_to_type (try checkpoint first, then separate file)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'idx_to_type' in checkpoint:
        idx_to_type = checkpoint['idx_to_type']
        # Convert string keys to int if needed
        if all(isinstance(k, str) for k in idx_to_type.keys()):
            idx_to_type = {int(k): v for k, v in idx_to_type.items()}
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        # Fallback to separate JSON file
        if idx_to_type_path.exists():
            with open(idx_to_type_path, "r") as f:
                idx_to_type = json.load(f)
            idx_to_type = {int(k): v for k, v in idx_to_type.items()}
        else:
            # Hardcoded fallback (should match Notebook 3's order)
            idx_to_type = {
                0: "ENFJ", 1: "ENFP", 2: "ENTJ", 3: "ENTP",
                4: "ESFJ", 5: "ESFP", 6: "ESTJ", 7: "ESTP",
                8: "INFJ", 9: "INFP", 10: "INTJ", 11: "INTP",
                12: "ISFJ", 13: "ISFP", 14: "ISTJ", 15: "ISTP"
            }
    
    # Load model
    model = MBTIClassifier(input_dim=len(feature_cols))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model with {len(feature_cols)} features on {device}")
    print(f"✅ Loaded scaler with {scaler.mean_.shape[0]} features")
    
    return model, scaler, device, feature_cols, idx_to_type


def predict_mbti(features_vector, model, scaler, device, feature_cols, idx_to_type):
    """
    Predict MBTI type from 45-feature vector.
    
    Args:
        features_vector: numpy array of shape (1, 45) - already aggregated, NOT scaled
        model: MBTIClassifier instance
        scaler: StandardScaler fitted on 45 features
        device: torch device
        feature_cols: list of 45 feature names
        idx_to_type: dict mapping index to MBTI type string
    
    Returns:
        dict with MBTI prediction and percentages
    """
    # Ensure features_vector is 2D
    if features_vector.ndim == 1:
        features_vector = features_vector.reshape(1, -1)
    
    # Scale features
    scaled = scaler.transform(features_vector)
    x = torch.tensor(scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Ensure idx_to_type has int keys
    idx_to_type_int = {int(k): v for k, v in idx_to_type.items()}
    
    # Aggregate 16 classes into 4 MBTI axes
    lp = {
        "E": 0.0, "I": 0.0, "S": 0.0, "N": 0.0,
        "T": 0.0, "F": 0.0, "J": 0.0, "P": 0.0
    }
    
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
    
    # Build result
    result = {"mbti": "", "percentages": {}}
    
    axes = [
        (("E", "I"), 0),
        (("S", "N"), 1),
        (("T", "F"), 2),
        (("J", "P"), 3)
    ]
    
    for (letter1, letter2), pos in axes:
        if lp[letter1] >= lp[letter2]:
            dominant, percentage = letter1, lp[letter1]
        else:
            dominant, percentage = letter2, lp[letter2]
        
        result["percentages"][letter1] = float(lp[letter1])
        result["percentages"][letter2] = float(lp[letter2])
        result[f"{letter1}/{letter2}"] = (dominant, float(percentage))
        result["mbti"] += dominant
    
    # Add full class probabilities
    result["all_probs"] = {idx_to_type_int[i]: float(p) for i, p in enumerate(probs) if i in idx_to_type_int}
    
    return result