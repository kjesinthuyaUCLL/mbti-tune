import torch
import numpy as np
import json
import joblib
import os
from src.model import MBTIClassifier


# ============================================================
# 1. Load model, scaler, and feature list
# ============================================================

def load_model_and_scaler():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load feature list (45 features)
    features_path = os.path.join(base_dir, "data", "processed", "mbti_features.json")
    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    # Load scaler
    scaler_path = os.path.join(base_dir, "data", "processed", "mbti_scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Load classifier
    model_path = os.path.join(base_dir, "data", "processed", "mbti_classifier.pth")
    model = MBTIClassifier(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, scaler, device, feature_cols


# ============================================================
# 2. Predict softmax probabilities for 16 MBTI types
# ============================================================

def predict_softmax(features_vector, model, scaler, device):
    scaled = scaler.transform(features_vector)
    x = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


# ============================================================
# 3. Convert softmax → dominant letters (E/I, S/N, T/F, J/P)
# ============================================================

def compute_letter_probabilities(probs, idx_to_type):
    indices_E = [i for i, t in idx_to_type.items() if t[0] == "E"]
    indices_I = [i for i, t in idx_to_type.items() if t[0] == "I"]
    indices_S = [i for i, t in idx_to_type.items() if t[1] == "S"]
    indices_N = [i for i, t in idx_to_type.items() if t[1] == "N"]
    indices_T = [i for i, t in idx_to_type.items() if t[2] == "T"]
    indices_F = [i for i, t in idx_to_type.items() if t[2] == "F"]
    indices_J = [i for i, t in idx_to_type.items() if t[3] == "J"]
    indices_P = [i for i, t in idx_to_type.items() if t[3] == "P"]

    return {
        "E": float(probs[indices_E].sum()),
        "I": float(probs[indices_I].sum()),
        "S": float(probs[indices_S].sum()),
        "N": float(probs[indices_N].sum()),
        "T": float(probs[indices_T].sum()),
        "F": float(probs[indices_F].sum()),
        "J": float(probs[indices_J].sum()),
        "P": float(probs[indices_P].sum()),
    }


def get_mbti_type(lp):
    EI = "E" if lp["E"] >= lp["I"] else "I"
    SN = "S" if lp["S"] >= lp["N"] else "N"
    TF = "T" if lp["T"] >= lp["F"] else "F"
    JP = "J" if lp["J"] >= lp["P"] else "P"
    return EI + SN + TF + JP


# ============================================================
# 4. Full prediction pipeline
# ============================================================

def predict_mbti(feature_dict, model, scaler, device, feature_cols, idx_to_type):
    row = np.array([[feature_dict[col] for col in feature_cols]], dtype=np.float32)

    probs = predict_softmax(row, model, scaler, device)

    lp = compute_letter_probabilities(probs, idx_to_type)
    mbti = get_mbti_type(lp)

    return {
        "mbti": mbti,
        "E/I": round(100 * max(lp["E"], lp["I"]), 1),
        "S/N": round(100 * max(lp["S"], lp["N"]), 1),
        "T/F": round(100 * max(lp["T"], lp["F"]), 1),
        "J/P": round(100 * max(lp["J"], lp["P"]), 1),
        "softmax": {idx_to_type[i]: float(probs[i]) for i in range(16)}
    }
