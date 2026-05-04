import torch
import numpy as np
import os
import joblib
import json
import pickle
from src.model import PlaylistClassifier, PretrainedEncoder

def load_model_and_scaler():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load feature list
    features_path = os.path.join(base_dir, "models", "pretrain_features.json")
    with open(features_path, "r") as f:
        feature_cols = json.load(f)
    input_dim = len(feature_cols)

    # Load scaler
    scaler_path = os.path.join(base_dir, "models", "pretrain_scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load encoder
    encoder = PretrainedEncoder(input_dim=input_dim)
    encoder_weights = os.path.join(base_dir, "models", "encoder_114k_weights.pth")
    state = torch.load(encoder_weights, map_location=device)
    encoder.layers.load_state_dict(state)
    encoder.to(device)
    encoder.eval()

    # Load playlist classifier
    classifier = PlaylistClassifier(encoder)
    classifier_weights = os.path.join(base_dir, "models", "playlist_classifier_best.pth")
    classifier.load_state_dict(torch.load(classifier_weights, map_location=device))
    classifier.to(device)
    classifier.eval()

    return classifier, scaler, device, feature_cols


def predict_mbti(features_vector, model, scaler, device):
    """Scale 1×42 feature vector and predict MBTI percentages."""
    scaled = scaler.transform(features_vector)
    x = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(x)).cpu().numpy()[0]

    return {
        "E": float(preds[0]),
        "N": float(preds[1]),
        "T": float(preds[2]),
        "J": float(preds[3])
    }


def get_mbti_type(pred):
    """Convert percentages to MBTI letters."""
    return (
        ("E" if pred["E"] > 0.5 else "I") +
        ("N" if pred["N"] > 0.5 else "S") +
        ("T" if pred["T"] > 0.5 else "F") +
        ("J" if pred["J"] > 0.5 else "P")
    )
