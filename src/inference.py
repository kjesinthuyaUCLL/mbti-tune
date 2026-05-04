import torch
import numpy as np
import os
import joblib
from src.model import MusicAutoencoder, MBTIPredictor

def load_model_and_scaler():
    """Load the trained PyTorch model and the StandardScaler"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Scaler
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    # fallback to scaler_new or others if scaler.pkl doesn't exist
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(base_dir, 'models', 'scaler_aggregated.pkl')
        
    scaler = joblib.load(scaler_path)
    
    # Reconstruct Model Architecture
    # input_dim is 45 based on features.json
    input_dim = 45 
    
    encoder = MusicAutoencoder(input_dim=input_dim).encoder
    model = MBTIPredictor(encoder=encoder, input_dim=input_dim)
    
    # Load weights
    model_path = os.path.join(base_dir, 'models', 'final_mbti_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'models', 'model_state_dict.pt')
        
    # Load state dict handling CPU/GPU mismatch
    # Trusted local checkpoint; PyTorch 2.6 defaults weights_only=True and can reject pickled objects.
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False),
        strict=False
    )
    model.to(device)
    model.eval()
    
    return model, scaler, device

def predict_mbti(features_vector, model, scaler, device):
    """
    Given a 1x45 numpy array of features, scale it and predict MBTI percentages.
    Returns dictionary with E, N, T, J percentages.
    """
    # Scale features
    scaled_features = scaler.transform(features_vector)
    
    # Convert to tensor
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor_features)
        # outputs are already passed through sigmoid in MBTIPredictor (or not?)
        # Let's check src/model.py. In MBTIPredictor forward: e = torch.sigmoid(...)
        # So outputs are already 0-1.
        percentages = outputs[0].cpu().numpy()
        
    return {
        'E': float(percentages[0]),
        'N': float(percentages[1]),
        'T': float(percentages[2]),
        'J': float(percentages[3])
    }

def get_mbti_type(predictions):
    """Convert percentages to 4-letter MBTI type"""
    mbti = ""
    mbti += "E" if predictions['E'] > 0.5 else "I"
    mbti += "N" if predictions['N'] > 0.5 else "S"
    mbti += "T" if predictions['T'] > 0.5 else "F"
    mbti += "J" if predictions['J'] > 0.5 else "P"
    return mbti
