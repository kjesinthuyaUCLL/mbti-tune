"""
Test script to verify model loading and basic functionality
Place this in: test_scripts/01_check_model_loading.py
Run with: python test_scripts/01_check_model_loading.py
"""

import os
import sys
import pickle
import json
import torch
import numpy as np

# Add parent directory to path to access src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_pretrain_model():
    """Check the pretrain scaler and features"""
    print("\n" + "="*60)
    print("CHECKING PRETRAIN MODEL FILES")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check features.json
    features_path = os.path.join(base_dir, "models", "pretrain_features.json")
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
        print(f"✅ pretrain_features.json loaded: {len(features)} features")
        print(f"   First 5 features: {features[:5]}")
        print(f"   Last 5 features: {features[-5:]}")
    else:
        print(f"❌ pretrain_features.json not found at {features_path}")
        return None
    
    # Check scaler
    scaler_path = os.path.join(base_dir, "models", "pretrain_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ pretrain_scaler.pkl loaded")
        print(f"   Scaler type: {type(scaler)}")
        if hasattr(scaler, 'mean_'):
            print(f"   Mean shape: {scaler.mean_.shape}")
            print(f"   Scale shape: {scaler.scale_.shape}")
            print(f"   First 3 means: {scaler.mean_[:3]}")
    else:
        print(f"❌ pretrain_scaler.pkl not found at {scaler_path}")
        return None
    
    return features, scaler

def check_song_classifier():
    """Check the song classifier model"""
    print("\n" + "="*60)
    print("CHECKING SONG CLASSIFIER")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "song_classifier.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ song_classifier.pt not found at {model_path}")
        return None
    
    try:
        # Try loading with map_location for CPU
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"✅ song_classifier.pt loaded successfully")
        print(f"   Model type: {type(model)}")
        
        # Check if it's a state dict or full model
        if isinstance(model, dict):
            print("   Model is a state dictionary")
            print(f"   Keys: {list(model.keys())[:10]}...")
            
            # Check for specific layer keys
            layer_keys = [k for k in model.keys() if 'weight' in k or 'bias' in k]
            print(f"   Found {len(layer_keys)} weight/bias layers")
            
            # Try to infer input/output dimensions
            if 'network.0.weight' in model:
                input_dim = model['network.0.weight'].shape[1]
                print(f"   Input dimension: {input_dim}")
            if 'network.11.weight' in model:
                output_dim = model['network.11.weight'].shape[0]
                print(f"   Output dimension: {output_dim}")
            
        elif hasattr(model, 'state_dict'):
            print("   Model is a full torch.nn.Module")
            print(f"   Model architecture: {type(model).__name__}")
            if hasattr(model, 'forward'):
                print("   Model has forward method ✅")
        else:
            print(f"   Model format: {type(model)}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading song_classifier.pt: {e}")
        return None

def test_model_prediction(features, scaler, model):
    """Test making a prediction with dummy data"""
    print("\n" + "="*60)
    print("TESTING MODEL PREDICTION")
    print("="*60)
    
    if features is None or scaler is None:
        print("❌ Cannot test: features or scaler missing")
        return
    
    # Create dummy data (random values for 42 features)
    dummy_input = np.random.randn(1, len(features))
    
    # Scale using pretrain scaler
    try:
        dummy_scaled = scaler.transform(dummy_input)
        print(f"✅ Dummy data scaled: shape {dummy_scaled.shape}")
    except Exception as e:
        print(f"❌ Scaling failed: {e}")
        return
    
    # Convert to torch tensor
    dummy_tensor = torch.FloatTensor(dummy_scaled)
    
    # Try to make prediction
    try:
        if isinstance(model, dict):
            # Load state dict into a model class
            from src.model import MBTIPredictor
            # You might need to adjust input/output dimensions
            model_net = MBTIPredictor(input_dim=len(features), output_dim=4)
            model_net.load_state_dict(model, strict=False)
            model_net.eval()
            
            with torch.no_grad():
                output = model_net(dummy_tensor)
            print(f"✅ Model prediction successful!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output values: {output.numpy()}")
            
        elif hasattr(model, 'eval'):
            model.eval()
            with torch.no_grad():
                output = model(dummy_tensor)
            print(f"✅ Model prediction successful!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output values: {output.numpy()}")
        else:
            print("⚠️ Cannot make prediction: model format not recognized")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def main():
    print("\n" + "🔍 MODEL VALIDATION SUITE".center(60))
    
    # Check all models
    features, scaler = check_pretrain_model()
    model = check_song_classifier()
    
    # Test prediction
    if model is not None:
        test_model_prediction(features, scaler, model)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()