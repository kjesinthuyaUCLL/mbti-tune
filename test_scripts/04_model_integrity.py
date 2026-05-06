"""
Check which models are available and their compatibility
"""

import os
import sys
import torch
import joblib
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_available_models():
    """List all models and their properties"""
    print("\n" + "="*60)
    print("AVAILABLE MODELS CHECK")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    models_to_check = [
        "song_classifier.pt",
        "playlist_classifier_best.pth",
        "playlist_classifier.pth",
        "encoder_114k_weights.pth"
    ]
    
    for model_file in models_to_check:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n📁 {model_file}:")
        
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024  # KB
            print(f"  ✅ Exists ({size:.1f} KB)")
            
            try:
                model = torch.load(model_path, map_location='cpu')
                if isinstance(model, dict):
                    print(f"  📦 Type: State Dictionary")
                    print(f"  🔑 Keys: {len(model.keys())} total")
                    
                    # Show first 5 keys
                    keys = list(model.keys())[:5]
                    print(f"  📝 First keys: {keys}")
                    
                    # Check input/output dimensions
                    for key in model.keys():
                        if 'weight' in key and 'network.0.weight' in key:
                            print(f"  ➡️ Input dim: {model[key].shape[1]}")
                        if 'weight' in key and 'network.11.weight' in key:
                            print(f"  ⬅️ Output dim: {model[key].shape[0]}")
                            break
                elif hasattr(model, 'state_dict'):
                    print(f"  📦 Type: nn.Module")
                    print(f"  🏗️ Class: {model.__class__.__name__}")
                else:
                    print(f"  📦 Type: {type(model)}")
                    
            except Exception as e:
                print(f"  ❌ Error loading: {e}")
        else:
            print(f"  ❌ Not found")

def check_feature_compatibility():
    """Check if feature lists match model expectations"""
    print("\n" + "="*60)
    print("FEATURE COMPATIBILITY CHECK")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load feature list
    features_path = os.path.join(base_dir, "models", "pretrain_features.json")
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    print(f"\n📊 pretrain_features.json: {len(features)} features")
    
    # Check song classifier (expects 10 features)
    song_model_path = os.path.join(base_dir, "models", "song_classifier.pt")
    if os.path.exists(song_model_path):
        model = torch.load(song_model_path, map_location='cpu')
        if isinstance(model, dict):
            input_dim = None
            for key in model.keys():
                if 'network.0.weight' in key:
                    input_dim = model[key].shape[1]
                    break
            if input_dim:
                print(f"\n🎵 Song Classifier expects: {input_dim} features")
                if input_dim == 10:
                    print(f"  ✅ Compatible with first {input_dim} features")
                    print(f"  📝 First {input_dim} features: {features[:10]}")
                else:
                    print(f"  ⚠️ Mismatch! Expected {input_dim}, have {len(features)}")
    
    # Check scaler
    scaler_path = os.path.join(base_dir, "models", "pretrain_scaler.pkl")
    scaler = joblib.load(scaler_path)
    print(f"\n📊 Scaler expects: {scaler.mean_.shape[0]} features")
    print(f"  ✅ Scalers match: {scaler.mean_.shape[0] == len(features)}")

def main():
    print("\n" + "🔍 MODEL INTEGRITY CHECK".center(60))
    check_available_models()
    check_feature_compatibility()
    print("\n" + "="*60)

if __name__ == "__main__":
    main()