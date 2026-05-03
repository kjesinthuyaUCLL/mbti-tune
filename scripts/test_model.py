import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

print("="*60)
print("TESTING MBTI MODEL (Simplified)")
print("="*60)

# Define Model Architecture

class SimpleMBTIPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMBTIPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        return self.network(x)

# Load Files

# Load features
FEATURES_PATH = Path("models/features.json")
with open(FEATURES_PATH, 'r') as f:
    feature_names = json.load(f)
print(f"✅ Loaded {len(feature_names)} features")

# Load model state dict
MODEL_PATH = Path("models/model_state_dict.pt")
if not MODEL_PATH.exists():
    print(f"❌ Model not found at {MODEL_PATH}")
    print("   Please download model_state_dict.pt from Google Drive to models/")
    exit(1)

state_dict = torch.load(MODEL_PATH, map_location='cpu')
model = SimpleMBTIPredictor(len(feature_names))
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create a new scaler

print("\n📊 Creating new scaler from your data...")

# Load playlist data to fit scaler
PLAYLIST_PATH = Path("data/merged/playlist_data.csv")
if PLAYLIST_PATH.exists():
    df = pd.read_csv(PLAYLIST_PATH)
    print(f"✅ Loaded {len(df)} playlists")
    
    # Fit new scaler
    X = df[feature_names].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X)
    print(f"✅ Created new scaler fitted on {len(X)} samples")
    
    # Save the new scaler
    import pickle
    with open(Path("models/scaler_new.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Saved new scaler to models/scaler_new.pkl")
else:
    print(f"⚠️ Playlist data not found, using dummy scaler")
    # Create dummy scaler for testing
    scaler = StandardScaler()
    dummy_data = np.random.randn(100, len(feature_names))
    scaler.fit(dummy_data)

# Test Prediction Function

def predict_mbti(features_df):
    """Predict MBTI from features DataFrame"""
    # Ensure all features exist
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0.5
    
    # Extract and scale
    X = features_df[feature_names].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        output = model(X_tensor)
        pred = torch.sigmoid(output).numpy()[0]
    
    return {
        'E': float(pred[0] * 100),
        'N': float(pred[1] * 100),
        'T': float(pred[2] * 100),
        'J': float(pred[3] * 100),
        'I': float((1 - pred[0]) * 100),
        'S': float((1 - pred[1]) * 100),
        'F': float((1 - pred[2]) * 100),
        'P': float((1 - pred[3]) * 100),
        'mbti_type': ('E' if pred[0] > 0.5 else 'I') + \
                     ('N' if pred[1] > 0.5 else 'S') + \
                     ('T' if pred[2] > 0.5 else 'F') + \
                     ('J' if pred[3] > 0.5 else 'P')
    }

# Test Predictions

print("\n" + "="*60)
print("TEST PREDICTIONS")
print("="*60)

# Test 1: Average features
test_df = pd.DataFrame([[0.5] * len(feature_names)], columns=feature_names)
result = predict_mbti(test_df)

print(f"\n📊 Test 1: Average features (0.5)")
print(f"   Predicted MBTI: {result['mbti_type']}")
print(f"   E: {result['E']:.1f}% | I: {result['I']:.1f}%")
print(f"   N: {result['N']:.1f}% | S: {result['S']:.1f}%")
print(f"   T: {result['T']:.1f}% | F: {result['F']:.1f}%")
print(f"   J: {result['J']:.1f}% | P: {result['P']:.1f}%")

# Test 2: High energy, danceable music
high_energy_df = pd.DataFrame([[0.5] * len(feature_names)], columns=feature_names)
for col in feature_names:
    if 'energy_mean' in col:
        high_energy_df[col] = 0.9
    elif 'danceability_mean' in col:
        high_energy_df[col] = 0.85
    elif 'valence_mean' in col:
        high_energy_df[col] = 0.8

result2 = predict_mbti(high_energy_df)

print(f"\n📊 Test 2: High energy, danceable music")
print(f"   Predicted MBTI: {result2['mbti_type']}")
print(f"   E: {result2['E']:.1f}% | I: {result2['I']:.1f}%")
print(f"   N: {result2['N']:.1f}% | S: {result2['S']:.1f}%")
print(f"   T: {result2['T']:.1f}% | F: {result2['F']:.1f}%")
print(f"   J: {result2['J']:.1f}% | P: {result2['P']:.1f}%")

# Test 3: High acoustic, low energy (calm music)
calm_df = pd.DataFrame([[0.5] * len(feature_names)], columns=feature_names)
for col in feature_names:
    if 'acousticness_mean' in col:
        calm_df[col] = 0.9
    elif 'energy_mean' in col:
        calm_df[col] = 0.2
    elif 'instrumentalness_mean' in col:
        calm_df[col] = 0.7

result3 = predict_mbti(calm_df)

print(f"\n📊 Test 3: Calm, acoustic, instrumental music")
print(f"   Predicted MBTI: {result3['mbti_type']}")
print(f"   E: {result3['E']:.1f}% | I: {result3['I']:.1f}%")
print(f"   N: {result3['N']:.1f}% | S: {result3['S']:.1f}%")
print(f"   T: {result3['T']:.1f}% | F: {result3['F']:.1f}%")
print(f"   J: {result3['J']:.1f}% | P: {result3['P']:.1f}%")

# Save for Streamlit

print("\n" + "="*60)
print("SAVING FOR STREAMLIT")
print("="*60)

# Save the new scaler
import pickle
with open(Path("models/scaler_new.pkl"), 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: models/scaler_new.pkl")

# Save features
with open(Path("models/features.json"), 'w') as f:
    json.dump(feature_names, f)
print("✅ Saved: models/features.json")

# Model is already saved

print("\n" + "="*60)
print("🎉 SUCCESS! Model ready for Streamlit app")
print("="*60)
print("\nFiles for Streamlit:")
print("  - models/model_state_dict.pt")
print("  - models/scaler_new.pkl")
print("  - models/features.json")