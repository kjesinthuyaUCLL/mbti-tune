from pathlib import Path
import json
import pandas as pd
import pickle
import torch

print("="*70)
print("MBTI-TUNE PROJECT INTEGRITY CHECK")
print("="*70)

# ---------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root:     {PROJECT_ROOT}")
print(f"Model directory:  {MODEL_DIR}")
print(f"Data directory:   {DATA_DIR}")

# ---------------------------------------------------------
# Check model files
# ---------------------------------------------------------
print("\n--- MODEL FILES ---")
required_models = [
    "encoder_114k_weights.pth",
    "playlist_classifier_best.pth",
    "pretrain_features.json",
    "pretrain_scaler.pkl",
    "song_classifier.pt"
]

for m in required_models:
    path = MODEL_DIR / m
    print(f"{m:<30} {'FOUND' if path.exists() else 'MISSING'}")

# ---------------------------------------------------------
# Load pretrain features
# ---------------------------------------------------------
print("\n--- PRETRAIN FEATURE LIST ---")
pretrain_features_path = MODEL_DIR / "pretrain_features.json"

if pretrain_features_path.exists():
    with open(pretrain_features_path, "r") as f:
        pretrain_features = json.load(f)
    print(f"Loaded {len(pretrain_features)} features from pretrain_features.json")
else:
    print("pretrain_features.json NOT FOUND")
    pretrain_features = []

# ---------------------------------------------------------
# Inspect playlist_data.csv
# ---------------------------------------------------------
print("\n--- PLAYLIST DATASET (merged/playlist_data.csv) ---")
playlist_path = DATA_DIR / "merged" / "playlist_data.csv"

if playlist_path.exists():
    df_playlist = pd.read_csv(playlist_path)
    print(f"Playlist rows: {len(df_playlist)}")
    print(f"Playlist columns: {len(df_playlist.columns)}")
else:
    print("playlist_data.csv NOT FOUND")
    df_playlist = None

# ---------------------------------------------------------
# Compare playlist columns vs. pretrain features
# ---------------------------------------------------------
if df_playlist is not None:
    playlist_cols = df_playlist.columns.tolist()
    missing_in_playlist = [f for f in pretrain_features if f not in playlist_cols]

    print("\nMissing features in playlist_data.csv:")
    if missing_in_playlist:
        for f in missing_in_playlist:
            print("  -", f)
    else:
        print("  None — playlist_data.csv matches pretrain_features.json")

# ---------------------------------------------------------
# Inspect deduplicated song files
# ---------------------------------------------------------
print("\n--- SONG DATASET (deduplicated/*.csv) ---")
dedup_dir = DATA_DIR / "deduplicated"

if dedup_dir.exists():
    files = list(dedup_dir.glob("*.csv"))
    print(f"Found {len(files)} MBTI song files")
    for f in files:
        print(" ", f.name)
else:
    print("deduplicated/ directory NOT FOUND")

# ---------------------------------------------------------
# Inspect pretrain scaler
# ---------------------------------------------------------
print("\n--- SCALER CHECK ---")
scaler_path = MODEL_DIR / "pretrain_scaler.pkl"

if scaler_path.exists():
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully")
        print("Scaler expects feature count:", len(scaler.mean_))
    except Exception as e:
        print("Scaler exists but failed to load:", e)
else:
    print("pretrain_scaler.pkl NOT FOUND")

# ---------------------------------------------------------
# Inspect playlist classifier input size
# ---------------------------------------------------------
print("\n--- PLAYLIST CLASSIFIER INPUT CHECK ---")
model_path = MODEL_DIR / "playlist_classifier_best.pth"

if model_path.exists():
    try:
        state = torch.load(model_path, map_location="cpu")
        # Find first linear layer weight
        first_layer = [k for k in state.keys() if "encoder.0.weight" in k]
        if first_layer:
            weight = state[first_layer[0]]
            print("Playlist classifier expects input dim:", weight.shape[1])
        else:
            print("Could not detect encoder input layer")
    except Exception as e:
        print("Model exists but failed to load:", e)
else:
    print("playlist_classifier_best.pth NOT FOUND")

print("\n" + "="*70)
print("INTEGRITY CHECK COMPLETE")
print("="*70)
