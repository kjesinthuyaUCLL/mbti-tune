from pathlib import Path
import torch
import torch.nn as nn
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("="*60)
print("UPDATED MODEL PERFORMANCE EVALUATION (PLAYLIST CLASSIFIER)")
print("="*60)

# ============================================================
# Resolve paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

print("Script directory:", SCRIPT_DIR)
print("Project root:", PROJECT_ROOT)
print("Model directory:", MODEL_DIR)
print("Data directory:", DATA_DIR)

# ============================================================
# Load feature list (42 features)
# ============================================================

with open(MODEL_DIR / "pretrain_features.json", "r") as f:
    feature_names = json.load(f)

print(f"Loaded {len(feature_names)} pretrained features")

# ============================================================
# Load playlist dataset
# ============================================================

df = pd.read_csv(DATA_DIR / "merged" / "playlist_data.csv")
print(f"Loaded {len(df)} playlists")

# Ensure all features exist
missing = [c for c in feature_names if c not in df.columns]
if missing:
    print("Missing features detected:", missing)
    for col in missing:
        df[col] = 0.0

X = df[feature_names].values.astype(np.float32)
y = df[['E','N','T','J']].values.astype(np.float32)

# ============================================================
# Load scaler
# ============================================================

with open(MODEL_DIR / "pretrain_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X)

# ============================================================
# Load correct model architecture
# ============================================================

class PretrainedEncoder(nn.Module):
    def __init__(self, input_dim=42, encoding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class PlaylistClassifier(nn.Module):
    def __init__(self, encoder, encoding_dim=16):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        z = self.encoder(x)
        return torch.sigmoid(self.classifier(z))


# Load encoder
encoder = PretrainedEncoder(input_dim=len(feature_names))
encoder.load_state_dict(torch.load(MODEL_DIR / "encoder_114k_weights.pth", map_location="cpu"))

# Load classifier
model = PlaylistClassifier(encoder)
model.load_state_dict(torch.load(MODEL_DIR / "playlist_classifier_best.pth", map_location="cpu"))
model.eval()

print("Model loaded successfully")

# ============================================================
# Test split
# ============================================================

_, X_test, _, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=df["personality"]
)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ============================================================
# Predictions
# ============================================================

with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

pred_pct = predictions * 100
true_pct = y_test * 100

# ============================================================
# Metrics
# ============================================================

dim_names = ["E","N","T","J"]

print("\nMETRICS PER DIMENSION")
print("="*60)

for i, dim in enumerate(dim_names):
    mae = mean_absolute_error(true_pct[:,i], pred_pct[:,i])
    rmse = np.sqrt(mean_squared_error(true_pct[:,i], pred_pct[:,i]))
    r2 = r2_score(true_pct[:,i], pred_pct[:,i])
    acc15 = np.mean(np.abs(pred_pct[:,i] - true_pct[:,i]) < 15)
    letter_acc = np.mean((predictions[:,i] > 0.5) == (y_test[:,i] > 0.5))

    print(f"\n{dim}:")
    print(f"  MAE:  {mae:.2f}%")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  R²:   {r2:.3f}")
    print(f"  Acc@15%: {acc15:.1%}")
    print(f"  Letter Accuracy: {letter_acc:.1%}")

# ============================================================
# Overall
# ============================================================

overall_mae = np.mean(np.abs(pred_pct - true_pct))
overall_acc15 = np.mean(np.abs(pred_pct - true_pct) < 15)
overall_letter_acc = np.mean((predictions > 0.5) == (y_test > 0.5))

print("\n" + "="*60)
print("OVERALL PERFORMANCE")
print("="*60)
print(f"Overall MAE: {overall_mae:.2f}%")
print(f"Overall Acc@15%: {overall_acc15:.1%}")
print(f"Overall Letter Accuracy: {overall_letter_acc:.1%}")
