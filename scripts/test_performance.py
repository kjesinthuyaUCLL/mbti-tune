import torch
import numpy as np
import joblib
import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PretrainedEncoder, PlaylistClassifier

print("\n==============================")
print(" MBTI MODEL DIAGNOSTIC REPORT")
print("==============================\n")

# ---------------------------------------------------------
# 1. Load feature list
# ---------------------------------------------------------
try:
    with open("models/pretrain_features.json", "r") as f:
        feature_cols = json.load(f)
    print(f"Loaded {len(feature_cols)} features.")
except Exception as e:
    print("❌ ERROR: Could not load pretrain_features.json:", e)
    exit()

# ---------------------------------------------------------
# 2. Load scaler
# ---------------------------------------------------------
try:
    scaler = joblib.load("models/pretrain_scaler.pkl")
    print("Scaler loaded successfully.")
except Exception as e:
    print("❌ ERROR: Could not load scaler:", e)
    exit()

# ---------------------------------------------------------
# 3. Load encoder
# ---------------------------------------------------------
try:
    encoder = PretrainedEncoder(input_dim=len(feature_cols), encoding_dim=16)

    raw_state = torch.load("models/encoder_114k_weights.pth", map_location="cpu")

    fixed_state = {}
    for k, v in raw_state.items():
        fixed_state[f"encoder.{k}"] = v

    encoder.load_state_dict(fixed_state)
    encoder.eval()
    print("Encoder loaded successfully (after key remapping).")

except Exception as e:
    print("❌ ERROR loading encoder:", e)
    exit()

# ---------------------------------------------------------
# 4. Load playlist classifier
# ---------------------------------------------------------
try:
    model = PlaylistClassifier(encoder)
    model.load_state_dict(torch.load("models/playlist_classifier_best.pth", map_location="cpu"))
    model.eval()
    print("Playlist classifier loaded successfully.")
except Exception as e:
    print("❌ ERROR loading playlist classifier:", e)
    exit()

# ---------------------------------------------------------
# 5. Generate a random valid input
# ---------------------------------------------------------
x = np.random.rand(1, len(feature_cols)).astype(np.float32)
x_scaled = scaler.transform(x)
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

# ---------------------------------------------------------
# 6. Run inference
# ---------------------------------------------------------
with torch.no_grad():
    output = model(x_tensor).numpy()[0]

print("\nModel output:", output)
print("Sum of outputs:", output.sum())

# ---------------------------------------------------------
# 7. Detect constant-output bug
# ---------------------------------------------------------
rounded = [round(v, 3) for v in output]
if len(set(rounded)) == 1:
    print("\n⚠️ WARNING: Model outputs are constant.")
    print("This indicates one of the following:")
    print("- Feature order mismatch")
    print("- Wrong scaler")
    print("- Wrong model architecture")
    print("- Wrong input dimension")
else:
    print("\n✅ Output varies — model is behaving normally.")

# ---------------------------------------------------------
# 8. Print model parameter count
# ---------------------------------------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total_params:,}")

# ---------------------------------------------------------
# 9. Optional: Evaluate accuracy if validation data exists
# ---------------------------------------------------------
try:
    import pandas as pd
    df = pd.read_csv("data/training/val.pt")  # adjust if needed

    X = df[feature_cols].values
    y = df[["E", "N", "T", "J"]].values

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor).numpy()

    mse = ((preds - y) ** 2).mean()
    print(f"\nValidation MSE: {mse:.4f}")

except Exception:
    print("\n(No validation dataset found — skipping accuracy test.)")

print("\nDiagnostic complete.\n")
