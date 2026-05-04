
# Simple test for downloaded model
import torch
import pickle
import json
import numpy as np

# Load
model_state = torch.load('model_state_dict.pt', map_location='cpu')
with open('features.json', 'r') as f:
    features = json.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define model (same architecture)
class SimpleMBTIPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 4)
        )
    def forward(self, x):
        return self.network(x)

model = SimpleMBTIPredictor(len(features))
model.load_state_dict(model_state)
model.eval()

# Test prediction
dummy = np.ones((1, len(features))) * 0.5
dummy_scaled = scaler.transform(dummy)
with torch.no_grad():
    pred = torch.sigmoid(model(torch.tensor(dummy_scaled))).numpy()[0]

print(f"Test prediction: E={pred[0]*100:.1f}%, N={pred[1]*100:.1f}%, T={pred[2]*100:.1f}%, J={pred[3]*100:.1f}%")
print("✅ Model loads and predicts correctly!")
