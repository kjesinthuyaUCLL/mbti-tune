"""
Experiment: Polynomial Feature Engineering (Feature Interaction)
Instead of adding external data (which failed due to API limits), 
we create new synthetic features by multiplying existing ones.
E.g., Energy * Danceability.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load the best dataset (SMOTE) ─────────────────────────────────────────
df = pd.read_csv('data/processed/mbti_smote.csv')

exclude_cols = ['mbti', 'function_pair', 'playlist_name', 'playlist_id']
feature_cols = [c for c in df.columns if c not in exclude_cols]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(0, inplace=True)

X = df[feature_cols].values
y_labels = df['mbti'].values
unique_labels = sorted(list(set(y_labels)))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
y = np.array([label_to_idx[l] for l in y_labels])

print(f"Original features shape: {X.shape}")

# ── 2. Create Polynomial Features (Degree 2, Interactions Only) ──────────────
# interactions_only=True means we do A*B, but not A^2.
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"Polynomial features shape: {X_poly.shape}")

# ── 3. Standardize and Split ─────────────────────────────────────────────────
sc = StandardScaler()
X_poly = sc.fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset  = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ── 4. Train FNN Model ───────────────────────────────────────────────────────
class FNNModel(nn.Module):
    def __init__(self, input_dim, num_classes=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

input_dim = X_train.shape[1]
model = FNNModel(input_dim=input_dim).to(device)

criterion = nn.CrossEntropyLoss()
# Use weight_decay (L2 regularization) to prevent overfitting on the huge number of features
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("\nTraining FNN on Polynomial Features (100 epochs)...")
best_macro_f1 = 0
best_bal_acc = 0

for epoch in range(100):
    model.train()
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        preds = model(x_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for x_b, y_b in test_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        preds = model(x_b).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_b.cpu().numpy())

bal_acc = balanced_accuracy_score(all_targets, all_preds)
macro_f1 = f1_score(all_targets, all_preds, average='macro')

print("\n--- RESULTS ---")
print(f"Original FNN (45 features):  Macro F1 = 35.05%")
print(f"Polynomial FNN ({input_dim} features): Macro F1 = {macro_f1*100:.2f}%")
print(f"Balanced Accuracy: {bal_acc*100:.2f}%")

# Axis-level Evaluation
axis_accuracies = {'E/I': 0, 'S/N': 0, 'T/F': 0, 'J/P': 0}
for t, p in zip(all_targets, all_preds):
    target_str = unique_labels[t]
    pred_str = unique_labels[p]
    
    if target_str[0] == pred_str[0]: axis_accuracies['E/I'] += 1
    if target_str[1] == pred_str[1]: axis_accuracies['S/N'] += 1
    if target_str[2] == pred_str[2]: axis_accuracies['T/F'] += 1
    if target_str[3] == pred_str[3]: axis_accuracies['J/P'] += 1

total = len(all_targets)
print("\n--- AXIS ACCURACIES ---")
for k in axis_accuracies:
    val = (axis_accuracies[k] / total) * 100
    print(f"{k}: {val:.2f}%")

from sklearn.metrics import confusion_matrix
import os
cm = confusion_matrix(all_targets, all_preds)
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/cm_poly.npy', cm)
print("\nConfusion matrix saved to data/processed/cm_poly.npy")

# ── 5. Save Model and Transforms for Inference ───────────────────────────────
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/best_poly_fnn.pth')
joblib.dump(poly, 'models/poly_transformer.pkl')
joblib.dump(sc, 'models/scaler.pkl')
print("\nModel, Scaler, and Polynomial features saved to models/ directory.")
