"""
Step 3: Multimodal Ablation Study
====================================
Trains 7 configurations and compares them:

  Config 1: Audio only         (18 features)
  Config 2: Genres only        (384 features, Sentence-BERT)
  Config 3: Lyrics only        (6 VADER features)
  Config 4: Audio + Genres     (18 + 384 = 402)
  Config 5: Audio + Lyrics     (18 + 6 = 24)
  Config 6: Genres + Lyrics    (384 + 6 = 390)
  Config 7: Audio + Genres + Lyrics  (18 + 384 + 6 = 408)  ← Full Dual-Branch

Justified by Week 3 (NLP):
  - FNN for tabular data
  - Sentence-BERT for text understanding (encoder-based model)
  - VADER for sentiment (polarity analysis)
  - Ablation = scientific comparison of modality contributions

Results saved to: evaluation/multimodal_ablation.csv
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading features...")
df_base    = pd.read_csv('data/processed/raw_playlist_features.csv')
genres_emb = np.load('data/processed/genres_embeddings.npy')         # (N, 384)
df_lyrics  = pd.read_csv('data/processed/lyrics_features.csv')        # (N, 6)

assert len(df_base) == len(genres_emb) == len(df_lyrics), "Row mismatch!"
print(f"  Playlists: {len(df_base)}")

# ── Build feature matrices ───────────────────────────────────────────────────
AUDIO_FEAT_COLS = [c for c in df_base.columns
                   if c.endswith('_mean') or c.endswith('_stdev')]

X_audio  = df_base[AUDIO_FEAT_COLS].fillna(0).values.astype(np.float32)   # (N,18)
X_genres = genres_emb.astype(np.float32)                                    # (N,384)
X_lyrics = df_lyrics.values.astype(np.float32)                              # (N,6)

# Labels
MBTI_TYPES = sorted(df_base['mbti'].unique())
label_map  = {m: i for i, m in enumerate(MBTI_TYPES)}
y = np.array([label_map[m] for m in df_base['mbti']])
n_classes  = len(MBTI_TYPES)

print(f"  Audio features:  {X_audio.shape[1]}")
print(f"  Genres features: {X_genres.shape[1]}")
print(f"  Lyrics features: {X_lyrics.shape[1]}")
print(f"  Classes:         {n_classes}  {MBTI_TYPES}")

CONFIGS = {
    'Audio only':              X_audio,
    'Genres only':             X_genres,
    'Lyrics only':             X_lyrics,
    'Audio + Genres':          np.hstack([X_audio, X_genres]),
    'Audio + Lyrics':          np.hstack([X_audio, X_lyrics]),
    'Genres + Lyrics':         np.hstack([X_genres, X_lyrics]),
    'Audio + Genres + Lyrics': np.hstack([X_audio, X_genres, X_lyrics]),
}

# ── Model ────────────────────────────────────────────────────────────────────
class FNN(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.3):
        super().__init__()
        h1 = min(512, input_dim * 2)
        h2 = min(256, h1 // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 64),        nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

# ── Training with 5-Fold CV (more reliable on small dataset) ─────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

def run_kfold(X: np.ndarray, y: np.ndarray, config_name: str, k: int = 5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores, bal_accs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Standardize
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr).astype(np.float32)
        X_va = sc.transform(X_va).astype(np.float32)

        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.LongTensor(y_tr)),
            batch_size=32, shuffle=True
        )

        model = FNN(X_tr.shape[1], n_classes).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit  = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

        model.train()
        for epoch in range(80):
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.from_numpy(X_va).to(device)).argmax(1).cpu().numpy()

        f1_scores.append(f1_score(y_va, val_preds, average='macro', zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_va, val_preds))

    return np.mean(f1_scores), np.std(f1_scores), np.mean(bal_accs)

# --- Run ablation -------------------------------------------------------------
results = []
print(f"{'Config':<30} {'Macro-F1 (mean+/-std)':<25} {'Bal-Acc'}")
print("-" * 70)

for config_name, X_feat in CONFIGS.items():
    mean_f1, std_f1, mean_bal = run_kfold(X_feat, y, config_name)
    print(f"{config_name:<30} {mean_f1:.4f} +/- {std_f1:.4f}          {mean_bal:.4f}")
    results.append({
        'Config':        config_name,
        'Macro_F1_mean': round(mean_f1, 4),
        'Macro_F1_std':  round(std_f1, 4),
        'Balanced_Acc':  round(mean_bal, 4),
        'Input_Dims':    X_feat.shape[1]
    })

os.makedirs('evaluation', exist_ok=True)
df_res = pd.DataFrame(results)
df_res.to_csv('evaluation/multimodal_ablation.csv', index=False)
print("\nResults saved to evaluation/multimodal_ablation.csv")
print("\nFull table:")
print(df_res.to_string(index=False))
