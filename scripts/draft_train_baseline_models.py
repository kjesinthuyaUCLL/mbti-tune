import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import os


class FNNModel(nn.Module):
    def __init__(self, input_dim=45, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class CNN1DModel(nn.Module):
    def __init__(self, input_dim=45, num_classes=16):
        super().__init__()
        # input shape for conv1d: (batch_size, channels, sequence_length)
        # we treat our 45 features as a sequence of length 45 with 1 channel
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 11, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension
        x = self.conv(x)
        return self.fc(x)

class SelfAttentionModel(nn.Module):
    """
    Feature-wise Self-Attention Classifier.
    Each of the 45 audio features is a 'token'. Multi-Head Self-Attention lets
    every feature interact with every other simultaneously.
    Week 3: 'Self-Attention: every word interacts directly with every other
    word simultaneously.'
    Added Residual + LayerNorm as in real Transformer blocks (Week 3).
    """
    def __init__(self, input_dim=45, num_classes=16, embed_dim=16, num_heads=4):
        super().__init__()
        self.embedding  = nn.Linear(1, embed_dim)
        self.attention  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm       = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(-1)              # (B, 45, 1)
        x = self.embedding(x)            # (B, 45, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)      # Residual + LayerNorm
        return self.classifier(x)


def train_model(model, train_loader, test_loader, device, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_macro_f1 = 0
    best_bal_acc = 0
    
    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
    # Final Eval
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
    return bal_acc, macro_f1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = {
    'Base (Imbalanced)': 'data/processed/mbti_base.csv',
    'Random Oversampling': 'data/processed/mbti_ros.csv',
    'SMOTE': 'data/processed/mbti_smote.csv',
    'VAE': 'data/processed/mbti_balanced.csv'
}

models_to_test = {
    'FNN (Tabular)': FNNModel,
    '1D CNN (Pattern)': CNN1DModel,
    'Self-Attention (Transformer)': SelfAttentionModel
}

results = []
print("Starting Mega-Experiment: Comparing 3 architectures across 4 datasets...")

for data_name, data_path in datasets.items():
    if not os.path.exists(data_path):
        print(f"Skipping {data_name}, file not found.")
        continue
        
    print(f"\n--- Loading {data_name} ---")
    df = pd.read_csv(data_path)
    
    # Ignore metadata columns
    exclude_cols = ['mbti', 'function_pair', 'playlist_name', 'playlist_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Ensure numerical types to avoid object arrays
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)
    
    X = df[feature_cols].values
    y_labels = df['mbti'].values
    unique_labels = sorted(list(set(y_labels)))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in y_labels])
    
    # Simple 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    for model_name, ModelClass in models_to_test.items():
        print(f"Training {model_name} on {data_name}...")
        model = ModelClass().to(device)
        bal_acc, macro_f1 = train_model(model, train_loader, test_loader, device, epochs=40)
        
        print(f"-> Macro F1: {macro_f1:.4f} | Balanced Acc: {bal_acc:.4f}")
        results.append({
            'Dataset_Balancing': data_name,
            'Architecture': model_name,
            'Balanced_Accuracy': round(bal_acc, 4),
            'Macro_F1': round(macro_f1, 4)
        })

df_results = pd.DataFrame(results)
os.makedirs('evaluation', exist_ok=True)
df_results.to_csv('evaluation/model_comparison.csv', index=False)
print("\nAll experiments finished. Results saved to evaluation/model_comparison.csv")
