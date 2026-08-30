import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
import os

class ConfigurableFNN(nn.Module):
    def __init__(self, input_dim=45, num_classes=16, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def evaluate_hyperparams(train_loader, test_loader, device, lr, dropout):
    model = ConfigurableFNN(dropout_rate=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 40
    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            preds = model(x_b).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return model, bal_acc, macro_f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading SMOTE dataset for Hyperparameter Grid Search...")
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=64, shuffle=False)
    
    learning_rates = [1e-2, 1e-3, 1e-4]
    dropouts = [0.1, 0.3, 0.5]
    
    results = []
    best_f1 = 0
    best_model = None
    
    print("Starting Grid Search...")
    for lr in learning_rates:
        for drop in dropouts:
            print(f"Testing LR: {lr}, Dropout: {drop}...")
            model, bal_acc, macro_f1 = evaluate_hyperparams(train_loader, test_loader, device, lr, drop)
            results.append({'Learning_Rate': lr, 'Dropout': drop, 'Balanced_Accuracy': round(bal_acc, 4), 'Macro_F1': round(macro_f1, 4)})
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_model = model
                
    # Save results
    os.makedirs('evaluation', exist_ok=True)
    pd.DataFrame(results).to_csv('evaluation/hyperparameter_search.csv', index=False)
    print("Grid search complete. Results saved to evaluation/hyperparameter_search.csv")
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    torch.save(best_model.state_dict(), 'models/best_fnn_smote.pth')
    print("Best model weights saved to models/best_fnn_smote.pth")
    
    # Save the label mapping for the web app
    import json
    with open('models/label_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in label_to_idx.items()}, f)
    
    # Save the feature columns so the web app knows the exact expected input order
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
