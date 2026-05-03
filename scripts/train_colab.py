import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_autoencoder(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """
    Train autoencoder on unlabeled songs
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            x = batch[0].to(device)
            _, reconstructed = model(x)
            loss = criterion(reconstructed, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                _, reconstructed = model(x)
                loss = criterion(reconstructed, x)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return train_losses, val_losses


def train_classifier(model, train_loader, val_loader, dimension_weights, epochs=100, lr=0.001, device='cuda'):
    """
    Train MBTI predictor with frequency weighting
    """
    model = model.to(device)
    
    # Separate loss weights for each dimension (handles imbalance)
    pos_weights = torch.tensor([dimension_weights[d]['positive'] for d in range(4)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)  # Note: we use logits, but our model outputs sigmoid
    
    # Better to use BCELoss with manual weighting
    criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            x, y, w = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            predictions = model(x)
            loss_per_sample = criterion(predictions, y).mean(dim=1)
            weighted_loss = (loss_per_sample * w).mean()
            
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, w = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                predictions = model(x)
                loss_per_sample = criterion(predictions, y).mean(dim=1)
                weighted_loss = (loss_per_sample * w).mean()
                val_loss += weighted_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.6f})")
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate on test set with multiple metrics
    """
    model.eval()
    predictions = []
    targets = []
    weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y, w = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pred = model(x)
            predictions.append(pred.cpu())
            targets.append(y.cpu())
            weights.append(w.cpu())
    
    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    weights = torch.cat(weights, dim=0).numpy()
    
    # Calculate metrics per dimension
    dim_names = ['Extraversion (E)', 'Intuition (N)', 'Thinking (T)', 'Judging (J)']
    results = {}
    
    for i, name in enumerate(dim_names):
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        
        # Weighted MAE
        weighted_mae = np.sum(np.abs(predictions[:, i] - targets[:, i]) * weights) / np.sum(weights)
        
        # Correlation
        corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        
        # Binary accuracy (threshold at 0.5)
        binary_pred = (predictions[:, i] > 0.5).astype(int)
        binary_true = (targets[:, i] > 0.5).astype(int)
        accuracy = np.mean(binary_pred == binary_true)
        
        results[name] = {
            'MAE': mae,
            'Weighted_MAE': weighted_mae,
            'Correlation': corr,
            'Accuracy': accuracy
        }
    
    return results


def plot_training_curves(train_losses, val_losses, title="Training Curves"):
    """Plot loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("✓ Saved training_curves.png")