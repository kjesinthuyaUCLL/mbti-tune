import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


print("Loading data...")
path = 'data/raw/mbti_playlists/*.csv'
files = glob.glob(path)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding='latin-1')
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.fillna(0)
feature_cols = [c for c in df_all.columns if c not in ['mbti', 'function_pair', 'playlist_name', 'playlist_id']]
print(f"Total shape: {df_all.shape}, Numeric features: {len(feature_cols)}")

X_raw = df_all[feature_cols].values
labels = df_all['mbti'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar to prevent NaNs when exponentiated
        logvar = torch.clamp(logvar, min=-20, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize by batch size
    kld /= x.size(0) * x.size(1)
    return recon_loss + beta * kld, recon_loss, kld


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(feature_cols)
latent_dim = 16
vae = VAE(input_dim, latent_dim).to(device)

optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
dataset = TensorDataset(torch.FloatTensor(X_scaled))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

epochs = 150
print("Training VAE...")
vae.train()
for epoch in range(epochs):
    total_loss, total_recon, total_kld = 0, 0, 0
    for batch in dataloader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(x_batch)
        loss, recon, kld = vae_loss(recon_batch, x_batch, mu, logvar, beta=0.5)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Recon: {total_recon/len(dataloader):.4f} | KLD: {total_kld/len(dataloader):.4f}")


print("\nAugmenting minority classes...")
vae.eval()
target_samples = 310 # Max class is 307
augmented_data = []
augmented_labels = []

class_counts = df_all['mbti'].value_counts()

with torch.no_grad():
    for mbti_class in class_counts.index:
        idx = np.where(labels == mbti_class)[0]
        x_class = torch.FloatTensor(X_scaled[idx]).to(device)
        
        # Get latent distribution for this class
        mu_class, logvar_class = vae.encode(x_class)
        
        # We can either resample from the global class distribution or individual points
        # Let's sample by randomly picking a real point's mu and logvar, and reparameterizing
        num_to_generate = max(0, target_samples - len(idx))
        if num_to_generate > 0:
            # Pick random real samples as base for generation
            rand_idx = torch.randint(0, len(idx), (num_to_generate,))
            sampled_mu = mu_class[rand_idx]
            sampled_logvar = logvar_class[rand_idx]
            
            # Generate new latent vectors
            z_new = vae.reparameterize(sampled_mu, sampled_logvar)
            
            # Decode to synthetic features
            x_synthetic = vae.decode(z_new).cpu().numpy()
            
            augmented_data.append(x_synthetic)
            augmented_labels.extend([mbti_class] * num_to_generate)
            print(f"Generated {num_to_generate} samples for {mbti_class}")

if augmented_data:
    X_synthetic = np.vstack(augmented_data)
    # Inverse transform to keep data in original scale before saving
    X_synthetic_raw = scaler.inverse_transform(X_synthetic)
    
    df_synthetic = pd.DataFrame(X_synthetic_raw, columns=feature_cols)
    df_synthetic['mbti'] = augmented_labels
    df_synthetic['function_pair'] = 'synthetic'
    df_synthetic['playlist_name'] = 'synthetic_playlist'
    df_synthetic['playlist_id'] = 'synthetic_id'
    
    df_balanced = pd.concat([df_all, df_synthetic], ignore_index=True)
else:
    df_balanced = df_all

print(f"\nOriginal dataset shape: {df_all.shape}")
print(f"Balanced dataset shape: {df_balanced.shape}")
print("New class counts:\n", df_balanced['mbti'].value_counts())

os.makedirs('../data/processed', exist_ok=True)
df_balanced.to_csv('data/processed/mbti_balanced.csv', index=False)
print("Saved balanced dataset to data/processed/mbti_balanced.csv")

# Save VAE models just in case
torch.save(vae.state_dict(), 'data/processed/vae_model.pth')
