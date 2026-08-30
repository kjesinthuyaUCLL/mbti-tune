import torch
import torch.nn as nn

# FINAL ARCHITECTURE

class PolynomialFNN(nn.Module):
    """
    Final Model: Deep Feedforward Neural Network for Polynomial Features
    Input: 1035 features (43 base features + polynomial interactions)
    Architecture: 4-layer FNN with BatchNorm and Dropout (30%) to prevent overfitting.
    """
    def __init__(self, input_dim: int = 1035, num_classes: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): 
        return self.net(x)


# ARCHIVED / INTERMEDIATE MODELS 
# (Kept for methodological documentation and Oral Defense)

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE)
    Purpose: Attempted to generate synthetic tabular data to balance the dataset.
    Outcome: Discarded. Probabilistic generation created blurry decision boundaries 
    (Macro-F1 dropped to 25.3%). Replaced by SMOTE.
    """
    def __init__(self, input_dim: int = 45, latent_dim: int = 16):
        super(VAE, self).__init__()
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-20, max=10)
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


class CNN1DModel(nn.Module):
    """
    1D-Convolutional Neural Network
    Purpose: Treat 45 audio features as a 1D sequence to extract local patterns.
    Outcome: Discarded. Tabular data features are independent and do not possess 
    spatial/sequential relationships, making convolutions ineffective.
    """
    def __init__(self, input_dim: int = 45, num_classes: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 11, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.conv(x))


class SelfAttentionModel(nn.Module):
    """
    Self-Attention Classifier
    Purpose: Let every feature dynamically attend to every other feature (NLP approach).
    Outcome: Discarded. Underperformed standard FNNs due to lack of large-scale 
    pretraining required for transformers to effectively learn feature interactions.
    """
    def __init__(self, input_dim: int = 45, num_classes: int = 16,
                 embed_dim: int = 16, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embedding  = nn.Linear(1, embed_dim)
        self.attention  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm       = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)
        return self.classifier(x)