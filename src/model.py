# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicAutoencoder(nn.Module):
    """
    Autoencoder for pretraining on music features
    Learns compressed representations of songs (16-dim bottleneck)
    """
    def __init__(self, input_dim, encoding_dim=16):
        super(MusicAutoencoder, self).__init__()
        
        # Encoder: input_dim -> 128 -> 64 -> encoding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, encoding_dim),
        )
        
        # Decoder: encoding_dim -> 64 -> 128 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # For normalized features (0-1 range)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        """Get latent representation only"""
        return self.encoder(x)


class MBTIPredictor(nn.Module):
    """
    Multi-output regression model for 4 MBTI dimensions
    Uses pretrained encoder + custom classification heads
    """
    def __init__(self, encoder, input_dim, hidden_dim=64, dropout=0.3):
        super(MBTIPredictor, self).__init__()
        
        # Freeze encoder initially (will be unfrozen later for fine-tuning)
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoding_dim = 16
        
        # Classification heads (shared backbone + 4 separate heads)
        self.shared = nn.Sequential(
            nn.Linear(self.encoding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Four separate output heads (one per MBTI dimension)
        self.head_E = nn.Linear(hidden_dim // 2, 1)  # Extraversion
        self.head_N = nn.Linear(hidden_dim // 2, 1)  # Intuition
        self.head_T = nn.Linear(hidden_dim // 2, 1)  # Thinking
        self.head_J = nn.Linear(hidden_dim // 2, 1)  # Judging
    
    def forward(self, x):
        # Get latent representation from pretrained encoder
        latent = self.encoder(x)
        
        # Shared processing
        features = self.shared(latent)
        
        # Four outputs (sigmoid to get 0-1 percentages)
        e = torch.sigmoid(self.head_E(features))
        n = torch.sigmoid(self.head_N(features))
        t = torch.sigmoid(self.head_T(features))
        j = torch.sigmoid(self.head_J(features))
        
        # Concatenate: [E, N, T, J]
        return torch.cat([e, n, t, j], dim=1)
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True


class FrequencyWeightedLoss(nn.Module):
    """
    BCE Loss with sample weighting (more frequent songs matter more)
    """
    def __init__(self):
        super(FrequencyWeightedLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, predictions, targets, weights):
        # predictions: (batch, 4), targets: (batch, 4), weights: (batch,)
        loss_per_sample = self.bce(predictions, targets).mean(dim=1)  # Mean across dimensions
        weighted_loss = (loss_per_sample * weights).mean()
        return weighted_loss