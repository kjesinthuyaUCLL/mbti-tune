"""
Model definitions for MBTI prediction system
Compatible with all three notebooks
"""
import torch
import torch.nn as nn


class SongAutoencoder(nn.Module):
    """Autoencoder for individual songs (Notebook 1)"""
    def __init__(self, input_dim: int = 9, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def encode(self, x):
        return self.encoder(x)


class PlaylistLSTMEncoder(nn.Module):
    """LSTM encoder for playlists (Notebook 2)"""
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, mask=None):
        # x: [B, T, F]
        if mask is not None:
            lengths = mask.sum(dim=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        z = self.fc(h_last)
        return z


class MBTIClassifier(nn.Module):
    """
    MBTI classifier with transfer learning from song autoencoder (Notebook 3)
    """
    def __init__(self, input_dim: int = 42, latent_dim: int = 16, 
                 pretrained_encoder_path: str = None):
        super().__init__()
        
        # Encoder (matches Notebook 1 structure)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Load pretrained weights if provided
        if pretrained_encoder_path:
            try:
                state = torch.load(pretrained_encoder_path, map_location='cpu')
                # Handle different save formats
                if 'encoder_state_dict' in state:
                    self.encoder.load_state_dict(state['encoder_state_dict'])
                elif 'model_state_dict' in state:
                    # Extract encoder part if full model saved
                    encoder_state = {k.replace('encoder.', ''): v 
                                   for k, v in state['model_state_dict'].items() 
                                   if k.startswith('encoder.')}
                    self.encoder.load_state_dict(encoder_state)
                else:
                    self.encoder.load_state_dict(state)
                print(f"✅ Loaded pretrained encoder from {pretrained_encoder_path}")
            except Exception as e:
                print(f"⚠️ Could not load pretrained weights: {e}")
                print("   Training from scratch instead")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 binary outputs (E, S, T, J)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z
    
    def predict(self, x):
        """Return probabilities for each MBTI dimension"""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)


def load_song_embeddings_model(model_path: str, input_dim: int = 9, device: str = 'cpu'):
    """Load the song autoencoder model (Notebook 1)"""
    model = SongAutoencoder(input_dim=input_dim, latent_dim=32)
    state = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_playlist_encoder_model(model_path: str, input_dim: int = 12, 
                                 hidden_dim: int = 128, latent_dim: int = 64, 
                                 device: str = 'cpu'):
    """Load the playlist LSTM encoder model (Notebook 2)"""
    model = PlaylistLSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim, 
                                 latent_dim=latent_dim)
    state = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_mbti_classifier(model_path: str, input_dim: int = 42, 
                          pretrained_encoder_path: str = None, device: str = 'cpu'):
    """Load the MBTI classifier model (Notebook 3)"""
    model = MBTIClassifier(input_dim=input_dim, 
                           pretrained_encoder_path=pretrained_encoder_path)
    state = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model