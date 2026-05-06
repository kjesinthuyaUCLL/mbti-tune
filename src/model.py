import torch
import torch.nn as nn

class SongAutoencoder(nn.Module):
    """Matches Architecture from Notebook 1"""
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


class PlaylistLSTMEncoder(nn.Module):
    """Matches Architecture from Notebook 2"""
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            lengths = mask.sum(dim=1).long().cpu()
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(x)
        else:
            _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class MBTIClassifier(nn.Module):
    """Matches Architecture from Notebook 3"""
    def __init__(self, input_dim: int = 45, hidden_dim: int = 128, num_classes: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)