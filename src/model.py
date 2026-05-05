import torch
import torch.nn as nn

# ============================================================
# 1. Pretrained Encoder (MATCHES TRAINING NOTEBOOK)
# ============================================================

class PretrainedEncoder(nn.Module):
    def __init__(self, input_dim=42, encoding_dim=16):
        super().__init__()

        # EXACT architecture used during training
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, encoding_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# 2. Playlist Classifier (MAIN MBTI MODEL)
# ============================================================

class PlaylistClassifier(nn.Module):
    def __init__(self, encoder, encoding_dim=16):
        super().__init__()

        # Correct module nesting to match saved weights
        self.music_encoder = nn.Module()
        self.music_encoder.encoder = encoder.encoder

        # EXACT classifier structure from training
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 4)
        )

    def forward(self, x):
        latent = self.music_encoder.encoder(x)
        logits = self.classifier(latent)
        return torch.sigmoid(logits)


# ============================================================
# 3. Song-Level MBTI Classifier
# ============================================================

class SongMBTIClassifier(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.network(x)
