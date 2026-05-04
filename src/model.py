import torch
import torch.nn as nn

# ============================================================
# 1. Pretrained Encoder (MATCHES EXACT SAVED WEIGHTS)
# ============================================================

class PretrainedEncoder(nn.Module):
    """
    EXACT architecture used in the training notebook.
    Matches encoder_114k_weights.pth perfectly.
    """
    def __init__(self, input_dim=42, encoding_dim=16):
        super().__init__()

        # EXACT layer order from your notebook
        self.layers = nn.Sequential(
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
        return self.layers(x)


# ============================================================
# 2. Playlist Classifier (main MBTI model)
# ============================================================

class PlaylistClassifier(nn.Module):
    """
    EXACT architecture used when playlist_classifier_best.pth was saved.
    """
    def __init__(self, encoder, encoding_dim=16):
        super().__init__()

        # The saved model expects this name
        self.music_encoder = nn.Module()
        self.music_encoder.encoder = encoder.layers  # match saved keys EXACTLY

        # EXACT classifier structure (0–7)
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 64),      # 0
            nn.BatchNorm1d(64),               # 1
            nn.ReLU(),                        # 2
            nn.Dropout(0.3),                  # 3

            nn.Linear(64, 32),                # 4
            nn.BatchNorm1d(32),               # 5
            nn.ReLU(),                        # 6

            nn.Linear(32, 4)                  # 7
        )

    def forward(self, x):
        latent = self.music_encoder.encoder(x)
        logits = self.classifier(latent)
        return torch.sigmoid(logits)


# ============================================================
# 3. Song-Level MBTI Classifier
# ============================================================

class SongMBTIClassifier(nn.Module):
    """
    Song-level MBTI predictor (10 audio features).
    Matches MBTI_Song_Classifier.ipynb
    """
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
