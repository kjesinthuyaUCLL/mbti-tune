# Technical Implementation Guide - MBTI Tune

This document provides strict technical guidelines, preferred libraries, and implementation strategies derived from the MBTI Tune project development.

**Copilot Instructions:** Use this document to dictate *how* the code should be structured, which libraries to use, and what specific implementation patterns to follow when assisting.

**Note:** Primary AI assistant used was DeepSeek; team member used Copilot for development.

---

## 1. Environment & Core Frameworks

### Python & Dependencies
- **Python Version:** 3.10 to 3.13
- **Virtual Environment:** `venv` (activated with `venv\Scripts\activate` on Windows)
- **Install:** `pip install -r requirements.txt`

### Deep Learning Framework
- **PyTorch** (CPU/GPU agnostic, no GPU required for inference)
- **Model Files:** Saved as `.pth` using `torch.save()`

### Data Science Stack
- **pandas** - Data manipulation and aggregation
- **numpy** - Numerical operations
- **scikit-learn** - StandardScaler, train_test_split, metrics
- **joblib** - Save/load scalers and models

### API Integrations
- **Spotipy** - Spotify API wrapper for OAuth and track fetching
- **Google Gemini** - LLM for psychological analysis and summarization
- **Groq** - Fallback LLM for lyrics summarization
- **LRCLIB** - Free lyrics API (no authentication required)

---

## 2. Device Handling (Mandatory)

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# All tensors must be moved to device:
tensor = tensor.to(device)

# For inference (Streamlit app):
device = torch.device('cpu')  # Force CPU for deployment
```

---

## 3. Core PyTorch Coding Standards

### Architecture Definition
Use **Object-Oriented PyTorch** with `nn.Module`:

```python
import torch.nn as nn

class SongAutoencoder(nn.Module):
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
```

### MBTI Classifier Architecture

```python
class MBTIClassifier(nn.Module):
    """
    Architecture: input_dim → 64 → 32 → 16 → 16 (num_classes)
    Dropout rates: 0.3, 0.3, 0.15
    """
    def __init__(self, input_dim: int, num_classes: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
```

### Hyperparameter Exposure
All models must expose:
- learning rate (`lr=1e-3`)
- batch size (`batch_size=1024` for autoencoder, `16` for classifier)
- number of layers
- hidden dimensions (128, 64, 32)
- dropout rates (0.3, 0.15)
- activation functions (ReLU)
- latent dimension (32 for autoencoder)

---

## 4. Training Loop Structure (Required)

### Autoencoder Training

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        
        optimizer.zero_grad()
        x_hat, z = model(xb)
        loss = criterion(x_hat, yb)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    
    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            x_hat, z = model(xb)
            loss = criterion(x_hat, yb)
            val_losses.append(loss.item())
    
    scheduler.step(np.mean(val_losses))
    
    # Save best model
    if np.mean(val_losses) < best_val_loss:
        best_val_loss = np.mean(val_losses)
        torch.save(model.state_dict(), MODEL_PATH)
```

### Classifier Training with Class Weights

```python
# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

# Early stopping
if patience_counter >= 20:
    print(f"Early stopping at epoch {epoch}")
    break
```

---

## 5. Visualization Requirements

Use `matplotlib` to plot:
- Autoencoder reconstruction loss (train vs validation)
- Classifier training & validation loss
- SHAP summary plots for explainability
- Confusion matrix for classification results
- t-SNE visualization of latent space

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## 6. Data Handling

### PyTorch Datasets

```python
from torch.utils.data import Dataset, DataLoader

class SongDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # Autoencoder: input = target

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
```

### Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test (stratified for classification)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
```

---

## 7. Audio Feature Processing

### Standard Audio Features (9 features)

```python
AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 'acousticness',
    'instrumentalness', 'speechiness', 'loudness', 'tempo', 'liveness'
]
```

### Feature Aggregation (43 features)

```python
def build_features_from_tracks(tracks_data):
    """Aggregates individual track data into 43 statistical features"""
    df = pd.DataFrame(tracks_data)
    res = {}
    
    # 1. Means and Standard Deviations (18 features)
    for col in AUDIO_FEATURES:
        res[f"{col}_mean"] = float(df[col].mean())
        res[f"{col}_stdev"] = float(df[col].std() if len(df) > 1 else 0.0)
    
    # 2. Key and Mode counts (24 features)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for name in key_names:
        res[f"{name}Major_count"] = 0.0
        res[f"{name}minor_count"] = 0.0
    
    # 3. Track count (1 feature)
    res["track_count"] = float(len(df))
    
    return res  # 43 features total
```

### Transfer Learning Features (128 features)

Generated by passing 9 audio features through autoencoder encoder, then aggregating:
- Mean of 32-dim latent vectors
- Standard deviation
- Minimum
- Maximum

**Total features: 43 + 128 = 171 features**

---

## 8. Fallback Simulation (Required for API Failures)

### Real Song Database (44,000+ songs)

```python
def load_song_database():
    """Load pre-processed 1M song database for realistic fallback"""
    db_dir = Path("data/processed_song_database")
    sample_features = db_dir / "sample_features_100k.npy"
    sample_metadata = db_dir / "sample_metadata_100k.parquet"
    
    if sample_features.exists():
        return np.load(sample_features), pd.read_parquet(sample_metadata)
    return None, None
```

### Fallback Simulation Logic

```python
def generate_simulated_features(track_name, artist_name):
    """Try database first, then Beta distribution fallback"""
    db_features = generate_simulated_features_from_database(track_name, artist_name)
    if db_features is not None:
        return db_features
    return generate_simulated_features_beta(track_name, artist_name)
```

---

## 9. Inference Stabilization (Critical Fix)

### Feature Clipping & Down-weighting

```python
def stabilize_features(features_vector, scaler, feature_cols, clip_range=(-3, 3)):
    """Apply stabilization to prevent extreme predictions"""
    stabilized = features_vector.copy()
    
    # 1. Clip extreme values
    stabilized = np.clip(stabilized, clip_range[0], clip_range[1])
    
    # 2. Down-weight low-variance features (std < 0.15)
    small_std_threshold = 0.15
    for i in range(len(feature_cols)):
        if scaler.scale_[i] < small_std_threshold:
            stabilized[i] = stabilized[i] * 0.3  # Reduce influence by 70%
    
    return stabilized.astype(np.float32)
```

### Temperature Scaling

```python
def predict_mbti(features_vector, model, scaler, device, feature_cols, idx_to_type, temperature=4.0):
    """Predict with temperature scaling to reduce overconfidence"""
    logits = model(x)
    logits = logits / temperature  # Temperature > 1 reduces confidence
    probs = torch.softmax(logits, dim=1)
    
    # Add small smoothing to prevent extremes
    smoothing = 0.005
    probs = (1 - smoothing) * probs + smoothing / len(probs)
    probs = probs / probs.sum()
    
    return result
```

---

## 10. LLM Integration (Gemini & Groq)

### Gemini Setup

```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(prompt)
```

### Groq Fallback

```python
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)
```

### Lyrics Pipeline
1. Fetch from LRCLIB API (timeout=10 seconds)
2. Detect language and translate to English (Gemini)
3. Summarize lyrics (Groq or Gemini)
4. Generate psychological breakdown (Gemini)

---

## 11. Streamlit Application

### Required Components

```python
import streamlit as st

st.set_page_config(page_title="MBTI Tune", layout="centered")

# OAuth Flow
auth_url = oauth.get_authorize_url()
st.markdown(f'<a href="{auth_url}" class="spotify-login-btn">Connect Spotify</a>', unsafe_allow_html=True)

# Cached model loading
@st.cache_resource
def load_assets():
    return load_model_and_scaler()

# Display MBTI with custom bars
st.markdown(f"""
<div class="mbti-bar-container">
    <div class="mbti-labels">
        <span>Introversion ({left_val:.0f}%)</span>
        <span>({right_val:.0f}%) Extraversion</span>
    </div>
    <div class="mbti-track">
        <div style="width: {left_val}%; background: linear-gradient(90deg, #ff9a9e, #fbc2eb);"></div>
        <div style="width: {right_val}%; background: linear-gradient(90deg, #a18cd1, #fbc2eb);"></div>
    </div>
</div>
""", unsafe_allow_html=True)
```

---

## 12. Explainability - SHAP (Planned)

```python
import shap

# Create wrapper for PyTorch model
def predict_for_shap(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

# KernelExplainer for any model
explainer = shap.KernelExplainer(predict_for_shap, X_train[:100])
shap_values = explainer.shap_values(X_test[:50])

# Visualizations
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
shap.waterfall_plot(shap.Explanation(values=shap_values[0], ...))
```

---

## 13. Strict "Do Not" Rules

### ❌ Do NOT use:
- TensorFlow / Keras
- HuggingFace Transformers
- Any LLM for training (only for interpretation)
- GANs, VAEs (only the implemented autoencoder)
- Reinforcement Learning
- LangChain, Docker, Kubernetes

### ❌ Do NOT:
- Write monolithic scripts (use modular `src/` structure)
- Skip loss/metric logging
- Skip device handling
- Skip evaluation metrics (accuracy, precision, recall, F1, MAE)
- Hardcode file paths (use `Path(__file__).parent.parent`)
- Use absolute imports (use relative imports within package)

---

## 14. Project Structure

```
mbti-tune/
├── app/
│   └── app.py                 # Streamlit application
├── src/
│   ├── model.py               # Neural network definitions
│   ├── inference.py           # Model loading + prediction
│   ├── spotify_utils.py       # Spotify API + feature extraction
│   ├── lyrics_utils.py        # Lyrics fetching
│   ├── gemini_utils.py        # Gemini API
│   └── groq_utils.py          # Groq API fallback
├── data/
│   ├── processed/             # Model files (.pth, .pkl, .json)
│   ├── raw/                   # Raw data (not in git)
│   └── processed_song_database/  # 100k song sample for fallback
├── notebooks/                 # Jupyter notebooks (1-4)
├── requirements.txt
└── .env                       # API keys (not in git)
```

---

## 15. Key Performance Metrics

| Model | Metric | Value |
|-------|--------|-------|
| Autoencoder | Reconstruction error | 3.83% |
| Playlist Classifier | Test Accuracy | 33-37% |
| Playlist Classifier | E/I Axis | 76.7% |
| Playlist Classifier | T/F Axis | 75.0% |
| Playlist Classifier | S/N Axis | 69.7% |
| Playlist Classifier | J/P Axis | 65.8% |

---

## 16. Summary

This updated guide reflects the **exact technologies and methods** used in the MBTI Tune project:

- ✅ PyTorch autoencoder (9→128→64→32→64→128→9)
- ✅ Transfer learning classifier (171→64→32→16→16)
- ✅ Spotify audio features + fallback database (44,000+ real songs)
- ✅ LRCLIB lyrics + Gemini/Groq summarization
- ✅ Streamlit frontend with custom CSS
- ✅ Inference stabilization (clipping + down-weighting + temperature scaling)
- ✅ SHAP explainability (planned/implemented)

---