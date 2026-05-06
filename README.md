# **MBTI Tune**

Predict your MBTI personality dimensions from your Spotify listening habits.

**Team Members:** Angela and Marwa

---

## **What This Project Does**

- Connects to your Spotify account
- Fetches your top 20 tracks
- Extracts audio features (with fallback to backup dataset or simulated features when Spotify API fails)
- Aggregates 12 audio features into 45 statistical features (means, standard deviations, key/mode counts)
- Predicts 16 MBTI personality types using a PyTorch neural network
- Fetches lyrics via LRCLIB API
- Summarizes lyrics using **Gemini AI**
- Generates a personalized psychological breakdown with SHAP explainability

---

## **Datasets Used**

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| **Spotify Tracks Dataset** | 114,000 tracks | [Kaggle - Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | Autoencoder pretraining on 9 audio features |
| **Raw MBTI Playlists (Folder)** | 324 playlists (16 folders, ~20 each) | Crowdsourced Spotify playlists | LSTM autoencoder training for playlist embeddings |
| **Aggregated MBTI Dataset** | 4,081 playlists | Public MBTI dataset (Kaggle) | Final classifier training (45 features) |
| **Spotify 1M Songs (Backup)** | ~1,000,000 songs | External dataset | Fallback when Spotify API fails |

---

## **Notebooks Documentation**

### **1. `MBTI_Tracks_Autoencoder.ipynb`**

**Purpose:** Train a neural autoencoder to compress 9 audio features into 32-dimensional song embeddings.

**Input Dataset:** `spotify_tracks.csv` (114,000 tracks, 9 audio features)
- Features: danceability, energy, valence, acousticness, instrumentalness, speechiness, loudness, tempo, liveness

**Architecture:**
```
Encoder: 9 → 128 → 64 → 32
Decoder: 32 → 64 → 128 → 9
```
- BatchNorm1d after each hidden layer
- ReLU activation
- MSELoss for reconstruction

**Output Files:**
| File | Description |
|------|-------------|
| `song_dataset_clean.csv` | Standardized audio features (9 columns) |
| `song_scaler.pkl` | StandardScaler for normalizing new songs |
| `song_embeddings.npy` | 32-dim embeddings (114,000 × 32) |

**Training Results:**
- Best validation MSE: 0.0024
- Global MAE: 0.037 (3.7% error per feature)

> **Note:** These song embeddings were explored but **not used** in the final pipeline. The classifier uses aggregated statistics instead.

---

### **2. `MBTI_Playlist_FineTune_Encoder.ipynb`**

**Purpose:** Train an LSTM autoencoder to compress variable-length playlists into 64-dimensional playlist embeddings.

**Input Dataset:** `raw_playlists/` folder (16 MBTI subfolders, each with ~20 CSV files)
- Each CSV contains 12 audio features per song in a playlist
- Features: BPM, Energy, Popularity, Dance, Acoustic, Instrumental, Valence, Speech, Live, Loud (Db), Time Signature, #

**Architecture:**
```
Encoder: LSTM(12 → 128) → Linear(128 → 64)
Decoder: LSTM(64 → 128) → Linear(128 → 12)
```
- Batch-first LSTM with packed sequences (handles variable lengths)
- Masked MSE loss to ignore padding

**Output Files:**
| File | Description |
|------|-------------|
| `playlist_embeddings.npy` | 64-dim embeddings (324 × 64) |
| `playlist_metadata.csv` | Mapping: playlist_id | mbti |
| `playlist_encoder.pth` | LSTM model weights |

**Training Results:**
- Best validation MSE: 1.19
- Global MAE: 0.64 (64% reconstruction error)

> **Note:** The 324 playlists were insufficient for robust training. This notebook is **exploratory only** - final classifier uses the larger aggregated dataset instead.

---

### **3. `MBTI_Playlist_Classifier.ipynb`**

**Purpose:** Train a multi-class neural network to predict MBTI types from aggregated playlist statistics.

**Input Dataset:** `spotify-mbti-playlists.csv` (4,081 playlists, 45 features)
- 9 audio features × 2 (mean + std) = 18 features
- 24 key/mode count features (C Major, C# Major, C minor, etc.)
- 1 track_count feature
- Target: 16 MBTI types

**Architecture:**
```
MLP: 45 → 128 (ReLU + Dropout 0.2) → 128 (ReLU + Dropout 0.2) → 16
```

**Data Split:** 70% train / 15% validation / 15% test (stratified by MBTI)

**Output Files:**
| File | Description |
|------|-------------|
| `mbti_classifier.pth` | Trained model + idx_to_type mapping |
| `mbti_scaler.pkl` | StandardScaler for 45 features |
| `mbti_features.json` | List of 45 feature names in order |
| `idx_to_type.json` | Index to MBTI type mapping |

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| Test Accuracy | ~40-50% (16 classes) |
| Global MAE | ~0.35 (35% error per dimension) |
| Best Validation Loss | ~1.19 |

**SHAP Explainability:** Integrated to show which audio features most influence each MBTI dimension.

---

## **Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT APP (app.py)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SPOTIFY OAUTH (spotify_utils.py)                    │
│                     Fetches user's top 20 tracks                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUDIO FEATURES EXTRACTION (spotify_utils.py)             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ Spotify API  │───▶│ Backup CSV   │───▶│ Simulated    │                  │
│  │ (deprecated) │    │ (1M songs)   │    │ (fallback)   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE AGGREGATION (spotify_utils.py)                   │
│         20 tracks × 12 features → 45 aggregated features                    │
│         (means + stds + key/mode counts + track_count)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCALING + PREDICTION (inference.py)                      │
│         mbti_scaler.pkl → MBTIClassifier → 16 class probabilities          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AXIS AGGREGATION (inference.py)                          │
│         16 classes → 4 dimensions (E/I, S/N, T/F, J/P)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LYRICS FETCHING (lyrics_utils.py)                        │
│         LRCLIB API → Gemini summary of lyrics themes                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEMINI ANALYSIS (gemini_utils.py)                        │
│         Combines MBTI + percentages + artists + lyrics → personality breakdown│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## **Model Performance (Final Classifier)**

| Dimension | Interpretation |
|-----------|----------------|
| **E/I** | Extraversion vs Introversion |
| **S/N** | Sensing vs Intuition |
| **T/F** | Thinking vs Feeling |
| **J/P** | Judging vs Perceiving |

**Note:** The classifier predicts 16 discrete types, then aggregates to dimension percentages.

---

## **Tools & Versions**

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Main language |
| PyTorch | 2.0+ | Deep learning framework |
| NumPy | 1.24+ | Numerical operations |
| Pandas | 2.0+ | Data manipulation |
| scikit-learn | 1.3+ | Scaling, train/test split |
| Streamlit | 1.28+ | Web application |
| Spotipy | 2.23+ | Spotify API wrapper |
| Google Gemini | Latest | LLM for summaries |
| SHAP | 0.43+ | Model explainability |
| Matplotlib/Seaborn | Latest | Visualizations |

---

## **Installation**

### 1. Clone the repo
```bash
git clone https://github.com/kjesinthuyaUCLL/mbti-tune.git
cd mbti-tune
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## **requirements.txt**

```txt
# Core
streamlit==1.32.0
spotipy==2.23.0
pandas==2.2.1
numpy==1.26.4
requests==2.31.0
python-dotenv==1.0.1

# AI/ML
torch==2.2.0
torchvision==0.17.0
scikit-learn==1.4.2
joblib==1.3.2

# Gemini/LLM
google-generativeai==0.5.2

# Visualization & Explainability
matplotlib==3.8.3
seaborn==0.13.2
shap==0.43.0

# Utilities
protobuf==4.25.3
```

---

## **Environment Variables (.env)**

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8501
GOOGLE_API_KEY=your_gemini_api_key
```

---

## **Running the App**

```bash
streamlit run app/app.py
```

Then open http://localhost:8501

---

## **Project Structure**

```
mbti-tune/
├── app/
│   └── app.py                    # Streamlit web application
├── src/
│   ├── model.py                  # Neural network architectures
│   ├── inference.py              # Model loading + prediction
│   ├── spotify_utils.py          # Spotify API + feature extraction
│   ├── lyrics_utils.py           # Lyrics fetching + summarization
│   └── gemini_utils.py           # Gemini personality analysis
├── data/
│   ├── raw/
│   │   ├── spotify_data.csv      # Backup dataset (1M songs)
│   │   └── raw_playlists/        # Original playlist CSVs (16 folders)
│   └── processed/
│       ├── mbti_classifier.pth   # Trained classifier
│       ├── mbti_scaler.pkl       # 45-feature scaler
│       ├── mbti_features.json    # Feature names
│       ├── idx_to_type.json      # MBTI label mapping
│       └── song_scaler.pkl       # Song-level scaler (from Notebook 1)
├── notebooks/
│   ├── MBTI_Tracks_Autoencoder.ipynb
│   ├── MBTI_Playlist_FineTune_Encoder.ipynb
│   └── MBTI_Playlist_Classifier.ipynb
├── scripts/
│   └── compress_spotify_dataset.py  # Compress large backup dataset
├── .env                          # API keys (not committed)
├── .gitignore
└── requirements.txt
```

---

## **Fallback Strategy**

The app handles API failures gracefully:

1. **Spotify API** (preferred) - deprecated but tried first
2. **Backup dataset** - 1M songs CSV with precomputed features
3. **Simulated features** - Realistic random values based on track name hash
4. **Lyrics fallback** - LRCLIB → Alternative API → Gemini guess

---

## **Known Limitations**

- Spotify audio features API is **deprecated** (403 errors). The app relies on backup dataset + simulation.
- Limited playlist data for LSTM autoencoder training (324 playlists vs 4,081 for classifier).
- Chinese/non-English tracks may have limited lyrics availability.
- Gemini free tier has rate limits.