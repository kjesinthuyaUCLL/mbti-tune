# MBTI Tune Project

## Project Overview

An AI-powered web application that predicts MBTI personality types from Spotify listening habits using a PyTorch neural network with transfer learning and LLM-powered lyrical analysis.

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/kjesinthuyaUCLL/mbti-tune.git
cd mbti-tune
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up Environment Variables**
Create a `.env` file in the root directory (see the Environment Variables section below for required keys).

5. **Run the Application**
```bash
streamlit run app/app.py
```

---

## Notebooks

### 1. `MBTI_Tracks_Autoencoder.ipynb`

**Purpose:** Creates 32-dimensional song embeddings from 114,000 tracks using 9 audio features.

**Methodology:**
- Input: 114,000 Spotify tracks with 9 audio features (danceability, energy, valence, acousticness, instrumentalness, speechiness, loudness, tempo, liveness)
- Architecture: Autoencoder with encoder (9→128→64→32) and decoder (32→64→128→9)
- Training: 20 epochs, 90/10 train/val split, MSE loss, Adam optimizer
- Output: 32-dim latent representations that preserve 96.2% of original information

**Results:**
- Validation MSE: 0.00227
- Mean reconstruction error: 3.83%
- Best reconstructed feature: instrumentalness (2.96% error)

**Output Files:**
- `song_embeddings.npy` (114,000 × 32 dimensions)
- `autoencoder_model.pth` (encoder/decoder weights)
- `song_scaler.pkl` (StandardScaler for audio features)
- `song_dataset_clean.csv` (processed dataset)

---

### 2. `MBTI_Song_Classifier.ipynb`

**Purpose:** Attempted to predict MBTI type from individual songs.

**Status:** Exploratory - Not used in production

**Findings:**
- Individual songs have too much variance for reliable personality prediction
- Test accuracy: 11.72% (only 1.87x random baseline)
- Significant class imbalance and prediction bias observed
- This experiment informed the decision to aggregate at playlist level

**Lesson Learned:** Song-level classification is insufficient; playlist aggregation provides more stable predictions.

**Output Files (not used in production):**
- `song_mbti_classifier.pth`
- `song_mbti_scaler.pkl`
- `song_predictions.csv`

---

### 3. `MBTI_Playlist_Classifier.ipynb`

**Purpose:** Aggregates 20 songs to playlist level and predicts final MBTI type. This is the production model powering the Streamlit application.

**Dataset:**
- Source A: `mbti_playlists/` - 4,201 playlists with pre-computed statistics (43 features)
- Source B: `raw_playlists/` - 326 playlists with raw audio data for embedding generation
- Merged dataset: 4,201 playlists × 171 features

**Feature Engineering (171 total):**
- Statistical features (43): means, standard deviations of 9 audio features + key/mode counts + track count
- Transfer learning features (128): aggregated autoencoder embeddings (mean, std, min, max of 32-dim song vectors)

**Model Architecture:**
```
Input (171) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(16) → BatchNorm → ReLU → Dropout(0.15)
           → Linear(16) → Output
```

**Training:**
- Loss: CrossEntropyLoss with class weights
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=15)
- Epochs: 200 with early stopping
- Batch size: 16

**Results:**
- Test accuracy: 33.12% (5.3x random baseline of 6.25%)
- Perfect calibration: when confidence >50%, accuracy is 100%

**Axis Performance:**
| Axis | Accuracy | Interpretation |
|------|----------|----------------|
| E/I (Extraversion vs Introversion) | 76.7% | Strong correlation with music taste |
| T/F (Thinking vs Feeling) | 75.0% | Strong correlation |
| S/N (Sensing vs Intuition) | 69.7% | Moderate correlation |
| J/P (Judging vs Perceiving) | 67.2% | Weakest correlation |

**SHAP Analysis - Top 5 Features:**
1. `track_count` - Number of songs in playlist (provides more signal)
2. `acousticness_mean` - Average acousticness (distinguishes acoustic vs electronic)
3. `acousticness_stdev` - Variety in acousticness within playlist
4. `speechiness_mean` - Amount of spoken word content
5. `danceability_mean` - Danceability of music

**Output Files (used in Streamlit):**
- `mbti_classifier.pth` (model weights)
- `mbti_scaler.pkl` (feature scaler)
- `mbti_features.json` (feature name list)
- `mbti_model_simplified.pth` (simplified version)

---

### 4. `Genre_Classifier.ipynb`

**Purpose:** Exploratory analysis of genre classification using audio features.

**Status:** Exploratory - Not used in MBTI prediction pipeline

**Dataset:** 14,136 songs, 24 genres (artificially balanced at 589 songs each)

**Results:**
- Test accuracy: 26.91% (6.4x random baseline)
- Best genre: Classical (57% F1-score)
- Worst genre: Rock (3.6% F1-score)

**Critical Discovery - Data Quality Issue:**
Regional pop genres (k-pop, j-pop, cantopop) are musically inconsistent:
- K-POP: only 43% classified as k-pop, 20% as country
- J-POP: 31% classified as country, 15% as power-pop
- CANTOPOP: 54% cantopop, 25% country

**Implication:** Genre labels in training data reflect cultural/linguistic origin, not consistent audio characteristics. This explains challenges in song-level classification.

**Output Files (not used in production):**
- `genre_classifier.pkl` (Random Forest model)
- `genre_scaler.pkl` (feature scaler)
- `regional_genre_mapping.json` (genre confusion mapping)

---

## Complete Data Pipeline

```
Phase 1: Autoencoder Pretraining (Notebook 1)
114,000 raw tracks (9 audio features)
    ↓ StandardScaler
Standardized features
    ↓ Train autoencoder (9→128→64→32→64→128→9)
32-dim song embeddings
    ↓
autoencoder_model.pth, song_embeddings.npy, song_scaler.pkl

Phase 2: Playlist Classification (Notebook 3)
Two data sources:

Source A: mbti_playlists/ (4,201 playlists)
    └── 43 statistical features

Source B: raw_playlists/ (326 playlists with raw songs)
    └── Generate 32-dim embeddings via encoder
    └── Aggregate to 128 features (mean, std, min, max)

Merge: 4,201 playlists × 171 features
    └── 678 playlists have real transfer features
    └── 3,523 playlists have zeros for transfer features

Train classifier: 171 → 64 → 32 → 16 → 16
    ↓
mbti_classifier.pth, mbti_scaler.pkl, mbti_features.json

Phase 3: Streamlit Application
User Spotify login → Top 20 tracks → Feature extraction
    ↓
Apply inference stabilization (clipping + down-weighting + smoothing)
    ↓
MBTI prediction with axis confidence scores
    ↓
Display results + audio radar chart + lyrical analysis
```

---

## Inference Stabilization (Post-Training Fix)

**Problem Identified:** 48 out of 171 features had near-zero variance in training data (key counts, some transfer embeddings). During inference, any non-zero value in these features caused extreme predictions (100% on specific axes).

**Solution Applied (in `inference.py`):**

1. **Feature Clipping:** All scaled features clipped to range [-3, 3]
   - Prevents any single feature from dominating predictions

2. **Down-weighting Low-Variance Features:** Features with training standard deviation < 0.15 have their influence reduced by 70%
   - Identified 48 problematic features including key counts and some transfer dimensions

3. **Probability Smoothing:** Adds 0.5% minimum probability to each MBTI type
   - Prevents 0% or 100% predictions

4. **Temperature Scaling:** Logits divided by temperature=4.0 before softmax
   - Softens overconfident predictions

**Result:** Predictions transformed from 100% extremes to balanced 45-55% ranges while preserving model accuracy.

---

## Fallback Song Database

**Purpose:** Provides realistic audio features when Spotify API fails (rate limits, token expiration, network issues).

**Source:** 1 million song Spotify dataset processed into 100,000 song sample

**Processing:**
- Loads CSV with 1,159,765 songs
- Extracts 9 audio features
- Fits StandardScaler on full dataset
- Saves 100,000 random sample for fast inference

**Database Contents:**
- `sample_features_100k.npy`: 100,000 × 9 scaled audio features
- `sample_metadata_100k.parquet`: Track names, artists, genres
- `song_scaler_1m.pkl`: Scaler fitted on full 1M dataset

**Usage in `spotify_utils.py`:**
1. Attempt Spotify API call
2. If API fails → Select deterministic song from database based on track name hash
3. Return real audio features from similar genre
4. Fallback to Beta distribution if database unavailable

**Benefits:**
- Deterministic (same track always gets same simulated features)
- Realistic feature distributions matching actual music
- App never fails due to API issues

---

## Streamlit Application (`app/app.py`)

**Features:**
- Spotify OAuth login with automatic token refresh
- Fetches user's top 20 tracks (last 6 months)
- Extracts 9 audio features per track (API or database fallback)
- Generates 128 transfer embeddings via autoencoder
- Predicts MBTI type with axis-wise confidence scores
- Displays interactive radar chart for audio features
- Shows top 5 tracks with album art
- Fetches lyrics via LRCLIB API
- Summarizes lyrics using Groq (fallback to Gemini)
- Generates psychological personality profile using Gemini API

**Technical Stack:**
- Frontend: Streamlit with custom CSS
- ML Framework: PyTorch, scikit-learn
- APIs: Spotify (Spotipy), Google Gemini, Groq
- Data: Pandas, NumPy, Plotly
- Explainability: SHAP

---

## Environment Variables (`.env`)

```
# Spotify credentials - BOTH naming conventions
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8501

# Spotipy library expects these names (add these lines)
SPOTIPY_CLIENT_ID=your_spotipy_client_id
SPOTIPY_CLIENT_SECRET=your_spotipy_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8501

# Google Gemini API
GOOGLE_API_KEY=your_google_api_key

# Groq API
GROQ_API_KEY=your_groq_api_key
```

---

## Model Performance Summary

| Model | Accuracy | vs Random | Notes |
|-------|----------|-----------|-------|
| Autoencoder | 96.2% reconstruction | N/A | Preserves audio information in 32-dim space |
| Song Classifier | 11.72% | 1.87x | Exploratory - not used in production |
| Playlist Classifier | 33.12% | 5.3x | Production model with perfect calibration |
| Genre Classifier | 26.91% | 6.4x | Exploratory - data quality analysis only |

---

## Key Technical Contributions

1. **Transfer Learning Pipeline:** Autoencoder pretrained on 114k unlabeled tracks, then encoder used for playlist classification
2. **Playlist Aggregation:** Demonstrated that individual songs have too much variance for personality prediction; playlist-level aggregation (20 songs) achieves 33% accuracy
3. **Inference Stabilization:** Identified 48 low-variance features causing extreme predictions; implemented clipping, down-weighting, and smoothing for balanced outputs
4. **Fallback Database:** Built 100k song database from 1M dataset for API failure recovery
5. **LLM Integration:** Groq/Gemini for lyrical analysis and psychological profiling
6. **Model Explainability:** SHAP analysis revealing top predictive features

---

## Known Limitations

1. **Data Labeling:** MBTI labels come from playlist creators' assumptions, not actual personality tests
2. **Transfer Learning Underutilized:** Only 16% of training playlists have autoencoder embeddings due to dataset mismatch; primary accuracy comes from statistical features
3. **Small Playlist Dataset:** 4,200 playlists for 16-class classification limits generalization
4. **Regional Genre Noise:** Training data contains ambiguous regional genre labels with inconsistent audio characteristics
5. **Lyrics Dependency:** Lyrics fetching from LRCLIB can be slow (5-15 seconds) and may fail for some tracks

---

## Future Improvements

1. Align datasets to provide autoencoder embeddings for 100% of training playlists
2. Collect user MBTI self-reports for ground truth labels
3. Expand dataset with more playlists for better generalization
4. Implement parallel lyrics fetching to reduce wait times
5. Fine-tune LLM on music psychology literature for better personality analysis