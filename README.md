# 🎵 MBTI Tune

**Predict your MBTI personality dimensions from your Spotify listening habits**

**Team Members:** Angela and Marwa

---

## What This Project Does

- Connects to your Spotify account
- Fetches your top 20 tracks
- Extracts audio features (danceability, energy, valence, acousticness, instrumentalness, speechiness, loudness, tempo, liveness)
- Aggregates features into 171 statistical features (43 audio stats + 128 transfer learning embeddings)
- Predicts 16 MBTI personality types using a PyTorch neural network with transfer learning
- Fetches lyrics via LRCLIB API (with backup)
- Summarizes lyrics using **Groq AI** (fallback to Gemini)
- Generates a personalized psychological breakdown with SHAP explainability

---

## How It Works: Simple Explanation

1. **You log in with Spotify** - The app asks for permission to see your top tracks
2. **We analyze your music** - Extract audio features like danceability, energy, tempo
3. **AI predicts your personality** - A neural network trained on 4,000+ playlists predicts your MBTI type
4. **We read your lyrics** - Fetch lyrics from your top songs and summarize themes
5. **AI explains the results** - Groq/Gemini writes a personalized personality analysis

---

## Datasets Used

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| **Spotify Tracks Dataset** | 114,000 tracks | Kaggle | Autoencoder pretraining (song embeddings) |
| **Raw MBTI Playlists** | 324 playlists | Crowdsourced | LSTM autoencoder exploration |
| **Aggregated MBTI Dataset** | 4,200 playlists | Public MBTI dataset | Final classifier training (43 audio stats) |
| **Organized Genre Dataset** | ~48,000 songs | Built from 1M songs | Genre classifier exploration (balanced by genre) |

---

## Notebooks Documentation

### 1. `MBTI_Tracks_Autoencoder.ipynb`

**Purpose:** Train an autoencoder to compress songs into 32-number fingerprints.

**Input:** 114,000 songs × 9 audio features

**Architecture:**
```
Encoder: 9 → 128 → 64 → 32
Decoder: 32 → 64 → 128 → 9
```

**Outputs:**
- `song_embeddings.npy` - 32-dim embeddings for each song
- `song_scaler.pkl` - Normalization tool for new songs
- `song_dataset_clean.csv` - Cleaned standardized data

**Result:** 3.7% reconstruction error - good at capturing song characteristics

> **Note:** These embeddings were used for transfer learning in the final classifier.

---

### 2. `MBTI_Playlist_FineTune_Encoder.ipynb`

**Purpose:** Explore LSTM autoencoder on playlist sequences (EXPLORATORY ONLY)

**Input:** 324 playlists with song sequences

**Architecture:** LSTM encoder → 64-dim latent → LSTM decoder

**Outputs:**
- `playlist_embeddings.npy` - 64-dim playlist fingerprints
- `playlist_metadata.csv` - Playlist to MBTI mapping

**Result:** 64% reconstruction error - limited data (324 playlists) made this approach less effective. Not used in final pipeline.

---

### 3. `MBTI_Playlist_Classifier.ipynb`

**Purpose:** Main MBTI classifier using transfer learning from song autoencoder

**Input:** 4,200 playlists × 171 features (43 stats + 128 transfer embeddings)

**Architecture:**
```
Input (171) → 64 (BatchNorm + ReLU + Dropout) → 32 → 16 → 16 outputs
```

**Features:**
- 9 means + 9 standard deviations = 18 features
- 24 key/mode counts (C Major, C minor, etc.)
- 1 track count
- 128 transfer learning embeddings (from song autoencoder)

**Outputs:**
- `mbti_classifier.pth` - Trained model for Streamlit
- `mbti_scaler.pkl` - Feature normalizer
- `mbti_features.json` - List of 171 feature names

**Performance:**
- Test accuracy: 36.77% (16 classes, random baseline is 6.25%)
- E/I axis accuracy: 76.7%
- T/F axis accuracy: 75.0%
- S/N axis accuracy: 69.7%
- Average confidence: 62.7%

**Explainability:** SHAP integrated to show which audio features influence predictions

---

### 4. `MBTI_Genre_Classifier.ipynb` (EXPLORATORY)

**Purpose:** Classify songs into true music genres (ignoring regional labels like "K-pop", "J-pop")

**Input:** Organized dataset (~48,000 songs balanced across genres)

**Features:** 9 audio features (same as autoencoder)

**Architecture:** Random Forest Classifier

**Genres targeted:** Pop, Rock, Electronic, Hip Hop, R&B/Soul, Jazz, Classical, Metal, Latin, Folk/Acoustic, Blues, Country

**Purpose:** Exploratory - not used in final app, but shows how audio features can distinguish true music genres

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT APP (app.py)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SPOTIFY OAUTH (spotify_utils.py)                │
│              Fetches user's top 20 tracks + album art           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              AUDIO FEATURES EXTRACTION (spotify_utils.py)       │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │ Spotify API  │───▶│ Simulated features (fallback)        │   │
│  │ (deprecated) │    │ (realistic random values)            │   │
│  └──────────────┘    └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE AGGREGATION (spotify_utils.py)        │
│         20 tracks → 43 stats + 128 transfer embeddings         │
│                      = 171 total features                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SCALING + PREDICTION (inference.py)            │
│         mbti_scaler.pkl → MBTIClassifier → 16 probabilities    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AXIS AGGREGATION (inference.py)               │
│         16 classes → 4 dimensions (E/I, S/N, T/F, J/P)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LYRICS FETCHING (lyrics_utils.py)              │
│    LRCLIB API → Lyrics.ovh (backup) → Groq summary             │
│         Searches top 20 tracks until finding 3 with lyrics     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AI ANALYSIS (gemini_utils.py)                  │
│         Groq (primary) → Gemini (fallback) → personality       │
│         breakdown combining MBTI + music + lyrics              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Performance (Final Classifier)

| Axis | Accuracy | Meaning |
|------|----------|---------|
| **E/I** (Extraversion vs Introversion) | 76.7% | Very good - music clearly reflects social energy |
| **T/F** (Thinking vs Feeling) | 75.0% | Very good - musical preference correlates with decision style |
| **S/N** (Sensing vs Intuition) | 69.7% | Good - abstract vs concrete thinking reflected in music |
| **J/P** (Judging vs Perceiving) | 65.8% | Moderate - planning style less reflected in music |

**Overall:** 36.77% accuracy on 16 classes (random baseline is 6.25%)

The model learns real patterns - Extraverts prefer higher energy music, Thinkers prefer more complex structures, etc.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Main programming language |
| PyTorch | Neural networks (autoencoder, classifier) |
| scikit-learn | Feature scaling, train/test split |
| Streamlit | Web application framework |
| Spotipy | Spotify API wrapper |
| Groq API | Primary LLM for lyrics summarization (fast, high limits) |
| Google Gemini API | Fallback LLM for personality analysis |
| SHAP | Model explainability (feature importance) |
| LRCLIB API | Primary lyrics source |
| Lyrics.ovh API | Backup lyrics source |
| Matplotlib/Seaborn | Visualizations |

---

## How the Lyrics Search Works

1. Looks through your top 20 tracks (not just first 3)
2. Tries LRCLIB API first (best coverage)
3. Falls back to Lyrics.ovh API
4. Summarizes lyrics using Groq (fast, 14,400 requests/day)
5. Only shows error if NO lyrics found in all 20 tracks

---

## Fallback Strategy

The app handles failures gracefully:

| Component | Primary | Fallback 1 | Fallback 2 |
|-----------|---------|------------|------------|
| **Audio Features** | Spotify API | Simulated features | - |
| **LLM** | Groq | Gemini | Built-in template |
| **Lyrics** | LRCLIB | Lyrics.ovh | AI-generated guess |
| **Album Art** | Spotify API | Gradient placeholder | - |

---

## Environment Variables (.env)

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8501
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

---

## Running the App

```bash
streamlit run app/app.py
```

Then open http://localhost:8501

---

## Project Structure

```
mbti-tune/
├── app/
│   └── app.py                 # Streamlit web app
├── src/
│   ├── model.py               # Neural network architectures
│   ├── inference.py           # Model loading + prediction
│   ├── spotify_utils.py       # Spotify API + feature extraction
│   ├── lyrics_utils.py        # Lyrics fetching + summarization
│   ├── gemini_utils.py        # AI personality analysis
│   └── groq_utils.py          # Groq API integration
├── data/
│   ├── raw/
│   │   ├── mbti_playlists/    # Aggregated stats (4,200 playlists)
│   │   ├── raw_playlists/     # Original playlist CSVs (324)
│   │   └── pretrain/          # Spotify tracks (114,000 songs)
│   └── processed/
│       ├── mbti_classifier.pth    # Trained model
│       ├── mbti_scaler.pkl        # Feature scaler
│       ├── mbti_features.json     # 171 feature names
│       └── song_embeddings.npy    # 32-dim song embeddings
├── notebooks/                  # Jupyter notebooks
│   ├── MBTI_Tracks_Autoencoder.ipynb
│   ├── MBTI_Playlist_FineTune_Encoder.ipynb
│   ├── MBTI_Playlist_Classifier.ipynb
│   └── MBTI_Genre_Classifier.ipynb (exploratory)
├── scripts/                    # Utility scripts
│   ├── diagnose_models.py
│   ├── diagnose_imbalance.py
│   └── analyze_genres.py
├── .env                        # API keys (not committed)
└── .gitignore
```

---

## Known Limitations

| Limitation | Explanation |
|------------|-------------|
| **Spotify API deprecated** | Audio features endpoint returns 403 errors. App uses simulated features as fallback. |
| **Transfer learning complexity** | The 128 embedding features are set to zero for individual users (no pre-computed playlist embeddings). |
| **Limited training data** | 4,200 playlists is small for 16-class classification |
| **Chinese/non-English tracks** | Lyrics availability is limited |
| **Groq/Gemini rate limits** | Free tier has request limits (handled with fallbacks) |

---

## Results Summary

| Aspect | Achievement |
|--------|-------------|
| **E/I prediction** | 76.7% accuracy - Excellent |
| **T/F prediction** | 75.0% accuracy - Excellent |
| **Overall accuracy** | 36.77% - Good (6x better than random) |
| **Transfer learning** | Successfully used song autoencoder → playlist classifier |
| **Fallback handling** | Graceful degradation when APIs fail |
| **Full pipeline** | Spotify → Features → Model → Lyrics → AI analysis |

---

## Acknowledgments

- Spotify for the API
- Kaggle for datasets
- Groq for high-limit free LLM access
- LRCLIB for lyrics API
- SHAP for model explainability