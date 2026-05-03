# MBTI Tune

Predict your MBTI personality dimensions from your Spotify listening habits.

**Team Members:** Angela and Marwa

---

## What This Project Does

- Connects to your Spotify account
- Analyzes your top 20 tracks
- Predicts 4 MBTI dimension percentages (E%, N%, T%, J%)
- Shows a fun personality description using Gemini AI

---

## Datasets Used

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| Song-level playlists | 32,367 songs | Manually collected from 16 MBTI folders | Autoencoder pretraining |
| Playlist-level aggregated | 4,201 playlists | Downloaded pre-processed dataset | Classifier training |
| Unlabeled songs | 113,000 songs | External dataset | Not used |

---

## Project Progress

| Step | What We Did | Output |
|------|-------------|--------|
| 1 | Collected 16 personality folders with Spotify playlist CSVs | Raw data |
| 2 | Combined songs per personality, added song frequency | Processed data |
| 3 | Removed duplicate songs across personalities | 32,367 unique songs |
| 4 | Used playlist-level aggregated dataset | 4,201 playlists with 45 features |
| 5 | Trained autoencoder on songs (unsupervised) | Learned music patterns |
| 6 | Trained classifier on playlists (supervised) | MBTI prediction model |
| 7 | Trained song-level classifier for comparison | Proved playlist aggregation is better |
| 8 | Tested both models locally | Ready for deployment |

---

## Why Two Models

| Model | What It Does | Data Used |
|-------|--------------|-----------|
| Autoencoder | Learns general music patterns | 32,367 songs (no labels) |
| Playlist Classifier | Predicts MBTI dimensions | 4,201 playlists (with MBTI labels) |
| Song Classifier | Individual song predictions (for comparison) | 32,367 songs (with labels) |

This approach is called **Transfer Learning**.

---

## Model Performance Comparison

| Model | Overall MAE | Letter Accuracy |
|-------|-------------|-----------------|
| Playlist Classifier | 37.7% | 68.1% |
| Song Classifier | 45.5% | 61.4% |

**Conclusion:** Playlist aggregation performs better because personality is revealed through listening patterns, not individual songs.

---

## Tools & Versions

| Tool | Version |
|------|---------|
| Python | 3.9+ |
| PyTorch | 2.1.0 |
| NumPy | 1.24.3 |
| Pandas | 2.0.3 |
| scikit-learn | 1.3.0 |
| Streamlit | 1.28.0 |
| Spotify API | Latest |
| Google Gemini API | 0.4.0 |
| Google Colab | T4 GPU |

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/kjesinthuyaUCLL/mbti-tune.git
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

## requirements.txt

```txt
numpy==1.24.3
pandas==2.0.3
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.0
spotipy==2.23.0
streamlit==1.28.0
google-generativeai==0.4.0
shap==0.43.0
matplotlib==3.7.2
seaborn==0.12.2
python-dotenv==1.0.0
tqdm==4.66.1
joblib==1.3.2
```

---

## Google Colab Training

Three notebooks were used:

| Notebook | Purpose | Output |
|----------|---------|--------|
| `MBTI_Tune_Training.ipynb` | Train autoencoder on 32,367 songs | `autoencoder.pth` |
| `MBTI_Playlist_Training.ipynb` | Train playlist classifier | `playlist_classifier.pt` |
| `MBTI_Song_Classifier.ipynb` | Train song classifier for comparison | `song_classifier.pt` |

### Google Colab Setup

Create a folder named `mbti_tune_data` in Google Drive. Upload your entire `data/` folder inside it.

Run notebooks in this order:
1. `MBTI_Tune_Training.ipynb`
2. `MBTI_Playlist_Training.ipynb`
3. `MBTI_Song_Classifier.ipynb`

---

## Testing the Model

```bash
python scripts/test_performance.py
```

**Expected output:**
```
✅ Model loaded with 54,084 parameters
🎉 SUCCESS! Model ready for Streamlit app
```

---

## Playlist Classifier Performance

| Dimension | MAE | Accuracy (within 15%) | Letter Acc |
|-----------|-----|----------------------|------------|
| E | 33.3% | 34.4% | 72.7% |
| N | 43.3% | 11.4% | 60.5% |
| T | 33.7% | 32.8% | 73.9% |
| J | 40.4% | 18.7% | 65.1% |
| **OVERALL** | **37.7%** | **24.3%** | **68.1%** |

---

## Files for Deployment

| File | Location | Size |
|------|----------|------|
| `playlist_classifier.pt` | `models/` | 223 KB |
| `scaler_new.pkl` | `models/` | 1.5 KB |
| `features.json` | `models/` | 0.8 KB |

---

## Environment Variables

Create a `.env` file for local testing:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
GEMINI_API_KEY=your_gemini_key
```