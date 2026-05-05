# **MBTI Tune**

Predict your MBTI personality dimensions from your Spotify listening habits.

**Team Members:** Angela and Marwa

---

## **What This Project Does**

- Connects to your Spotify account  
- Fetches your top 20 tracks  
- Extracts audio features (with fallback if Spotify blocks the API)  
- Predicts 4 MBTI dimension percentages: **E%, N%, T%, J%**  
- Fetches lyrics via LRCLIB  
- Translates + summarizes lyrics using **Gemini AI**  
- Generates a personalized psychological breakdown  

---

## **Datasets Used (Updated)**

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| **Song-level dataset** | 113,000 songs | External dataset | Autoencoder pretraining |
| **MBTI playlist dataset** | 4,816 playlists | Public MBTI dataset | Classifier training |
| **User top tracks** | Live | Spotify API | Real-time prediction |

> Note: The older 32,367‑song dataset and 16‑folder structure were used during early prototyping but are **not part of the final pipeline**.

---

## **Project Progress (Updated)**

| Step | What We Did | Output |
|------|-------------|--------|
| 1 | Cleaned and standardized MBTI playlist dataset | 4,816 playlists |
| 2 | Pretrained autoencoder on 113k songs | `encoder_114k_weights.pth` |
| 3 | Built transfer-learning classifier | `playlist_classifier.pt` |
| 4 | Implemented Spotify OAuth + fallback audio features | Robust live pipeline |
| 5 | Integrated LRCLIB for lyrics | Raw lyrics |
| 6 | Added Gemini translation + summarization | Clean English summaries |
| 7 | Added Gemini psychological breakdown | Personalized insights |
| 8 | Built Streamlit app | Ready for deployment |

---

## **Why This Architecture Works**

| Component | Purpose |
|----------|----------|
| **Autoencoder** | Learns general music patterns (unsupervised) |
| **Encoder (16‑dim latent vector)** | “Music fingerprint” of a playlist |
| **Playlist Classifier** | Predicts MBTI percentages from latent vector |
| **Lyrics Summaries (Gemini)** | Adds interpretability and user engagement |

This is a **transfer learning pipeline**:  
Unsupervised → Supervised → Real-time inference.

---

## **Model Performance (Updated)**

| Metric | Value |
|--------|--------|
| **Overall MAE** | **37.7%** |
| **Letter Accuracy** | **68.1%** |

Playlist-level aggregation performs better than song-level prediction because personality emerges from **listening patterns**, not individual tracks.

---

## **Tools & Versions (Updated)**

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| PyTorch | 2.1.0 |
| NumPy | 1.24.3 |
| Pandas | 2.0.3 |
| scikit-learn | 1.6.1 (required for scaler compatibility) |
| Streamlit | 1.28.0 |
| Spotipy | 2.23.0 |
| Google Gemini API | Latest |
| SHAP | 0.43.0 (explainability planned) |
| Google Colab | T4 GPU |

---

## **Installation**

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

## **requirements.txt (Updated)**

```txt
streamlit==1.32.0
spotipy==2.23.0
pandas==2.2.1
numpy==1.26.4
requests==2.31.0
python-dotenv==1.0.1
langdetect==1.0.9
google-generativeai==0.5.2
protobuf==4.25.3

torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0

scikit-learn==1.4.2
matplotlib==3.8.3
seaborn==0.13.2
```

---

## **Google Colab Training (Updated)**

Three notebooks were used:

| Notebook | Purpose | Output |
|----------|---------|--------|
| `MBTI_Tracks_Autoencoder.ipynb` | Pretrain autoencoder on 113k songs | `encoder_114k_weights.pth` |
| `MBTI_Playlist_Training.ipynb` | Train playlist classifier | `playlist_classifier.pt` |
| `MBTI_Song_Classifier.ipynb` | (Optional) Song-level classifier | `song_classifier.pt` |

### Colab Setup

Create:

```
/content/drive/MyDrive/mbti_tune_data/
```

Upload:
- `data/`
- `models/` (optional)

Run notebooks in order:
1. Autoencoder  
2. Playlist classifier  
3. Song classifier (optional)

---

## **Testing the Model**

```bash
python scripts/test_performance.py
```

Expected output:

```
Model loaded successfully
Ready for Streamlit app
```

---

## **Playlist Classifier Performance (Updated)**

| Dimension | MAE | Letter Accuracy |
|-----------|-----|-----------------|
| E | 33.3% | 72.7% |
| N | 43.3% | 60.5% |
| T | 33.7% | 73.9% |
| J | 40.4% | 65.1% |
| **OVERALL** | **37.7%** | **68.1%** |

---

## **Files for Deployment**

| File | Location | Size |
|------|----------|------|
| `playlist_classifier.pt` | `models/` | 223 KB |
| `encoder_114k_weights.pth` | `models/` | 180 KB |
| `pretrain_scaler.pkl` | `models/` | 1.5 KB |
| `pretrain_features.json` | `models/` | 0.8 KB |

---

## **Environment Variables**

Create a `.env` file:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
GOOGLE_API_KEY=your_gemini_key
```

---
