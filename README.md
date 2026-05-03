# MBTI Tune

Predict your MBTI personality dimensions from your Spotify listening habits.


## Team Members

**Angela and Marwa**

---

## What This Project Does

- Connects to your Spotify account  
- Analyzes your top 20 tracks  
- Predicts 4 MBTI dimension percentages (E%, N%, T%, J%)  
- Shows a fun personality description using Gemini AI  

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
| 7 | Tested model locally | Ready for deployment |

---

## Why Two Models

| Model | What It Does | Data Used |
|-------|--------------|-----------|
| Autoencoder | Learns general music patterns | 32,367 songs (no labels) |
| Classifier | Predicts MBTI dimensions | 4,201 playlists (with MBTI labels) |

This approach is called **Transfer Learning**.

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

Two notebooks were used:

### Notebook 1: `MBTI_Tune_Training.ipynb`

- Trains autoencoder on 32,367 songs  
- Runtime: T4 GPU
- Output: `autoencoder.pth`  

### Notebook 2: `MBTI_Playlist_Training.ipynb`

- Trains classifier on 4,201 playlists  
- Runtime: T4 GPU
- Output: `model_state_dict.pt`, `scaler.pkl`, `features.json`  

---

## Testing the Model

```bash
python scripts/test_model.py
```

**Expected output:**
```
✅ Model loaded with 54,084 parameters
✅ Created new scaler fitted on 4201 samples
🎉 SUCCESS! Model ready for Streamlit app
```

---

## Model Performance

| Dimension | MAE   | Accuracy (within 15%) | Letter Acc |
|-----------|-------|----------------------|------------|
| E         | 33.3% |   34.4%              | 72.7%      |
| N         | 43.3% |   11.4%              | 60.5%      |
| T         | 33.7% |   32.8%              | 73.9%      |
| J         | 40.4% |   18.7%              | 65.1%      |
| OVERALL   | 37.7% |   24.3%              | 68.1%      |

---

## Files for Deployment

| File                | Location | Size   |
|---------------------|----------|--------|
| model_state_dict.pt | models/  | 223 KB |
| scaler_new.pkl      | models/  | 1.5 KB |
| features.json       | models/  | 0.8 KB |

---

## Environment Variables

Create a `.env` file:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
GEMINI_API_KEY=your_gemini_key
```

---
