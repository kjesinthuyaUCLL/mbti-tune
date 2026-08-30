# MBTI Tune

A web application that predicts MBTI personality types from Spotify listening habits. Users log in with Spotify, and the app extracts audio features from their top tracks, runs them through a trained neural network, and generates a psychological profile combining model predictions with lyric semantics.

---

## Setup

```bash
git clone https://github.com/kjesinthuyaUCLL/mbti-tune.git
cd mbti-tune
pip install -r requirements.txt
```

Create a `.env` file in the root:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

Run:

```bash
streamlit run app.py
```

---

## Model Architecture

The final model uses **Polynomial Feature Engineering** combined with a **Feedforward Neural Network (FNN)**.

From the original 43 audio features (means, standard deviations, key/mode distributions extracted from a user's top 20 tracks), degree-2 interaction terms are generated — i.e., pairwise products like `energy × danceability`, `tempo × valence` — expanding the feature space to **1,035 dimensions**. These are standardized and passed through a 4-layer FNN.

```
Input (1035)
  → Linear(256) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(128) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(64)  + BatchNorm + ReLU + Dropout(0.3)
  → Linear(16)  [output: 16 MBTI classes]
```

Training used Adam (lr=1e-3, weight_decay=1e-4) with CrossEntropyLoss on the SMOTE-balanced dataset over 100 epochs.

---

## Results

| Model | Macro F1 | Notes |
|---|---|---|
| Random baseline | 6.25% | 16-class uniform random |
| Best classical ML (Random Forest) | 31.7% | SMOTE-Tomek, all features |
| Baseline FNN (43 features, SMOTE) | 35.1% | No polynomial expansion |
| **Polynomial FNN (final)** | **41.1%** | 1,035 features, SMOTE |

Per-axis binary accuracy (random baseline = 50%):

| Axis | Accuracy |
|---|---|
| E/I (Extraversion / Introversion) | 74.9% |
| T/F (Thinking / Feeling) | 74.8% |
| S/N (Sensing / Intuition) | 63.5% |
| J/P (Judging / Perceiving) | 62.4% |

---

## Research Progression

The model went through several iterations before reaching the final architecture:

**1. Data balancing — VAE vs SMOTE**
An initial attempt used a Variational Autoencoder to generate synthetic playlists and balance the dataset. Probabilistic generation on tabular statistics produced blurry feature distributions, and Macro F1 dropped to 25.3%. SMOTE was adopted instead, preserving geometric feature relationships and raising the baseline to 35.1%.

**2. Autoencoder removed from pipeline**
An autoencoder was trained on audio features to produce compact song embeddings for transfer learning. Empirical tests showed it was compressing personality-relevant variance rather than preserving it. The encoder was removed and the pipeline reverted to raw statistical features.

**3. Multimodal learning — lyrics + audio**
An ablation study explored combining audio features with lyrics embeddings (VADER sentiment) and genre embeddings (Sentence-BERT). The best multimodal configuration reached 27.1% F1. Spotify and lyrics API rate limits prevented scaling the approach to the full training set.

**4. Polynomial Feature Engineering**
With external data inaccessible at scale, we applied degree-2 polynomial interaction terms to the existing 43 features. This expanded the feature set to 1,035 dimensions and improved Macro F1 to 41.1%.

---

## Inference Pipeline

1. User authenticates via Spotify OAuth
2. Top 20 tracks (last 6 months) are fetched; audio features are pulled from the Spotify API
3. If the API is unavailable (rate limits, token expiry), features are approximated from a 100k-song fallback database using deterministic hashing on track name
4. Features are aggregated into a 43-dimensional playlist vector
5. `PolynomialFeatures` expands this to 1,035 dimensions; `StandardScaler` normalizes
6. The FNN outputs logits for 16 MBTI classes; temperature scaling (T=4.0) softens overconfident predictions
7. Lyrics are fetched via LRCLIB and summarized using Groq (Mixtral) or Gemini as fallback
8. A full psychological profile is generated via LLM, combining model predictions with lyric themes

---

## Limitations

- MBTI labels are self-assigned by playlist creators, not validated by psychometric tests
- The dataset contains ~4,200 playlists across 16 classes, which limits generalization on minority types (ESTJ, ESFJ)
- The J/P axis shows the weakest correlation with audio features; it would benefit most from lyric-based signals
- Spotify's audio features API is rate-limited, affecting inference reliability in production

---

## Project Structure

```
retake_notebooks/     research notebooks (numbered, with status headers)
OLD_notebooks/        earlier versions kept for reference
scripts/              training and evaluation scripts
src/                  inference, model, Spotify/lyrics/LLM utilities
data/                 raw playlists, processed datasets, song database
models/               trained model weights and transformers
evaluation/           metrics, plots, ablation results
app.py                Streamlit application entry point
```