# Advanced AI Project – MBTI Tune

---

## 1. Deliverables & Grading Rubric

### **Technical Depth (2 pts)**
The project demonstrates advanced AI techniques through:
- A **PyTorch Autoencoder** pretrained on 114,000 songs (unsupervised learning → 32-dim song embeddings) with 3.83% reconstruction error
- **Transfer Learning** from song autoencoder to playlist classifier (128 embedding features aggregated as mean, std, min, max)
- **Multi-class Neural Network** (171 features → 64 → 32 → 16 → 16 MBTI types)
- **SHAP Explainability** for model interpretability (identifying 48 low-variance features and top predictive features)
- **Groq + Gemini LLMs** for lyric summarization and personality analysis

### **Implementation (3 pts)**
- Clean, modular PyTorch code (autoencoder, classifier, training loops with learning rate scheduling)
- Device‑agnostic GPU support (CPU/GPU fallback)
- Streamlit frontend with Spotify OAuth integration and automatic token refresh
- Robust fallback system (Spotify API → 1M song database → Beta distribution simulation)
- Album art display from Spotify API
- Inference stabilization (feature clipping, down-weighting low-variance features, temperature scaling)

### **Analysis & Evaluation (2 pts)**
- Loss curves for autoencoder and classifier training
- Classification metrics (33.12% accuracy on 16 classes, random baseline 6.25%)
- Axis‑specific accuracy (E/I: 76.7%, T/F: 75.0%, S/N: 69.7%, J/P: 65.8%)
- SHAP feature importance plots revealing top 5 predictive features
- Model bias analysis identifying 48 problematic features (key counts, transfer dimensions)

### **Innovation & Creativity (2 pts)**
- Predicting personality from music using compressed "music fingerprints"
- Combining numerical audio features (171 dimensions) with LLM‑based lyric interpretation
- Real‑time Spotify integration with graceful fallback to 100k-song database
- Transfer learning from 114k songs → 4,200 playlists
- Inference stabilization to transform extreme predictions to balanced 45-55% ranges

### **Defense (3 pts)**
- Clear explanation of Autoencoder → Transfer Learning → Classifier pipeline
- Ability to justify design choices (latent dimensions, loss functions, architecture)
- Understanding of Groq/Gemini's role in summarization and psychological interpretation

---

## 2. Theoretical Foundations (Matched to Real Implementation)

### A. Deep Learning & Optimization
- PyTorch models with explicit forward pass, loss, backward, optimizer step
- Adam optimizer with ReduceLROnPlateau scheduling (patience=15, factor=0.5)
- Autoencoder bottleneck (9 → 32 dimensions) for representation learning (3.83% reconstruction error)
- BatchNorm + Dropout (0.3, 0.15) for regularization
- CrossEntropyLoss with class weights to handle imbalance

### B. Transfer Learning
- Pretrained song autoencoder (114k songs) → Playlist classifier (4,200 playlists)
- 128 embedding features transferred via aggregation (mean, std, min, max of 32-dim song vectors)
- Encoder weights frozen during playlist classifier training

### C. NLP (LLM Integration)
- **Groq LLM** (primary): Llama 3.3 70B for fast lyric summarization
- **Google Gemini** (fallback): Personality analysis and psychological interpretation
- Used for:
  - Lyrics theme summarization (20 tracks)
  - Personality breakdown generation based on MBTI + listening history
  - Engaging user-facing analysis

### D. Model Explainability
- **SHAP (SHapley Additive exPlanations)** integrated with KernelExplainer
- Feature importance analysis revealing top predictive features: track_count, acousticness_mean, acousticness_stdev, speechiness_mean, danceability_mean
- Identification of 48 low-variance features causing extreme predictions

### E. Multimodal (Inference-Time)
- Audio features (171-dim) + Text lyrics → Combined personality analysis via LLM prompt
- Not full multimodal training, but inference-time fusion

### F. Reinforcement Learning
- Not used in this project

### G. Fine‑Tuning & Large Models
- Transfer learning applied to the encoder (from song to playlist level)
- No full LLM fine‑tuning
- No LoRA required

---

## 3. AI Techniques Used (Course Requirement Check)

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| **Neural Networks** | Autoencoder, Multi-class Classifier | ✅ |
| **Transfer Learning** | Song autoencoder → Playlist classifier | ✅ |
| **LLM Integration** | Groq (primary) + Gemini (fallback) | ✅ |
| **Multimodal** | Audio features + Lyrics text (combined in LLM prompt) | ✅ |
| **Explainability** | SHAP feature importance with KernelExplainer | ✅ |
| **Realistic Data** | 114k songs + 4,200 playlists + live Spotify API | ✅ |
| **New Technology** | PyTorch, Groq API, SHAP (first-time use) | ✅ |

---

## 4. Project Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW (Training)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  114k songs (9 features)                                            │
│       ↓                                                             │
│  Autoencoder (9 → 128 → 64 → 32 → 64 → 128 → 9)                    │
│       ↓                                                             │
│  3.83% reconstruction error → Encoder weights saved                 │
│       ↓                                                             │
│  Encode 54k raw songs → 32-dim embeddings                          │
│       ↓                                                             │
│  Aggregate to 4,200 playlists (mean, std, min, max = 128 features) │
│       ↓                                                             │
│  Merge with 43 statistical features (means, stds, key counts)      │
│       ↓                                                             │
│  4,200 playlists × 171 features                                     │
│       ↓                                                             │
│  Classifier (171 → 64 → 32 → 16 → 16) → 33.12% MBTI accuracy       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW (Inference)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User logs into Spotify                                             │
│       ↓                                                             │
│  Fetch top 20 tracks                                                │
│       ↓                                                             │
│  Try Spotify API → On failure, use 100k-song database               │
│       ↓                                                             │
│  Extract 9 audio features per track → Aggregate 43 stats           │
│       ↓                                                             │
│  Encode songs via autoencoder → Aggregate 128 embeddings           │
│       ↓                                                             │
│  Apply inference stabilization:                                     │
│    - Clip features to [-3, 3]                                       │
│    - Down-weight 48 low-variance features by 70%                    │
│    - Temperature scaling (temperature=4.0)                          │
│    - Probability smoothing                                          │
│       ↓                                                             │
│  MBTI prediction with balanced axis percentages (45-55% ranges)    │
│       ↓                                                             │
│  Fetch lyrics → Groq summary → Gemini personality analysis         │
│       ↓                                                             │
│  Display results + SHAP explainability                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Model Performance Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Autoencoder Reconstruction** | 3.83% MAE | 96.2% information preserved in 32-dim |
| **Playlist Classifier Accuracy** | 33.12% | 5.3x better than random (6.25%) |
| **E/I Axis** | 76.7% | Very good - music reflects social energy |
| **T/F Axis** | 75.0% | Very good - music reflects decision style |
| **S/N Axis** | 69.7% | Good - abstract vs concrete thinking |
| **J/P Axis** | 65.8% | Moderate - planning style less reflected |
| **Model Calibration** | Perfect | 100% accuracy when confidence >50% |

### SHAP Top 5 Features:
1. **track_count** - Number of songs in playlist (more signal = better prediction)
2. **acousticness_mean** - Average acousticness (acoustic vs electronic)
3. **acousticness_stdev** - Variety in acousticness within playlist
4. **speechiness_mean** - Amount of spoken word content
5. **danceability_mean** - Danceability of music

---

## 6. Fallback Strategy (Robustness)

| Component | Primary | Fallback 1 | Fallback 2 |
|-----------|---------|------------|------------|
| **Audio Features** | Spotify API | 100k-song database (real songs) | Beta distribution simulation |
| **LLM** | Groq (Llama 3.3 70B) | Gemini 2.0 Flash | Template fallback |
| **Lyrics** | LRCLIB | Lyrics.ovh | "No lyrics found" message |
| **Album Art** | Spotify API | Gradient placeholder | - |
| **Transfer Embeddings** | Autoencoder encoder | Zeros (128-dim) | - |

---

## 7. Strict Directives (Followed)

1. ✅ Keep architecture simple and PyTorch‑centric
2. ✅ No multimodal BERT fusion (used inference-time combination only)
3. ✅ No over‑engineering (Docker, LangChain, etc.) - used Streamlit + scripts
4. ✅ Maintain clean training loops and evaluation metrics
5. ✅ SHAP explainability implemented with KernelExplainer

---

## 8. Notebooks & Their Purpose

| Notebook | Purpose | Status |
|----------|---------|--------|
| `MBTI_Tracks_Autoencoder.ipynb` | Pretrain song autoencoder (9→32) on 114k tracks | ✅ Complete |
| `MBTI_Song_Classifier.ipynb` | Song-level MBTI classification (exploratory - failed) | ✅ Complete (not used) |
| `MBTI_Playlist_Classifier.ipynb` | Main MBTI classifier with transfer learning (171 features → 16 types) | ✅ Complete |
| `Genre_Classifier.ipynb` | Genre classification exploration (revealed data quality issues) | ✅ Complete (exploratory) |

---

## 9. Tools & Technologies Used

| Tool | Purpose |
|------|---------|
| **PyTorch** | Neural networks (autoencoder, classifier) |
| **scikit-learn** | Feature scaling, train/test split, StandardScaler |
| **Streamlit** | Web application framework with custom CSS |
| **Spotipy** | Spotify API wrapper (top tracks, audio features, OAuth) |
| **Groq API** | Primary LLM (Llama 3.3 70B) for lyric summarization |
| **Google Gemini API** | Fallback LLM for personality analysis |
| **SHAP** | Model explainability with KernelExplainer |
| **LRCLIB / Lyrics.ovh** | Lyrics APIs |
| **Matplotlib/Seaborn/Plotly** | Visualizations (loss curves, confusion matrix, radar chart) |
| **Joblib** | Model serialization (.pkl files) |

---

## 10. Summary of Achievements

| Requirement | Achievement |
|-------------|-------------|
| **AI-driven solution** | Neural network predicts MBTI from music with 33% accuracy (5.3x random) |
| **Advanced ML** | Autoencoder (3.83% error) + Transfer Learning + SHAP explainability |
| **LLM Integration** | Groq + Gemini for lyric summarization & personality analysis |
| **Multimodal** | Audio features (171-dim) + Lyrics text combined in LLM prompt |
| **Realistic data** | 114k songs, 4,200 playlists, 1M song database, live Spotify API |
| **New technology** | PyTorch, Groq API, SHAP (first-time use in this project) |
| **Robustness** | 3-layer fallback for audio features, dual LLM fallback, inference stabilization |

---
