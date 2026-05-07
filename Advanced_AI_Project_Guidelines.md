# Advanced AI Project – MBTI Tune

This document defines the operational boundaries and theoretical context for the MBTI Tune project.  
The implementation strictly follows the course requirements and uses only the techniques covered in the curriculum.

---

## 1. Deliverables & Grading Rubric

### **Technical Depth (2 pts)**
The project demonstrates advanced AI techniques through:
- A **PyTorch Autoencoder** pretrained on 114,000 songs (unsupervised learning → 32-dim song embeddings)
- **Transfer Learning** from song autoencoder to playlist classifier (128 embedding features)
- **LSTM Autoencoder** for playlist sequence processing (exploratory)
- **Multi-class Neural Network** (171 features → 64 → 32 → 16 → 16 MBTI types)
- **SHAP Explainability** for model interpretability
- **Groq + Gemini LLMs** for lyric summarization and personality analysis

### **Implementation (3 pts)**
- Clean, modular PyTorch code (autoencoder, LSTM encoder, classifier, training loops)
- Device‑agnostic GPU support (CPU/GPU fallback)
- Streamlit frontend with Spotify OAuth integration
- Robust fallback system (Spotify API → Simulated features | Groq → Gemini → Template)
- Album art display from Spotify API

### **Analysis & Evaluation (2 pts)**
- Loss curves for autoencoder and classifier training
- Classification metrics (36.77% accuracy on 16 classes, random baseline 6.25%)
- Axis‑specific accuracy (E/I: 76.7%, T/F: 75.0%, S/N: 69.7%, J/P: 65.8%)
- SHAP feature importance plots
- Model bias analysis and recommendations

### **Innovation & Creativity (2 pts)**
- Predicting personality from music using compressed "music fingerprints"
- Combining numerical audio features (171 dimensions) with LLM‑based lyric interpretation
- Real‑time Spotify integration with graceful fallback handling
- Transfer learning from 114k songs → 4,200 playlists

### **Defense (3 pts)**
- Clear explanation of Autoencoder → Transfer Learning → Classifier pipeline
- Ability to justify design choices (latent dimensions, loss functions, architecture)
- Understanding of Groq/Gemini's role in summarization and psychological interpretation

---

## 2. Theoretical Foundations (Matched to Real Implementation)

### A. Deep Learning & Optimization
- PyTorch models with explicit forward pass, loss, backward, optimizer step
- Adam optimizer with ReduceLROnPlateau scheduling
- Autoencoder bottleneck (9 → 32 dimensions) for representation learning
- LSTM for sequence processing (exploratory)
- BatchNorm + Dropout for regularization

### B. Transfer Learning
- Pretrained song autoencoder (114k songs) → Playlist classifier (4,200 playlists)
- 128 embedding features transferred from song-level to playlist-level
- Encoder weights frozen, classifier head trainable

### C. NLP (LLM Integration)
- **Groq LLM** (primary): Llama 3.3 70B for fast lyric summarization (14,400 req/day limit)
- **Google Gemini** (fallback): Personality analysis and psychological interpretation
- Used for:
  - Lyrics theme summarization
  - Personality breakdown generation
  - Engaging user-facing analysis

### D. Model Explainability
- **SHAP (SHapley Additive exPlanations)** integrated
- Feature importance analysis for model transparency
- KernelExplainer for neural network interpretation

### E. Multimodal (Planned, Light)
- Audio features + Text lyrics → Combined personality analysis
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
| **Neural Networks** | Autoencoder, LSTM, Multi-class Classifier | ✅ |
| **Transfer Learning** | Song autoencoder → Playlist classifier | ✅ |
| **LLM Integration** | Groq (primary) + Gemini (fallback) | ✅ |
| **Multimodal** | Audio features + Lyrics text (combined in prompt) | ✅ |
| **Explainability** | SHAP feature importance | ✅ |
| **Realistic Data** | 114k songs + 4,200 playlists + Spotify API | ✅ |
| **New Technology** | PyTorch, Groq API, SHAP (first time) | ✅ |

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
│  Encoder weights → Transfer Learning                                │
│       ↓                                                             │
│  4,200 playlists (43 stats + 128 transferred = 171 features)       │
│       ↓                                                             │
│  Classifier (171 → 64 → 32 → 16) → MBTI prediction                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW (Inference)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User logs into Spotify                                             │
│       ↓                                                             │
│  Fetch top 20 tracks (Spotify API → Simulated fallback)            │
│       ↓                                                             │
│  Extract 9 audio features per track → Aggregate 43 stats           │
│       ↓                                                             │
│  Add 128 zero embeddings (no playlist context for individual)      │
│       ↓                                                             │
│  MBTI Classifier → 16 probabilities → 4 axis percentages           │
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
| **Autoencoder Reconstruction** | 3.7% MAE | Excellent song representation |
| **Playlist Classifier Accuracy** | 36.77% | 6x better than random (6.25%) |
| **E/I Axis** | 76.7% | Very good - music reflects social energy |
| **T/F Axis** | 75.0% | Very good - music reflects decision style |
| **S/N Axis** | 69.7% | Good - abstract vs concrete thinking |
| **J/P Axis** | 65.8% | Moderate - planning style less reflected |

---

## 6. Fallback Strategy (Robustness)

| Component | Primary | Fallback 1 | Fallback 2 |
|-----------|---------|------------|------------|
| **Audio Features** | Spotify API | Simulated features (deterministic) | - |
| **LLM** | Groq (Llama 3.3 70B) | Gemini 2.0 Flash | Template fallback |
| **Lyrics** | LRCLIB | Lyrics.ovh | AI‑generated guess |
| **Album Art** | Spotify API | Gradient placeholder | - |
| **Features** | 171 with transfer | 43 stats only | - |

---

## 7. Strict Directives (Followed)

1. ✅ Keep architecture simple and PyTorch‑centric
2. ✅ No multimodal BERT fusion (used inference-time combination only)
3. ✅ No over‑engineering (Docker, LangChain, etc.) - used Streamlit + scripts
4. ✅ Maintain clean training loops and evaluation metrics
5. ✅ SHAP explainability implemented

---

## 8. Notebooks & Their Purpose

| Notebook | Purpose | Status |
|----------|---------|--------|
| `MBTI_Tracks_Autoencoder.ipynb` | Pretrain song autoencoder (9→32) | ✅ Complete |
| `MBTI_Playlist_FineTune_Encoder.ipynb` | LSTM playlist encoder (exploratory) | ✅ Complete |
| `MBTI_Playlist_Classifier.ipynb` | Main MBTI classifier + transfer learning | ✅ Complete |
| `MBTI_Genre_Classifier.ipynb` | Genre classification exploration | ✅ Complete (exploratory) |

---

## 9. Tools & Technologies Used

| Tool | Purpose |
|------|---------|
| **PyTorch** | Neural networks (autoencoder, LSTM, classifier) |
| **scikit-learn** | Feature scaling, train/test split |
| **Streamlit** | Web application framework |
| **Spotipy** | Spotify API wrapper |
| **Groq API** | Primary LLM (Llama 3.3 70B, 14,400 req/day) |
| **Google Gemini API** | Fallback LLM for personality analysis |
| **SHAP** | Model explainability (feature importance) |
| **LRCLIB / Lyrics.ovh** | Lyrics APIs |
| **Matplotlib/Seaborn** | Visualizations |

---

## 10. Summary of Achievements

| Requirement | Achievement |
|-------------|-------------|
| **AI-driven solution** | Neural network predicts MBTI from music |
| **Advanced ML** | Autoencoder + Transfer Learning + LSTM + SHAP |
| **LLM Integration** | Groq + Gemini for lyric summarization & personality |
| **Multimodal** | Audio features + Lyrics text combined |
| **Realistic data** | 114k songs, 4,200 playlists, live Spotify API |
| **New technology** | PyTorch, Groq API, SHAP (first-time use) |

---

**Project successfully meets all course requirements with a complete end‑to‑end pipeline from data processing to deployed Streamlit application.** 🚀