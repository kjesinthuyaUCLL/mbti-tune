# Advanced AI Project – MBTI Tune

This document defines the operational boundaries and theoretical context for the MBTI Tune project.  
The implementation strictly follows the course requirements and uses only the techniques covered in the curriculum.

---

## 1. Deliverables & Grading Rubric

### **Technical Depth (2 pts)**
The project demonstrates advanced AI techniques through:
- A **PyTorch Autoencoder** pretrained on 113,000 songs (unsupervised learning).
- A **Transfer Learning Classifier** using the pretrained encoder to predict MBTI dimensions.
- A **Gemini‑powered NLP pipeline** for lyric translation, summarization, and psychological interpretation.

### **Implementation (3 pts)**
- Clean, modular PyTorch code (autoencoder, classifier, training loops).
- Device‑agnostic GPU support.
- Streamlit frontend with Spotify OAuth.
- Robust Spotify fallback for audio features.

### **Analysis & Evaluation (2 pts)**
- Loss curves for pretraining and fine‑tuning.
- Regression metrics (MAE, RMSE, R²).
- Interpretation of MBTI dimension outputs.
- **SHAP explainability planned** (not yet implemented).

### **Innovation & Creativity (2 pts)**
- Predicting personality from music using a compressed “music fingerprint”.
- Combining numerical audio features with LLM‑based lyric interpretation.
- Real‑time Spotify integration.

### **Defense (3 pts)**
- Clear explanation of autoencoder → encoder → classifier pipeline.
- Ability to justify design choices (latent dimension, loss functions, transfer learning).
- Understanding of Gemini’s role in summarization and interpretation.

---

## 2. Theoretical Foundations (Matched to Real Implementation)

### A. Deep Learning & Optimization
- PyTorch models with explicit forward pass, loss, backward, optimizer step.
- Adam optimizer with learning rate scheduling.
- Autoencoder bottleneck for representation learning.

### B. NLP (Used Indirectly)
- Gemini LLM used for:
  - Language detection  
  - Translation  
  - Summarization  
  - Psychological interpretation  
- No training of NLP models.

### C. Generative AI
- Gemini used as a generative text model for explanations.

### D. Reinforcement Learning
- Not used in this project.

### E. Fine‑Tuning & Large Models
- Transfer learning applied to the encoder.
- No full LLM fine‑tuning.
- No LoRA required.

---

## 3. Strict Directives
1. Keep architecture simple and PyTorch‑centric.
2. No multimodal BERT fusion (not implemented).
3. No over‑engineering (Docker, LangChain, etc.).
4. Maintain clean training loops and evaluation metrics.
5. SHAP explainability may be added later.
