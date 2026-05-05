# **Technical Implementation Guide for Advanced AI Project**

This document provides strict technical guidelines, preferred libraries, and implementation strategies derived directly from the course’s practical lab exercises.  
**Copilot Instructions:** Use this document to dictate *how* the code should be structured, which libraries to use, and what specific implementation patterns to follow when assisting the student.

---

# **1. Environment & Core Frameworks**

### ✔ Python & Dependencies
- **Python Version:** 3.10 to 3.13  
- **Virtual Environment:** `venv` recommended  
- **Install:** `pip install -r requirements.txt`

### ✔ Deep Learning Framework
- **PyTorch** (GPU‑ready, device‑agnostic)

### ✔ Device Handling (Mandatory)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# All tensors must be moved to device:
tensor = tensor.to(device)
```

---

# **2. Core PyTorch Coding Standards**

### ✔ Architecture Definition
Use **Object‑Oriented PyTorch**:
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    def forward(self, x):
        ...
```

### ✔ Hyperparameter Exposure
All models must expose:
- learning rate  
- batch size  
- number of layers  
- hidden dimensions  
- dropout  
- activation functions  

### ✔ Training Loop Structure (Required)
```python
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### ✔ Visualization (Required)
Use `matplotlib` to plot:
- Autoencoder reconstruction loss  
- Classifier training & validation loss  

### ✔ Data Handling
Use:
```python
from torch.utils.data import Dataset, DataLoader
```

---

# **3. Domain‑Specific Technical Implementations (Updated to Real Project)**

Your project uses **audio features**, **autoencoders**, **transfer learning**, and **LLM‑based summarization**.  
No CV, RL, GANs, or Transformers are used — so they are removed.

---

## **3.1 Audio Feature Processing (Spotify + Fallback)**

### ✔ Real Spotify Features
- Fetch top 20 tracks  
- Extract 49 audio features  
- Aggregate:
  - mean  
  - standard deviation  
  - key/mode counts  

### ✔ Fallback (Required)
If Spotify blocks `audio-features` (403):
- Simulate realistic values  
- Maintain same feature schema  

This ensures the model **always** receives valid input.

---

## **3.2 Autoencoder (Unsupervised Pretraining)**

### ✔ Purpose
Learn a compressed “music fingerprint” from 113,000 songs.

### ✔ Architecture
- Encoder: 49 → 128 → 64 → 16  
- Decoder: 16 → 64 → 128 → 49  

### ✔ Loss
- MSELoss

### ✔ Optimizer
- Adam  
- Learning rate scheduling recommended

---

## **3.3 Transfer Learning Classifier**

### ✔ Input
- 16‑dim latent vector from pretrained encoder

### ✔ Output
- 4 MBTI percentages: **E, N, T, J**

### ✔ Architecture
- MLP head: 16 → 64 → 32 → 4  
- Activation: ReLU  
- Output: Sigmoid (0–1 range)

### ✔ Loss
- MSELoss

### ✔ Evaluation Metrics
- MAE  
- RMSE  
- R²  

---

## **3.4 Lyrics Pipeline (LLM‑Based)**

### ✔ Lyrics Source
- LRCLIB API  
- Filter top 20 → pick first 3 tracks with lyrics  

### ✔ Processing Steps
1. **Language detection**  
2. **Translation to English** (Gemini)  
3. **Summarization** (Gemini)  
4. **Psychological breakdown** (Gemini)

### ✔ Purpose
- Not used for training  
- Used only for **interpretation** in the Streamlit app  

---

## **3.5 Streamlit Application**

### ✔ Required Components
- Spotify OAuth login  
- Display MBTI percentages  
- Display progress bars  
- Show lyric summaries  
- Show Gemini psychological breakdown  

### ✔ Device
- CPU inference is sufficient  
- Model loaded via PyTorch  

---

## **3.6 Explainability (Planned Feature)**

### ✔ SHAP (To Be Added Later)
- KernelExplainer or DeepExplainer  
- Visualizations:
  - Feature importance  
  - Waterfall plots  
  - Summary plots  

---

# **4. Strict “Do Not” Rules for Copilot**

These rules ensure the project stays within course scope.

### ❌ Do NOT use:
- TensorFlow / Keras  
- HuggingFace Transformers  
- BERT / GPT embeddings  
- GANs, VAEs (other than your autoencoder), Diffusion models  
- Reinforcement Learning  
- LangChain, Docker, Kubernetes  
- Any multimodal fusion not implemented  
- Any black‑box code without logging  

### ❌ Do NOT:
- Write monolithic scripts  
- Skip loss/metric logging  
- Skip device handling  
- Skip evaluation metrics  

---

# **5. Summary**

This updated guide now reflects the **exact technologies and methods** used in your real MBTI Tune project:

- PyTorch autoencoder  
- Transfer learning classifier  
- Spotify audio features + fallback  
- LRCLIB lyrics  
- Gemini summarization  
- Streamlit frontend  
- SHAP planned  

