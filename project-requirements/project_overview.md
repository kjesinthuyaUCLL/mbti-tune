# **Comprehensive Project Overview & Rubric Alignment: MBTI Tune (Updated)**

## **1. Course Project Requirements Overview**
* **Project Duration:** ~60 hours per student.  
* **Team Size:** Individual or Pair (Angela and Marwa).  
* **Core Scope:** Build an AI‑driven solution using advanced machine learning techniques beyond basic supervised learning.  
* **Technology Mandate:** Must use at least one advanced technology (e.g., LLMs, Computer Vision, Reinforcement Learning, Multimodal models, Image Generation) and a new‑to‑you technology (e.g., PyTorch).  
* **Data:** Must use realistic, real‑world data.  
* **Deliverables:**  
  * **GitHub Repository** with clean code and documentation  
  * **Technical Report (~2 pages)**  
  * **15‑minute Oral Defense**  

---

## **2. Project Status Update: MBTI Tune (Updated to Real Implementation)**

**Group Members:** Angela and Marwa

### **General Idea & Scope**
MBTI Tune is an AI application that predicts a user’s Myers‑Briggs personality dimensions (E, N, T, J) based on their Spotify listening habits.  
The system combines:

- **Numerical audio features** from Spotify  
- **Unsupervised representation learning** via a PyTorch autoencoder  
- **Transfer learning** for MBTI prediction  
- **LLM‑based lyric summarization** for interpretability  

This replaces traditional shallow classifiers with a modern deep learning pipeline.

---

## **Investigated Technologies & AI Architecture (Updated)**

### **Deep Learning Framework**
- Fully implemented in **PyTorch**  
- GPU‑ready training loops  
- Modular architecture (autoencoder + classifier)

### **Model Architecture (Real Implementation)**

#### **1. Autoencoder (Unsupervised Pretraining)**
- Trained on **113,000 songs**  
- Learns a compressed **16‑dimensional “music fingerprint”**  
- Architecture:  
  - Encoder: 49 → 128 → 64 → 16  
  - Decoder: 16 → 64 → 128 → 49  

#### **2. Transfer Learning Classifier**
- Input: 16‑dim latent vector  
- Output: 4 MBTI percentages (E, N, T, J)  
- Architecture: 16 → 64 → 32 → 4  
- Loss: MSE  
- Optimizer: Adam  

#### **3. Lyrics Interpretation (LLM‑Based)**
- Lyrics fetched via **LRCLIB**  
- Gemini used for:  
  - Language detection  
  - Translation  
  - Summarization  
  - Psychological breakdown  

> Lyrics are **not** used for training — only for interpretability.

#### **4. Live Data Pipeline**
- Spotify OAuth  
- Fetch top 20 tracks  
- Extract audio features  
- **Fallback simulation** when Spotify blocks audio‑features API  
- Aggregate → Encoder → Classifier  

---

## **Data Collection & Prototyping (Updated)**

### **Foundational Dataset**
- **113,000 songs** with 49 audio features  
- Used for autoencoder pretraining  

### **MBTI Dataset**
- **4,816 playlists** labeled with MBTI types  
- Used for supervised fine‑tuning  

### **Live User Data**
- Spotify top tracks fetched in real time  
- Ensures predictions reflect actual listening habits  

---

## **Challenges & Advanced Solutions (Updated)**

### **1. Spotify API Limitations**
- Spotify often blocks `audio-features` → implemented **robust fallback simulation**  
- Ensures consistent model input  

### **2. Missing Lyrics**
- LRCLIB used instead of Genius  
- Filter top 20 → pick first 3 tracks with lyrics  

### **3. Multilingual Lyrics**
- Gemini translation ensures consistent English summaries  

### **4. Overfitting**
- Dropout + transfer learning  
- Latent bottleneck reduces noise  

### **5. Explainability**
- SHAP explainability **planned**  
- Will show which audio features influence each MBTI dimension  

---

## **3. Rubric Alignment Strategy (Updated)**

To achieve the highest grades, MBTI Tune targets the “Excellent” criteria across all categories.

| **Rubric Criteria** | **Target Score** | **How MBTI Tune Achieves This (Updated)** |
|---------------------|------------------|-------------------------------------------|
| **Technical Depth** | 2 / 2 | Uses advanced AI techniques: autoencoder pretraining, transfer learning, multi‑output regression, and LLM‑based summarization. |
| **Implementation** | 3 / 3 | Clean PyTorch code, modular architecture, Streamlit UI, Spotify OAuth, fallback logic, and Gemini integration. |
| **Analysis & Evaluation** | 2 / 2 | Regression metrics (MAE, RMSE, R²), loss curves, and planned SHAP explainability. |
| **Innovation & Creativity** | 2 / 2 | Novel idea: predicting personality from music using compressed audio representations + LLM‑based lyric interpretation. |
| **Defense** | 3 / 3 | Clear architecture, strong justification for design choices, and interpretable outputs via Gemini summaries. |

---

