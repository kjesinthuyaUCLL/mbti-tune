# Comprehensive Project Overview & Rubric Alignment: MBTI Tune

## 1. Course Project Requirements Overview
* **Project Duration:** ~60 hours per student.
* **Team Size:** Individual or Pair (Angela and Marwa)[cite: 2].
* **Core Scope:** Be an AI-driven solution that applies advanced machine learning techniques beyond basic supervised learning.
* **Technology Mandate:** Must use at least one advanced technology (e.g., LLMs, Computer Vision, Reinforcement Learning, Multimodal models, Image Generation) and a relevant, new-to-you technology (e.g., PyTorch, TensorFlow).
* **Data:** Work with realistic data (curated dataset or real-world input).
* **Deliverables:**
  * **Code Repository (GitHub):** Well-structured with clear documentation and a README (setup instructions, project overview).
  * **Technical Report (~2 pages):** Simple, straightforward overview including Introduction, Data, Model & Methods, Results & Evaluation, Contributions, and Challenges/Future Work. GenAI can be used for proofreading, but the core technical input must be original.

---

## 2. Project Status Update: MBTI Tune[cite: 2]

**Group Members:** Angela and Marwa[cite: 2]

### General Idea & Scope[cite: 2]
* The project is an advanced multimodal Al application predicting a person's Myers-Briggs Type Indicator (MBTI) based on Spotify listening habits[cite: 2].
* It uses a Multi-modal Deep Neural Network that simultaneously processes numerical audio features and NLP embeddings from song lyrics, replacing traditional basic classifiers[cite: 2].

### Investigated Technologies & Al Architecture[cite: 2]
* **Deep Learning Framework:** Custom Neural Network built in PyTorch (or TensorFlow/Keras)[cite: 2].
* **Dual-Branch Input System:**[cite: 2]
  * **Branch 1 (Numerical):** A Dense Neural Network (DNN) processing standardized Spotify audio features (danceability, energy, valence, etc.) from the user's top 20 tracks[cite: 2].
  * **Branch 2 (NLP/Textual):** Uses the Genius API to fetch lyrics and a pre-trained Transformer model (e.g., BERT via Hugging Face) to extract semantic text embeddings[cite: 2].
* **Fusion and Classification:** Outputs from both branches are concatenated and passed through fully connected layers to predict the 16 MBTI classes[cite: 2].
* **Generative LLM Integration:** Uses Google Gemini API (or OpenAI API) to generate a personalized psychological breakdown explaining the connection between music taste and cognitive functions, based on predicted MBTI, top genres, and lyrics sentiment[cite: 2].

### Data Collection & Prototyping[cite: 2]
* **Foundational Dataset:** 4,816 playlists with 49 columns of average audio features labeled by MBTI type[cite: 2].
* **Live Data Pipeline:** Uses Python scripts and the Spotipy library to authenticate users via Spotify API and fetch top tracks, ensuring live data matches the training format[cite: 2].

### Challenges & Advanced Solutions[cite: 2]
* **Data Augmentation:** Fetching lyrics via Genius API to train the NLP branch; alternative includes extracting feature vectors from track titles/artist metadata using pre-trained LLMs if API rate limits are an issue[cite: 2].
* **Class Imbalance:** Addressing the overrepresentation of certain MBTI types (e.g., INFPs over ESTJs) using SMOTE or custom class weights in the PyTorch loss function[cite: 2].
* **Explainability:** Implementing confusion matrices and Explainable AI (SHAP values) to interpret Neural Network decisions based on specific audio features or lyric sentiments[cite: 2].

### Deployment[cite: 2]
* **Frontend:** Streamlit for Spotify OAuth login and interactive results display[cite: 2].
* **Hosting:** Final PyTorch model and app deployed on Hugging Face Spaces for a complete MLOps pipeline[cite: 2].

---

## 3. Rubric Alignment Strategy

To achieve the highest grades, the MBTI Tune project targets the "Excellent" criteria across all grading categories[cite: 3].

| Rubric Criteria[cite: 3] | Target Score[cite: 3] | How MBTI Tune Achieves This |
| :--- | :--- | :--- |
| **Technical Depth**[cite: 3] | 2 / 2 points[cite: 3] | Integrates advanced AI techniques (Multi-modal Deep Neural Networks, BERT text embeddings, and Generative LLMs)[cite: 2], going well beyond basic machine learning models[cite: 3]. |
| **Implementation**[cite: 3] | 3 / 3 points[cite: 3] | Will deliver clean, efficient, and well-documented code[cite: 3] structured into an end-to-end MLOps pipeline featuring Spotipy, PyTorch, and Streamlit on Hugging Face Spaces[cite: 2]. |
| **Analysis and Evaluation**[cite: 3] | 2 / 2 points[cite: 3] | Goes beyond numerical evaluation by providing meaningful assessment and interpretation[cite: 3] through confusion matrices and Explainable AI (SHAP values) to map features to predictions[cite: 2]. |
| **Innovation and Creativity**[cite: 3] | 2 / 2 points[cite: 3] | Presents a novel and original idea[cite: 3] by fusing Spotify audio metrics with semantic lyric embeddings into a dual-branch neural network to predict psychological types[cite: 2]. |
| **Defense**[cite: 3] | 3 / 3 points[cite: 3] | Will provide a clear, well-organized, and complete presentation with excellent answers to technical questions during the final defense[cite: 3]. |