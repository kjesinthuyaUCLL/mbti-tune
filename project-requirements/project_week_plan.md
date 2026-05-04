# [cite_start]MBTI Tune - Complete 1-Week Project Plan [cite: 140]

## [cite_start]1. Project Overview [cite: 142]
[cite_start]MBTI Tune is an Al web application that predicts a person's Myers-Briggs personality dimensions from their Spotify listening habits[cite: 143]. [cite_start]Unlike basic classifiers that output a single type (e.g., "INTJ"), our model outputs percentage scores for each of the four MBTI dimensions, providing nuanced insights into personality[cite: 144].

### [cite_start]Example Output [cite: 145]
* [cite_start]**Extraversion (E):** 78% [cite: 147]
* [cite_start]**Intuition (N):** 65% [cite: 148, 149]
* [cite_start]**Thinking (T):** 45% [cite: 148, 150]
* [cite_start]**Judging (J):** 82% [cite: 151, 152]

[cite_start]**Result:** ENFJ Type (E=78%, N=65%, F=55%, J=82%)[cite: 154, 155]. [cite_start]You lean toward Extraversion, Moderate Intuition preference, Slight Feeling preference, Strong Judging preference[cite: 153].

[cite_start]**Gemini Says:** [cite: 157]
> [cite_start]"You're the kind of person who listens to Hozier when you need to feel deep emotions, but also blasts Lizzo when you need to hype yourself up. [cite: 158, 159] [cite_start]Your 78% Extraversion explains why you love live concert recordings, and your 82% Judging means you've probably organized your Spotify playlists by mood, genre, AND color. [cite: 160] [cite_start]Classic ENFJ energy - you're everyone's therapist friend who also knows how to party." [cite: 161]

[cite_start]**What influenced this prediction?** [cite: 162]
* [cite_start]High energy (0.82) and high valence (0.76) → Extraversion [cite: 163]
* [cite_start]Variable tempo (high standard deviation) → Intuition [cite: 164]
* [cite_start]Low acousticness (0.23) and high speechiness (0.18) → Thinking preference [cite: 165]
* [cite_start]Consistent listening patterns → Judging preference [cite: 166]

---

## [cite_start]2. Tools & Resources [cite: 180]

| Category | Tool | Purpose |
| :--- | :--- | :--- |
| **Core** | Python 3.9+ | [cite_start]Main programming language [cite: 182] |
| **Core** | PyTorch | [cite_start]Neural network framework [cite: 182] |
| **Core** | VS Code / GitHub | [cite_start]Code editor and Version control [cite: 182] |
| **Data** | Pandas / NumPy / Scikit-learn | [cite_start]Load, manipulate, split, scale, and evaluate datasets [cite: 182] |
| **APIs** | Spotify API | [cite_start]Fetch user's top tracks and audio features [cite: 182] |
| **APIs** | Google Gemini API | [cite_start]Generate funny personality descriptions [cite: 182] |
| **APIs** | Genius API | [cite_start]Fetch song lyrics for deeper analysis (+6-9 hours) [cite: 186] |
| **Web & Hosting** | Streamlit | [cite_start]Build web app using only Python [cite: 182] |
| **Web & Hosting** | Hugging Face Spaces | [cite_start]Host the live web app [cite: 182] |
| **Explainability** | SHAP | [cite_start]Explain which features influenced each prediction [cite: 182] |

---

## [cite_start]3. AI Architecture & Models [cite: 187]

| Component | Type | Purpose |
| :--- | :--- | :--- |
| **Autoencoder** | Unsupervised neural network | [cite_start]Pre-train on 113k songs to learn general music patterns [cite: 188] |
| **Encoder** | Dense layers (128-64-16) | [cite_start]Compresses 49 audio features into 16 "music fingerprint" numbers [cite: 188] |
| **Decoder** | Dense layers (16-64-128-49) | [cite_start]Reconstructs original features (only used during pretraining) [cite: 188] |
| **Classifier** | Dense layers (16-64-32-4) | [cite_start]Takes the 16 fingerprint numbers → outputs 4 dimension percentages [cite: 188] |

[cite_start]**Why This Architecture?** [cite: 188]
* [cite_start]**Pretraining on 113k songs:** Transfer learning - model learns general music patterns before personality task[cite: 188].
* [cite_start]**Bottleneck layer (16 neurons):** Forces model to learn compressed, essential features[cite: 188].
* [cite_start]**Multi-output regression (4 dimensions):** More nuanced than 16-class classification[cite: 188].
* [cite_start]**Sigmoid output activation:** Converts raw scores to 0-1 percentages[cite: 188].
* [cite_start]**Dropout layers (0.3):** Prevents overfitting on small dataset[cite: 188].

---

## [cite_start]4. Data Pipeline [cite: 196]

[cite_start]**Datasets We Have:** [cite: 199]
1.  **Song dataset:** 113,000 rows. 49 audio features per song. [cite_start]Purpose: Unsupervised pretraining[cite: 200].
2.  **MBTI playlists:** 4,816 rows. 49 audio features + MBTI type. [cite_start]Purpose: Supervised training[cite: 200].
3.  **User data:** Live. Top 20 tracks from Spotify login. [cite_start]Purpose: Real-time prediction[cite: 200].

[cite_start]**How Data Flows:** [cite: 201]
* [cite_start]**PHASE 1: PRETRAINING:** 49 audio features → Autoencoder → 49 reconstructed features (learns compressed "music fingerprint")[cite: 203, 204, 205].
* [cite_start]**PHASE 2: TRANSFER LEARNING:** 49 features → Pretrained Encoder → 16 compressed features → 4 dimension percentages (E%, N%, T%, J%)[cite: 206, 207, 209, 210].
* [cite_start]**PHASE 3: USER PREDICTION:** Top 20 tracks → Average 49 features → Trained model → Dimension percentages → MBTI type[cite: 211, 212, 213].

[cite_start]**Hardware Infrastructure:** [cite: 214]
* [cite_start]**Model training (Pretraining & Fine-tuning):** Google Colab with free T4 GPU for acceleration[cite: 215].
* [cite_start]**Web app hosting:** Hugging Face Spaces (CPU is sufficient for fast inference)[cite: 215].
* [cite_start]**Local development:** Personal laptops[cite: 215].

---

## [cite_start]5. 7-Day Execution Plan [cite: 221]

[cite_start]**Total Estimated Time:** 100 hours (50 hours per person)[cite: 292, 293].

* [cite_start]**Day 1: Foundation & Data Preparation (16 hours total)** [cite: 222, 286]
    * [cite_start]Goal: Environment setup, datasets downloaded, folder structures created, and data preprocessed[cite: 223, 225, 232, 236, 240, 246].
    * [cite_start]*Deliverable:* Clean, preprocessed data ready for training[cite: 250].
* [cite_start]**Day 2: Build & Train Autoencoder (16 hours total)** [cite: 251, 286]
    * [cite_start]Goal: Build architecture in PyTorch, train on 113k songs via Colab GPU, and test reconstruction quality[cite: 252, 253].
    * [cite_start]*Deliverable:* Trained autoencoder model saved to file[cite: 253].
* [cite_start]**Day 3: Build & Train Classifier (16 hours total)** [cite: 253, 286]
    * [cite_start]Goal: Load pretrained encoder, add classification layers (64-32-4), implement class weights for imbalance, and train[cite: 253].
    * [cite_start]*Deliverable:* Trained classifier model that outputs 4 dimension percentages[cite: 253].
* [cite_start]**Day 4: SHAP Explainability (12 hours total)** [cite: 253, 286]
    * [cite_start]Goal: Create SHAP explainer wrapper, generate feature importance plots, and interpret results[cite: 254, 257].
    * [cite_start]*Deliverable:* SHAP visualizations showing feature importance for each dimension[cite: 257].
* [cite_start]**Day 5: Spotify Integration (12 hours total)** [cite: 257, 286]
    * [cite_start]Goal: Register app, setup OAuth, write fetching/calculating functions, and test with personal accounts[cite: 257].
    * [cite_start]*Deliverable:* Working Spotify login and data fetching[cite: 257].
* [cite_start]**Day 6: Streamlit Web App + Gemini API (16 hours total)** [cite: 257, 286]
    * [cite_start]Goal: Build app structure, integrate model predictions and progress bars, and set up Gemini API prompt[cite: 257].
    * [cite_start]*Deliverable:* Working web app running locally[cite: 265].
* [cite_start]**Day 7: Testing, Deployment, Documentation (12 hours total)** [cite: 266, 286]
    * [cite_start]Goal: Deploy to Hugging Face Spaces, set up API keys, end-to-end testing, and prepare documentation/demo[cite: 267, 270, 273, 275, 278, 281].
    * [cite_start]*Deliverable:* Live web app accessible via public URL[cite: 284].

---

## [cite_start]6. Success Checklist [cite: 308]
* [cite_start]User can log in with Spotify[cite: 309].
* [cite_start]App fetches top 20 tracks and calculates average features[cite: 310].
* [cite_start]Neural network outputs E%, N%, T%, J% (0-100)[cite: 311].
* [cite_start]Progress bars and MBTI type displayed[cite: 312].
* [cite_start]Gemini generates unique description based on percentages + artists[cite: 313].
* [cite_start]SHAP explanations show feature importance[cite: 314].
* [cite_start]App deployed on Hugging Face Spaces[cite: 315].
* [cite_start]Documentation complete[cite: 316].