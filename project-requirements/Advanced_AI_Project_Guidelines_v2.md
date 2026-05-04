# Advanced AI Project - Guidelines and Theoretical Context for Copilot

This document serves as the strict operational context for the AI Copilot assisting in the development of the "Advanced AI" course project. 
The Copilot must strictly adhere to these instructions, course requirements, and theoretical bounds to prevent over-engineering and ensure the student meets the exact grading criteria.

## 1. Deliverables & Grading Rubric
The project will be evaluated based on the following deliverables: a complete Code Repository, a Technical Report, and a 15-minute Individual Oral Defense.

**Rubric:**
* **Technical Depth (2 pts):** Use advanced AI techniques beyond the basics. Combining techniques (e.g., LLM + Computer Vision, or Reinforcement Learning + CV) is highly encouraged.
* **Implementation (3 pts):** Code must be clean, efficient, well-documented, and written in **PyTorch**.
* **Analysis and Evaluation (2 pts):** Meaningful assessment, valid metric tracking, and critical discussion of results.
* **Innovation and Creativity (2 pts):** Novelty and originality of the chosen approach.
* **Defense (3 pts):** Clarity, organization, and completeness during the oral presentation.

## 2. Theoretical Foundations & Allowed Scope
The Copilot must base its architectural choices and code generation *only* on the following theoretical concepts covered in the course:

### A. Mathematical Background & Deep Learning Basics
* **Framework:** Use **PyTorch**. Ensure device agnosticism (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`). Code must be ready for Google Colab/GPU execution.
* **Optimization:** Use standard optimizers like **Stochastic Gradient Descent (SGD)** or **Adam** (which combines momentum and adaptive learning rates).
* **Training Loop:** Must explicitly handle the Forward Pass, Loss Calculation, Backpropagation (`loss.backward()`), and Gradient Updates (`optimizer.step()`). Manage dynamic learning rates (starting high, ending low).

### B. Natural Language Processing (NLP)
* **Architectures:** Use Word Embeddings and Transformer-based models. 
* **Evaluation Metrics:** * *Classification:* Accuracy, Precision, Recall, F1-Score (do not rely solely on accuracy for imbalanced datasets).
    * *Generation/Translation:* BLEU (overlap with reference), ROUGE (summarization), Perplexity (model uncertainty/probability distribution).

### C. Generative AI & Image Generation
* **Concepts:** Understand the difference between Discriminative (predicting classes/boundaries) and Generative (modeling data distribution to create new samples) models.
* **Architectures allowed:** Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Diffusion Models (e.g., Stable Diffusion).
* **Text-to-Image:** Linking text to images via models like CLIP.

### D. Reinforcement Learning (RL)
* **Framework:** Agent interacting with an Environment to maximize cumulative reward over Trajectories (states, actions, rewards). Use a discount factor ($\gamma$).
* **Q-Learning (Value-based):** Updating Q-values rather than state values. Use the **Temporal Difference (TD) update** rule. Understand TD Target and TD Error.
* **Deep RL:** Using deep neural networks in simulations to estimate policies or values when the state space is too large.

### E. Fine-Tuning & Large Models
* **Constraint:** DO NOT attempt to train massive Foundation Models from scratch due to hardware and data limits.
* **Methodology:** Use Pre-trained models and apply Parameter-Efficient Fine-Tuning.
* **LoRA (Low-Rank Adaptation):** When fine-tuning, implement or utilize LoRA. Freeze the original large weight matrix ($W$) and only train the smaller, low-rank decomposition matrices ($A$ and $B$) to drastically reduce VRAM usage and trainable parameters.

## 3. Strict Directives for the Copilot
1.  **No Over-engineering:** Do not introduce complex frameworks, external orchestrators (like LangChain, unless explicitly requested), or advanced deployment architectures (like Docker/Kubernetes) that are outside the scope of this AI theory course.
2.  **Focus on PyTorch & Implementation:** Maximize the "Implementation" score. Provide modular Python scripts or well-structured Jupyter Notebooks.
3.  **Evaluate Rigorously:** Always include code blocks dedicated to generating plots (Loss curves, Confusion Matrices, generated image grids) and calculating the specific metrics mentioned in the theory section.
