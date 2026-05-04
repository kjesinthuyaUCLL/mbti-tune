# Technical Implementation Guide for Advanced AI Project

This document provides strict technical guidelines, preferred libraries, and implementation strategies derived directly from the course's practical lab exercises. 
**Copilot Instructions:** Use this document to dictate *how* the code should be structured, which libraries to use, and what specific implementation patterns to follow when assisting the student.

## 1. Environment & Core Frameworks
* **Python Version:** Python 3.10 to 3.13 (Avoid 3.14 for compatibility).
* **Deep Learning Framework:** **PyTorch** (Targeting v2.10 features).
* **Device Agnosticism:** ALL PyTorch code must explicitly handle device placement:
    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Tensors must be moved to device: tensor.to(device)
    ```
* **Dependency Management:** Code should assume a virtual environment (`venv`). If providing setup instructions, include `pip install -r requirements.txt`.

## 2. Core PyTorch Coding Standards
* **Architecture Definition:** Always use Object-Oriented PyTorch (`class MyModel(nn.Module):`). Explicitly define `__init__` and `forward` methods.
* **Hyperparameter Tuning:** Code must easily expose hyperparameters (number of layers, neurons, learning rate, batch size, activation functions) for easy experimentation.
* **Training Loop Structure:**
    * Standard pattern: `optimizer.zero_grad()`, `output = model(data)`, `loss = criterion(output, target)`, `loss.backward()`, `optimizer.step()`.
    * **Visualization:** ALWAYS include code to plot training and validation Loss Curves using `matplotlib.pyplot`.
* **Data Handling:** Use `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`.

## 3. Domain-Specific Technical Implementations

### 3.1. Computer Vision (CNNs, Object Detection)
* **Libraries:** `torchvision` (datasets, transforms, models), `PIL` (Python Imaging Library), `matplotlib`.
* **Implementation Patterns:**
    * Use `torchvision.transforms` for data augmentation and normalization.
    * For object detection/segmentation tasks, ensure bounding boxes are correctly parsed and plotted (e.g., using `PIL.ImageDraw` to draw red rectangles for scores > 0.7).
    * Always ensure proper tensor dimensions (e.g., handling image channels, flattening when connecting CNNs to MLPs).

### 3.2. NLP & Transformers
* **Architectures:** From basic RNNs/LSTMs (handling vanishing gradients) to Transformers.
* **Transformers Implementation:** * Implement and visualize **Positional Encodings** to show how sequences are handled without recurrence.
    * Implement **Attention Mechanisms** (Self-Attention) to demonstrate dynamic weighting of input sequences.
* **Pipelines:** Tokenization -> Embedding -> Positional Encoding -> Attention/Transformer Blocks -> Output generation.

### 3.3. Reinforcement Learning (RL)
* **Libraries:** `Stable-Baselines3`, standard `gym` or Hugging Face environments.
* **Implementation Patterns:**
    * *Tabular Q-Learning:* For simple discrete environments (like FrozenLake or Taxi-v3), implement the Q-table and update rule from scratch (without neural networks).
    * *Deep RL:* For continuous/complex environments (like Lunar Lander or Doom), use Deep Q-Networks (DQN). If relying on libraries, implement using `Stable-Baselines3`.
    * *Saving/Loading:* Include code to save the trained agent and load it for inference/rendering.

### 3.4. Generative AI (Images)
* **Custom Models (GANs & VAEs):**
    * Implement separate loops for the Generator and Discriminator in GANs.
    * Handle latent space vectors (`z_dim`) explicitly.
    * Use `torchvision.utils.make_grid` and `torchvision.utils.save_image` to visualize generated batches at various epochs.
* **Stable Diffusion:**
    * **Library:** Hugging Face `diffusers` and `transformers`.
    * **Components:** Code should explicitly interact with the `StableDiffusionPipeline`.
    * Demonstrate knowledge of the underlying components: `VAE` (Latent diffusion), `Tokenizer` & `Text Encoder` (CLIP), `UNet`, and the `Scheduler`.
    * Implement variations like `Img2Img`, `Inpainting`, or `Depth2Img` pipelines if applicable.

## 4. Strict "Do Not" Rules for Copilot
* **DO NOT** use TensorFlow or Keras. PyTorch only.
* **DO NOT** provide black-box solutions without logging. Always log loss and metrics.
* **DO NOT** write monolithic scripts. Separate models, datasets, training loops, and evaluation into functions or separate cells if using Jupyter format.
* **DO NOT** evaluate models without proper, un-annotated evaluation datasets or metrics. (e.g., classification requires an annotated dataset to properly evaluate).
