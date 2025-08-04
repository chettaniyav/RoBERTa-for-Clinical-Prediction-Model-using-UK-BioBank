---

# ⚕️ MLM for Clinical Contextual Language Modeling & CVD Prediction

This repository contains the implementation of a **Masked Language Modeling (MLM)**-based approach for pre-training and fine-tuning a RoBERTa-based model on clinical data, followed by downstream classification tasks for **Cardiovascular Disease (CVD)** prediction.

---

## 🧠 Overview

We leverage **Masked Language Modeling (MLM)** to pre-train a transformer model on synthetic and real-world clinical data. The model is then fine-tuned to predict cardiovascular conditions such as **Stroke**, **Angina**, and **Heart Attack** based on electronic health records (EHR).

---

## 📚 Masked Language Modeling (MLM)

* **Masking Strategy**:

  * 15% of tokens in each sequence are selected for prediction.

    * 80% replaced with `[MASK]`
    * 10% replaced with a random word
    * 10% left unchanged
* Helps model learn word dependencies and context in clinical narratives.

---

## ⚙️ Pre-Training Details

* **Dataset**: Synthetic dataset (Denaxas, 2020)
* **Model**: RoBERTa (no custom embedding layers used)
* **Hardware**: Cloud TPU (128GB memory)
* **Epochs**: 3
* **Batch Size**: 512
* **Max Sequence Length**: 512
* **Optimizer**: AdamW

  * `lr=2e-5`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8`, `weight_decay=0.01`

The model was saved after pre-training for future downstream fine-tuning.

---

## 🔁 Fine-Tuning with Real-World Data

* **Dataset**: Subset-of-HES from UK Biobank (half a million EHRs)
* **Hardware**: iCSF cluster (2TB RAM, no GPU/TPU)
* **Epochs**: 5
* **Batch Size**: 32
* **Max Sequence Length**: 128
* **Optimizer**: AdamW

  * `lr=5e-5`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8`, `weight_decay=0.01`

Logging was performed every 200 steps. The model was saved locally for CVD prediction fine-tuning.

---

## ❤️ CVD Prediction

After MLM fine-tuning, a **classification head** was added:

$$
P(CVD = 1 | h_{[CLS]}) = \sigma(Wh_{[CLS]} + b)
$$

* Output layer: Sigmoid activated linear layer on top of `[CLS]` token.
* **Conditions Modeled**:

  * Stroke
  * Angina
  * Heart Attack
* **Training**:

  * Epochs: 5
  * Batch Size: 1028
  * Optimizer: AdamW (`lr=2e-5`, same β/ε/weight decay)

---

## 🧪 Baseline Models

We compared our transformer-based model with baseline models:

* **Architectures**:

  * Bi-RNN
  * Bi-LSTM

* **Layers**:

  * Input Embedding: 128-dim
  * Hidden: 128-dim with 0.5 dropout
  * FC Layers: \[256, ReLU] → \[128, ReLU]
  * Output: Sigmoid classification

* **Optimizer**: AdamW (`lr=1e-2`)

* **Loss**: Binary Cross Entropy

---

## ✅ Model Outcomes

* Predicts CVD likelihood from structured and unstructured EHR data.
* Especially effective at identifying **high-risk** individuals for:

  * **Stroke**
  * **Heart Attack**
  * **Angina**

---

## 🧪 Validation Strategy

* **Internal Validation**:

  * 15% held-out test set
  * Bootstrap resampling
* **Metrics**:
<img width="688" height="141" alt="image" src="https://github.com/user-attachments/assets/eb0f678a-5080-4e5b-9b14-6369ec687636" />

All preprocessing steps mirrored those used in training phases for both SOTA and baseline models.

---

## 🔧 Requirements

* Python 3.8+
* Transformers (Hugging Face)
* PyTorch
* scikit-learn
* pandas
* tqdm

---

## 🧾 Citation

*Research paper under review. Citation details will be updated upon publication.*

---

## 📬 Contact

For questions, reach out via [issues](https://github.com/your-repo/issues) or email at vermachettaniya6666@gmail.com .

---
