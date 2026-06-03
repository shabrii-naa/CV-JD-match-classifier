# CV-Job Match Classifier

Model machine learning untuk memprediksi kesesuaian antara CV kandidat dan job description, dilengkapi dengan analisis skill gap.

---

## Overview

Pipeline menerima teks CV dan job description mentah, melakukan preprocessing, lalu menghasilkan match probability dan daftar skill yang belum dimiliki kandidat.

```
CV Text + JD Text → clean_text() → TF-IDF Vectorizer → Neural Network → Match Probability + Skill Gap
```

---

## Model Architecture

| Komponen | Detail |
|---|---|
| Input | TF-IDF combined CV + JD (15.000 fitur, n-gram 1–2) |
| Hidden layers | Dense 256 → 128 → 64 + BatchNorm + Dropout |
| Residual | Skip connection dari Dense-1 ke Dense-3 |
| Attention | SkillAttentionLayer (dim=64) |
| Output | Dense(1, sigmoid) → match probability |
| Loss | WeightedBCELoss (W_POS boost 2.5× untuk class imbalance) |
| Optimizer | Adam (lr=3e-4, clipnorm=1.0) |
| Training | Custom `tf.GradientTape` loop + Early Stopping + ReduceLROnPlateau |

---

## Performance

| Metric | Value |
|---|---|
| F1 (class match) | 0.84 |
| AUC | 0.80 |
| Accuracy | 76.7% |
| Optimal Threshold | 0.50 |
| Dataset | 600 samples, rasio 70:30 (match:no-match) |

> Model dilatih pada dataset terbatas (600 baris). F1 dan AUC diprioritaskan sebagai metrik utama karena adanya class imbalance.

---

## Artifacts

| File | Deskripsi |
|---|---|
| `skill_gap_model.keras` | Trained Keras model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer — **harus dipakai, jangan di-fit ulang** |
| `skill_labels.json` | Skill taxonomy untuk gap analysis |
| `logs/` | TensorBoard training logs |

---

## Preprocessing Contract

Fungsi `clean_text()` wajib dijalankan sebelum vectorization. Harus konsisten antara training dan serving.

```python
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

## Inference

```python
import pickle, re
import numpy as np
from tensorflow import keras

# Load artifacts (sekali saat startup)
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = keras.models.load_model("skill_gap_model.keras")

# Predict
combined = clean_text(cv_text) + " " + clean_text(jd_text)
X        = tfidf.transform([combined]).toarray().astype("float32")
prob     = float(model.predict(X, verbose=0)[0][0])
is_match = prob >= 0.50
```

---

## TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Project Structure

```
├── Capstone_Project.ipynb     # Training notebook
├── skill_gap_model.keras      # Trained model
├── tfidf_vectorizer.pkl       # Fitted vectorizer
├── skill_labels.json          # Skill taxonomy
├── logs/                      # TensorBoard logs
└── README.md
```

---

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- scikit-learn (TF-IDF, metrics)
- NumPy, Pandas
- TensorBoard
