# 🧠 Emotion Classification with Fine-Tuned XLNet

A fine-tuned XLNet model for multi-class emotion detection in text, built with Hugging Face Transformers.

---

## 📌 Overview

This project fine-tunes the pretrained `xlnet-base-cased` model on an emotion-labeled text dataset to classify input sentences into one of four emotional categories:

| Label | Emotion |
|-------|---------|
| 0 | Anger |
| 1 | Fear |
| 2 | Joy |
| 3 | Sadness |

---

## 🗂️ Dataset

The dataset consists of three CSV files with `text` and `label` columns:

- `emotion-labels-train.csv`
- `emotion-labels-test.csv`
- `emotion-labels-val.csv`

Classes are **balanced by downsampling** to prevent bias toward majority emotions.

---

## ⚙️ Pipeline

```
Raw Text
   ↓  Clean (remove mentions, emojis, special characters)
   ↓  Balance classes
   ↓  Encode labels (LabelEncoder)
   ↓  Tokenize (XLNetTokenizer, max_length=128)
   ↓  Fine-tune (XLNetForSequenceClassification)
   ↓  Evaluate (accuracy per epoch)
   ↓  Save model → Inference pipeline
```

---

## 🛠️ Tech Stack

- **Model:** `xlnet-base-cased` (Hugging Face)
- **Framework:** Hugging Face `transformers`, `datasets`, `evaluate`
- **Training:** Hugging Face `Trainer` API
- **Text Cleaning:** `cleantext`, `re`
- **Environment:** Google Colab

---

## 🚀 How to Run

1. Upload the dataset CSVs to `/content/` in Google Colab
2. Run all cells in order
3. The fine-tuned model will be saved to `fine_tuned_model/`

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `xlnet-base-cased` |
| Num labels | 4 |
| Max token length | 128 |
| Epochs | 3 |
| Eval strategy | Per epoch |
| Metric | Accuracy |

---

## 🔍 Inference Example

```python
from transformers import pipeline, XLNetForSequenceClassification, XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForSequenceClassification.from_pretrained("fine_tuned_model")

clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
clf("I can't believe this happened, I'm so frustrated!")
# Output: [{'label': 'anger', 'score': 0.91}]
```

---

## 📁 Project Structure

```
├── fineTunning_a_model_using_XLNet.ipynb   # Main notebook
├── fine_tuned_model/                        # Saved model weights
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
└── README.md
```

---

## 💡 Notes

- Training was run on a small subset (100 samples) for demonstration. For production use, train on the full dataset.
- XLNet uses permutation-based language modeling, giving it an advantage over BERT-style models in capturing bidirectional context.
