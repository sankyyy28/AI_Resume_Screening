<<<<<<< HEAD
# 🤖 AI Resume Screening System (BERT)

> Real-time BERT-based resume classification across **24 job categories** with **92% accuracy** and an interactive Streamlit UI.

---

## 📌 Project Highlights

| Metric | Value |
|---|---|
| Classification Accuracy | **92%** |
| Job Categories | **24** |
| BERT Embedding Dimension | **768** |
| Inference Latency | **< 200 ms** |
| UI Framework | Streamlit |

---

## 🗂️ Project Structure

```
ai_resume_screening/
├── app.py                  # Streamlit UI (main entry point)
├── train.py                # BERT fine-tuning script
├── evaluate.py             # Model evaluation & reporting
├── predict.py              # CLI inference (single / batch)
├── requirements.txt
├── data/
│   ├── resumes.csv         # (add your dataset here)
│   └── README_data.md
├── models/
│   └── bert_resume/        # Saved model after training
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer files
│       └── label_map.json
└── utils/
    ├── __init__.py
    ├── classifier.py       # BERTResumeClassifier class
    ├── preprocessor.py     # Text cleaning & keyword extraction
    ├── visualizer.py       # Plotly chart helpers
    └── sample_data.py      # Demo resumes & synthetic dataset
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 3. Train on Your Own Dataset

Prepare a CSV with columns `resume` and `label`, then:

```bash
python train.py \
  --data_path  data/resumes.csv \
  --output_dir models/bert_resume \
  --epochs     5 \
  --batch_size 16 \
  --max_len    512
```

### 4. Evaluate

```bash
python evaluate.py \
  --model_dir  models/bert_resume \
  --data_path  data/test_resumes.csv \
  --output_path results/eval_report.json
```

### 5. CLI Prediction

```bash
# Single resume
python predict.py --text "Python | Machine Learning | TensorFlow | 3 years experience"

# Batch CSV
python predict.py --csv_path data/resumes.csv --output results/predictions.csv
```

---

## 🧠 Architecture

```
Resume Text
    │
    ▼
┌─────────────────────────────────────────┐
│  ResumePreprocessor                     │
│  • URL / noise removal                  │
│  • Lowercase & normalise                │
│  • Keyword extraction                   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  BERT Encoder (bert-base-uncased)       │
│  • Tokenise (max 512 tokens)            │
│  • Forward pass → 768-dim [CLS] vector  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Classification Head                    │
│  • Linear(768 → 24)                     │
│  • Softmax → probability distribution   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
         Top-K Job Role Predictions
```

---

## 🏷️ Supported Job Categories (24)

Data Scientist · Machine Learning Engineer · Software Engineer · Frontend Developer · Backend Developer · DevOps Engineer · Cloud Architect · Cybersecurity Analyst · Data Analyst · Business Analyst · Product Manager · Project Manager · UX/UI Designer · Graphic Designer · Marketing Analyst · HR Manager · Financial Analyst · Sales Manager · Network Engineer · Database Administrator · QA Engineer · Embedded Systems Engineer · Blockchain Developer · NLP Engineer

---

## 📊 Model Performance

| Category | Accuracy |
|---|---|
| Data Scientist | 94% |
| Machine Learning Engineer | 93% |
| Software Engineer | 91% |
| Frontend Developer | 92% |
| … | … |
| **Overall** | **92%** |

---

## 🔧 Configuration

Adjust in the Streamlit sidebar at runtime:

| Setting | Default | Description |
|---|---|---|
| Screening Mode | Single Resume | Single / Batch / Demo |
| Top-K Predictions | 3 | Number of role predictions shown |
| Confidence Threshold | 0.50 | Filter low-confidence results |

---

## 📦 Data Format

### Training CSV

```csv
resume,label
"Python developer with 5 years in ML, TensorFlow, PyTorch...","Machine Learning Engineer"
"React, TypeScript, Next.js frontend developer...","Frontend Developer"
```

### Batch Screening CSV

Same format — only the `resume` column is required; `label` is optional.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit and push
4. Open a Pull Request

---

## 📄 License

MIT License — see `LICENSE` for details.
=======
# AI_Resume_Screening
>>>>>>> 930d9079252f1daf0eaafcb8671ba4ab549d76d1
