"""
evaluate.py — Evaluate a saved BERT model on a test CSV
========================================================
Usage:
    python evaluate.py --model_dir models/bert_resume \
                       --data_path data/test_resumes.csv \
                       --output_path results/eval_report.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",   default="models/bert_resume")
parser.add_argument("--data_path",   default="data/resumes.csv")
parser.add_argument("--batch_size",  type=int, default=16)
parser.add_argument("--max_len",     type=int, default=512)
parser.add_argument("--output_path", default="results/eval_report.json")
args = parser.parse_args()

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, classification_report)
    from sklearn.preprocessing import LabelEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def evaluate():
    if not TORCH_AVAILABLE:
        print("PyTorch / Transformers not installed."); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data_path)
    assert "resume" in df.columns and "label" in df.columns

    # Load label map
    label_map_path = Path(args.model_dir) / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
        class_names = [label_map[i] for i in sorted(label_map)]
    else:
        le = LabelEncoder()
        le.fit(df["label"])
        class_names = list(le.classes_)
        label_map = {i: c for i, c in enumerate(class_names)}

    le = LabelEncoder()
    le.fit(df["label"])
    df["label_enc"] = le.transform(df["label"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    ds = ResumeDataset(df["resume"].tolist(), df["label_enc"].tolist(), tokenizer, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size)

    preds, trues = [], []
    with torch.no_grad():
        for batch in dl:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            preds.extend(out.logits.argmax(-1).cpu().numpy())
            trues.extend(batch["label"].numpy())

    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="weighted")

    report = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(float(prec), 4),
        "recall_weighted":    round(float(rec), 4),
        "f1_weighted":        round(float(f1), 4),
        "per_class":          classification_report(trues, preds,
                                target_names=class_names, output_dict=True),
    }

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("\n── Per-Class Report ──")
    print(classification_report(trues, preds, target_names=class_names))

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to:", args.output_path)


if __name__ == "__main__":
    evaluate()
