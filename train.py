"""
train.py — BERT Fine-tuning Script for Resume Classification
============================================================
This script fine-tunes bert-base-uncased on a labelled resume dataset.

Usage
-----
    python train.py --data_path data/resumes.csv \
                    --output_dir models/bert_resume \
                    --epochs 5 \
                    --batch_size 16 \
                    --max_len 512

Requirements
------------
    pip install transformers torch scikit-learn pandas tqdm
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune BERT for resume classification")
parser.add_argument("--data_path",   default="data/resumes.csv")
parser.add_argument("--output_dir",  default="models/bert_resume")
parser.add_argument("--model_name",  default="bert-base-uncased")
parser.add_argument("--epochs",      type=int,   default=5)
parser.add_argument("--batch_size",  type=int,   default=16)
parser.add_argument("--max_len",     type=int,   default=512)
parser.add_argument("--lr",          type=float, default=2e-5)
parser.add_argument("--test_size",   type=float, default=0.2)
parser.add_argument("--seed",        type=int,   default=42)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Lazy imports (only needed at training time) ───────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AdamW, get_linear_schedule_with_warmup,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Dataset ───────────────────────────────────────────────────────────────────
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    if not TORCH_AVAILABLE:
        print("PyTorch / Transformers not installed. Run:\n  pip install torch transformers")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(args.data_path)
    assert "resume" in df.columns and "label" in df.columns, \
        "CSV must have 'resume' and 'label' columns."

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    num_labels = len(le.classes_)
    print(f"Classes ({num_labels}): {list(le.classes_)}")

    # Save label mapping
    label_map = {i: c for i, c in enumerate(le.classes_)}
    with open(Path(args.output_dir) / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        df["resume"].tolist(), df["label_enc"].tolist(),
        test_size=args.test_size, random_state=args.seed, stratify=df["label_enc"],
    )

    # Tokeniser + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels).to(device)

    train_ds  = ResumeDataset(X_tr,  y_tr,  tokenizer, args.max_len)
    val_ds    = ResumeDataset(X_val, y_val, tokenizer, args.max_len)
    train_dl  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl    = DataLoader(val_ds,   batch_size=args.batch_size)

    total_steps = len(train_dl) * args.epochs
    optimizer   = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_acc = 0.0
    history  = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask, labels=lbl)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbl  = batch["label"].to(device)
                out  = model(input_ids=ids, attention_mask=mask, labels=lbl)
                val_loss += out.loss.item()
                preds.extend(out.logits.argmax(-1).cpu().numpy())
                trues.extend(lbl.cpu().numpy())

        acc = accuracy_score(trues, preds)
        print(f"\nEpoch {epoch}: train_loss={tr_loss/len(train_dl):.4f} "
              f"val_loss={val_loss/len(val_dl):.4f} val_acc={acc:.4f}")

        history["train_loss"].append(tr_loss / len(train_dl))
        history["val_loss"].append(val_loss / len(val_dl))
        history["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Best model saved (acc={best_acc:.4f})")

    print("\n── Classification Report ──")
    print(classification_report(trues, preds, target_names=list(le.classes_)))
    print(f"\nBest validation accuracy: {best_acc:.4f}")

    with open(Path(args.output_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete. Model saved to:", args.output_dir)


if __name__ == "__main__":
    train()
