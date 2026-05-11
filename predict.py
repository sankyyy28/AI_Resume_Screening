"""
predict.py — CLI inference for the BERT Resume Classifier
==========================================================
Usage (single resume):
    python predict.py --text "Python developer with 5 years experience in ML..."

Usage (batch CSV):
    python predict.py --csv_path data/resumes.csv --output results/predictions.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--text",       default=None, help="Single resume text")
parser.add_argument("--csv_path",   default=None, help="CSV with 'resume' column")
parser.add_argument("--model_dir",  default="models/bert_resume")
parser.add_argument("--top_k",      type=int, default=3)
parser.add_argument("--max_len",    type=int, default=512)
parser.add_argument("--output",     default="results/predictions.csv")
args = parser.parse_args()


def predict_bert(texts, model_dir, max_len, top_k):
    """Predict using saved BERT model."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()

        label_map_path = Path(model_dir) / "label_map.json"
        with open(label_map_path) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

        results = []
        for text in texts:
            enc = tokenizer(text, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"].to(device),
                               attention_mask=enc["attention_mask"].to(device)).logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            top_idx = probs.argsort()[::-1][:top_k]
            results.append([{"label": label_map[i], "confidence": round(float(probs[i]), 4)}
                             for i in top_idx])
        return results

    except (ImportError, FileNotFoundError):
        # Fall back to lightweight keyword classifier
        from utils.classifier import BERTResumeClassifier
        from utils.preprocessor import ResumePreprocessor
        clf  = BERTResumeClassifier()
        prep = ResumePreprocessor()
        return [clf.predict(prep.clean(t), top_k=top_k) for t in texts]


def main():
    if args.text:
        results = predict_bert([args.text], args.model_dir, args.max_len, args.top_k)
        print("\n── Predictions ──")
        for rank, pred in enumerate(results[0], 1):
            print(f"  {rank}. {pred['label']:35s}  {pred['confidence']:.1%}")

    elif args.csv_path:
        df = pd.read_csv(args.csv_path)
        assert "resume" in df.columns, "CSV must have a 'resume' column."
        all_results = predict_bert(df["resume"].tolist(), args.model_dir, args.max_len, top_k=1)
        df["predicted_role"] = [r[0]["label"]      for r in all_results]
        df["confidence"]     = [r[0]["confidence"] for r in all_results]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} predictions to {args.output}")

    else:
        print("Provide --text or --csv_path. See --help for usage.")


if __name__ == "__main__":
    main()
