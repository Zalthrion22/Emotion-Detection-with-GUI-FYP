"""Inference using the fine-tuned DistilBERT model.

"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import joblib

from src.config import DISTILBERT_DIR
from src.text_preprocessing import clean_text


class DistilBertNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> tuple:
    if not (DISTILBERT_DIR / "config.json").exists():
        raise DistilBertNotTrainedError(
            f"No DistilBERT model found at {DISTILBERT_DIR}. Train it first:\n"
            "    python -m src.train_text_distilbert\n"
            "or use the Colab runbook at notebooks/train_distilbert_colab.md"
        )
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise DistilBertNotTrainedError(
            "DistilBERT inference requires torch + transformers. "
            f"Install with: pip install -r requirements-advanced.txt ({e})"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
    model.eval()
    le = joblib.load(DISTILBERT_DIR / "label_encoder.joblib")
    return model, tokenizer, le


def predict_emotion(text: str) -> Tuple[str, float]:
    cleaned = clean_text(text or "")
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")

    model, tokenizer, le = _load()
    import torch

    enc = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        ).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    idx = int(probs.argmax())
    return le.inverse_transform([idx])[0], float(probs[idx])


def predict_with_probs(text: str) -> Tuple[str, float, Dict[str, float]]:

    cleaned = clean_text(text or "")
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")

    model, tokenizer, le = _load()
    import torch

    enc = tokenizer(
        cleaned, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).logits[0]
    probs_arr = torch.softmax(logits, dim=-1).cpu().numpy()
    probs = {str(le.classes_[i]): float(probs_arr[i]) for i in range(len(le.classes_))}
    idx = int(probs_arr.argmax())
    return str(le.inverse_transform([idx])[0]), float(probs_arr[idx]), probs
