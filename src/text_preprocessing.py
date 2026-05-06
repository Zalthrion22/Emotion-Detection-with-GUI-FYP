"""Text cleaning and GoEmotions loading utilities.

GoEmotions TSV format (from Google Research GitHub):
    column 0: raw text
    column 1: comma-separated integer label IDs (0..27)
    column 2: Reddit comment ID

"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import GOEMOTIONS_LABEL_MAP, TARGET_EMOTIONS, TEXT_DATA_DIR

# Index <-> name for GoEmotions' 28 labels (from the official emotions.txt).
GOEMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_PLACEHOLDER_RE = re.compile(r"\[\s*name\s*\]", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7e]")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _PLACEHOLDER_RE.sub(" ", s)
    s = _NON_PRINTABLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _first_target_label(label_id_csv: str) -> Optional[str]:
    for raw in label_id_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            idx = int(raw)
        except ValueError:
            continue
        if 0 <= idx < len(GOEMOTIONS_LABELS):
            mapped = GOEMOTIONS_LABEL_MAP.get(GOEMOTIONS_LABELS[idx])
            if mapped in TARGET_EMOTIONS:
                return mapped
    return None


def load_goemotions(split: str, raw_dir: Path = TEXT_DATA_DIR) -> pd.DataFrame:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"split must be train/dev/test, got {split!r}")
    path = Path(raw_dir) / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing GoEmotions split at {path}. Download the TSVs from "
            "https://github.com/google-research/google-research/tree/master/goemotions/data"
        )
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str,
        keep_default_na=False,
        quoting=3,  # GoEmotions text contains stray quotes
    )
    df["emotion"] = df["label_ids"].map(_first_target_label)
    df = df.dropna(subset=["emotion"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    return df[["text", "emotion"]].reset_index(drop=True)


def load_goemotions_all(raw_dir: Path = TEXT_DATA_DIR) -> pd.DataFrame:
    frames = []
    for split in ("train", "dev", "test"):
        d = load_goemotions(split, raw_dir=raw_dir)
        d["split"] = split
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
