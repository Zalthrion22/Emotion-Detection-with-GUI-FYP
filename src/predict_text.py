"""Text emotion inference for the GUI.

"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import joblib
import numpy as np

from src.config import (
    TEXT_LABEL_ENCODER_PATH,
    TEXT_LINEARSVC_PATH,
    TEXT_LOGREG_PATH,
    TEXT_MODEL_PATH,
    TEXT_VECTORIZER_PATH,
)
from src.text_preprocessing import clean_text


class ModelNotTrainedError(RuntimeError):
    """Raised when the saved text-model artifacts are missing."""


# Display key -> filesystem path. ``"best"`` maps to whichever classifier
# the training script saved as the best macro-F1 winner.
_TEXT_MODEL_PATHS = {
    "best": TEXT_MODEL_PATH,
    "logreg": TEXT_LOGREG_PATH,
    "linearsvc": TEXT_LINEARSVC_PATH,
}


@lru_cache(maxsize=4)
def _load(model_name: str = "best") -> tuple:
    if model_name not in _TEXT_MODEL_PATHS:
        raise ValueError(
            f"Unknown text model {model_name!r}. "
            f"Options: {sorted(_TEXT_MODEL_PATHS)}"
        )
    model_path = _TEXT_MODEL_PATHS[model_name]
    for p in (model_path, TEXT_VECTORIZER_PATH, TEXT_LABEL_ENCODER_PATH):
        if not p.exists():
            raise ModelNotTrainedError(
                f"Missing artifact: {p}. Train the text models first with:\n"
                "    python -m src.train_text_models"
            )
    model = joblib.load(model_path)
    vec = joblib.load(TEXT_VECTORIZER_PATH)
    le = joblib.load(TEXT_LABEL_ENCODER_PATH)
    return model, vec, le


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_emotion(text: str, model_name: str = "best") -> Tuple[str, float]:
    cleaned = clean_text(text or "")
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")

    model, vec, le = _load(model_name)
    X = vec.transform([cleaned])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        # Defensive fallback
        scores = model.decision_function(X)[0]
        idx = int(np.argmax(scores))
        confidence = float(_softmax(scores)[idx])

    return le.inverse_transform([idx])[0], confidence


def predict_with_probs(
    text: str, model_name: str = "best"
) -> Tuple[str, float, Dict[str, float]]:
    cleaned = clean_text(text or "")
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")

    model, vec, le = _load(model_name)
    X = vec.transform([cleaned])

    if hasattr(model, "predict_proba"):
        probs_arr = model.predict_proba(X)[0]
    else:
        scores = model.decision_function(X)[0]
        probs_arr = _softmax(scores)

    probs = {str(le.classes_[i]): float(probs_arr[i]) for i in range(len(le.classes_))}
    idx = int(np.argmax(probs_arr))
    return str(le.inverse_transform([idx])[0]), float(probs_arr[idx]), probs
