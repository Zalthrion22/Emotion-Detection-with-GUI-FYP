"""Speech emotion inference for the GUI.

"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

from src.config import (
    SPEECH_LABEL_ENCODER_PATH,
    SPEECH_MODEL_PATH,
    SPEECH_RF_PATH,
    SPEECH_SVM_PATH,
)
from src.speech_features import extract_mfcc_features, load_audio


SUPPORTED_EXTENSIONS = {".wav"}


class SpeechModelNotTrainedError(RuntimeError):
    """Raised when the saved speech-model artifacts are missing."""


# Display key -> filesystem path. ``"best"`` maps to whichever classifier
# the training script saved as the best macro-F1 winner.
_SPEECH_MODEL_PATHS = {
    "best": SPEECH_MODEL_PATH,
    "random_forest": SPEECH_RF_PATH,
    "svm_rbf": SPEECH_SVM_PATH,
}


@lru_cache(maxsize=4)
def _load(model_name: str = "best") -> tuple:
    if model_name not in _SPEECH_MODEL_PATHS:
        raise ValueError(
            f"Unknown speech model {model_name!r}. "
            f"Options: {sorted(_SPEECH_MODEL_PATHS)}"
        )
    model_path = _SPEECH_MODEL_PATHS[model_name]
    for p in (model_path, SPEECH_LABEL_ENCODER_PATH):
        if not p.exists():
            raise SpeechModelNotTrainedError(
                f"Missing artifact: {p}. Train the speech models first with:\n"
                "    python -m src.train_speech_models"
            )
    return joblib.load(model_path), joblib.load(SPEECH_LABEL_ENCODER_PATH)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_emotion(wav_path, model_name: str = "best") -> Tuple[str, float]:
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{p.suffix}'. Only {sorted(SUPPORTED_EXTENSIONS)} are supported."
        )

    try:
        y = load_audio(p)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read audio file: {e}") from e

    feats = extract_mfcc_features(y).reshape(1, -1)
    model, le = _load(model_name)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        scores = model.decision_function(feats)[0]
        idx = int(np.argmax(scores))
        confidence = float(_softmax(scores)[idx])

    return le.inverse_transform([idx])[0], confidence

def predict_with_probs(
    wav_path, model_name: str = "best"
) -> Tuple[str, float, Dict[str, float]]:
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{p.suffix}'. Only {sorted(SUPPORTED_EXTENSIONS)} are supported."
        )

    try:
        y = load_audio(p)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read audio file: {e}") from e

    feats = extract_mfcc_features(y).reshape(1, -1)
    model, le = _load(model_name)

    if hasattr(model, "predict_proba"):
        probs_arr = model.predict_proba(feats)[0]
    else:
        scores = model.decision_function(feats)[0]
        probs_arr = _softmax(scores)

    probs = {str(le.classes_[i]): float(probs_arr[i]) for i in range(len(le.classes_))}
    idx = int(np.argmax(probs_arr))
    return str(le.inverse_transform([idx])[0]), float(probs_arr[idx]), probs
