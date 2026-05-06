"""Integration tests for src/predict_speech.py.

Skipped automatically if the trained speech-model artifacts are missing.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import predict_speech
from src.config import (
    SAMPLE_RATE,
    SPEECH_LABEL_ENCODER_PATH,
    SPEECH_MODEL_PATH,
)
from src.predict_speech import (
    SpeechModelNotTrainedError,
    predict_emotion,
    predict_with_probs,
)

pytestmark = pytest.mark.skipif(
    not (SPEECH_MODEL_PATH.exists() and SPEECH_LABEL_ENCODER_PATH.exists()),
    reason="Speech model artifacts missing -- run python -m src.train_speech_models first.",
)

VALID_LABELS = {"anger", "fear", "joy", "sadness", "surprise", "neutral"}


def _write_sine_wav(path: Path, duration_s: float = 1.0, freq: float = 220.0) -> None:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), y, SAMPLE_RATE, subtype="PCM_16")


def test_predict_returns_label_and_confidence(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav)
    label, conf = predict_emotion(wav)
    assert label in VALID_LABELS
    assert 0.0 <= conf <= 1.0


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_emotion(tmp_path / "does_not_exist.wav")


def test_unsupported_extension_raises(tmp_path):
    other = tmp_path / "fake.mp3"
    other.write_bytes(b"not really an mp3")
    with pytest.raises(ValueError, match="Unsupported file type"):
        predict_emotion(other)


def test_silent_audio_raises(tmp_path):
    # All-zero wav: librosa.effects.trim removes everything, which the
    # loader translates into a ValueError.
    silent = tmp_path / "silent.wav"
    silent_data = np.zeros(SAMPLE_RATE, dtype=np.float32)
    sf.write(str(silent), silent_data, SAMPLE_RATE, subtype="PCM_16")
    with pytest.raises(ValueError):
        predict_emotion(silent)


# model_name dispatch
@pytest.mark.parametrize("model_name", ["best", "random_forest", "svm_rbf"])
def test_each_model_name_returns_valid_prediction(tmp_path, model_name):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav)
    label, conf = predict_emotion(wav, model_name=model_name)
    assert label in VALID_LABELS
    assert 0.0 <= conf <= 1.0


# predict_with_probs surface
@pytest.mark.parametrize("model_name", ["best", "random_forest", "svm_rbf"])
def test_predict_with_probs_returns_full_distribution(tmp_path, model_name):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav)
    label, conf, probs = predict_with_probs(wav, model_name=model_name)
    assert label in VALID_LABELS
    assert 0.0 <= conf <= 1.0
    assert set(probs.keys()) == VALID_LABELS
    for v in probs.values():
        assert 0.0 <= v <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    top = max(probs, key=probs.get)
    assert top == label
    assert abs(probs[label] - conf) < 1e-9


def test_unknown_model_name_raises_value_error(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav)
    with pytest.raises(ValueError, match="Unknown speech model"):
        predict_emotion(wav, model_name="bogus")


def test_missing_artifacts_raises(monkeypatch, tmp_path):
    # Patch the path the "best" key resolves to inside the per-model
    # dispatch dict introduced for the GUI's model dropdown.
    monkeypatch.setitem(
        predict_speech._SPEECH_MODEL_PATHS, "best", tmp_path / "missing.joblib"
    )
    predict_speech._load.cache_clear()
    try:
        wav = tmp_path / "x.wav"
        _write_sine_wav(wav)
        with pytest.raises(SpeechModelNotTrainedError):
            predict_emotion(wav)
    finally:
        predict_speech._load.cache_clear()
