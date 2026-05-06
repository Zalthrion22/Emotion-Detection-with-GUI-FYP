"""Integration tests for src/predict_text.py.

Skipped automatically if the trained text-model artifacts are missing.

"""
from __future__ import annotations

import pytest

from src import predict_text
from src.config import (
    TEXT_LABEL_ENCODER_PATH,
    TEXT_MODEL_PATH,
    TEXT_VECTORIZER_PATH,
)
from src.predict_text import ModelNotTrainedError, predict_emotion, predict_with_probs

pytestmark = pytest.mark.skipif(
    not (
        TEXT_MODEL_PATH.exists()
        and TEXT_VECTORIZER_PATH.exists()
        and TEXT_LABEL_ENCODER_PATH.exists()
    ),
    reason="Text model artifacts missing -- run python -m src.train_text_models first.",
)

VALID_LABELS = {"anger", "fear", "joy", "sadness", "surprise", "neutral"}


def test_predict_returns_label_and_confidence():
    label, conf = predict_emotion("I am so happy today!")
    assert label in VALID_LABELS
    assert 0.0 <= conf <= 1.0


def test_predict_obviously_happy_text_is_joy_or_neutral():
    # Not asserting precise probabilities 
    # positive sentence is *not* classified as a negative emotion.
    label, _ = predict_emotion("I love this so much, it is wonderful!")
    assert label in {"joy", "neutral"}


def test_predict_obviously_negative_text_is_negative_or_neutral():
    label, _ = predict_emotion("I am furious and outraged about this disaster")
    assert label in {"anger", "sadness", "fear", "neutral"}


def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        predict_emotion("")


def test_whitespace_input_raises_value_error():
    with pytest.raises(ValueError):
        predict_emotion("   \n\t  ")


def test_input_that_cleans_to_empty_raises_value_error():
    # A URL on its own becomes empty after clean_text() strips it.
    with pytest.raises(ValueError):
        predict_emotion("https://example.com")


# model_name dispatch
@pytest.mark.parametrize("model_name", ["best", "logreg", "linearsvc"])
def test_each_model_name_returns_valid_prediction(model_name):
    """Every key in the dispatch dict must resolve to a working classifier."""
    label, conf = predict_emotion("I am so happy today!", model_name=model_name)
    assert label in VALID_LABELS
    assert 0.0 <= conf <= 1.0


# predict_with_probs surface
@pytest.mark.parametrize("model_name", ["best", "logreg", "linearsvc"])
def test_predict_with_probs_returns_full_distribution(model_name):
    label, conf, probs = predict_with_probs("I am so happy today!", model_name=model_name)
    assert label in VALID_LABELS
    assert 0.0 <= conf <= 1.0
    # All six classes present in the dict.
    assert set(probs.keys()) == VALID_LABELS
    # Probabilities are valid: in [0, 1] and sum approximately to 1.
    for v in probs.values():
        assert 0.0 <= v <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    # Top class agrees with the dict.
    top = max(probs, key=probs.get)
    assert top == label
    assert abs(probs[label] - conf) < 1e-9


def test_predict_with_probs_empty_input_raises():
    with pytest.raises(ValueError):
        predict_with_probs("")


def test_unknown_model_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown text model"):
        predict_emotion("hello", model_name="bogus")


def test_each_model_name_caches_separately():
    predict_text._load.cache_clear()
    predict_emotion("hi", model_name="logreg")
    predict_emotion("hi", model_name="linearsvc")
    info = predict_text._load.cache_info()
    # Two distinct keys cached, no hits yet between them.
    assert info.currsize == 2


def test_missing_artifacts_raises(monkeypatch, tmp_path):
    # Patch the path the "best" key resolves to inside the per-model
    # dispatch dict introduced for the GUI's model dropdown.
    monkeypatch.setitem(
        predict_text._TEXT_MODEL_PATHS, "best", tmp_path / "missing.joblib"
    )
    predict_text._load.cache_clear()
    try:
        with pytest.raises(ModelNotTrainedError):
            predict_emotion("hello world")
    finally:
        predict_text._load.cache_clear()
