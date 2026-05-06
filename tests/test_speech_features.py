"""Unit tests for src/speech_features.py.

"""
from __future__ import annotations

import numpy as np

from src.config import N_MFCC, SAMPLE_RATE
from src.speech_features import (
    extract_mfcc_features,
    parse_ravdess_actor,
    parse_ravdess_label,
)


# parse_ravdess_label
def test_parse_label_anger():
    # Emotion code 05 = angry -> anger
    assert parse_ravdess_label("03-01-05-01-01-01-12.wav") == "anger"


def test_parse_label_neutral_and_calm_both_map_to_neutral():
    assert parse_ravdess_label("03-01-01-01-01-01-12.wav") == "neutral"
    assert parse_ravdess_label("03-01-02-01-01-01-12.wav") == "neutral"


def test_parse_label_disgust_dropped():
    # Emotion code 07 = disgust -> dropped
    assert parse_ravdess_label("03-01-07-01-01-01-12.wav") is None


def test_parse_label_handles_directory_in_path():
    assert (
        parse_ravdess_label("/tmp/Actor_12/03-01-04-01-01-01-12.wav") == "sadness"
    )


def test_parse_label_malformed():
    assert parse_ravdess_label("nope.wav") is None
    assert parse_ravdess_label("03-01.wav") is None


# parse_ravdess_actor
def test_parse_actor_extracts_int():
    assert parse_ravdess_actor("03-01-05-01-01-01-07.wav") == 7
    assert parse_ravdess_actor("Actor_12/03-01-05-01-01-01-12.wav") == 12


def test_parse_actor_malformed():
    assert parse_ravdess_actor("nope.wav") is None


# extract_mfcc_features
def test_mfcc_returns_fixed_length_vector():
    # 1-second 440 Hz sine wave
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    feats = extract_mfcc_features(y, sr=SAMPLE_RATE)
    assert feats.shape == (2 * N_MFCC,)
    assert np.all(np.isfinite(feats))


def test_mfcc_length_invariance():
    # Clips of different durations should both produce same-shape feature
    rng = np.random.default_rng(0)
    short = rng.standard_normal(SAMPLE_RATE).astype(np.float32)        # 1s
    longer = rng.standard_normal(SAMPLE_RATE * 3).astype(np.float32)   # 3s
    a = extract_mfcc_features(short)
    b = extract_mfcc_features(longer)
    assert a.shape == b.shape == (2 * N_MFCC,)


def test_mfcc_distinguishes_signals():
    # Two qualitatively different signals should produce non-identical features.
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1
    a = extract_mfcc_features(sine)
    b = extract_mfcc_features(noise)
    assert not np.allclose(a, b)
