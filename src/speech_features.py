"""Audio loading, RAVDESS label parsing, and MFCC feature extraction.

RAVDESS filename schema: ``MM-VC-EE-EI-ST-RE-AC.wav``

    MM  modality   (03 = audio-only)
    VC  vocal      (01 = speech, 02 = song)
    EE  emotion    (01..08 -- see RAVDESS_EMOTION_MAP)
    EI  intensity  (01 normal, 02 strong)
    ST  statement  (01 or 02)
    RE  repetition (01 or 02)
    AC  actor      (01..24, odd = male, even = female)

"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import librosa
import numpy as np

from src.config import (
    N_MFCC,
    RAVDESS_EMOTION_MAP,
    SAMPLE_RATE,
    SPEECH_DATA_DIR,
)


def parse_ravdess_label(filename: str) -> Optional[str]:
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 7:
        return None
    return RAVDESS_EMOTION_MAP.get(parts[2])


def parse_ravdess_actor(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 7:
        return None
    try:
        return int(parts[6])
    except ValueError:
        return None


def load_audio(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    if float(np.max(np.abs(y))) < 1e-4:
        raise ValueError(f"Audio is silent (peak amplitude below threshold): {path}")
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError(f"Audio is silent after trim: {path}")
    return y


def extract_mfcc_features(
    y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    return np.concatenate([mfcc_mean, mfcc_std], axis=0)


def iter_wav_files(speech_dir=SPEECH_DATA_DIR) -> Iterable[Path]:
    return sorted(Path(speech_dir).rglob("*.wav"))


def build_dataset(
    speech_dir=SPEECH_DATA_DIR, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[str] = []
    actors: List[int] = []
    skipped = 0
    files = list(iter_wav_files(speech_dir))
    for i, p in enumerate(files):
        label = parse_ravdess_label(p.name)
        actor = parse_ravdess_actor(p.name)
        if label is None or actor is None:
            skipped += 1
            continue
        try:
            y = load_audio(p)
            f = extract_mfcc_features(y)
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"   skip {p.name}: {e}")
            continue
        feats.append(f)
        labels.append(label)
        actors.append(actor)
        if verbose and (i + 1) % 200 == 0:
            print(f"   processed {i + 1}/{len(files)} files...")
    if not feats:
        raise RuntimeError(
            f"No usable audio found under {speech_dir}. "
            "Did you unzip RAVDESS into data/speech/ravdess/?"
        )
    if verbose:
        print(f"   total kept: {len(feats)}, skipped: {skipped}")
    return np.vstack(feats), np.array(labels), np.array(actors)
