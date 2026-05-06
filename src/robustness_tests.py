"""Run prescribed robustness tests and write a report-ready markdown table.
Originally for report but kept in

Outputs
-------
outputs/reports/robustness_tests.md
outputs/reports/robustness_audio/   
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from src.config import REPORTS_DIR, SAMPLE_RATE
from src.predict_speech import predict_emotion as predict_speech
from src.predict_text import predict_emotion as predict_text
from src.speech_features import iter_wav_files, parse_ravdess_label

# Distil BERT is optional 
def _try_distilbert() -> Optional[Callable]:
    try:
        from src.predict_text_distilbert import predict_emotion as p
        p("warm-up")  # trigger lazy load + token_type_ids fix etc.
        return p
    except Exception:
        return None


TEXT_CASES = [
    ("empty",         "",
     "should raise (empty after cleaning)"),
    ("very short",    "fine",
     "neutral or weak signal"),
    ("slang",         "I'm so done lol",
     "negative or neutral"),
    ("misspelled",    "I am so angery",
     "anger ideally; misspelling reduces TF-IDF coverage"),
    ("sarcasm",       "Great, another disaster",
     "anger/sadness ideally; bag-of-words is fooled by 'Great'"),
    ("mixed emotion", "I am happy but also nervous",
     "joy or fear -- the model has to pick one"),
    ("long",          ("I have been waiting for this moment for years. "
                       "It feels surreal that I finally made it. "
                       "I cannot stop smiling."),
     "joy"),
]


def _safe_input(text: str, limit: int = 80) -> str:
    s = repr(text)
    return s if len(s) <= limit else (s[: limit - 3] + "...")


def _run_text(predictor: Optional[Callable], text: str) -> str:
    if predictor is None:
        return "(unavailable)"
    try:
        label, conf = predictor(text)
        return f"{label} ({conf * 100:.1f}%)"
    except ValueError as e:
        return f"REJECTED: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def _make_sine_wav(path: Path, duration_s: float, freq: float = 220.0,
                   amp: float = 0.3) -> None:
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    y = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), y, SAMPLE_RATE, subtype="PCM_16")


def _pick_ravdess_clip(target_label: str) -> Optional[Path]:
    for p in iter_wav_files():
        if parse_ravdess_label(p.name) == target_label:
            return p
    return None


def _run_speech(wav_path) -> str:
    try:
        label, conf = predict_speech(wav_path)
        return f"{label} ({conf * 100:.1f}%)"
    except (ValueError, FileNotFoundError) as e:
        return f"REJECTED: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def main() -> None:
    print("=== Text robustness ===")
    distilbert = _try_distilbert()
    text_rows = []
    for name, text, expected in TEXT_CASES:
        classical = _run_text(predict_text, text)
        bert = _run_text(distilbert, text)
        print(f"  {name:>15s}: classical={classical} | distilbert={bert}")
        text_rows.append({
            "case": name, "input": _safe_input(text), "expected": expected,
            "classical": classical, "distilbert": bert,
        })

    print()
    print("=== Speech robustness ===")
    audio_dir = REPORTS_DIR / "robustness_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    speech_rows = []

    valid = _pick_ravdess_clip("anger")
    if valid is not None:
        result = _run_speech(valid)
        speech_rows.append({
            "case": "valid .wav (RAVDESS anger)", "input": valid.name,
            "expected": "anger", "result": result,
        })
        print(f"  valid .wav (RAVDESS anger):    {result}")

    fake = audio_dir / "fake.mp3"
    fake.write_bytes(b"definitely not an mp3")
    result = _run_speech(fake)
    speech_rows.append({
        "case": "unsupported file type", "input": "fake.mp3 (random bytes)",
        "expected": "REJECTED", "result": result,
    })
    print(f"  unsupported file type:         {result}")

    short = audio_dir / "short.wav"
    _make_sine_wav(short, duration_s=0.2)
    result = _run_speech(short)
    speech_rows.append({
        "case": "short audio (0.2s)", "input": "synthetic 0.2s sine",
        "expected": "any (low confidence)", "result": result,
    })
    print(f"  short audio (0.2s):            {result}")

    quiet = audio_dir / "quiet.wav"
    _make_sine_wav(quiet, duration_s=1.0, amp=0.0001)
    result = _run_speech(quiet)
    speech_rows.append({
        "case": "very quiet audio", "input": "synthetic 1s sine, amp 1e-4",
        "expected": "REJECTED (silent)", "result": result,
    })
    print(f"  very quiet audio:              {result}")

    for dur in (1.0, 3.0, 6.0):
        f = audio_dir / f"len_{int(dur)}s.wav"
        _make_sine_wav(f, duration_s=dur)
        result = _run_speech(f)
        speech_rows.append({
            "case": f"length {int(dur)}s",
            "input": f"synthetic {int(dur)}s sine",
            "expected": "any (length should not crash)", "result": result,
        })
        print(f"  length {int(dur)}s:                    {result}")

    missing = audio_dir / "does_not_exist.wav"
    if missing.exists():
        missing.unlink()
    result = _run_speech(missing)
    speech_rows.append({
        "case": "missing file", "input": "does_not_exist.wav",
        "expected": "REJECTED", "result": result,
    })
    print(f"  missing file:                  {result}")

    out = ["# Robustness tests", "",
           "Generated by `python -m src.robustness_tests`.", "",
           "## Text", "",
           "| Case | Input | Expected | Classical (LogReg) | DistilBERT |",
           "|---|---|---|---|---|"]
    for r in text_rows:
        out.append(
            f"| {r['case']} | `{r['input']}` | {r['expected']} | "
            f"{r['classical']} | {r['distilbert']} |"
        )
    out += ["", "## Speech (SVM-RBF)", "",
            "| Case | Input | Expected | Result |",
            "|---|---|---|---|"]
    for r in speech_rows:
        out.append(
            f"| {r['case']} | {r['input']} | {r['expected']} | {r['result']} |"
        )

    out_path = REPORTS_DIR / "robustness_tests.md"
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
