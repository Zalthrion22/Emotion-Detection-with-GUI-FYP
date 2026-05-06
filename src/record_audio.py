"""Microphone capture wrapper used by the GUI's Record button.

Public API:
    record_microphone(duration_seconds: float = 4.0) -> Path

Records mono float32 audio at the project's SAMPLE_RATE from the default
input device, saves it as 16-bit PCM `.wav` in
``outputs/recordings/mic_recording.wav``, and returns the path.

The output file is **always overwritten** at the same path so the GUI does
not accumulate stale recordings on disk between sessions.

Errors raise ``MicrophoneError`` with a user-readable message:
- ``sounddevice`` not installed
- no microphone available (PortAudio reports no input device)
- recording call itself fails (driver / permission errors)
- the captured buffer is silent (mic muted, wrong device, OS permission denied)

The lazy ``import sounddevice`` keeps it out of GUI startup -- it is only
imported the first time the Record button is clicked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import RECORDINGS_DIR, SAMPLE_RATE


class MicrophoneError(RuntimeError):
    """Raised when microphone recording cannot complete cleanly."""


# Minimum peak amplitude for a recording to be considered non-silent.
# Matches the threshold used in ``speech_features.load_audio``.
_SILENT_PEAK_THRESHOLD = 1e-4


def record_microphone(duration_seconds: float = 4.0) -> Path:
    """Record from the default microphone and return the saved .wav path."""
    if duration_seconds <= 0:
        raise ValueError(
            f"duration_seconds must be > 0, got {duration_seconds}"
        )

    try:
        import sounddevice as sd
    except ImportError as e:  # pragma: no cover -- sounddevice is in requirements.txt
        raise MicrophoneError(
            "Microphone support requires the `sounddevice` package. "
            "Install with:  pip install -r requirements.txt"
        ) from e

    n_frames = int(SAMPLE_RATE * duration_seconds)
    try:
        recording = sd.rec(
            n_frames,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:
        # PortAudio surfaces "no default input", "Invalid device", permission
        # errors etc. as exceptions; we fold all of them into a single
        # readable MicrophoneError.
        msg = str(e)
        lowered = msg.lower()
        if "no default input" in lowered or "invalid device" in lowered or "no input" in lowered:
            raise MicrophoneError(
                f"No microphone detected ({msg}). Check that a microphone "
                "is connected and that the OS has granted permission."
            ) from e
        raise MicrophoneError(f"Microphone recording failed: {msg}") from e

    audio = np.asarray(recording).flatten()
    if audio.size == 0:
        raise MicrophoneError("Microphone returned an empty buffer.")
    peak = float(np.max(np.abs(audio)))
    if peak < _SILENT_PEAK_THRESHOLD:
        raise MicrophoneError(
            "Microphone recorded only silence. Check that the correct "
            "input device is selected and that it is not muted."
        )

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RECORDINGS_DIR / "mic_recording.wav"
    sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
    return out_path
