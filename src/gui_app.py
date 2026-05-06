"""Desktop GUI for dual-modality emotion detection.

"""
from __future__ import annotations

import threading
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable, Deque, Dict, Optional, Tuple

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.predict_speech import (
    SpeechModelNotTrainedError,
    predict_with_probs as predict_speech_probs,
)
from src.predict_text import (
    ModelNotTrainedError,
    predict_with_probs as predict_text_probs,
)
from src.predict_text_distilbert import (
    DistilBertNotTrainedError,
    predict_with_probs as predict_distilbert_probs,
)

# look
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("green")

WINDOW_TITLE = "Dual-Modality Emotion Detection"
WINDOW_GEOMETRY = "1100x740"

TARGET_EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise", "neutral"]

EMOTION_COLOURS = {
    "anger":    "#d62728",
    "fear":     "#9467bd",
    "joy":      "#2ca02c",
    "sadness":  "#1f77b4",
    "surprise": "#ff7f0e",
    "neutral":  "#7f7f7f",
}

ERROR_COLOUR = ("#b00020", "#ff5252")
DEFAULT_COLOUR = ("gray20", "gray85")
SUBDUED_COLOUR = ("gray35", "gray70")
STATUS_COLOUR = ("gray15", "gray85")

TEXT_BUTTON_LABEL = "Predict"
SPEECH_BUTTON_LABEL = "Predict"
PLAY_AUDIO_BUTTON_LABEL = "Play selected audio"

TEXT_MODEL_DISPLAY_TO_KEY = {
    "Logistic Regression": "logreg",
    "Linear SVM":          "linearsvc",
    "DistilBERT":          "distilbert",
}
SPEECH_MODEL_DISPLAY_TO_KEY = {
    "Random Forest": "random_forest",
    "SVM-RBF":       "svm_rbf",
}

RECORDING_DURATION = 4.0
HISTORY_LIMIT = 3
TEXT_PREVIEW_MAX = 50

HELP_INTRO = (
    "Type a sentence or load / record a .wav clip "
    "then click Predict. The bar chart on the right "
    "shows the model's confidence in each of the six emotions."
)
HELP_CONFIDENCE_LINE = (
    "Confidence below ~50% means the model is unsure; treat the result "
    "as a guess, not a measurement."
)
TEXT_TAB_HELP = HELP_INTRO + "\n\n" + HELP_CONFIDENCE_LINE
SPEECH_TAB_HELP = (
    HELP_INTRO
    + "\n\nFor best results with the speech model:"
    + "\n  • Speak clearly and with intent ."
    + "\n  • Record in a quiet room and keep ~15 cm away from the microphone."
    + "\n  • " + HELP_CONFIDENCE_LINE
)


def _truncate(text: str, limit: int = TEXT_PREVIEW_MAX) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ConfidenceChart
class ConfidenceChart:
    """Embedded matplotlib bar chart showing per-class confidences (0-100%)."""

    def __init__(self, parent_widget):
        self.fig = Figure(figsize=(5.0, 2.6), dpi=100)
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_subplot(111)
        self._draw_empty()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_widget)
        self.tk_widget = self.canvas.get_tk_widget()
        # Suppress the white halo that Tk paints around the canvas widget.
        self.tk_widget.configure(highlightthickness=0, borderwidth=0)

    # styling
    def _style_axes(self):
        self.ax.set_facecolor("white")
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Confidence (%)", fontsize=9)
        self.ax.set_xticks(range(len(TARGET_EMOTIONS)))
        self.ax.set_xticklabels(TARGET_EMOTIONS, rotation=18, ha="right", fontsize=9)
        self.ax.tick_params(axis="y", labelsize=8)
        self.ax.grid(axis="y", alpha=0.25)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

    def _draw_empty(self):
        self.ax.clear()
        self._style_axes()
        self.fig.tight_layout()

    # public API
    def update(self, probs: Dict[str, float]) -> None:
        self.ax.clear()
        self._style_axes()
        values = [probs.get(e, 0.0) * 100 for e in TARGET_EMOTIONS]
        colours = [EMOTION_COLOURS[e] for e in TARGET_EMOTIONS]
        bars = self.ax.bar(range(len(TARGET_EMOTIONS)), values, color=colours)
        for bar, v in zip(bars, values):
            if v >= 1.0:
                self.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 1.5,
                    f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def reset(self) -> None:
        self._draw_empty()
        self.canvas.draw_idle()


# RecentResults
class RecentResultsPanel:
    """Shows the last ``HISTORY_LIMIT`` predictions as compact rows."""

    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.entries: Deque[dict] = deque(maxlen=HISTORY_LIMIT)
        self._row_widgets: list = []

        self.title_label = ctk.CTkLabel(
            parent_widget,
            text="Recent Predictions",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        self.title_label.pack(anchor="w", padx=4, pady=(0, 4))

        self.body = ctk.CTkFrame(parent_widget, fg_color="transparent")
        self.body.pack(fill="x", expand=False, padx=2)
        self._render()

    def push(
        self, *, emotion: str, confidence: float, model: str, input_preview: str
    ) -> None:
        self.entries.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "emotion": emotion,
            "confidence": confidence,
            "model": model,
            "input": input_preview,
        })
        self._render()

    def clear(self) -> None:
        self.entries.clear()
        self._render()

    def _clear_rows(self):
        for w in self._row_widgets:
            w.destroy()
        self._row_widgets = []

    def _render(self):
        self._clear_rows()
        if not self.entries:
            empty = ctk.CTkLabel(
                self.body,
                text="No predictions yet.",
                font=ctk.CTkFont(size=13),
                text_color=SUBDUED_COLOUR,
                anchor="w",
            )
            empty.pack(anchor="w", padx=4, pady=2)
            self._row_widgets.append(empty)
            return
        for entry in self.entries:
            row = ctk.CTkFrame(self.body, fg_color="transparent")
            row.pack(fill="x", pady=1)

            ctk.CTkLabel(
                row, text=entry["time"],
                font=ctk.CTkFont(size=12),
                text_color=SUBDUED_COLOUR,
                width=70, anchor="w",
            ).pack(side="left", padx=(2, 6))

            colour = EMOTION_COLOURS.get(entry["emotion"], "#000000")
            ctk.CTkLabel(
                row,
                text=f"{entry['emotion']}  {entry['confidence']*100:.1f}%",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=colour, width=160, anchor="w",
            ).pack(side="left", padx=4)

            ctk.CTkLabel(
                row,
                text=f"{entry['model']}  ·  {entry['input']}",
                font=ctk.CTkFont(size=12),
                text_color=SUBDUED_COLOUR,
                anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=4)

            self._row_widgets.append(row)


# EmotionDetectorApp
class EmotionDetectorApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_GEOMETRY)
        self.minsize(960, 680)

        self._app_icon = tk.PhotoImage(width=16, height=16)
        self._app_icon.put("black", to=(0, 0, 16, 16))
        try:
            self.iconphoto(True, self._app_icon)
        except tk.TclError:
            pass

        self._wav_path: Optional[Path] = None
        self._build_layout()

    # root layout
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        ctk.CTkLabel(
            header, text=WINDOW_TITLE,
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="by Gautam Mukthi",
            font=ctk.CTkFont(size=14, slant="italic"),
            text_color=SUBDUED_COLOUR,
        ).pack(anchor="w")

        # Tabs
        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
        text_tab = self.tabs.add("Text")
        speech_tab = self.tabs.add("Speech")
        self._build_text_tab(text_tab)
        self._build_speech_tab(speech_tab)

        # Status bar
        self.status_var = ctk.StringVar(value="Ready.")
        self.status_label = ctk.CTkLabel(
            self, textvariable=self.status_var,
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
            text_color=STATUS_COLOUR,
        )
        self.status_label.grid(row=2, column=0, sticky="ew", padx=22, pady=(4, 14))

    # text tab
    def _build_text_tab(self, tab) -> None:
        tab.grid_columnconfigure((0, 1), weight=1, uniform="cols")
        tab.grid_rowconfigure(1, weight=1)

        self._build_help_panel(tab, TEXT_TAB_HELP)

        # input
        left = ctk.CTkFrame(tab)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)

        # Model dropdown row
        model_row = ctk.CTkFrame(left, fg_color="transparent")
        model_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        ctk.CTkLabel(
            model_row, text="Model", font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(0, 8))
        self.text_model_var = ctk.StringVar(value="Logistic Regression")
        ctk.CTkOptionMenu(
            model_row,
            values=list(TEXT_MODEL_DISPLAY_TO_KEY.keys()),
            variable=self.text_model_var,
            width=200,
        ).pack(side="left")

        self.text_box = ctk.CTkTextbox(left, wrap="word")
        self.text_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=4)

        button_row = ctk.CTkFrame(left, fg_color="transparent")
        button_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 14))
        button_row.grid_columnconfigure((0, 1), weight=1, uniform="b")
        self.text_predict_btn = ctk.CTkButton(
            button_row, text=TEXT_BUTTON_LABEL, command=self._on_text_predict,
        )
        self.text_predict_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.text_clear_btn = ctk.CTkButton(
            button_row, text="Clear",
            command=self._on_text_clear,
            fg_color=("#a0a0a0", "#555555"),
            hover_color=("#888888", "#444444"),
        )
        self.text_clear_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # output
        right = ctk.CTkFrame(tab)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self._build_output_panel(right, modality="text")

    # speech tab
    def _build_speech_tab(self, tab) -> None:
        tab.grid_columnconfigure((0, 1), weight=1, uniform="cols")
        tab.grid_rowconfigure(1, weight=1)

        self._build_help_panel(tab, SPEECH_TAB_HELP)

        # input
        left = ctk.CTkFrame(tab)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.grid_columnconfigure(0, weight=1)

        # Model dropdown row
        model_row = ctk.CTkFrame(left, fg_color="transparent")
        model_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        ctk.CTkLabel(
            model_row, text="Model", font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(0, 8))
        self.speech_model_var = ctk.StringVar(value="SVM-RBF")
        ctk.CTkOptionMenu(
            model_row,
            values=list(SPEECH_MODEL_DISPLAY_TO_KEY.keys()),
            variable=self.speech_model_var,
            width=200,
        ).pack(side="left")

        self.browse_btn = ctk.CTkButton(
            left, text="Browse .wav file...", command=self._on_browse_audio,
        )
        self.browse_btn.grid(row=1, column=0, sticky="ew", padx=14, pady=4)

        self.record_btn = ctk.CTkButton(
            left,
            text=f"Record from microphone ({int(RECORDING_DURATION)}s)",
            command=self._on_record,
            fg_color=("#bb3333", "#aa3333"),
            hover_color=("#a52929", "#992525"),
        )
        self.record_btn.grid(row=2, column=0, sticky="ew", padx=14, pady=4)

        self.play_audio_btn = ctk.CTkButton(
            left,
            text=PLAY_AUDIO_BUTTON_LABEL,
            command=self._on_play_audio,
            state="disabled",
        )
        self.play_audio_btn.grid(row=3, column=0, sticky="ew", padx=14, pady=4)

        self.selected_file_label = ctk.CTkLabel(
            left, text="No file selected.",
            font=ctk.CTkFont(size=13), text_color=SUBDUED_COLOUR,
            anchor="w", wraplength=380,
        )
        self.selected_file_label.grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 4))

        button_row = ctk.CTkFrame(left, fg_color="transparent")
        button_row.grid(row=5, column=0, sticky="ew", padx=14, pady=(12, 14))
        button_row.grid_columnconfigure((0, 1), weight=1, uniform="b")
        self.speech_predict_btn = ctk.CTkButton(
            button_row, text=SPEECH_BUTTON_LABEL,
            command=self._on_speech_predict, state="disabled",
        )
        self.speech_predict_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.speech_clear_btn = ctk.CTkButton(
            button_row, text="Clear",
            command=self._on_speech_clear,
            fg_color=("#a0a0a0", "#555555"),
            hover_color=("#888888", "#444444"),
        )
        self.speech_clear_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # output
        right = ctk.CTkFrame(tab)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self._build_output_panel(right, modality="speech")

    # help panel (shared)
    def _build_help_panel(self, tab, help_body: str) -> None:
        help_frame = ctk.CTkFrame(tab)
        help_frame.grid(
            row=0, column=0, columnspan=2, sticky="ew",
            padx=8, pady=(8, 0),
        )
        help_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            help_frame, text="How to use this model",
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            help_frame, text=help_body,
            font=ctk.CTkFont(size=12), justify="left", anchor="w",
            text_color=SUBDUED_COLOUR, wraplength=1020,
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

    def _build_output_panel(self, container, modality: str) -> None:
        # Result display
        result_frame = ctk.CTkFrame(container, fg_color="transparent")
        result_frame.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 4))
        ctk.CTkLabel(
            result_frame, text="Predicted emotion",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUBDUED_COLOUR, anchor="w",
        ).pack(anchor="w")
        result_label = ctk.CTkLabel(
            result_frame, text="No prediction yet.",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=DEFAULT_COLOUR, anchor="w",
        )
        result_label.pack(anchor="w", pady=(2, 0))
        confidence_label = ctk.CTkLabel(
            result_frame, text="",
            font=ctk.CTkFont(size=14),
            text_color=SUBDUED_COLOUR, anchor="w",
        )
        confidence_label.pack(anchor="w", pady=(0, 4))

        # Chart
        chart_frame = ctk.CTkFrame(container)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=4)
        chart = ConfidenceChart(chart_frame)
        chart.tk_widget.pack(fill="both", expand=True, padx=4, pady=4)

        # History
        history_frame = ctk.CTkFrame(container, fg_color="transparent")
        history_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 14))
        history = RecentResultsPanel(history_frame)

        if modality == "text":
            self.text_result_label = result_label
            self.text_confidence_label = confidence_label
            self.text_chart = chart
            self.text_history = history
        else:
            self.speech_result_label = result_label
            self.speech_confidence_label = confidence_label
            self.speech_chart = chart
            self.speech_history = history

    # text events
    def _on_text_predict(self) -> None:
        text = self.text_box.get("0.0", "end").strip()
        if not text:
            self._set_status("Error: invalid input", error=True)
            self._set_result_error(
                self.text_result_label, self.text_confidence_label,
                "Please type something first.",
            )
            return
        display_name = self.text_model_var.get()
        model_key = TEXT_MODEL_DISPLAY_TO_KEY.get(display_name, "logreg")
        is_distilbert = (model_key == "distilbert")

        if is_distilbert and self._is_distilbert_cold():
            self._set_status(
                "Loading DistilBERT, this may take a few seconds..."
            )
        else:
            self._set_status("Predicting...")

        self._run_prediction(
            modality="text",
            buttons=(self.text_predict_btn, self.text_clear_btn),
            button_labels=(TEXT_BUTTON_LABEL, "Clear"),
            task=lambda: self._dispatch_text_prediction(text, model_key),
            input_preview=_truncate(text),
            model_label=display_name,
        )

    def _on_text_clear(self) -> None:
        self.text_box.delete("0.0", "end")
        self.text_result_label.configure(text="No prediction yet.", text_color=DEFAULT_COLOUR)
        self.text_confidence_label.configure(text="")
        self.text_chart.reset()
        self._set_status("Ready.")

    def _dispatch_text_prediction(
        self, text: str, model_key: str
    ) -> Tuple[str, float, Dict[str, float]]:
        if model_key == "distilbert":
            return predict_distilbert_probs(text)
        return predict_text_probs(text, model_name=model_key)

    def _is_distilbert_cold(self) -> bool:
        """Return True if the DistilBERT artifacts have not yet been loaded
        into the lru_cache. Used to display a clearer status message on
        the first DistilBERT prediction."""
        try:
            from src.predict_text_distilbert import _load
            return _load.cache_info().currsize == 0
        except Exception:
            return False

    # speech events
    def _on_browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a .wav audio file",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".wav":
            self._set_status("Error: invalid input", error=True)
            self._set_result_error(
                self.speech_result_label, self.speech_confidence_label,
                f"Unsupported file type: {p.suffix or '(none)'}.",
            )
            self._wav_path = None
            self.speech_predict_btn.configure(state="disabled")
            self.selected_file_label.configure(text="No file selected.")
            self.play_audio_btn.configure(state="disabled")
            return
        self._wav_path = p
        self.selected_file_label.configure(text=f"Selected: {p.name}")
        self.speech_predict_btn.configure(state="normal")
        self.play_audio_btn.configure(state="normal")
        self._set_status(f"Selected {p.name}")

    def _on_record(self) -> None:
        self.record_btn.configure(state="disabled", text="Recording...")
        self.browse_btn.configure(state="disabled")
        self.speech_predict_btn.configure(state="disabled")
        self.speech_clear_btn.configure(state="disabled")
        self.play_audio_btn.configure(state="disabled")
        self._set_status(f"Recording {int(RECORDING_DURATION)} seconds...")

        def worker():
            try:
                from src.record_audio import (
                    MicrophoneError, record_microphone,
                )
                path = record_microphone(duration_seconds=RECORDING_DURATION)
            except MicrophoneError as exc:
                msg = str(exc)
                self.after(0, lambda: self._on_recording_failed(msg))
                return
            except Exception as exc:
                msg = f"Microphone recording failed: {exc}"
                self.after(0, lambda: self._on_recording_failed(msg))
                return
            self.after(0, lambda: self._on_recording_done(path))

        threading.Thread(target=worker, daemon=True).start()

    def _on_recording_done(self, path: Path) -> None:
        self._wav_path = path
        self.selected_file_label.configure(text=f"Recorded: {path.name}")
        self.record_btn.configure(
            state="normal",
            text=f"Record from microphone ({int(RECORDING_DURATION)}s)",
        )
        self.browse_btn.configure(state="normal")
        self.speech_predict_btn.configure(state="normal")
        self.speech_clear_btn.configure(state="normal")
        self.play_audio_btn.configure(state="normal")
        self._set_status("Recording complete. Click Predict to classify.")

    def _on_recording_failed(self, msg: str) -> None:
        self.record_btn.configure(
            state="normal",
            text=f"Record from microphone ({int(RECORDING_DURATION)}s)",
        )
        self.browse_btn.configure(state="normal")
        self.speech_clear_btn.configure(state="normal")
        if self._wav_path is not None and self._wav_path.exists():
            self.speech_predict_btn.configure(state="normal")
            self.play_audio_btn.configure(state="normal")
        else:
            self.speech_predict_btn.configure(state="disabled")
            self.play_audio_btn.configure(state="disabled")
        self._set_result_error(
            self.speech_result_label, self.speech_confidence_label, msg,
        )
        self._set_status("Error: microphone recording failed", error=True)

    def _on_play_audio(self) -> None:
        if self._wav_path is None or not self._wav_path.exists():
            self._set_status("Error: audio file not found", error=True)
            self.play_audio_btn.configure(state="disabled")
            messagebox.showerror(
                "Playback error",
                "Audio file not found. Please select or record a .wav file first.",
            )
            return

        wav = self._wav_path
        self.play_audio_btn.configure(state="disabled", text="Playing...")
        self._set_status(f"Playing {wav.name}...")

        def workerTwo():
            try:
                import sounddevice as sd
                import soundfile as sf

                audio, sample_rate = sf.read(str(wav), dtype="float32")
                if getattr(audio, "size", 0) == 0:
                    raise ValueError("Selected audio file is empty.")

                sd.stop()
                sd.play(audio, sample_rate)
                sd.wait()

            except ImportError:
                msg = (
                    "Audio playback requires sounddevice and soundfile. "
                    "Install them with: pip install sounddevice soundfile"
                )
                self.after(0, lambda: self._on_playback_failed(msg))
                return
            except Exception as exc:
                msg = f"Audio playback failed: {exc}"
                self.after(0, lambda: self._on_playback_failed(msg))
                return
            self.after(0, self._on_playback_done)
        threading.Thread(target=workerTwo, daemon=True).start()

    def _on_playback_done(self) -> None:
        if self._wav_path is not None and self._wav_path.exists():
            self.play_audio_btn.configure(state="normal", text=PLAY_AUDIO_BUTTON_LABEL)
        else:
            self.play_audio_btn.configure(state="disabled", text=PLAY_AUDIO_BUTTON_LABEL)
        self._set_status("Playback complete.")

    def _on_playback_failed(self, msg: str) -> None:
        if self._wav_path is not None and self._wav_path.exists():
            self.play_audio_btn.configure(state="normal", text=PLAY_AUDIO_BUTTON_LABEL)
        else:
            self.play_audio_btn.configure(state="disabled", text=PLAY_AUDIO_BUTTON_LABEL)

        self._set_status("Error: audio playback failed", error=True)
        messagebox.showerror("Playback error", msg)

    def _on_speech_predict(self) -> None:
        if self._wav_path is None or not self._wav_path.exists():
            self._set_status("Error: invalid input", error=True)
            self._set_result_error(
                self.speech_result_label, self.speech_confidence_label,
                "Audio file not found. Please re-select.",
            )
            self.speech_predict_btn.configure(state="disabled")
            return
        wav = self._wav_path
        display_name = self.speech_model_var.get()
        model_key = SPEECH_MODEL_DISPLAY_TO_KEY.get(display_name, "svm_rbf")

        self._set_status("Predicting...")
        self._run_prediction(
            modality="speech",
            buttons=(self.speech_predict_btn, self.speech_clear_btn),
            button_labels=(SPEECH_BUTTON_LABEL, "Clear"),
            task=lambda: predict_speech_probs(wav, model_name=model_key),
            input_preview=wav.name,
            model_label=display_name,
        )

    def _on_speech_clear(self) -> None:
        self._wav_path = None
        self.selected_file_label.configure(text="No file selected.")
        self.speech_predict_btn.configure(state="disabled")
        self.play_audio_btn.configure(state="disabled", text=PLAY_AUDIO_BUTTON_LABEL)
        self.speech_result_label.configure(text="No prediction yet.", text_color=DEFAULT_COLOUR)
        self.speech_confidence_label.configure(text="")
        self.speech_chart.reset()
        self._set_status("Ready.")

    # other shared logic
    def _run_prediction(
        self,
        *,
        modality: str,
        buttons: Tuple,
        button_labels: Tuple[str, str],
        task: Callable[[], Tuple[str, float, Dict[str, float]]],
        input_preview: str,
        model_label: str,
    ) -> None:
        # Disable interactive controls
        for b in buttons:
            b.configure(state="disabled")
        buttons[0].configure(text="Predicting...")

        def worker():
            try:
                emotion, confidence, probs = task()
                self.after(
                    0,
                    lambda: self._on_prediction_success(
                        modality, emotion, confidence, probs,
                        input_preview, model_label,
                    ),
                )
            except (
                ModelNotTrainedError,
                SpeechModelNotTrainedError,
                DistilBertNotTrainedError,
            ) as exc:
                msg = str(exc)
                title = (
                    "DistilBERT not available"
                    if isinstance(exc, DistilBertNotTrainedError)
                    else "Model not trained"
                )
                self.after(0, lambda m=msg, t=title: messagebox.showerror(t, m))
                self.after(0, lambda: self._on_prediction_error(modality, "Error: missing model file"))
            except (FileNotFoundError, ValueError) as exc:
                msg = str(exc)
                self.after(0, lambda: self._on_prediction_error(modality, msg))
            except Exception as exc:  # defensive catch-all
                msg = f"Unexpected error: {exc}"
                self.after(0, lambda: self._on_prediction_error(modality, msg))
            finally:
                self.after(
                    0,
                    lambda: self._restore_buttons(buttons, button_labels),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _restore_buttons(self, buttons, button_labels):
        for b, label in zip(buttons, button_labels):
            b.configure(state="normal", text=label)
        # speech_predict_btn should re-enable only if we still have a wav.
        if buttons and buttons[0] is self.speech_predict_btn:
            if self._wav_path is None or not self._wav_path.exists():
                self.speech_predict_btn.configure(state="disabled")

    # results
    def _on_prediction_success(
        self,
        modality: str,
        emotion: str,
        confidence: float,
        probs: Dict[str, float],
        input_preview: str,
        model_label: str,
    ) -> None:
        if modality == "text":
            self.text_result_label.configure(
                text=emotion, text_color=EMOTION_COLOURS.get(emotion, "#000000"),
            )
            self.text_confidence_label.configure(
                text=f"{confidence * 100:.1f}% confidence", text_color=SUBDUED_COLOUR,
            )
            self.text_chart.update(probs)
            self.text_history.push(
                emotion=emotion, confidence=confidence,
                model=model_label, input_preview=input_preview,
            )
        else:
            self.speech_result_label.configure(
                text=emotion, text_color=EMOTION_COLOURS.get(emotion, "#000000"),
            )
            self.speech_confidence_label.configure(
                text=f"{confidence * 100:.1f}% confidence", text_color=SUBDUED_COLOUR,
            )
            self.speech_chart.update(probs)
            self.speech_history.push(
                emotion=emotion, confidence=confidence,
                model=model_label, input_preview=input_preview,
            )
        self._set_status("Prediction complete.")

    def _on_prediction_error(self, modality: str, msg: str) -> None:
        if modality == "text":
            self._set_result_error(
                self.text_result_label, self.text_confidence_label, msg,
            )
        else:
            self._set_result_error(
                self.speech_result_label, self.speech_confidence_label, msg,
            )
        if "Missing artifact" in msg or "DistilBERT" in msg or "Model not loaded" in msg:
            self._set_status("Error: missing model file", error=True)
        elif "empty" in msg.lower() or "Unsupported" in msg or "not found" in msg.lower():
            self._set_status("Error: invalid input", error=True)
        else:
            self._set_status("Error during prediction", error=True)

    def _set_result_error(self, result_label, confidence_label, msg: str) -> None:
        result_label.configure(text="Error", text_color=ERROR_COLOUR)
        confidence_label.configure(text=msg, text_color=ERROR_COLOUR)

    # status
    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_var.set(message)
        self.status_label.configure(
            text_color=ERROR_COLOUR if error else STATUS_COLOUR,
        )


def main() -> None:
    app = EmotionDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
