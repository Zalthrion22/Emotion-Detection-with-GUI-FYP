"""Build a unified comparison table across all trained models.
Originally used when writing report, but included for reproducibility

Outputs

outputs/reports/all_models_comparison.csv
outputs/reports/all_models_comparison.md
outputs/figures/all_models_f1_macro.png
"""
from __future__ import annotations

import json
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR

METRIC_COLS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
]


def _load_text_classical() -> List[Dict]:
    path = REPORTS_DIR / "text_model_comparison.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, index_col=0)
    rows = []
    for name, row in df.iterrows():
        rows.append({
            "model": name,
            "modality": "text",
            **{c: float(row[c]) for c in METRIC_COLS if c in row.index},
        })
    return rows


def _load_distilbert() -> List[Dict]:
    path = REPORTS_DIR / "text_distilbert_metrics.json"
    if not path.exists():
        return []
    metrics = json.loads(path.read_text())
    return [{
        "model": "distilbert",
        "modality": "text",
        **{c: float(metrics[c]) for c in METRIC_COLS if c in metrics},
    }]


def _load_speech() -> List[Dict]:
    path = REPORTS_DIR / "speech_model_comparison.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, index_col=0)
    rows = []
    for name, row in df.iterrows():
        rows.append({
            "model": name,
            "modality": "speech",
            **{c: float(row[c]) for c in METRIC_COLS if c in row.index},
        })
    return rows


def _to_markdown(df: pd.DataFrame) -> str:
    cols = ["modality", "model"] + METRIC_COLS
    df = df[cols].copy()
    for c in METRIC_COLS:
        df[c] = df[c].map(lambda v: f"{v:.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _plot_f1_bar(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df_sorted = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    colours = ["#1f77b4" if m == "text" else "#ff7f0e" for m in df_sorted["modality"]]
    bars = ax.bar(df_sorted["model"], df_sorted["f1_macro"], color=colours)
    ax.set_ylabel("Macro-F1 score")
    ax.set_title("Model comparison - macro F1 (text in blue, speech in orange)")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, df_sorted["f1_macro"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = _load_text_classical() + _load_distilbert() + _load_speech()
    if not rows:
        raise SystemExit(
            "No trained-model artifacts found. Train at least one model first."
        )
    df = pd.DataFrame(rows)
    df = df[["modality", "model"] + METRIC_COLS]
    df = df.sort_values(["modality", "f1_macro"], ascending=[True, False]).reset_index(drop=True)

    out_csv = REPORTS_DIR / "all_models_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print()
    print(df.to_string(index=False))
    print()

    md = _to_markdown(df)
    out_md = REPORTS_DIR / "all_models_comparison.md"
    out_md.write_text(md + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")

    out_png = FIGURES_DIR / "all_models_f1_macro.png"
    _plot_f1_bar(df, out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
