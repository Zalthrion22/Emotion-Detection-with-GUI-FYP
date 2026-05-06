"""Train and compare two classical text models on GoEmotions (6-class).


Outputs

models/text_vectorizer.joblib
models/text_label_encoder.joblib
models/text_model.joblib                  
outputs/figures/text_confusion_<model>.png
outputs/reports/text_classification_report_<model>.txt
outputs/reports/text_model_comparison.csv
outputs/reports/text_best_model.json
"""
from __future__ import annotations

import json
from typing import Dict

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend so this works over SSH / in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.config import (
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_EMOTIONS,
    TEXT_LABEL_ENCODER_PATH,
    TEXT_LINEARSVC_PATH,
    TEXT_LOGREG_PATH,
    TEXT_MODEL_PATH,
    TEXT_VECTORIZER_PATH,
)
from src.text_preprocessing import load_goemotions


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def _plot_confusion_matrix(cm: np.ndarray, label_names: list, title: str, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list,
) -> Dict[str, float]:
    report = classification_report(
        y_true, y_pred, target_names=label_names, digits=4, zero_division=0
    )
    (REPORTS_DIR / f"text_classification_report_{model_name}.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    _plot_confusion_matrix(
        cm,
        label_names,
        f"Confusion matrix - {model_name}",
        FIGURES_DIR / f"text_confusion_{model_name}.png",
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main() -> None:
    print("[1/5] Loading GoEmotions...")
    train_df = load_goemotions("train")
    dev_df = load_goemotions("dev")
    test_df = load_goemotions("test")
    train_df = pd.concat([train_df, dev_df], ignore_index=True)
    print(f"   train+dev rows: {len(train_df)}, test rows: {len(test_df)}")
    print("   class distribution (train+dev):")
    print(train_df["emotion"].value_counts().to_string())

    print("[2/5] Encoding labels and vectorising text...")
    le = LabelEncoder().fit(TARGET_EMOTIONS)
    y_train = le.transform(train_df["emotion"])
    y_test = le.transform(test_df["emotion"])
    label_names = list(le.classes_)

    vec = _build_vectorizer()
    X_train = vec.fit_transform(train_df["text"])
    X_test = vec.transform(test_df["text"])

    # LinearSVC is wrapped in CalibratedClassifierCV so we can produce
    # probability-style confidence scores in the GUI.
    candidates = {
        "logreg": LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "linearsvc": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            cv=3,
        ),
    }

    print("[3/5] Training & evaluating models...")
    results: Dict[str, Dict[str, float]] = {}
    fitted: Dict[str, object] = {}
    for name, est in candidates.items():
        print(f"   - {name}")
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        results[name] = _evaluate(name, y_test, y_pred, label_names)
        fitted[name] = est

    print("[4/5] Comparison:")
    comparison = pd.DataFrame(results).T.round(4)
    print(comparison.to_string())
    comparison.to_csv(REPORTS_DIR / "text_model_comparison.csv")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"   Best by macro-F1: {best_name}")

    print("[5/5] Saving models + vectoriser + label encoder...")
    # Best-model pointer (preserved for backward compatibility)
    joblib.dump(fitted[best_name], TEXT_MODEL_PATH)
    joblib.dump(vec, TEXT_VECTORIZER_PATH)
    joblib.dump(le, TEXT_LABEL_ENCODER_PATH)
    # Per-model artifacts so the GUI dropdown can offer either choice
    joblib.dump(fitted["logreg"], TEXT_LOGREG_PATH)
    joblib.dump(fitted["linearsvc"], TEXT_LINEARSVC_PATH)
    meta = {"best_model": best_name, "metrics": results[best_name]}
    (REPORTS_DIR / "text_best_model.json").write_text(json.dumps(meta, indent=2))
    print(
        f"   Saved: {TEXT_MODEL_PATH.name} (best={best_name}), "
        f"{TEXT_LOGREG_PATH.name}, {TEXT_LINEARSVC_PATH.name}, "
        f"{TEXT_VECTORIZER_PATH.name}, {TEXT_LABEL_ENCODER_PATH.name}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
