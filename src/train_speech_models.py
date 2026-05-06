"""Train and compare two MFCC-based speech models on RAVDESS (6-class).


Outputs

models/speech_model.joblib                     
models/speech_label_encoder.joblib
outputs/figures/speech_confusion_<model>.png
outputs/reports/speech_classification_report_<model>.txt
outputs/reports/speech_model_comparison.csv
outputs/reports/speech_best_model.json
"""
from __future__ import annotations

import json
from typing import Dict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import (
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SPEECH_LABEL_ENCODER_PATH,
    SPEECH_MODEL_PATH,
    SPEECH_RF_PATH,
    SPEECH_SVM_PATH,
    TARGET_EMOTIONS,
)
from src.speech_features import build_dataset


def _plot_confusion_matrix(cm, label_names, title, out_path):
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


def _evaluate(model_name, y_true, y_pred, label_names):
    report = classification_report(
        y_true, y_pred, target_names=label_names, digits=4, zero_division=0
    )
    (REPORTS_DIR / f"speech_classification_report_{model_name}.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    _plot_confusion_matrix(
        cm,
        label_names,
        f"Confusion matrix - {model_name}",
        FIGURES_DIR / f"speech_confusion_{model_name}.png",
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main() -> None:
    print("[1/5] Building speech dataset (this can take a couple of minutes)...")
    X, y_str, actors = build_dataset()
    print(f"   samples: {X.shape[0]}, features: {X.shape[1]}")
    print("   class distribution:")
    for cls, n in pd.Series(y_str).value_counts().items():
        print(f"     {cls}: {n}")

    print("[2/5] Encoding labels and creating speaker-independent split...")
    le = LabelEncoder().fit(TARGET_EMOTIONS)
    y = le.transform(y_str)
    label_names = list(le.classes_)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=actors))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_actors = sorted(set(actors[test_idx].tolist()))
    print(f"   train samples: {len(X_train)}, test samples: {len(X_test)}")
    print(f"   held-out actor IDs: {test_actors}")

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
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
    comparison.to_csv(REPORTS_DIR / "speech_model_comparison.csv")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"   Best by macro-F1: {best_name}")

    print("[5/5] Saving models + label encoder...")
    # Best-model pointer (preserved for backward compatibility)
    joblib.dump(fitted[best_name], SPEECH_MODEL_PATH)
    joblib.dump(le, SPEECH_LABEL_ENCODER_PATH)
    # Per-model artifacts so the GUI dropdown can offer choices
    joblib.dump(fitted["random_forest"], SPEECH_RF_PATH)
    joblib.dump(fitted["svm_rbf"], SPEECH_SVM_PATH)
    meta = {
        "best_model": best_name,
        "metrics": results[best_name],
        "test_actors": test_actors,
    }
    (REPORTS_DIR / "speech_best_model.json").write_text(json.dumps(meta, indent=2))
    print(
        f"   Saved: {SPEECH_MODEL_PATH.name} (best={best_name}), "
        f"{SPEECH_RF_PATH.name}, {SPEECH_SVM_PATH.name}, "
        f"{SPEECH_LABEL_ENCODER_PATH.name}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
