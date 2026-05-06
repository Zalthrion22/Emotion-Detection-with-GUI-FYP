"""Fine-tune DistilBERT on the 6-class GoEmotions subset.


Outputs

models/distilbert_text/                   
models/distilbert_text/label_encoder.joblib
outputs/figures/text_confusion_distilbert.png
outputs/reports/text_classification_report_distilbert.txt
outputs/reports/text_distilbert_metrics.json
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DISTILBERT_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_EMOTIONS,
)
from src.text_preprocessing import load_goemotions

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:  # only triggers without optional deps
    raise SystemExit(
        "DistilBERT dependencies are missing.\n"
        "Install with: pip install -r requirements-advanced.txt\n"
        f"Underlying error: {e}"
    )

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


class WeightedTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


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


def main() -> None:
    print(f"[1/6] Loading GoEmotions (target classes: {TARGET_EMOTIONS}) ...")
    train_df = load_goemotions("train")
    dev_df = load_goemotions("dev")
    test_df = load_goemotions("test")
    train_df = pd.concat([train_df, dev_df], ignore_index=True)
    print(f"   train+dev rows: {len(train_df)}, test rows: {len(test_df)}")

    le = LabelEncoder().fit(TARGET_EMOTIONS)
    label_names = list(le.classes_)
    y_train = le.transform(train_df["emotion"])
    y_test = le.transform(test_df["emotion"])

    counts = pd.Series(y_train).value_counts().sort_index().to_numpy()
    class_weights = torch.tensor(
        (counts.sum() / (len(counts) * counts)).astype(np.float32),
    )
    print(
        "   class weights:",
        {label_names[i]: round(float(class_weights[i]), 3) for i in range(len(label_names))},
    )

    print("[2/6] Tokenising...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(
        list(train_df["text"]),
        truncation=True, padding="max_length", max_length=MAX_LENGTH,
    )
    test_enc = tokenizer(
        list(test_df["text"]),
        truncation=True, padding="max_length", max_length=MAX_LENGTH,
    )
    train_ds = TextDataset(train_enc, y_train)
    test_ds = TextDataset(test_enc, y_test)

    print("[3/6] Building model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label={i: lab for i, lab in enumerate(label_names)},
        label2id={lab: i for i, lab in enumerate(label_names)},
    )

    DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(DISTILBERT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=RANDOM_STATE,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    print("[4/6] Training (this is the slow part)...")
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )
    trainer.train()

    print("[5/6] Evaluating on test split...")
    pred = trainer.predict(test_ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print("   metrics:", {k: round(v, 4) for k, v in metrics.items()})

    report = classification_report(
        y_test, y_pred, target_names=label_names, digits=4, zero_division=0
    )
    (REPORTS_DIR / "text_classification_report_distilbert.txt").write_text(report)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(label_names))))
    _plot_confusion_matrix(
        cm, label_names, "Confusion matrix - distilbert",
        FIGURES_DIR / "text_confusion_distilbert.png",
    )
    (REPORTS_DIR / "text_distilbert_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("[6/6] Saving model artifacts...")
    trainer.save_model(str(DISTILBERT_DIR))
    tokenizer.save_pretrained(str(DISTILBERT_DIR))
    joblib.dump(le, DISTILBERT_DIR / "label_encoder.joblib")
    print(f"   Saved to {DISTILBERT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
