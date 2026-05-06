# Training DistilBERT on Google Colab

This guide explains how to fine-tune the DistilBERT text model on the
GoEmotions dataset Google Colab, then download the
trained artefacts back into your local project so the GUI and the
benchmark scripts can use them.

A single CPU laptop takes too long for this (roughly 30–60 minutes per
epoch). On a T4 GPU the whole run completes in **about 10–15 minutes**
end-to-end for 3 epochs over the 6-class GoEmotions subset.


## Step 1 — Open a new Colab notebook with GPU

Go to <https://colab.research.google.com> and create a new notebook.

Then turn on the GPU runtime:

> **Runtime → Change runtime type → Hardware accelerator: T4 GPU**


## Step 2 — Upload the project

In the Colab sidebar, click the folder icon and use **Upload to session
storage** to upload either `colab_upload.zip` (included in this copy) or
your own zip of the project. Then unzip it in a cell:

```python
!unzip -q /content/colab_upload.zip -d /content/emotion_detection_project
```

(If you cloned a Git repo instead, use
`!git clone <repo-url> /content/emotion_detection_project`.)

## Step 3 — Install dependencies

```python
%cd /content/emotion_detection_project
!pip install -q -r requirements-advanced.txt
```

`requirements-advanced.txt` pulls in `torch`, `transformers`,
`datasets` and `accelerate` on top of the base requirements. Colab
already has compatible CUDA drivers, so no extra setup is needed.

## Step 4 — Make sure GoEmotions is in place

If you uploaded a zip that does not contain the dataset, fetch the
official TSVs:

```python
!mkdir -p data/text/goemotions
!wget -q -O data/text/goemotions/train.tsv https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv
!wget -q -O data/text/goemotions/dev.tsv   https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv
!wget -q -O data/text/goemotions/test.tsv  https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv
```

You should now have three files totalling about 4 MB.

## Step 5 — Run training

```python
!python -m src.train_text_distilbert
```

This launches `src/train_text_distilbert.py`. It uses the following
settings (defined as constants at the top of the script):

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max token length | 128 (WordPiece) |
| Epochs | 3 |
| Batch size (train) | 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| FP16 mixed precision | enabled when a CUDA GPU is detected |
| Loss | cross-entropy with class weights inversely proportional to training distribution (custom `WeightedTrainer`) |

The script prints per-epoch evaluation metrics. At the end, the best
checkpoint by macro-F1 is loaded back and saved.

Expected wall-clock time on a T4: ~3–5 minutes per epoch, so ~10–15
minutes total.

## Step 6 — Where the artefacts are saved

After training, the following files should appear in
`models/distilbert_text/`:

- `model.safetensors` — fine-tuned weights
- `config.json` — model architecture
- `tokenizer.json`, `tokenizer_config.json` — DistilBERT tokenizer
- `label_encoder.joblib` — joblib of the sklearn `LabelEncoder` so the
  index returned by the model can be turned back into one of the six
  emotion strings
- `training_args.bin` — Hugging Face `TrainingArguments` snapshot

Evaluation outputs are also generated:

- `outputs/figures/text_confusion_distilbert.png`
- `outputs/reports/text_classification_report_distilbert.txt`
- `outputs/reports/text_distilbert_metrics.json`


## Step 7 — Download the trained artefacts

Zip just the parts you need so the download is small (~260 MB instead
of ~1 GB):

```python
!cd /content/emotion_detection_project && zip -r /content/distilbert_artifacts.zip \
    models/distilbert_text/config.json \
    models/distilbert_text/label_encoder.joblib \
    models/distilbert_text/model.safetensors \
    models/distilbert_text/tokenizer.json \
    models/distilbert_text/tokenizer_config.json \
    models/distilbert_text/training_args.bin \
    outputs/figures/text_confusion_distilbert.png \
    outputs/reports/text_classification_report_distilbert.txt \
    outputs/reports/text_distilbert_metrics.json

from google.colab import files
files.download("/content/distilbert_artifacts.zip")
```

## Step 8 — Place the files in your local project

Unzip `distilbert_artifacts.zip` at the **root of your local project**
(the folder that contains `run_app.py`). The model files should be at
`models/distilbert_text/...` and the figure / report files merge with
the existing `outputs/` directory.

## Step 9 — Confirm local DistilBERT inference works

From a terminal in the project root, with `requirements-advanced.txt`
already installed in your virtual environment:

```bash
python -c "from src.predict_text_distilbert import predict_emotion; print(predict_emotion('I am thrilled and over the moon!'))"
```

You should see something like `('joy', 0.94)`. The first call takes
~20 seconds because the weights are loaded into RAM; subsequent calls
are faster.

In the GUI, the same is true: the **DistilBERT** option in the Text-tab
dropdown loads the model on first use (the status bar shows "Loading
DistilBERT, this may take a few seconds…"), and stays loaded for the
rest of the session.
