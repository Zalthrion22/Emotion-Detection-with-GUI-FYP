# Dual-Modality Emotion Detection

A desktop app that classifies an utterance into one of six emotions
(**anger, fear, joy, sadness, surprise, neutral**) from either typed text
or a `.wav` clip / live microphone recording.

Final-year undergraduate project, Queen Mary University of London.

## Features

- Two independent pipelines, presented in a tabbed GUI:
  - **Text:** Logistic Regression, LinearSVC (Platt-calibrated), and
    fine-tuned DistilBERT, all over the GoEmotions corpus.
  - **Speech:** Random Forest and RBF-SVM over MFCC features extracted
    from RAVDESS, with a speaker-independent train/test split.
- Per-class confidence chart, predicted-emotion display, recent-results
  panel, status bar, and friendly error handling.
- Live microphone recording (4 seconds, default input device) with
  silent-buffer rejection.
- Lazy model loading and background threads so the window does not
  freeze while DistilBERT is loading or audio is being predicted.

## Folder Layout

    src/                        all pipeline + GUI source
      config.py                 paths, label mappings, hyperparameters
      text_preprocessing.py     clean_text(), GoEmotions label collapse
      speech_features.py        load_audio(), MFCC + delta features
      train_text_models.py      LogReg + LinearSVC training
      train_text_distilbert.py  DistilBERT fine-tuning (run in Colab)
      train_speech_models.py    Random Forest + SVM-RBF training
      predict_text.py           classical text inference
      predict_text_distilbert.py DistilBERT inference
      predict_speech.py         speech inference
      record_audio.py           microphone capture for the GUI
      gui_app.py                CustomTkinter desktop GUI
      evaluate_models.py        cross-modality comparison table
      robustness_tests.py       fixed text + audio robustness cases
      benchmark_latency.py      cold-start + warm latency benchmarks
    tests/                      pytest suite (48 tests)
    data/
      text/goemotions/          GoEmotions train/dev/test TSVs (included)
      speech/ravdess/           RAVDESS .wav files (NOT included, see below)
    models/                     trained .joblib + DistilBERT artefacts
    outputs/
      figures/                  confusion matrices + macro-F1 chart
      reports/                  classification reports, comparison CSVs,
                                latency, robustness
    notebooks/                  Colab guide for DistilBERT training
    run_app.py                  GUI launcher

`outputs/recordings/` and `logs/` are auto-created at runtime by the
GUI / `config.py` and are not included in this copy. The robustness
runner regenerates `outputs/reports/robustness_audio/` on demand.

## Requirements

Python 3.10+ on Windows, macOS or Linux. CPU is sufficient for the
classical models and for DistilBERT inference; a GPU is only needed if
you want to retrain DistilBERT locally.

Base dependencies:

    pip install -r requirements.txt

Add this only if you intend to load DistilBERT (~2 GB extra dependencies):

    pip install -r requirements-advanced.txt

## Setting Up

    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1        # PowerShell on Windows
    # or:  source .venv/bin/activate    # macOS / Linux

    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-advanced.txt   # optional, for DistilBERT

## Datasets

| Modality | Dataset | Where to put it |
|---|---|---|
| Text   | GoEmotions (Demszky et al., 2020) | `data/text/goemotions/{train,dev,test}.tsv` (already included in this copy) |
| Speech | RAVDESS Speech Audio-only (Livingstone & Russo, 2018) | `data/speech/ravdess/Audio_Speech_Actors_01-24/Actor_NN/*.wav` |

**RAVDESS is not included** because the audio is ~570 MB. Download it
from <https://zenodo.org/record/1188976> (file
`Audio_Speech_Actors_01-24.zip`), unzip into `data/speech/ravdess/`, and
keep the `Actor_01 ... Actor_24` folder names as released. The pipeline
expects 1,440 `.wav` files matching the RAVDESS filename schema.

If you only want to launch the GUI and try a few predictions, you do
**not** need RAVDESS at all — the included `.joblib` files are enough.

## Training the Text Models

LogReg + LinearSVC (~1 minute on CPU):

    python -m src.train_text_models

Outputs:
- `models/text_logreg.joblib`, `models/text_linearsvc.joblib`,
  `models/text_vectorizer.joblib`, `models/text_label_encoder.joblib`,
  `models/text_model.joblib` (the macro-F1 winner — currently LogReg).
- `outputs/reports/text_classification_report_<model>.txt`,
  `outputs/reports/text_model_comparison.csv`,
  `outputs/figures/text_confusion_<model>.png`.

DistilBERT — train in Google Colab using a free T4 GPU. See
[`notebooks/train_distilbert_colab.md`](notebooks/train_distilbert_colab.md)
for step-by-step instructions. The trained DistilBERT artefacts are
already included in this clean copy under `models/distilbert_text/`, so
you only need to retrain if you want to change hyperparameters.

## Training the Speech Models

Requires RAVDESS to be downloaded into `data/speech/ravdess/`.

    python -m src.train_speech_models

This takes about 3 minutes on CPU (most of the time is feature
extraction). It produces `models/speech_random_forest.joblib`,
`models/speech_svm_rbf.joblib`, the matching label encoder, and writes
classification reports + confusion matrices into `outputs/`.

## Running the GUI

    python run_app.py

The Text tab accepts typed input and lets you choose between LogReg,
LinearSVC and DistilBERT. The Speech tab accepts a `.wav` file or a
4-second microphone recording and lets you choose between Random Forest
and SVM-RBF. The right panel shows the predicted emotion, a confidence
percentage and a per-class confidence chart. The status bar at the
bottom narrates progress.

DistilBERT loads only when first selected, which takes ~20 seconds and
~600 MB of RAM.

## Running the Tests

    pytest tests/ -q

The full suite is 48 tests. Tests that need trained model artefacts
skip themselves if those artefacts are missing, so the suite still runs
on a fresh checkout before any training.

## Evaluation, Robustness, Latency

Cross-modality comparison table (depends on classification reports being
already generated by training):

    python -m src.evaluate_models

Robustness scenarios for both text and speech:

    python -m src.robustness_tests

Cold-start and warm latency benchmark for each available model:

    python -m src.benchmark_latency

Outputs land in `outputs/reports/` and `outputs/figures/`.

## Notes on Model Artefacts and Large Files

- The classical sklearn models (~36 MB total) are included in the repo.
- The DistilBERT weights (`models/distilbert_text/model.safetensors`,
  256 MB) exceed GitHub's 100 MB per-file limit, so they are **not**
  committed. Download `model.safetensors` from this repository's
  **Releases** page and place it under `models/distilbert_text/`. The
  rest of the DistilBERT artefacts (config, tokenizer, label encoder)
  *are* committed, so once `model.safetensors` is in place inference
  works immediately. If you would rather retrain from scratch, follow
  [`notebooks/train_distilbert_colab.md`](notebooks/train_distilbert_colab.md).
- The DistilBERT *checkpoints* directory (training intermediates,
  ~770 MB) is **not** included — it is only needed if you want to resume
  training, and you can regenerate it by re-running the Colab notebook.
- RAVDESS audio (~570 MB) is not included; see the dataset section.
- The GUI writes microphone captures to `outputs/recordings/mic_recording.wav`,
  overwriting the file each time. The folder is auto-created on first run.

## Known Limitations

- **Speech accuracy is modest** (macro-F1 around 0.50). RAVDESS is acted
  studio speech, so live microphone clips fall outside the training
  distribution and get systematically biased — soft, breathy or quiet
  recordings often land near the *fear* prototype. Random Forest is the
  more conservative dropdown choice if you see this happen.
- The system **always assigns one of the six labels**. There is no
  out-of-distribution rejection, so non-speech audio (e.g. a sine wave)
  still receives a prediction.
- Both pipelines are **English-only** — both datasets are English.
- DistilBERT is accurate but **slow to load (~20 s)**, so it is not the
  default text model.
- Confidence scores are not perfectly calibrated and should not be
  treated as probabilities.
- Usability evidence is from manual developer testing; no formal
  external user study was conducted.
