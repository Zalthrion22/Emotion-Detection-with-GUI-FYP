"""Cconfiguration

"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEXT_DATA_DIR = DATA_DIR / "text" / "goemotions"
SPEECH_DATA_DIR = DATA_DIR / "speech" / "ravdess"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
RECORDINGS_DIR = OUTPUTS_DIR / "recordings"

for _d in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR, RECORDINGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Target emotion classes
# present a consistent vocabulary regardless of which modality was used.
TARGET_EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise", "neutral"]

# 6-class mapping text
# Labels not listed here are dropped.
GOEMOTIONS_LABEL_MAP = {
    # anger
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    # fear
    "fear": "fear", "nervousness": "fear",
    # joy
    "joy": "joy", "amusement": "joy", "excitement": "joy",
    "gratitude": "joy", "love": "joy", "optimism": "joy",
    "pride": "joy", "relief": "joy", "admiration": "joy",
    # sadness
    "sadness": "sadness", "disappointment": "sadness", "grief": "sadness",
    "remorse": "sadness", "embarrassment": "sadness",
    # surprise
    "surprise": "surprise", "realization": "surprise",
    "confusion": "surprise", "curiosity": "surprise",
    # neutral
    "neutral": "neutral",
}

# 6-class mapping speech
# Filename schema: 03-01-EE-IN-ST-RE-AC.wav, where EE is the emotion code.
# calm is folded into neutral. Disgust is dropped
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": None,
    "08": "surprise",
}

# Audio / MFCC settings
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_AUDIO_SECONDS = 4.0

# Training hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Saved artifact paths
TEXT_MODEL_PATH = MODELS_DIR / "text_model.joblib"
TEXT_VECTORIZER_PATH = MODELS_DIR / "text_vectorizer.joblib"
TEXT_LABEL_ENCODER_PATH = MODELS_DIR / "text_label_encoder.joblib"

SPEECH_MODEL_PATH = MODELS_DIR / "speech_model.joblib"
SPEECH_LABEL_ENCODER_PATH = MODELS_DIR / "speech_label_encoder.joblib"

# Per-model artifacts so the GUI can offer dropdown
TEXT_LOGREG_PATH = MODELS_DIR / "text_logreg.joblib"
TEXT_LINEARSVC_PATH = MODELS_DIR / "text_linearsvc.joblib"
SPEECH_RF_PATH = MODELS_DIR / "speech_random_forest.joblib"
SPEECH_SVM_PATH = MODELS_DIR / "speech_svm_rbf.joblib"

DISTILBERT_DIR = MODELS_DIR / "distilbert_text"
