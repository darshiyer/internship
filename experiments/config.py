"""Shared configuration for all experiments."""
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUADRUPLETS_JSON = DATA_DIR / "quadruplets.json"
SPLITS_DIR       = RESULTS_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Sentence transformer base model (118M params, multilingual, fast on CPU/MPS)
BASE_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Translation model (NLLB-600M: supports MR, KN, HI, EN directly)
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB language codes
LANG_CODES = {
    "marathi":  "mar_Deva",
    "kannada":  "kan_Knda",
    "hindi":    "hin_Deva",
    "english":  "eng_Latn",
}

TRAIN_RATIO = 0.70
DEV_RATIO   = 0.10
TEST_RATIO  = 0.20

BATCH_SIZE  = 32
EPOCHS      = 4
WARMUP_STEPS = 100

TRANSLATION_TEST_SIZE = 100  # sentences for translation eval (CPU-safe)
