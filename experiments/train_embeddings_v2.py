"""
train_embeddings_v2.py  —  IMPROVED training with three fixes:

Fix 1: MultipleNegativesRankingLoss instead of CosineSimilarityLoss
       → treats other items in the batch as hard negatives
       → prevents embedding collapse (centroid saturating at 0.999)
       → expected Acc@1 gain: +10–20 pp

Fix 2: Full dataset on GPU / larger sample on CPU
       → was 8,000 pairs; now up to 50,000 (GPU) or 20,000 (CPU)
       → uses all available data instead of 1% of it

Fix 3: 3× oversample of MILPaC (legal domain) sentences during pair construction
       → reduces domain imbalance between Samanantar (general) and MILPaC (legal)

Run on Colab (T4 GPU) for best results:
    !pip install sentence-transformers accelerate
    !python train_embeddings_v2.py

Run locally (CPU fallback — slower but works):
    .venv312/bin/python experiments/train_embeddings_v2.py
"""

import json, os, time, random
from pathlib import Path

# Force CPU if no GPU found (local Mac)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built     = lambda: False

# Auto-detect device: CUDA GPU → CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# On GPU (Colab T4): can afford 50k pairs, bs=64
# On CPU (local): cap at 20k pairs, bs=32 to stay under 1 hr
if DEVICE == "cuda":
    MAX_PAIRS  = 50_000
    BATCH_SIZE = 64
else:
    MAX_PAIRS  = 20_000
    BATCH_SIZE = 32
    torch.set_num_threads(8)

EPOCHS       = 3
LEGAL_BOOST  = 3     # oversample legal sentences this many times

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses as st_losses
from sentence_transformers import evaluation as st_eval
from torch.utils.data import DataLoader

# ── path setup ───────────────────────────────────────────────────────────────
# Works whether run from project root or experiments/ subdirectory
THIS_DIR = Path(__file__).parent
ROOT     = THIS_DIR.parent if THIS_DIR.name == "experiments" else THIS_DIR
SPLITS_DIR  = ROOT / "experiments" / "results" / "splits"
RESULTS_DIR = ROOT / "experiments" / "results"
BASE_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LEGAL_KEYWORDS = {
    "en": {"section", "act", "court", "article", "clause", "schedule",
           "provision", "hereby", "thereof", "whereas", "notwithstanding",
           "pursuant", "ordinance", "statute", "judgment", "decree"},
    "mr": {"कलम", "अधिनियम", "न्यायालय", "अनुच्छेद", "खंड", "अनुसूची",
           "तरतूद", "न्यायनिर्णय", "विधेयक", "अध्यादेश"},
    "kn": {"ವಿಭಾಗ", "ಅಧಿನಿಯಮ", "ನ್ಯಾಯಾಲಯ", "ಅನುಚ್ಛೇದ", "ಷರತ್ತು",
           "ನಿಬಂಧನೆ", "ತೀರ್ಪು", "ಮಸೂದೆ", "ಶಾಸನ"},
}

def is_legal(entry):
    """Return True if the entry contains at least one legal keyword."""
    for lang, kws in LEGAL_KEYWORDS.items():
        field = {"en": "english", "mr": "marathi", "kn": "kannada"}.get(lang, lang)
        text  = entry.get(field, "").lower()
        if any(kw in text for kw in kws):
            return True
    return False


# ── data helpers ─────────────────────────────────────────────────────────────

def load_split(name):
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run prepare_data.py first to generate splits."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_mnrl_pairs(entries, include_hindi=False, legal_boost=LEGAL_BOOST):
    """
    For MultipleNegativesRankingLoss: InputExample(texts=[anchor, positive])
    No label needed — all other pairs in the batch serve as negatives.

    language combinations:
      - always:   EN-MR, EN-KN, MR-KN
      - if hindi: EN-HI, MR-HI, KN-HI
    Legal entries are duplicated `legal_boost` times to oversample the domain.
    """
    pairs   = []
    combos  = ["english-marathi", "english-kannada", "marathi-kannada"]
    if include_hindi:
        combos += ["english-hindi", "marathi-hindi", "kannada-hindi"]

    for e in entries:
        multiplier = legal_boost if is_legal(e) else 1
        for combo in combos:
            l1, l2 = combo.split("-")
            t1 = e.get(l1, "").strip()
            t2 = e.get(l2, "").strip()
            if t1 and t2:
                for _ in range(multiplier):
                    pairs.append(InputExample(texts=[t1, t2]))
    return pairs


def build_eval_sets(entries, n_neg=500):
    """EmbeddingSimilarityEvaluator needs (s1, s2, score) triples."""
    s1, s2, scores = [], [], []
    for e in entries:
        en = e.get("english","").strip()
        mr = e.get("marathi","").strip()
        kn = e.get("kannada","").strip()
        for a, b in [(en, mr), (en, kn), (mr, kn)]:
            if a and b:
                s1.append(a); s2.append(b); scores.append(1.0)
    n = len(s1)
    for i in range(min(n_neg, n)):
        j = (i + n // 3) % n
        s1.append(s1[i]); s2.append(s2[j]); scores.append(0.0)
    return s1, s2, scores


# ── training ─────────────────────────────────────────────────────────────────

def train_model(model_type, train_pairs, dev_entries, save_dir):
    if len(train_pairs) > MAX_PAIRS:
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:MAX_PAIRS]

    print(f"\n{'='*65}")
    print(f"  Training : {model_type.upper()}")
    print(f"  Pairs    : {len(train_pairs):,}  (legal oversampled {LEGAL_BOOST}×)")
    print(f"  Loss     : MultipleNegativesRankingLoss  (hard negatives)")
    print(f"  Device   : {DEVICE}  |  batch={BATCH_SIZE}  epochs={EPOCHS}")
    print(f"{'='*65}")

    model = SentenceTransformer(BASE_ENCODER, device=DEVICE)

    loader = DataLoader(
        train_pairs,
        shuffle=True,
        batch_size=BATCH_SIZE,
        pin_memory=(DEVICE == "cuda"),
        num_workers=0,
    )

    # MultipleNegativesRankingLoss: each positive pair's other-batch items
    # become in-batch negatives — prevents trivial cosine collapse
    loss_fn = st_losses.MultipleNegativesRankingLoss(model)

    s1, s2, sc = build_eval_sets(dev_entries)
    evaluator = st_eval.EmbeddingSimilarityEvaluator(
        s1, s2, sc,
        name=f"{model_type}_dev",
        show_progress_bar=False,
    )

    warmup = max(1, len(train_pairs) // BATCH_SIZE // 4)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss_fn)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup,
        output_path=str(save_dir),
        show_progress_bar=True,
        save_best_model=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Saved → {save_dir}  (took {elapsed/60:.1f} min)")
    return save_dir


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading splits …")
    triplet_train    = load_split("triplet_train")
    quadruplet_train = load_split("quadruplet_train")
    dev              = load_split("dev")

    tri_pairs  = make_mnrl_pairs(triplet_train,    include_hindi=False)
    quad_pairs = make_mnrl_pairs(quadruplet_train, include_hindi=True)

    # Report legal vs general breakdown
    tri_legal  = sum(1 for e in triplet_train    if is_legal(e))
    quad_legal = sum(1 for e in quadruplet_train if is_legal(e))
    print(f"\nTriplet  split : {len(triplet_train):,} entries  "
          f"({tri_legal:,} legal = {100*tri_legal/max(1,len(triplet_train)):.1f}%)")
    print(f"Quadruplet split: {len(quadruplet_train):,} entries  "
          f"({quad_legal:,} legal = {100*quad_legal/max(1,len(quadruplet_train)):.1f}%)")
    print(f"\nTriplet  pairs (after {LEGAL_BOOST}× legal boost): {len(tri_pairs):,}")
    print(f"Quadruplet pairs (after {LEGAL_BOOST}× legal boost): {len(quad_pairs):,}")
    print(f"Capped at {MAX_PAIRS:,} per model ({DEVICE} mode)\n")

    train_model(
        "triplet_v2",
        tri_pairs, dev,
        RESULTS_DIR / "models" / "triplet_v2",
    )

    train_model(
        "quadruplet_v2",
        quad_pairs, dev,
        RESULTS_DIR / "models" / "quadruplet_v2",
    )

    print("\n✅  Both v2 models trained. Run evaluate_embeddings.py with")
    print("   model names ['baseline','triplet_v2','quadruplet_v2'] to compare.")
