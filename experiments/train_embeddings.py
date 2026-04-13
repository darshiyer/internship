"""
Step 2 — Train sentence transformers: Triplet vs Quadruplet.
Forces CPU to avoid MPS OOM. Uses CosineSimilarityLoss for stability.
"""

import json, os, time, random
from pathlib import Path

# Completely disable MPS before ANY torch import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
# Monkey-patch MPS detection to always return False → forces CPU
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built     = lambda: False
torch.set_num_threads(8)

from config import BASE_ENCODER, SPLITS_DIR, RESULTS_DIR, WARMUP_STEPS
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses as st_losses
from sentence_transformers import evaluation as st_eval
from torch.utils.data import DataLoader

BATCH_SIZE = 32
EPOCHS     = 3
MAX_PAIRS  = 8000    # 97k available; 8k pairs = ~25 min total on CPU


# ── data helpers ─────────────────────────────────────────────────────────────

def load_split(name):
    with open(SPLITS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def make_triplet_pairs(entries):
    """(EN,MR) and (EN,KN) pairs only."""
    pairs = []
    for e in entries:
        en = e.get("english","").strip()
        mr = e.get("marathi","").strip()
        kn = e.get("kannada","").strip()
        if en and mr: pairs.append(InputExample(texts=[en, mr], label=1.0))
        if en and kn: pairs.append(InputExample(texts=[en, kn], label=1.0))
    return pairs


def make_quadruplet_pairs(entries):
    """All valid language-pair combinations."""
    pairs = []
    for e in entries:
        langs = {k: e.get(k,"").strip()
                 for k in ["english","marathi","kannada","hindi"]}
        present = [(v, k) for k, v in langs.items() if v]
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                pairs.append(InputExample(texts=[present[i][0], present[j][0]], label=1.0))
    return pairs


def build_eval_sets(entries, n_neg_each=200):
    """Build (sent1, sent2, score) for EmbeddingSimilarityEvaluator."""
    s1, s2, scores = [], [], []
    for e in entries:
        en = e.get("english","").strip()
        mr = e.get("marathi","").strip()
        kn = e.get("kannada","").strip()
        hi = e.get("hindi","").strip()
        for a, b in [(en,mr),(en,kn),(mr,kn),(en,hi),(mr,hi)]:
            if a and b:
                s1.append(a); s2.append(b); scores.append(1.0)
    # Negatives: shift by half
    n = len(s1)
    for i in range(min(n_neg_each, n)):
        j = (i + n//3) % n
        if s1[i] != s1[j]:
            s1.append(s1[i]); s2.append(s2[j]); scores.append(0.0)
    return s1, s2, scores


# ── training ─────────────────────────────────────────────────────────────────

def train_model(model_type, train_pairs, dev_entries, save_dir):
    if len(train_pairs) > MAX_PAIRS:
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:MAX_PAIRS]

    print(f"\n{'='*60}")
    print(f"  Training: {model_type.upper()}  pairs={len(train_pairs)}  "
          f"device=cpu  epochs={EPOCHS}")
    print(f"{'='*60}")

    model = SentenceTransformer(BASE_ENCODER, device="cpu")

    # Use CosineSimilarityLoss — simpler, more memory-efficient than MNRL
    loader = DataLoader(train_pairs, shuffle=True, batch_size=BATCH_SIZE,
                        pin_memory=False, num_workers=0)
    loss_fn = st_losses.CosineSimilarityLoss(model)

    s1, s2, sc = build_eval_sets(dev_entries)
    evaluator = st_eval.EmbeddingSimilarityEvaluator(s1, s2, sc,
                                                      name=f"{model_type}_dev",
                                                      show_progress_bar=False)

    save_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss_fn)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=min(WARMUP_STEPS, max(1, len(train_pairs)//BATCH_SIZE//4)),
        output_path=str(save_dir),
        show_progress_bar=True,
        save_best_model=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  →  {save_dir}")
    return save_dir


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    triplet_train    = load_split("triplet_train")
    quadruplet_train = load_split("quadruplet_train")
    dev              = load_split("dev")

    tri_pairs  = make_triplet_pairs(triplet_train)
    quad_pairs = make_quadruplet_pairs(quadruplet_train)

    print(f"Triplet  pairs (EN+MR+KN only) : {len(tri_pairs)}")
    print(f"Quadruplet pairs (all combos)   : {len(quad_pairs)}")
    print(f"Capped at {MAX_PAIRS} pairs per model for speed\n")

    train_model("triplet",
                tri_pairs, dev,
                RESULTS_DIR / "models" / "triplet")

    train_model("quadruplet",
                quad_pairs, dev,
                RESULTS_DIR / "models" / "quadruplet")

    print("\n✅ Both models trained and saved.")
