"""
Step 1 — Prepare train / dev / test splits from quadruplets.json.

Outputs (JSON lists saved to results/splits/):
  train.json   dev.json   test.json
  triplet_train.json   (entries with EN + MR + KN)
  quadruplet_train.json (all entries, with any 2+ langs)
"""

import json, random
from pathlib import Path
from config import QUADRUPLETS_JSON, SPLITS_DIR, TRAIN_RATIO, DEV_RATIO

random.seed(42)

def load_quadruplets():
    with open(QUADRUPLETS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    items = list(data.values())
    print(f"Loaded {len(items)} quadruplets.")
    return items


def analyse(items, label=""):
    has_mr  = sum(1 for x in items if x.get("marathi"))
    has_kn  = sum(1 for x in items if x.get("kannada"))
    has_hi  = sum(1 for x in items if x.get("hindi"))
    full_4  = sum(1 for x in items if x.get("marathi") and x.get("kannada") and x.get("hindi"))
    tri_kn  = sum(1 for x in items if x.get("marathi") and x.get("kannada") and not x.get("hindi"))
    tri_hi  = sum(1 for x in items if x.get("marathi") and x.get("hindi")   and not x.get("kannada"))
    legal   = sum(1 for x in items if x.get("source") == "milpac_legal")
    print(f"\n{'='*50}")
    print(f"  {label}  n={len(items)}")
    print(f"  Has Marathi : {has_mr}")
    print(f"  Has Kannada : {has_kn}")
    print(f"  Has Hindi   : {has_hi}")
    print(f"  Full 4-lang : {full_4}")
    print(f"  Triplet MR+KN (no HI) : {tri_kn}  [Samanantar general]")
    print(f"  Triplet MR+HI (no KN) : {tri_hi}  [MILPaC legal]")
    print(f"  Legal domain (MILPaC) : {legal}")


def split_and_save(items):
    random.shuffle(items)
    n       = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)

    train = items[:n_train]
    dev   = items[n_train : n_train + n_dev]
    test  = items[n_train + n_dev :]

    # Triplet subset: only entries with Marathi AND Kannada
    triplet_items = [x for x in items if x.get("marathi") and x.get("kannada")]
    n_tri = len(triplet_items)
    triplet_train = triplet_items[:int(n_tri * TRAIN_RATIO)]

    # Quadruplet training: use ALL entries (model sees Hindi too)
    quad_train = train

    for name, data in [
        ("train", train), ("dev", dev), ("test", test),
        ("triplet_train", triplet_train),
        ("quadruplet_train", quad_train),
    ]:
        path = SPLITS_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSplits saved to {SPLITS_DIR}")
    print(f"  train: {len(train)}  dev: {len(dev)}  test: {len(test)}")
    print(f"  triplet_train (MR+KN only): {len(triplet_train)}")
    print(f"  quadruplet_train (all):     {len(quad_train)}")
    return train, dev, test


if __name__ == "__main__":
    items = load_quadruplets()
    analyse(items, "Full Dataset")
    train, dev, test = split_and_save(items)
    analyse(test, "Test Set")
