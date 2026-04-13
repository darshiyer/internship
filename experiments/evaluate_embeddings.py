"""
Step 3 — Evaluate and compare Baseline vs Triplet vs Quadruplet models.

Metrics computed on the test set:
  1. Mean cosine similarity — aligned pairs (should be HIGH)
  2. Mean cosine similarity — random pairs   (should be LOW)
  3. Alignment gap = aligned_sim - random_sim  (key metric, higher = better)
  4. Alignment accuracy — for each query, does correct pair rank 1st?
  5. MRR — Mean Reciprocal Rank
  6. Language-wise analysis: EN-MR, EN-KN, EN-HI, MR-KN cross-sim
  7. Centroid bias: cosine distance of each Indic centroid from English centroid
"""

import json, numpy as np
from pathlib import Path
from config import BASE_ENCODER, RESULTS_DIR, SPLITS_DIR

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ── helpers ──────────────────────────────────────────────────────────────────

def load_split(name):
    with open(SPLITS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def model_path(name):
    p = RESULTS_DIR / "models" / name
    return str(p) if p.exists() else BASE_ENCODER


def embed(model, texts, batch=64):
    return model.encode(texts, batch_size=batch, show_progress_bar=False,
                        normalize_embeddings=True)


def mean_cos(vecs_a, vecs_b):
    sims = (vecs_a * vecs_b).sum(axis=1)
    return float(np.mean(sims))


def alignment_accuracy_mrr(q_embs, ref_embs):
    """For each query embedding, rank all ref embeddings by cosine similarity.
    Correct match = same index. Returns accuracy@1 and MRR."""
    sim = cosine_similarity(q_embs, ref_embs)   # (N, N)
    ranks = np.argsort(-sim, axis=1)
    acc = float(np.mean(ranks[:, 0] == np.arange(len(q_embs))))
    rr  = [1.0 / (np.where(ranks[i] == i)[0][0] + 1) for i in range(len(q_embs))]
    return acc, float(np.mean(rr))


def random_cos(vecs_a, vecs_b, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vecs_b))
    return mean_cos(vecs_a, vecs_b[idx])


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model_name, model, test_entries):
    print(f"\n  Evaluating: {model_name}")
    results = {}

    # Gather language-specific sentence lists
    pairs = {"en_mr": [], "en_kn": [], "en_hi": [], "mr_kn": [], "mr_hi": []}
    for e in test_entries:
        en = e.get("english","").strip()
        mr = e.get("marathi","").strip()
        kn = e.get("kannada","").strip()
        hi = e.get("hindi","").strip()
        if en and mr: pairs["en_mr"].append((en, mr))
        if en and kn: pairs["en_kn"].append((en, kn))
        if en and hi: pairs["en_hi"].append((en, hi))
        if mr and kn: pairs["mr_kn"].append((mr, kn))
        if mr and hi: pairs["mr_hi"].append((mr, hi))

    lang_results = {}
    for key, pair_list in pairs.items():
        if len(pair_list) < 5:
            continue
        a_texts = [p[0] for p in pair_list]
        b_texts = [p[1] for p in pair_list]
        a_emb = embed(model, a_texts)
        b_emb = embed(model, b_texts)
        aligned_sim = mean_cos(a_emb, b_emb)
        random_sim  = random_cos(a_emb, b_emb)
        gap         = aligned_sim - random_sim
        acc, mrr    = alignment_accuracy_mrr(a_emb, b_emb)
        lang_results[key] = {
            "n": len(pair_list),
            "aligned_sim": round(aligned_sim, 4),
            "random_sim":  round(random_sim, 4),
            "gap":         round(gap, 4),
            "accuracy_at_1": round(acc, 4),
            "mrr":         round(mrr, 4),
        }
        print(f"    {key:8s}  aligned={aligned_sim:.3f}  random={random_sim:.3f}"
              f"  gap={gap:.3f}  acc@1={acc:.3f}  mrr={mrr:.3f}  n={len(pair_list)}")

    # Centroid bias analysis (how Indic-centric are the embeddings?)
    en_sents = [e.get("english","") for e in test_entries if e.get("english")]
    mr_sents = [e.get("marathi","") for e in test_entries if e.get("marathi")]
    kn_sents = [e.get("kannada","") for e in test_entries if e.get("kannada")]
    hi_sents = [e.get("hindi","")   for e in test_entries if e.get("hindi")]

    centroids = {}
    for lang, sents in [("EN", en_sents), ("MR", mr_sents), ("KN", kn_sents), ("HI", hi_sents)]:
        if len(sents) >= 5:
            embs = embed(model, sents[:200])
            centroids[lang] = embs.mean(axis=0, keepdims=True)

    bias_results = {}
    if "EN" in centroids:
        en_c = centroids["EN"]
        for lang in ["MR", "KN", "HI"]:
            if lang in centroids:
                sim = float(cosine_similarity(en_c, centroids[lang])[0][0])
                bias_results[f"EN_{lang}_centroid_sim"] = round(sim, 4)
                print(f"    Centroid EN↔{lang}: {sim:.4f}")
        if "MR" in centroids and "KN" in centroids:
            sim = float(cosine_similarity(centroids["MR"], centroids["KN"])[0][0])
            bias_results["MR_KN_centroid_sim"] = round(sim, 4)
            print(f"    Centroid MR↔KN: {sim:.4f}  (higher = less English-biased, Indic languages closer)")

    results["lang_pairs"]  = lang_results
    results["centroid_bias"] = bias_results
    return results


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test = load_split("test")
    print(f"Test set size: {len(test)}")

    all_results = {}

    for model_name in ["baseline", "triplet", "quadruplet"]:
        path = model_path(model_name)
        print(f"\nLoading {model_name} from: {path}")
        model = SentenceTransformer(path)
        all_results[model_name] = evaluate_model(model_name, model, test)

    # Print comparison table
    print("\n" + "="*70)
    print("  SUMMARY: Alignment Gap (aligned_sim − random_sim)  [HIGHER IS BETTER]")
    print("="*70)
    header = f"  {'Pair':10s}  {'Baseline':>10s}  {'Triplet':>10s}  {'Quadruplet':>10s}"
    print(header)
    all_keys = set()
    for r in all_results.values():
        all_keys |= set(r.get("lang_pairs", {}).keys())
    for key in sorted(all_keys):
        row = f"  {key:10s}"
        for mn in ["baseline", "triplet", "quadruplet"]:
            v = all_results.get(mn, {}).get("lang_pairs", {}).get(key, {}).get("gap", "N/A")
            row += f"  {v:>10}"
        print(row)

    print("\n  CENTROID BIAS (EN-Indic centroid cosine similarity)  [LOWER = LESS BIAS]")
    bias_keys = set()
    for r in all_results.values():
        bias_keys |= set(r.get("centroid_bias", {}).keys())
    for key in sorted(bias_keys):
        row = f"  {key:25s}"
        for mn in ["baseline", "triplet", "quadruplet"]:
            v = all_results.get(mn, {}).get("centroid_bias", {}).get(key, "N/A")
            row += f"  {v:>10}"
        print(row)

    # Save
    out = RESULTS_DIR / "embedding_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Embedding results saved → {out}")
