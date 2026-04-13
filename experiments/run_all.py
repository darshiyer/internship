"""
Master runner — executes all experiments in sequence and saves a
unified results summary to results/final_results.json.

Run:
  cd experiments
  python run_all.py
"""

import subprocess, json, sys, time
from pathlib import Path
from config import RESULTS_DIR

PYTHON = sys.executable
EXP_DIR = Path(__file__).resolve().parent


def run_step(script, label):
    print(f"\n{'#'*60}")
    print(f"  STEP: {label}")
    print(f"{'#'*60}")
    t0 = time.time()
    result = subprocess.run([PYTHON, str(EXP_DIR / script)],
                            capture_output=False, text=True)
    elapsed = time.time() - t0
    status = "✅ Done" if result.returncode == 0 else f"❌ Failed (exit {result.returncode})"
    print(f"\n  {status}  in {elapsed:.1f}s")
    return result.returncode == 0


def merge_results():
    """Combine embedding + translation results into one summary."""
    summary = {}

    emb_path = RESULTS_DIR / "embedding_results.json"
    if emb_path.exists():
        with open(emb_path, encoding="utf-8") as f:
            summary["embedding"] = json.load(f)

    trans_path = RESULTS_DIR / "translation_results.json"
    if trans_path.exists():
        with open(trans_path, encoding="utf-8") as f:
            summary["translation"] = json.load(f)

    out = RESULTS_DIR / "final_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Unified results → {out}")

    # Print key findings
    print("\n" + "="*60)
    print("  KEY FINDINGS")
    print("="*60)

    if "embedding" in summary:
        print("\n  EMBEDDING ALIGNMENT (gap = aligned_sim - random_sim)")
        for model in ["baseline", "triplet", "quadruplet"]:
            mr_kn = summary["embedding"].get(model, {}).get("lang_pairs", {}).get("mr_kn", {})
            gap  = mr_kn.get("gap", "N/A")
            acc  = mr_kn.get("accuracy_at_1", "N/A")
            mrr  = mr_kn.get("mrr", "N/A")
            print(f"  {model:12s}  MR-KN gap={gap}  acc@1={acc}  mrr={mrr}")

    if "translation" in summary:
        t = summary["translation"]
        print("\n  TRANSLATION (BLEU / chrF / BERTScore F1)")
        for exp_key, label in [
            ("mr_to_kn_pivot", "MR→EN→KN pivot"),
            ("kn_to_mr_pivot", "KN→EN→MR pivot"),
            ("en_to_kn_direct","EN→KN  direct"),
        ]:
            r = t.get(exp_key, {})
            print(f"  {label:20s}  BLEU={r.get('corpus_bleu','N/A')}  "
                  f"chrF={r.get('corpus_chrf','N/A')}  "
                  f"BERT-F1={r.get('bertscore_f1','N/A')}")

        rt = t.get("round_trip_en_mr_en", {})
        print(f"  Round-trip EN→MR→EN  cosine={rt.get('cosine_similarity','N/A')}")

    return out


if __name__ == "__main__":
    total_start = time.time()
    steps = [
        ("prepare_data.py",       "Prepare train/dev/test splits"),
        ("train_embeddings.py",   "Train triplet & quadruplet sentence transformers"),
        ("evaluate_embeddings.py","Evaluate embedding alignment & bias"),
        ("translation_eval.py",   "Run pivot translation experiments (NLLB-600M)"),
    ]

    failed = []
    for script, label in steps:
        ok = run_step(script, label)
        if not ok:
            failed.append(label)
            print(f"  ⚠  Continuing despite failure in: {label}")

    print(f"\n{'='*60}")
    print("  MERGING RESULTS")
    print(f"{'='*60}")
    out = merge_results()

    total = time.time() - total_start
    print(f"\n{'='*60}")
    if failed:
        print(f"  ⚠  Completed with errors in: {', '.join(failed)}")
    else:
        print(f"  ✅ All experiments complete in {total/60:.1f} min")
    print(f"  Results: {out}")
    print(f"{'='*60}")
