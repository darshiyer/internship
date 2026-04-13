"""
Step 4 — Translation evaluation using NLLB-200-distilled-600M.

Experiments:
  A. Pivot translation: MR → EN → KN
  B. Pivot translation: KN → EN → MR
  C. Back-translation consistency: MR → EN → KN → EN  (round-trip)

Evaluation metrics:
  - BLEU   (sacrebleu)
  - chrF   (sacrebleu)
  - BERTScore (bert-score)
  - Round-trip consistency score (cosine similarity EN_original vs EN_backtranslated)

Test set: entries that have BOTH Marathi and Kannada (so we have a reference).
"""

import json, time, torch
from pathlib import Path
from config import TRANSLATION_MODEL, LANG_CODES, RESULTS_DIR, SPLITS_DIR, TRANSLATION_TEST_SIZE

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU, CHRF
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer


# ── Model loading ─────────────────────────────────────────────────────────────

def load_nllb():
    print(f"\nLoading translation model: {TRANSLATION_MODEL}")
    print("  (first run downloads ~2.5GB — please wait)")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    # Force CPU — NLLB-600M causes MPS OOM on Apple Silicon with large batches
    device = "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device)
    model.eval()
    print(f"  Loaded on device: {device}")
    return model, tokenizer, device


def translate_batch(texts, src_lang, tgt_lang, model, tokenizer, device, batch_size=8):
    tokenizer.src_lang = LANG_CODES[src_lang]
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(device)
        forced_bos = tokenizer.convert_tokens_to_ids(LANG_CODES[tgt_lang])
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=forced_bos,
                                 num_beams=4, max_length=256)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(decoded)
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(hypotheses, references, lang="kn"):
    bleu_scorer = BLEU(effective_order=True)
    chrf_scorer = CHRF()

    bleu_scores = [bleu_scorer.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]
    chrf_scores = [chrf_scorer.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]

    P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    bert_f1  = F1.mean().item()

    corpus_bleu = BLEU().corpus_score(hypotheses, [references]).score
    corpus_chrf = CHRF().corpus_score(hypotheses, [references]).score

    return {
        "corpus_bleu":  round(corpus_bleu, 4),
        "corpus_chrf":  round(corpus_chrf, 4),
        "mean_sent_bleu": round(sum(bleu_scores)/len(bleu_scores), 4),
        "mean_sent_chrf": round(sum(chrf_scores)/len(chrf_scores), 4),
        "bertscore_f1": round(bert_f1, 4),
        "n": len(hypotheses),
    }


def round_trip_score(original_en, back_translated_en, encoder_model):
    """Cosine similarity between original EN and round-tripped EN."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    embs_orig = encoder_model.encode(original_en, normalize_embeddings=True)
    embs_back = encoder_model.encode(back_translated_en, normalize_embeddings=True)
    sims = (embs_orig * embs_back).sum(axis=1)
    return round(float(np.mean(sims)), 4)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load test data — only entries with both MR and KN
    with open(SPLITS_DIR / "test.json", encoding="utf-8") as f:
        test_all = json.load(f)

    mr_kn_test = [e for e in test_all
                  if e.get("marathi") and e.get("kannada") and e.get("english")]
    mr_kn_test = mr_kn_test[:TRANSLATION_TEST_SIZE]
    print(f"Translation test set (MR+KN+EN): {len(mr_kn_test)} sentences")

    if len(mr_kn_test) < 5:
        print("⚠  Not enough MR+KN test entries. Results will be limited.")

    en_refs = [e["english"] for e in mr_kn_test]
    mr_sents = [e["marathi"] for e in mr_kn_test]
    kn_refs  = [e["kannada"] for e in mr_kn_test]

    # Load translation model
    nllb_model, nllb_tok, device = load_nllb()

    results = {}

    # ── Experiment A: MR → EN → KN (pivot) ─────────────────────────────────
    print("\n[Experiment A] Pivot: Marathi → English → Kannada")
    t0 = time.time()
    mr_to_en = translate_batch(mr_sents, "marathi", "english", nllb_model, nllb_tok, device)
    en_to_kn = translate_batch(mr_to_en, "english", "kannada", nllb_model, nllb_tok, device)
    elapsed_A = time.time() - t0
    metrics_A = compute_metrics(en_to_kn, kn_refs, lang="kn")
    metrics_A["time_seconds"] = round(elapsed_A, 1)
    results["mr_to_kn_pivot"] = metrics_A
    print(f"  corpus_bleu={metrics_A['corpus_bleu']}  "
          f"corpus_chrf={metrics_A['corpus_chrf']}  "
          f"bertscore_f1={metrics_A['bertscore_f1']}  "
          f"time={elapsed_A:.0f}s")

    # ── Experiment B: KN → EN → MR (pivot) ─────────────────────────────────
    print("\n[Experiment B] Pivot: Kannada → English → Marathi")
    t0 = time.time()
    kn_to_en = translate_batch(kn_refs,  "kannada", "english", nllb_model, nllb_tok, device)
    en_to_mr = translate_batch(kn_to_en, "english", "marathi", nllb_model, nllb_tok, device)
    elapsed_B = time.time() - t0
    metrics_B = compute_metrics(en_to_mr, mr_sents, lang="mr")
    metrics_B["time_seconds"] = round(elapsed_B, 1)
    results["kn_to_mr_pivot"] = metrics_B
    print(f"  corpus_bleu={metrics_B['corpus_bleu']}  "
          f"corpus_chrf={metrics_B['corpus_chrf']}  "
          f"bertscore_f1={metrics_B['bertscore_f1']}  "
          f"time={elapsed_B:.0f}s")

    # ── Experiment C: Round-trip EN → MR → EN ───────────────────────────────
    print("\n[Experiment C] Round-trip: English → Marathi → English")
    t0 = time.time()
    en_to_mr_fwd = translate_batch(en_refs,    "english", "marathi", nllb_model, nllb_tok, device)
    mr_back_en   = translate_batch(en_to_mr_fwd,"marathi","english", nllb_model, nllb_tok, device)
    elapsed_C = time.time() - t0

    encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    rt_score = round_trip_score(en_refs, mr_back_en, encoder)
    results["round_trip_en_mr_en"] = {
        "cosine_similarity": rt_score,
        "time_seconds": round(elapsed_C, 1),
        "n": len(en_refs),
        "interpretation": "Higher = more meaning preserved through translation cycle",
    }
    print(f"  round-trip cosine similarity = {rt_score}  time={elapsed_C:.0f}s")

    # ── Experiment D: EN → KN direct ────────────────────────────────────────
    print("\n[Experiment D] Direct: English → Kannada (no pivot, NLLB direct)")
    t0 = time.time()
    en_to_kn_direct = translate_batch(en_refs, "english", "kannada", nllb_model, nllb_tok, device)
    elapsed_D = time.time() - t0
    metrics_D = compute_metrics(en_to_kn_direct, kn_refs, lang="kn")
    metrics_D["time_seconds"] = round(elapsed_D, 1)
    results["en_to_kn_direct"] = metrics_D
    print(f"  corpus_bleu={metrics_D['corpus_bleu']}  "
          f"corpus_chrf={metrics_D['corpus_chrf']}  "
          f"bertscore_f1={metrics_D['bertscore_f1']}  "
          f"time={elapsed_D:.0f}s")

    # Save samples for inspection
    samples = []
    for i in range(min(10, len(mr_kn_test))):
        samples.append({
            "english_source":    en_refs[i],
            "marathi_input":     mr_sents[i],
            "kannada_reference": kn_refs[i],
            "pivot_mr_en":       mr_to_en[i],
            "pivot_mr_kn":       en_to_kn[i],
            "direct_en_kn":      en_to_kn_direct[i],
        })
    results["sample_outputs"] = samples

    # Save
    out = RESULTS_DIR / "translation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Translation results saved → {out}")
