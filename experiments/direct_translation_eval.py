"""
direct_translation_eval.py

Compares THREE translation strategies for Marathi↔Kannada:

  System A (baseline):  MR → [IndicTrans2 Indic-EN] → EN → [NLLB-200] → KN
  System B (proposed):  MR → [IndicTrans2 Indic-Indic] → HI → [IndicTrans2 Indic-Indic] → KN
  System C (proposed):  MR → [IndicTrans2 Indic-Indic] → KN  (NO pivot at all)

Run on Google Colab (T4 GPU) — models are large:
    !pip install transformers sentencepiece sacrebleu bert-score accelerate
    !python experiments/direct_translation_eval.py

This is the core novel contribution of the paper:
  - First benchmark comparing EN-pivot vs HI-pivot vs direct for MR↔KN
  - Uses 500 sentences (up from 100) for statistical reliability
"""

import json, time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent
ROOT       = THIS_DIR.parent if THIS_DIR.name == "experiments" else THIS_DIR
SPLITS_DIR = ROOT / "experiments" / "results" / "splits"
RESULTS    = ROOT / "experiments" / "results"

N_TEST = 500  # up from 100 — more statistically reliable

# ── lazy imports (heavy, only load when needed) ───────────────────────────────
def load_models():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading IndicTrans2 Indic-EN  (for System A first hop)…")
    tok_indic_en = AutoTokenizer.from_pretrained(
        "ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True)
    mdl_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True).to(device)

    print("Loading IndicTrans2 Indic-Indic  (for Systems B and C)…")
    tok_indic_indic = AutoTokenizer.from_pretrained(
        "ai4bharat/indictrans2-indic-indic-1B", trust_remote_code=True)
    mdl_indic_indic = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-indic-indic-1B", trust_remote_code=True).to(device)

    print("Loading NLLB-200 distilled-600M  (for System A second hop)…")
    from transformers import NllbTokenizer
    tok_nllb = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    mdl_nllb = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M").to(device)

    return {
        "indic_en":    (tok_indic_en,    mdl_indic_en),
        "indic_indic": (tok_indic_indic, mdl_indic_indic),
        "nllb":        (tok_nllb,        mdl_nllb),
    }, device


def translate_batch(tokenizer, model, texts, src_lang, tgt_lang,
                    device, batch_size=8, max_new_tokens=256):
    """Generic batch translation for any seq2seq model."""
    from transformers import pipeline
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        with __import__("torch").no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
                    if hasattr(tokenizer, "convert_tokens_to_ids") else None,
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        results.extend(decoded)
        if (i // batch_size) % 5 == 0:
            print(f"  Translated {min(i+batch_size, len(texts))}/{len(texts)}")
    return results


# ── IndicTrans2 wrapper (uses IndicTrans2 tokenizer conventions) ──────────────
def indic_translate(tokenizer, model, texts, src, tgt, device, batch_size=8):
    """
    IndicTrans2 uses language codes like 'mar_Deva', 'kan_Knda', 'hin_Deva', 'eng_Latn'
    and handles forced_bos differently — uses tokenizer's built-in lang support.
    """
    from IndicTransTokenizer import IndicProcessor   # pip install IndicTransTokenizer
    ip = IndicProcessor(inference=True)
    preprocessed = ip.preprocess_batch(texts, src_lang=src, tgt_lang=tgt)
    inputs = tokenizer(
        preprocessed,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        max_length=256,
    ).to(device)
    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=256,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=tgt)


# ── NLLB wrapper ──────────────────────────────────────────────────────────────
# NLLB language codes:  Marathi=mar_Deva  Kannada=kan_Knda  Hindi=hin_Deva  English=eng_Latn
NLLB_CODES = {
    "marathi":  "mar_Deva",
    "kannada":  "kan_Knda",
    "hindi":    "hin_Deva",
    "english":  "eng_Latn",
}

def nllb_translate(tokenizer, model, texts, src_lang, tgt_lang, device, batch_size=8):
    tokenizer.src_lang = src_lang
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(device)
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
        with __import__("torch").no_grad():
            out = model.generate(**inputs, forced_bos_token_id=forced_bos,
                                 max_new_tokens=256, num_beams=4)
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(hypotheses, references, lang_code="kn"):
    import sacrebleu
    from bert_score import score as bert_score

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    P, R, F1 = bert_score(hypotheses, references, lang=lang_code, verbose=False)
    return {
        "bleu":       round(bleu, 2),
        "chrf":       round(chrf, 2),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def round_trip_cosine(texts, translations, model_name="sentence-transformers/LaBSE"):
    """Encode original + back-translation, compute cosine similarity."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    sem_model = SentenceTransformer(model_name)
    e1 = sem_model.encode(texts,        normalize_embeddings=True)
    e2 = sem_model.encode(translations, normalize_embeddings=True)
    cosines = (e1 * e2).sum(axis=1)
    return round(float(np.mean(cosines)), 4)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # Load test data
    split_path = SPLITS_DIR / "test.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Test split not found at {split_path}")

    with open(split_path, encoding="utf-8") as f:
        test = json.load(f)

    # Filter to entries that have all three languages
    complete = [e for e in test
                if e.get("marathi") and e.get("kannada") and e.get("english")]
    complete = complete[:N_TEST]
    print(f"Test sentences: {len(complete)}")

    mr_texts  = [e["marathi"]  for e in complete]
    kn_refs   = [e["kannada"]  for e in complete]
    en_texts  = [e["english"]  for e in complete]

    models, device = load_models()
    tok_ie, mdl_ie       = models["indic_en"]
    tok_ii, mdl_ii       = models["indic_indic"]
    tok_nllb, mdl_nllb   = models["nllb"]

    all_results = {}

    # ── SYSTEM A: English pivot (MR→EN→KN) ────────────────────────────────────
    print("\n" + "="*60)
    print("SYSTEM A: MR → EN → KN  (English pivot, baseline)")
    print("="*60)
    t0 = time.time()

    print("  Step 1: MR → EN  (IndicTrans2 Indic-EN)")
    mr_to_en = indic_translate(tok_ie, mdl_ie, mr_texts,
                                src="mar_Deva", tgt="eng_Latn", device=device)

    print("  Step 2: EN → KN  (NLLB-200)")
    en_to_kn = nllb_translate(tok_nllb, mdl_nllb, mr_to_en,
                               src_lang="eng_Latn", tgt_lang="kan_Knda",
                               device=device)

    sysA_time = time.time() - t0
    sysA_metrics = compute_metrics(en_to_kn, kn_refs, lang_code="kn")
    sysA_rt = round_trip_cosine(mr_texts, en_to_kn)
    all_results["system_A_english_pivot"] = {
        **sysA_metrics,
        "round_trip_cosine": sysA_rt,
        "inference_time_sec": round(sysA_time, 1),
        "pipeline": "MR→EN→KN",
    }
    print(f"  BLEU={sysA_metrics['bleu']}  chrF={sysA_metrics['chrf']}  "
          f"BERT-F1={sysA_metrics['bertscore_f1']}  "
          f"RT-cos={sysA_rt}  time={sysA_time:.1f}s")

    # ── SYSTEM B: Hindi pivot (MR→HI→KN) ─────────────────────────────────────
    print("\n" + "="*60)
    print("SYSTEM B: MR → HI → KN  (Hindi pivot, linguistically closer)")
    print("="*60)
    t0 = time.time()

    print("  Step 1: MR → HI  (IndicTrans2 Indic-Indic)")
    mr_to_hi = indic_translate(tok_ii, mdl_ii, mr_texts,
                                src="mar_Deva", tgt="hin_Deva", device=device)

    print("  Step 2: HI → KN  (IndicTrans2 Indic-Indic)")
    hi_to_kn = indic_translate(tok_ii, mdl_ii, mr_to_hi,
                                src="hin_Deva", tgt="kan_Knda", device=device)

    sysB_time = time.time() - t0
    sysB_metrics = compute_metrics(hi_to_kn, kn_refs, lang_code="kn")
    sysB_rt = round_trip_cosine(mr_texts, hi_to_kn)
    all_results["system_B_hindi_pivot"] = {
        **sysB_metrics,
        "round_trip_cosine": sysB_rt,
        "inference_time_sec": round(sysB_time, 1),
        "pipeline": "MR→HI→KN",
    }
    print(f"  BLEU={sysB_metrics['bleu']}  chrF={sysB_metrics['chrf']}  "
          f"BERT-F1={sysB_metrics['bertscore_f1']}  "
          f"RT-cos={sysB_rt}  time={sysB_time:.1f}s")

    # ── SYSTEM C: Direct MR→KN (no pivot at all) ──────────────────────────────
    print("\n" + "="*60)
    print("SYSTEM C: MR → KN  (DIRECT — no pivot language)")
    print("="*60)
    t0 = time.time()

    print("  Translating: MR → KN  (IndicTrans2 Indic-Indic, single hop)")
    mr_to_kn_direct = indic_translate(tok_ii, mdl_ii, mr_texts,
                                       src="mar_Deva", tgt="kan_Knda", device=device)

    sysC_time = time.time() - t0
    sysC_metrics = compute_metrics(mr_to_kn_direct, kn_refs, lang_code="kn")
    sysC_rt = round_trip_cosine(mr_texts, mr_to_kn_direct)
    all_results["system_C_direct"] = {
        **sysC_metrics,
        "round_trip_cosine": sysC_rt,
        "inference_time_sec": round(sysC_time, 1),
        "pipeline": "MR→KN (direct)",
    }
    print(f"  BLEU={sysC_metrics['bleu']}  chrF={sysC_metrics['chrf']}  "
          f"BERT-F1={sysC_metrics['bertscore_f1']}  "
          f"RT-cos={sysC_rt}  time={sysC_time:.1f}s")

    # ── REVERSE DIRECTION: KN→MR ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("BONUS — Reverse direction KN→MR for all three systems")
    print("="*60)

    mr_refs = mr_texts  # now Marathi is the reference

    # KN→EN→MR
    kn_to_en = nllb_translate(tok_nllb, mdl_nllb, kn_refs,
                               "kan_Knda", "eng_Latn", device)
    en_to_mr = indic_translate(tok_ie, mdl_ie, kn_to_en,
                                "eng_Latn", "mar_Deva", device)  # uses indic-en reversed? use indic_indic
    # Actually use NLLB for EN→MR as well
    en_to_mr = nllb_translate(tok_nllb, mdl_nllb, kn_to_en,
                               "eng_Latn", "mar_Deva", device)
    rev_A = compute_metrics(en_to_mr, mr_refs, lang_code="mr")
    all_results["system_A_reverse_KN_MR"] = {**rev_A, "pipeline": "KN→EN→MR"}

    # KN→HI→MR
    kn_to_hi = indic_translate(tok_ii, mdl_ii, kn_refs, "kan_Knda", "hin_Deva", device)
    hi_to_mr = indic_translate(tok_ii, mdl_ii, kn_to_hi, "hin_Deva", "mar_Deva", device)
    rev_B = compute_metrics(hi_to_mr, mr_refs, lang_code="mr")
    all_results["system_B_reverse_KN_MR"] = {**rev_B, "pipeline": "KN→HI→MR"}

    # KN→MR direct
    kn_to_mr = indic_translate(tok_ii, mdl_ii, kn_refs, "kan_Knda", "mar_Deva", device)
    rev_C = compute_metrics(kn_to_mr, mr_refs, lang_code="mr")
    all_results["system_C_reverse_KN_MR"] = {**rev_C, "pipeline": "KN→MR (direct)"}

    # ── COMPARISON TABLE ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL COMPARISON TABLE  (MR→KN direction)")
    print("="*70)
    print(f"  {'System':<30} {'BLEU':>6} {'chrF':>7} {'BERT-F1':>9} {'RT-cos':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*9} {'-'*8}")
    for key in ["system_A_english_pivot", "system_B_hindi_pivot", "system_C_direct"]:
        r = all_results[key]
        print(f"  {r['pipeline']:<30} {r['bleu']:>6.2f} {r['chrf']:>7.2f} "
              f"{r['bertscore_f1']:>9.4f} {r['round_trip_cosine']:>8.4f}")

    print(f"\n  {'System':<30} {'BLEU':>6} {'chrF':>7} {'BERT-F1':>9}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*9}")
    for key in ["system_A_reverse_KN_MR", "system_B_reverse_KN_MR", "system_C_reverse_KN_MR"]:
        r = all_results[key]
        print(f"  {r['pipeline']:<30} {r['bleu']:>6.2f} {r['chrf']:>7.2f} "
              f"{r['bertscore_f1']:>9.4f}")

    # ── save results ──────────────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "direct_translation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅  Results saved → {out}")

    # Save sample translations for manual inspection
    samples = []
    for i in range(min(20, len(complete))):
        samples.append({
            "id": i+1,
            "source_marathi": mr_texts[i],
            "reference_kannada": kn_refs[i],
            "system_A_EN_pivot": en_to_kn[i],
            "system_B_HI_pivot": hi_to_kn[i],
            "system_C_direct":   mr_to_kn_direct[i],
        })
    with open(RESULTS / "translation_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"✅  Sample translations saved → {RESULTS}/translation_samples.json")
    print("\nUse translation_samples.json for human evaluation (human_eval_generate.py)")


if __name__ == "__main__":
    main()
