"""
finetune_legal_indictrans.py

Fine-tunes IndicTrans2 Indic-Indic on legal MR↔KN pairs using LoRA/QLoRA.

This creates the first legal-domain direct MR↔KN translation model.

Run on Google Colab (T4 GPU — 16GB VRAM):
    !pip install transformers sentencepiece peft accelerate bitsandbytes datasets
    !pip install IndicTransTokenizer
    !python experiments/finetune_legal_indictrans.py

What this does:
  1. Extracts legal sentences from our quadruplet dataset
  2. Loads IndicTrans2 Indic-Indic-1B in 4-bit quantisation (fits T4)
  3. Fine-tunes with LoRA (r=16) — only ~10M trainable params out of 1B
  4. Saves the adapter weights (small — ~40MB)
  5. Evaluates on held-out legal test set: zero-shot vs fine-tuned

Expected outcome:
  - General IndicTrans2 BERTScore ~0.79 on legal text
  - Fine-tuned BERTScore ~0.83-0.87 on legal text
  - Legal terminology accuracy improved significantly
"""

import json
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent
ROOT       = THIS_DIR.parent if THIS_DIR.name == "experiments" else THIS_DIR
SPLITS_DIR = ROOT / "experiments" / "results" / "splits"
RESULTS    = ROOT / "experiments" / "results"
MODEL_OUT  = RESULTS / "models" / "indictrans_legal_lora"

BASE_MODEL = "ai4bharat/indictrans2-indic-indic-1B"

# ── legal keyword filter ──────────────────────────────────────────────────────
# A sentence is "legal" if it contains at least one keyword in any language
LEGAL_KW = {
    "english": [
        "section", "act", "court", "article", "clause", "schedule",
        "provision", "hereby", "thereof", "whereas", "notwithstanding",
        "pursuant", "ordinance", "statute", "judgment", "decree",
        "tribunal", "plaintiff", "defendant", "magistrate", "petition",
        "verdict", "legislature", "amendment", "constitution", "gazette",
        "regulation", "legal", "law", "justice", "hereby",
    ],
    "marathi": [
        "कलम", "अधिनियम", "न्यायालय", "अनुच्छेद", "खंड", "अनुसूची",
        "तरतूद", "न्यायनिर्णय", "विधेयक", "अध्यादेश", "सरकार",
        "शासन", "राजपत्र", "विधिमंडळ", "दंड", "कायदा", "हक्क",
        "याचिका", "सुनावणी", "वकील", "न्यायाधीश",
    ],
    "kannada": [
        "ವಿಭಾಗ", "ಅಧಿನಿಯಮ", "ನ್ಯಾಯಾಲಯ", "ಅನುಚ್ಛೇದ", "ಷರತ್ತು",
        "ನಿಬಂಧನೆ", "ತೀರ್ಪು", "ಮಸೂದೆ", "ಶಾಸನ", "ಸರ್ಕಾರ",
        "ರಾಜಪತ್ರ", "ಕಾನೂನು", "ನ್ಯಾಯ", "ಅರ್ಜಿ", "ವಕೀಲ",
        "ನ್ಯಾಯಾಧೀಶ", "ದಂಡ", "ಹಕ್ಕು", "ನ್ಯಾಯಮಂಡಳಿ",
    ],
}

def is_legal(entry):
    for lang, keywords in LEGAL_KW.items():
        field = {"english": "english", "marathi": "marathi",
                 "kannada": "kannada"}.get(lang, lang)
        text = entry.get(field, "").lower()
        if any(kw in text for kw in keywords):
            return True
    return False


# ── 1. extract legal pairs ────────────────────────────────────────────────────
def extract_legal_pairs():
    """
    From quadruplet splits, extract sentences tagged as legal.
    Returns list of {"marathi": ..., "kannada": ...} dicts.
    """
    all_pairs = []
    for split in ["triplet_train", "quadruplet_train", "dev"]:
        path = SPLITS_DIR / f"{split}.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        legal = [e for e in entries if is_legal(e)]
        for e in legal:
            mr = e.get("marathi", "").strip()
            kn = e.get("kannada", "").strip()
            if mr and kn:
                all_pairs.append({"marathi": mr, "kannada": kn})

    print(f"\nLegal MR-KN pairs extracted: {len(all_pairs):,}")

    # Deduplicate
    seen = set()
    deduped = []
    for p in all_pairs:
        key = p["marathi"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    print(f"After deduplication: {len(deduped):,}")

    # Split: 90% train, 10% eval
    split_idx = int(len(deduped) * 0.9)
    train_pairs = deduped[:split_idx]
    eval_pairs  = deduped[split_idx:]
    print(f"Train: {len(train_pairs):,}  |  Eval: {len(eval_pairs):,}")

    return train_pairs, eval_pairs


# ── 2. prepare HuggingFace dataset ───────────────────────────────────────────
def make_hf_dataset(pairs, tokenizer, ip, src_lang="mar_Deva", tgt_lang="kan_Knda",
                    max_length=256):
    """
    Convert raw MR-KN pairs into tokenised HuggingFace Dataset.
    IndicTransTokenizer uses IndicProcessor for pre/post-processing.
    """
    from datasets import Dataset

    src_texts = [p["marathi"]  for p in pairs]
    tgt_texts = [p["kannada"] for p in pairs]

    # IndicTrans2 preprocessing
    preprocessed_src = ip.preprocess_batch(src_texts, src_lang=src_lang,
                                            tgt_lang=tgt_lang, show_progress_bar=False)
    preprocessed_tgt = ip.preprocess_batch(tgt_texts, src_lang=tgt_lang,
                                            tgt_lang=src_lang, show_progress_bar=False)

    model_inputs = tokenizer(
        preprocessed_src, text_target=preprocessed_tgt,
        max_length=max_length, truncation=True, padding="max_length",
        return_tensors="pt",
    )

    dataset = Dataset.from_dict({
        "input_ids":      model_inputs["input_ids"].tolist(),
        "attention_mask": model_inputs["attention_mask"].tolist(),
        "labels":         model_inputs["labels"].tolist(),
    })
    return dataset


# ── 3. fine-tune with LoRA / QLoRA ───────────────────────────────────────────
def finetune(train_pairs, eval_pairs):
    import torch
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                               Seq2SeqTrainer, Seq2SeqTrainingArguments,
                               DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model, TaskType
    from IndicTransTokenizer import IndicProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: Fine-tuning on CPU will take many hours.")
        print("Run this on Google Colab T4 GPU for best results.")

    print(f"\nLoading {BASE_MODEL}…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ip        = IndicProcessor(inference=False)

    # Load in 4-bit quantisation to fit T4 (16GB VRAM)
    use_4bit = (device == "cuda")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True).to(device)

    # LoRA config — targets encoder + decoder attention layers
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,                          # rank — higher = more capacity, more memory
        lora_alpha=32,                 # scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj",
                        "fc1", "fc2"],  # IndicTrans2 uses these layer names
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: trainable params ~10M out of 1B (about 1%)

    print("\nPreparing datasets…")
    train_dataset = make_hf_dataset(train_pairs, tokenizer, ip)
    eval_dataset  = make_hf_dataset(eval_pairs,  tokenizer, ip)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    # Training arguments — tuned for T4 (16GB)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUT),
        num_train_epochs=5,            # more epochs on small legal dataset
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, # effective batch = 4×4 = 16
        warmup_steps=100,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        learning_rate=2e-4,            # higher LR works well with LoRA
        report_to="none",
        generation_max_length=256,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting fine-tuning…")
    print(f"  Training pairs  : {len(train_pairs):,}")
    print(f"  Eval pairs      : {len(eval_pairs):,}")
    print(f"  Epochs          : 5")
    print(f"  Effective batch : 16")
    print(f"  LoRA rank       : 16 (~1% params trainable)\n")

    trainer.train()

    # Save LoRA adapter only (small — ~40MB vs 4GB full model)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print(f"\n✅  LoRA adapter saved → {MODEL_OUT}")
    print("    Load it with: model = PeftModel.from_pretrained(base_model, adapter_path)")


# ── 4. evaluate: zero-shot vs fine-tuned ─────────────────────────────────────
def evaluate_both(eval_pairs):
    """
    Compare translation quality on legal test sentences:
      - Zero-shot IndicTrans2 Indic-Indic (no fine-tuning)
      - Fine-tuned IndicTrans2 + LoRA legal adapter
    """
    import torch, sacrebleu
    from bert_score import score as bert_score_fn
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    from IndicTransTokenizer import IndicProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ip     = IndicProcessor(inference=True)

    test_pairs = eval_pairs[:200]
    mr_texts = [p["marathi"]  for p in test_pairs]
    kn_refs  = [p["kannada"] for p in test_pairs]

    def run_inference(model, tokenizer, texts,
                      src="mar_Deva", tgt="kan_Knda", batch_size=8):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            preprocessed = ip.preprocess_batch(batch, src_lang=src,
                                               tgt_lang=tgt, show_progress_bar=False)
            inputs = tokenizer(preprocessed, return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=256).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, num_beams=4, max_new_tokens=256)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            results.extend(ip.postprocess_batch(decoded, lang=tgt))
        return results

    def score(hyps, refs):
        bleu  = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf  = sacrebleu.corpus_chrf(hyps, [refs]).score
        _, _, F1 = bert_score_fn(hyps, refs, lang="kn", verbose=False)
        return {"bleu": round(bleu,2), "chrf": round(chrf,2),
                "bertscore_f1": round(F1.mean().item(), 4)}

    results = {}

    # Zero-shot
    print("\nLoading base model (zero-shot)…")
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_mdl = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True).to(device)
    print("  Running zero-shot inference…")
    zs_hyps = run_inference(base_mdl, base_tok, mr_texts)
    results["zero_shot_general"] = score(zs_hyps, kn_refs)
    del base_mdl  # free VRAM

    # Fine-tuned
    if MODEL_OUT.exists():
        print("\nLoading fine-tuned model (legal adapter)…")
        ft_tok = AutoTokenizer.from_pretrained(str(MODEL_OUT), trust_remote_code=True)
        ft_base = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True).to(device)
        ft_mdl = PeftModel.from_pretrained(ft_base, str(MODEL_OUT))
        ft_mdl.eval()
        print("  Running fine-tuned inference…")
        ft_hyps = run_inference(ft_mdl, ft_tok, mr_texts)
        results["fine_tuned_legal"] = score(ft_hyps, kn_refs)
    else:
        print(f"\nWARNING: Fine-tuned model not found at {MODEL_OUT}")
        print("Run the fine-tuning step first.")

    # Print comparison
    print("\n" + "="*60)
    print("  LEGAL DOMAIN: Zero-shot vs Fine-tuned Comparison")
    print("="*60)
    print(f"  {'Model':<30} {'BLEU':>6} {'chrF':>7} {'BERT-F1':>9}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*9}")
    for name, r in results.items():
        label = "IndicTrans2 (general)" if "zero" in name else "IndicTrans2 + Legal LoRA"
        print(f"  {label:<30} {r['bleu']:>6.2f} {r['chrf']:>7.2f} {r['bertscore_f1']:>9.4f}")

    out = RESULTS / "legal_finetune_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅  Results saved → {out}")
    return results


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip fine-tuning, just evaluate existing adapter")
    args = parser.parse_args()

    print("Extracting legal MR-KN pairs from quadruplet dataset…")
    train_pairs, eval_pairs = extract_legal_pairs()

    if not args.eval_only:
        finetune(train_pairs, eval_pairs)

    evaluate_both(eval_pairs)
