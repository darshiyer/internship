from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from evaluate import compute_bertscore, compute_bleu, compute_chrf
from translate import get_runtime_status, pivot_translate

app = FastAPI(title="Marathi-Kannada Legal MT API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    direction: str = Field("mr_to_kn", pattern="^(mr_to_kn|kn_to_mr)$")
    domain: str = Field("legal", pattern="^(legal|general)$")


class EvalRequest(BaseModel):
    hypothesis: str
    reference: str
    lang: str = "kn"


@app.get("/")
def root():
    return {"status": "ok", "project": "Marathi-Kannada Legal MT"}


@app.post("/translate")
def translate(req: TranslationRequest):
    start = time.time()
    try:
        result = pivot_translate(req.text, direction=req.direction, domain=req.domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["latency_ms"] = round((time.time() - start) * 1000, 2)
    return result


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    bleu = compute_bleu(req.hypothesis, req.reference)
    chrf = compute_chrf(req.hypothesis, req.reference)
    bert = compute_bertscore([req.hypothesis], [req.reference], lang=req.lang)
    return {"bleu": bleu, "chrf": chrf, "bertscore": bert}


@app.get("/model_info")
def model_info():
    runtime = get_runtime_status()
    return {
        **runtime,
        "pipeline": "Pivot (Marathi -> English -> Kannada) and reverse",
        "domain": "Legal",
        "supported_languages": ["Marathi (mar_Deva)", "Kannada (kan_Knda)", "English (eng_Latn)"],
        "models": [
            {
                "name": "IndicTrans2 Indic->En 1B",
                "hf": "https://huggingface.co/ai4bharat/indictrans2-indic-en-1B",
                "params": "1B",
                "training_data": "AI4Bharat multilingual corpora",
            },
            {
                "name": "InLegalTrans En->Indic 1B",
                "hf": "https://huggingface.co/law-ai/InLegalTrans-En2Indic-1B",
                "params": "1B",
                "training_data": "MILPaC + legal corpora",
            },
        ],
    }


def _count_lines_if_exists(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as file_obj:
        return sum(1 for _ in file_obj)


@app.get("/quadruplets")
def get_quadruplets(page: int = 0, page_size: int = 20, search: str = ""):
    q_path = DATA_DIR / "processed" / "quadruplets.json"
    if not q_path.exists():
        return {"quadruplets": [], "total": 0, "page": page, "page_size": page_size}
    with q_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = list(data.values())
    if search:
        s = search.lower()
        items = [
            item for item in items
            if s in item.get("english", "").lower()
            or s in item.get("marathi", "").lower()
            or s in item.get("kannada", "").lower()
            or s in item.get("hindi", "").lower()
        ]
    total = len(items)
    start = page * page_size
    return {
        "quadruplets": items[start: start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


@app.get("/dataset_stats")
def dataset_stats():
    glossary_path = DATA_DIR / "glossary" / "legal_glossary.tsv"
    eval_path = DATA_DIR / "evaluation" / "test_set.json"
    processed_quadruplets = DATA_DIR / "processed" / "quadruplets.json"
    en_mr_tsv = DATA_DIR / "processed" / "en_mr_legal.tsv"
    en_kn_tsv = DATA_DIR / "processed" / "en_kn_legal.tsv"

    glossary_entries = _count_lines_if_exists(glossary_path)

    test_set_size = 0
    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                test_set_size = len(data)

    # Prefer direct counts from processed bilingual TSVs (minus header row).
    en_mr_legal_pairs = max(_count_lines_if_exists(en_mr_tsv) - 1, 0)
    en_kn_legal_pairs = max(_count_lines_if_exists(en_kn_tsv) - 1, 0)

    quadruplets_count = 0
    if processed_quadruplets.exists():
        with processed_quadruplets.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                quadruplets_count = len(data)

    # If TSVs are not ready yet, use quadruplet count as a fallback signal.
    if en_mr_legal_pairs == 0 and en_kn_legal_pairs == 0 and quadruplets_count > 0:
        en_mr_legal_pairs = quadruplets_count
        en_kn_legal_pairs = quadruplets_count

    en_hi_tsv = DATA_DIR / "processed" / "en_hi_legal.tsv"
    en_hi_legal_pairs = max(_count_lines_if_exists(en_hi_tsv) - 1, 0)

    return {
        "en_mr_legal_pairs": en_mr_legal_pairs,
        "en_kn_legal_pairs": en_kn_legal_pairs,
        "en_hi_legal_pairs": en_hi_legal_pairs,
        "quadruplets_count": quadruplets_count,
        "glossary_entries": max(glossary_entries - 1, 0),
        "test_set_size": test_set_size,
    }
