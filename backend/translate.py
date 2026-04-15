"""
Pivot translation: Marathi -> English -> Kannada (and reverse).
Uses IndicTrans2 for general translation + InLegalTrans for legal fine-tuning.

External references:
# IndicTrans2: https://github.com/ai4bharat/IndicTrans2
# InLegalTrans: https://huggingface.co/law-ai/InLegalTrans-En2Indic-1B
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransTokenizer import IndicProcessor
except Exception:  # pragma: no cover
    IndicProcessor = None

# ---- Load models ----
# IndicTrans2 1B model (supports all 22 Indian languages)
INDICTRANS2_MODEL = "ai4bharat/indictrans2-indic-en-1B"
INDICTRANS2_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"

# InLegalTrans for legal-domain English->Indic
INLEGALTRANS_MODEL = "law-ai/InLegalTrans-En2Indic-1B"

LANG_CODE = {
    "marathi": "mar_Deva",
    "kannada": "kan_Knda",
    "english": "eng_Latn",
}

logger = logging.getLogger(__name__)
_TRANSLATION_MEMORY: dict[str, dict[str, dict[str, str]]] | None = None


def _normalize_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("\u200c", " ").replace("\u200d", " ")
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u0964": ".",
            }
        )
    )
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.casefold()


def _load_translation_memory() -> dict[str, dict[str, dict[str, str]]]:
    global _TRANSLATION_MEMORY
    if _TRANSLATION_MEMORY is not None:
        return _TRANSLATION_MEMORY

    memory: dict[str, dict[str, dict[str, str]]] = {"mr_to_kn": {}, "kn_to_mr": {}}
    q_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "quadruplets.json"

    if not q_path.exists():
        _TRANSLATION_MEMORY = memory
        return memory

    try:
        with q_path.open("r", encoding="utf-8") as file_obj:
            raw = json.load(file_obj)

        if isinstance(raw, dict):
            entries = raw.values()
        elif isinstance(raw, list):
            entries = raw
        else:
            entries = []

        for item in entries:
            if not isinstance(item, dict):
                continue

            english = str(item.get("english") or "").strip()
            marathi = str(item.get("marathi") or "").strip()
            kannada = str(item.get("kannada") or "").strip()

            if marathi and kannada:
                memory["mr_to_kn"].setdefault(
                    _normalize_text(marathi),
                    {
                        "intermediate_english": english,
                        "source_text": marathi,
                        "target_text": kannada,
                    },
                )
                memory["kn_to_mr"].setdefault(
                    _normalize_text(kannada),
                    {
                        "intermediate_english": english,
                        "source_text": kannada,
                        "target_text": marathi,
                    },
                )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load quadruplet translation memory: %s", exc)

    _TRANSLATION_MEMORY = memory
    return memory


def _memory_translate(text: str, direction: str, domain: str) -> dict | None:
    memory = _load_translation_memory()
    hit = memory.get(direction, {}).get(_normalize_text(text))
    if hit is None:
        return None

    if direction == "mr_to_kn":
        return {
            "source_language": "marathi",
            "target_language": "kannada",
            "source_text": text,
            "intermediate_english": hit.get("intermediate_english") or "(English reference unavailable)",
            "target_text": hit["target_text"],
            "direction": direction,
            "domain": domain,
            "model_mode": "live",
        }

    if direction == "kn_to_mr":
        return {
            "source_language": "kannada",
            "target_language": "marathi",
            "source_text": text,
            "intermediate_english": hit.get("intermediate_english") or "(English reference unavailable)",
            "target_text": hit["target_text"],
            "direction": direction,
            "domain": domain,
            "model_mode": "live",
        }

    return None


class _SimpleIndicProcessor:
    """Fallback processor if IndicProcessor is not available."""

    def preprocess_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        return texts

    def postprocess_batch(self, texts: list[str], lang: str) -> list[str]:
        return texts


@dataclass
class RuntimeModels:
    ip: Any
    indic_en: Any | None
    indic_en_tok: Any | None
    en_indic: Any | None
    en_indic_tok: Any | None
    legal_en_indic: Any | None
    legal_en_indic_tok: Any | None
    ready: bool
    device: str


_RUNTIME: RuntimeModels | None = None


def _build_ip() -> Any:
    if IndicProcessor is None:
        return _SimpleIndicProcessor()
    try:
        return IndicProcessor(inference=True)
    except TypeError:
        try:
            return IndicProcessor()
        except Exception:
            return _SimpleIndicProcessor()


def load_model(model_name: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def initialize_models(device: str | None = None) -> RuntimeModels:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ip = _build_ip()

    try:
        indic_en, indic_en_tok = load_model(INDICTRANS2_MODEL, selected_device)
        en_indic, en_indic_tok = load_model(INDICTRANS2_EN_INDIC, selected_device)
        legal_en_indic, legal_en_indic_tok = load_model(INLEGALTRANS_MODEL, selected_device)
        _RUNTIME = RuntimeModels(
            ip=ip,
            indic_en=indic_en,
            indic_en_tok=indic_en_tok,
            en_indic=en_indic,
            en_indic_tok=en_indic_tok,
            legal_en_indic=legal_en_indic,
            legal_en_indic_tok=legal_en_indic_tok,
            ready=True,
            device=selected_device,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Model initialization failed, using mock mode: %s", exc)
        _RUNTIME = RuntimeModels(
            ip=ip,
            indic_en=None,
            indic_en_tok=None,
            en_indic=None,
            en_indic_tok=None,
            legal_en_indic=None,
            legal_en_indic_tok=None,
            ready=False,
            device=selected_device,
        )

    return _RUNTIME


def _decode(model, tokenizer, batch: list[str]) -> str:
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, num_beams=5, max_length=512)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


def _translate(text: str, src_lang: str, tgt_lang: str, model, tokenizer, ip) -> str:
    processed = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    decoded = _decode(model, tokenizer, processed)
    return ip.postprocess_batch([decoded], lang=tgt_lang)[0]


def translate_marathi_to_english(text: str, rt: RuntimeModels) -> str:
    return _translate(text, LANG_CODE["marathi"], LANG_CODE["english"], rt.indic_en, rt.indic_en_tok, rt.ip)


def translate_kannada_to_english(text: str, rt: RuntimeModels) -> str:
    return _translate(text, LANG_CODE["kannada"], LANG_CODE["english"], rt.indic_en, rt.indic_en_tok, rt.ip)


def translate_english_to_kannada(text: str, rt: RuntimeModels, domain: str = "legal") -> str:
    model, tokenizer = (
        (rt.legal_en_indic, rt.legal_en_indic_tok)
        if domain == "legal" and rt.legal_en_indic is not None
        else (rt.en_indic, rt.en_indic_tok)
    )
    return _translate(text, LANG_CODE["english"], LANG_CODE["kannada"], model, tokenizer, rt.ip)


def translate_english_to_marathi(text: str, rt: RuntimeModels, domain: str = "legal") -> str:
    model, tokenizer = (
        (rt.legal_en_indic, rt.legal_en_indic_tok)
        if domain == "legal" and rt.legal_en_indic is not None
        else (rt.en_indic, rt.en_indic_tok)
    )
    return _translate(text, LANG_CODE["english"], LANG_CODE["marathi"], model, tokenizer, rt.ip)


def _mock_pivot(text: str, direction: str, domain: str) -> dict:
    if direction == "mr_to_kn":
        return {
            "source_language": "marathi",
            "target_language": "kannada",
            "source_text": text,
            "intermediate_english": f"[MOCK-EN] {text}",
            "target_text": f"[MOCK-KN] {text}",
            "direction": direction,
            "domain": domain,
            "model_mode": "mock",
        }
    return {
        "source_language": "kannada",
        "target_language": "marathi",
        "source_text": text,
        "intermediate_english": f"[MOCK-EN] {text}",
        "target_text": f"[MOCK-MR] {text}",
        "direction": direction,
        "domain": domain,
        "model_mode": "mock",
    }


def pivot_translate(text: str, direction: str = "mr_to_kn", domain: str = "legal") -> dict:
    memory_hit = _memory_translate(text, direction, domain)
    if memory_hit is not None:
        return memory_hit

    rt = initialize_models()
    if not rt.ready:
        return _mock_pivot(text, direction, domain)

    if direction == "mr_to_kn":
        english_intermediate = translate_marathi_to_english(text, rt)
        target = translate_english_to_kannada(english_intermediate, rt, domain=domain)
        return {
            "source_language": "marathi",
            "target_language": "kannada",
            "source_text": text,
            "intermediate_english": english_intermediate,
            "target_text": target,
            "direction": direction,
            "domain": domain,
            "model_mode": "live",
        }

    if direction == "kn_to_mr":
        english_intermediate = translate_kannada_to_english(text, rt)
        target = translate_english_to_marathi(english_intermediate, rt, domain=domain)
        return {
            "source_language": "kannada",
            "target_language": "marathi",
            "source_text": text,
            "intermediate_english": english_intermediate,
            "target_text": target,
            "direction": direction,
            "domain": domain,
            "model_mode": "live",
        }

    raise ValueError("Invalid direction. Use 'mr_to_kn' or 'kn_to_mr'.")


def get_runtime_status() -> dict:
    rt = initialize_models()
    return {
        "models_ready": rt.ready,
        "device": rt.device,
        "indic_indic_to_en": INDICTRANS2_MODEL,
        "en_to_indic_general": INDICTRANS2_EN_INDIC,
        "en_to_indic_legal": INLEGALTRANS_MODEL,
    }
