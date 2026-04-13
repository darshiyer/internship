"""
Prepare legal bilingual TSVs from raw corpora.

Outputs:
- data/processed/en_mr_legal.tsv (english, marathi, hindi)
- data/processed/en_kn_legal.tsv (english, kannada)

External references:
# MILPaC dataset: https://github.com/Law-AI/MILPaC
# English-Kannada legal corpus: https://www.futurebeeai.com/dataset/parallel-corpora/kannada-english-translated-parallel-corpus-for-legal-domain
# Samanantar: https://huggingface.co/datasets/ai4bharat/samanantar
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

try:
    from datasets import load_from_disk
except Exception:  # pragma: no cover
    load_from_disk = None

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"

DELIMITERS = ["\t", ",", "|"]

EN_KEYS = {"english", "en", "src", "source", "sentence_en", "text_en"}
MR_KEYS = {"marathi", "mr", "sentence_mr", "text_mr", "target_mr"}
KN_KEYS = {"kannada", "kn", "sentence_kn", "text_kn", "target_kn"}
HI_KEYS = {"hindi", "hi", "sentence_hi", "text_hi", "target_hi"}


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def _find_column(columns: list[str], aliases: set[str]) -> str | None:
    normalized = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for col in columns:
        low = col.lower().strip()
        if any(alias in low for alias in aliases):
            return col
    return None


def _read_tabular_pairs(path: Path, target_lang: str) -> list[dict]:
    pairs: list[dict] = []
    for delimiter in DELIMITERS:
        try:
            df = pd.read_csv(path, sep=delimiter, quoting=csv.QUOTE_MINIMAL)
            if df.shape[1] < 2:
                continue

            columns = list(df.columns)
            en_col = _find_column(columns, EN_KEYS)
            tgt_col = _find_column(columns, MR_KEYS if target_lang == "mr" else KN_KEYS)
            hi_col = _find_column(columns, HI_KEYS) if target_lang == "mr" else None

            if en_col is None or tgt_col is None:
                continue

            for _, row in df.iterrows():
                english = _clean(row.get(en_col, ""))
                target = _clean(row.get(tgt_col, ""))
                if not english or not target:
                    continue
                if target_lang == "mr":
                    pairs.append(
                        {
                            "english": english,
                            "marathi": target,
                            "hindi": _clean(row.get(hi_col, "")) if hi_col else "",
                        }
                    )
                else:
                    pairs.append({"english": english, "kannada": target})
            if pairs:
                return pairs
        except Exception:
            continue
    return []


def _line_pairs(en_file: Path, tgt_file: Path, target_lang: str) -> list[dict]:
    en_lines = en_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    tgt_lines = tgt_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    size = min(len(en_lines), len(tgt_lines))
    pairs: list[dict] = []
    for i in range(size):
        english = _clean(en_lines[i])
        target = _clean(tgt_lines[i])
        if not english or not target:
            continue
        if target_lang == "mr":
            pairs.append({"english": english, "marathi": target, "hindi": ""})
        else:
            pairs.append({"english": english, "kannada": target})
    return pairs


def _collect_parallel_files(root: Path, target_lang: str) -> list[dict]:
    pairs: list[dict] = []
    en_suffixes = [".en", ".eng", ".english"]
    tgt_suffixes = [".mr", ".marathi"] if target_lang == "mr" else [".kn", ".kannada"]

    for en_file in root.rglob("*"):
        if not en_file.is_file():
            continue
        en_name = en_file.name.lower()
        if not any(en_name.endswith(suf) for suf in en_suffixes):
            continue

        for tgt_suffix in tgt_suffixes:
            tgt_file = en_file.with_suffix(tgt_suffix)
            if tgt_file.exists():
                pairs.extend(_line_pairs(en_file, tgt_file, target_lang))

    return pairs


def _collect_tabular_files(root: Path, target_lang: str) -> list[dict]:
    pairs: list[dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            continue
        extracted = _read_tabular_pairs(path, target_lang)
        if extracted:
            pairs.extend(extracted)
    return pairs


def _collect_samanantar_pairs(dataset_path: Path, target_lang: str, limit: int | None) -> list[dict]:
    if load_from_disk is None or not dataset_path.exists():
        return []

    ds = load_from_disk(str(dataset_path))
    pairs: list[dict] = []
    count = 0
    for item in ds:
        # Samanantar records have keys: src (English), tgt (Indic).
        english = _clean(item.get("src") or item.get("en") or item.get("english") or "")
        target = _clean(item.get("tgt") or item.get(target_lang) or item.get("indic") or "")
        if not english or not target:
            continue
        if target_lang == "mr":
            pairs.append({"english": english, "marathi": target, "hindi": ""})
        elif target_lang == "kn":
            pairs.append({"english": english, "kannada": target})
        elif target_lang == "hi":
            pairs.append({"english": english, "hindi": target})
        count += 1
        if limit is not None and count >= limit:
            break
    return pairs


def _dedupe_pairs(pairs: list[dict], target_key: str) -> list[dict]:
    seen = set()
    deduped: list[dict] = []
    for pair in pairs:
        english = _clean(pair.get("english", ""))
        target = _clean(pair.get(target_key, ""))
        if not english or not target:
            continue
        key = (english.lower(), target.lower())
        if key in seen:
            continue
        seen.add(key)
        pair["english"] = english
        pair[target_key] = target
        deduped.append(pair)
    return deduped


def prepare_en_mr(limit_samanantar: int | None = 50000) -> list[dict]:
    pairs: list[dict] = []
    milpac = RAW_DIR / "MILPaC"
    if milpac.exists():
        pairs.extend(_collect_tabular_files(milpac, "mr"))
        pairs.extend(_collect_parallel_files(milpac, "mr"))

    pairs.extend(_collect_tabular_files(RAW_DIR, "mr"))

    samanantar_path = RAW_DIR / "samanantar_en_mr"
    pairs.extend(_collect_samanantar_pairs(samanantar_path, "mr", limit_samanantar))

    return _dedupe_pairs(pairs, "marathi")


def prepare_en_kn(limit_samanantar: int | None = 50000) -> list[dict]:
    pairs: list[dict] = []
    en_kn = RAW_DIR / "en_kn_legal"
    if en_kn.exists():
        pairs.extend(_collect_tabular_files(en_kn, "kn"))
        pairs.extend(_collect_parallel_files(en_kn, "kn"))

    pairs.extend(_collect_tabular_files(RAW_DIR, "kn"))

    samanantar_path = RAW_DIR / "samanantar_en_kn"
    pairs.extend(_collect_samanantar_pairs(samanantar_path, "kn", limit_samanantar))

    return _dedupe_pairs(pairs, "kannada")


def prepare_en_hi(limit_samanantar: int | None = 50000) -> list[dict]:
    """Prepare English–Hindi pairs from Samanantar EN-HI."""
    pairs: list[dict] = []
    samanantar_path = RAW_DIR / "samanantar_en_hi"
    pairs.extend(_collect_samanantar_pairs(samanantar_path, "hi", limit_samanantar))
    # Also scan any raw tabular files that might have Hindi columns.
    pairs.extend(_collect_tabular_files(RAW_DIR, "hi"))
    return _dedupe_pairs(pairs, "hindi")


def save_outputs(en_mr: list[dict], en_kn: list[dict], en_hi: list[dict] | None = None) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    en_mr_df = pd.DataFrame(en_mr, columns=["english", "marathi", "hindi"])
    en_kn_df = pd.DataFrame(en_kn, columns=["english", "kannada"])

    en_mr_path = PROCESSED_DIR / "en_mr_legal.tsv"
    en_kn_path = PROCESSED_DIR / "en_kn_legal.tsv"
    summary_path = PROCESSED_DIR / "preprocess_summary.json"

    en_mr_df.to_csv(en_mr_path, sep="\t", index=False)
    en_kn_df.to_csv(en_kn_path, sep="\t", index=False)

    summary: dict = {
        "en_mr_pairs": len(en_mr_df),
        "en_kn_pairs": len(en_kn_df),
        "en_mr_path": str(en_mr_path),
        "en_kn_path": str(en_kn_path),
    }

    if en_hi is not None:
        en_hi_df = pd.DataFrame(en_hi, columns=["english", "hindi"])
        en_hi_path = PROCESSED_DIR / "en_hi_legal.tsv"
        en_hi_df.to_csv(en_hi_path, sep="\t", index=False)
        summary["en_hi_pairs"] = len(en_hi_df)
        summary["en_hi_path"] = str(en_hi_path)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed legal bilingual TSV files.")
    parser.add_argument(
        "--limit-samanantar",
        type=int,
        default=50000,
        help="Max pairs per language from Samanantar. Use 0 for full local dataset.",
    )
    parser.add_argument("--skip-hindi", action="store_true", help="Skip Hindi preparation.")
    args = parser.parse_args()

    limit_value = None if args.limit_samanantar == 0 else args.limit_samanantar

    en_mr_pairs = prepare_en_mr(limit_samanantar=limit_value)
    en_kn_pairs = prepare_en_kn(limit_samanantar=limit_value)
    en_hi_pairs = None if args.skip_hindi else prepare_en_hi(limit_samanantar=limit_value)

    save_outputs(en_mr_pairs, en_kn_pairs, en_hi_pairs)


if __name__ == "__main__":
    main()
