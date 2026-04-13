"""
Build a 4-language legal glossary: English | Marathi | Kannada | Hindi.

External references:
# MILPaC dataset: https://github.com/Law-AI/MILPaC
# English-Kannada legal corpus: https://www.futurebeeai.com/dataset/parallel-corpora/kannada-english-translated-parallel-corpus-for-legal-domain
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from paddleocr import PaddleOCR


def extract_glossary_from_pdf(pdf_path: str) -> pd.DataFrame:
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(pdf_path, cls=True)
    rows = []
    for page in result:
        for line in page:
            rows.append(line[1][0])
    return pd.DataFrame({"raw_text": rows})


def build_unified_glossary(en_mr_path: str, en_kn_path: str) -> pd.DataFrame:
    en_mr = pd.read_csv(en_mr_path, sep="\t", names=["English", "Marathi", "Hindi"])
    en_kn = pd.read_csv(en_kn_path, sep="\t", names=["English", "Kannada"])
    merged = pd.merge(en_mr, en_kn, on="English", how="outer")
    merged["Missing_In"] = merged.apply(
        lambda row: ", ".join(
            [
                lang
                for lang, col in [("Marathi", "Marathi"), ("Kannada", "Kannada"), ("Hindi", "Hindi")]
                if pd.isna(row[col])
            ]
        )
        or None,
        axis=1,
    )
    return merged[["English", "Marathi", "Kannada", "Hindi", "Missing_In"]]


def save_glossary(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)


def load_glossary(glossary_tsv_path: str) -> pd.DataFrame:
    return pd.read_csv(glossary_tsv_path, sep="\t")


def inject_glossary_terms(text: str, glossary_df: pd.DataFrame, src_col: str, tgt_col: str) -> str:
    replaced = text
    pairs = glossary_df[[src_col, tgt_col]].dropna().values.tolist()
    for src_term, tgt_term in pairs:
        replaced = replaced.replace(str(src_term), str(tgt_term))
    return replaced
