"""
Build indexed sentence quadruplets: { index: { english, marathi, kannada, hindi } }.

Strategy
--------
English is used as the pivot key.  All three bilingual TSVs (EN-MR, EN-KN, EN-HI) are
loaded and indexed by a *normalised* English string so that minor whitespace / punctuation
differences do not block matching.

Matching tiers (applied in order, first hit wins):
  1. Exact lowercase match.
  2. Whitespace-collapsed + punctuation-stripped match.
  3. First-100-char fingerprint match (catches minor trailing-text differences).

Full quadruplets (all 4 languages present) are always preferred; triplets (EN/MR/KN,
Hindi missing) are kept as a fallback so the dataset is not unnecessarily small.

Structure follows the indexed-dict format agreed in Meeting 2:
  {
    "0": { "english": "…", "marathi": "…", "kannada": "…", "hindi": "…" },
    …
  }
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_exact(text: str) -> str:
    """Lowercase + strip."""
    return text.strip().lower()


def _norm_loose(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _norm_fingerprint(text: str, n: int = 100) -> str:
    """First n chars of loose-normalised text — tolerates trailing differences."""
    return _norm_loose(text)[:n]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_index(pairs: list[dict], value_key: str) -> tuple[dict, dict, dict]:
    """Return three dicts keyed by (exact, loose, fingerprint) norms."""
    exact: dict[str, str] = {}
    loose: dict[str, str] = {}
    finger: dict[str, str] = {}
    for p in pairs:
        raw_en = _nfc(p.get("english", ""))
        val = p.get(value_key, "")
        if not raw_en or not val:
            continue
        k1 = _norm_exact(raw_en)
        k2 = _norm_loose(raw_en)
        k3 = _norm_fingerprint(raw_en)
        if k1 not in exact:
            exact[k1] = val
        if k2 not in loose:
            loose[k2] = val
        if k3 not in finger:
            finger[k3] = val
    return exact, loose, finger


def _lookup(raw_en: str, exact: dict, loose: dict, finger: dict) -> str | None:
    v = exact.get(_norm_exact(raw_en))
    if v:
        return v
    v = loose.get(_norm_loose(raw_en))
    if v:
        return v
    v = finger.get(_norm_fingerprint(raw_en))
    return v


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_quadruplet_dict(
    en_mr_pairs: list[dict],
    en_kn_pairs: list[dict],
    en_hi_pairs: list[dict] | None = None,
    require_hindi: bool = False,
) -> dict:
    """
    Parameters
    ----------
    en_mr_pairs : list of {"english": …, "marathi": …}
    en_kn_pairs : list of {"english": …, "kannada": …}
    en_hi_pairs : list of {"english": …, "hindi": …}  (optional)
    require_hindi : if True, skip quadruplets where Hindi is missing.

    Returns
    -------
    Indexed dict:  { "0": {"english", "marathi", "kannada", "hindi"}, … }
    """
    kn_exact, kn_loose, kn_finger = _build_index(en_kn_pairs, "kannada")
    hi_exact, hi_loose, hi_finger = (
        _build_index(en_hi_pairs, "hindi") if en_hi_pairs else ({}, {}, {})
    )

    quadruplets: dict[str, dict] = {}
    stats = {"full": 0, "no_hindi": 0, "no_kannada": 0, "skipped": 0}
    idx = 0

    for item in en_mr_pairs:
        raw_en = _nfc(item.get("english", "").strip())
        marathi = item.get("marathi", "").strip()
        if not raw_en or not marathi:
            continue

        kannada = _lookup(raw_en, kn_exact, kn_loose, kn_finger) or ""
        hindi = _lookup(raw_en, hi_exact, hi_loose, hi_finger) or item.get("hindi", "")

        if not kannada:
            stats["no_kannada"] += 1
            continue  # Kannada is required for the pivot route to work.

        if require_hindi and not hindi:
            stats["no_hindi"] += 1
            stats["skipped"] += 1
            continue

        if hindi:
            stats["full"] += 1
        else:
            stats["no_hindi"] += 1

        quadruplets[str(idx)] = {
            "english": raw_en,
            "marathi": marathi,
            "kannada": kannada,
            "hindi": hindi,
        }
        idx += 1

    print(
        f"  Quadruplets built: {idx} total  "
        f"| full (4-lang): {stats['full']}  "
        f"| triplet (no Hindi): {stats['no_hindi']}  "
        f"| skipped (no Kannada): {stats['no_kannada']}"
    )
    return quadruplets


def save_quadruplets(quadruplets: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(quadruplets, fh, ensure_ascii=False, indent=2)
    print(f"  Saved {len(quadruplets)} quadruplets → {output_path}")


# ---------------------------------------------------------------------------
# TSV helpers
# ---------------------------------------------------------------------------

def _load_tsv(path: Path, lang_col: str) -> list[dict]:
    if not path.exists():
        print(f"  [WARN] TSV not found: {path}")
        return []
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    cols = list(df.columns)
    if "english" not in cols or lang_col not in cols:
        print(f"  [WARN] Expected columns 'english' and '{lang_col}' in {path.name}, got {cols}")
        return []
    records = df[["english", lang_col]].to_dict("records")
    print(f"  Loaded {len(records)} pairs from {path.name}")
    return records


def build_from_tsv(
    en_mr_tsv: str | Path,
    en_kn_tsv: str | Path,
    output_json: str | Path,
    en_hi_tsv: str | Path | None = None,
) -> dict:
    en_mr_pairs = _load_tsv(Path(en_mr_tsv), "marathi")
    en_kn_pairs = _load_tsv(Path(en_kn_tsv), "kannada")
    en_hi_pairs = _load_tsv(Path(en_hi_tsv), "hindi") if en_hi_tsv else None

    quadruplets = build_quadruplet_dict(en_mr_pairs, en_kn_pairs, en_hi_pairs)
    save_quadruplets(quadruplets, output_json)
    return quadruplets


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    processed = base / "processed"

    en_mr_default = processed / "en_mr_legal.tsv"
    en_kn_default = processed / "en_kn_legal.tsv"
    en_hi_default = processed / "en_hi_legal.tsv"
    output_default = processed / "quadruplets.json"

    if not en_mr_default.exists() or not en_kn_default.exists():
        print("Required input TSVs not found. Run prepare_legal_tsv.py first.")
        print(f"  Expected: {en_mr_default}")
        print(f"  Expected: {en_kn_default}")
    else:
        hi_path = en_hi_default if en_hi_default.exists() else None
        if hi_path:
            print(f"Hindi TSV found → {hi_path}")
        else:
            print("Hindi TSV not found — building triplets (EN/MR/KN) only.")
        build_from_tsv(en_mr_default, en_kn_default, output_default, hi_path)
