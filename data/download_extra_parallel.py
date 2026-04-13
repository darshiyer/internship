"""
download_extra_parallel.py
──────────────────────────
Downloads additional parallel corpora for EN↔MR, EN↔KN, EN↔HI:

  1. PMIndia v1      — PM of India speeches (government/policy domain)
                       EN-MR ~50k, EN-KN ~50k, EN-HI ~80k sentences
  2. OPUS-100        — Curated multilingual parallel sentences
                       EN-MR 27k, EN-KN 14k, EN-HI 100k (subset)

Outputs (auto-discovered by prepare_legal_tsv.py):
  data/raw/pmindia/en_mr_pmindia.tsv   → english | marathi
  data/raw/pmindia/en_kn_pmindia.tsv   → english | kannada
  data/raw/pmindia/en_hi_pmindia.tsv   → english | hindi
  data/raw/opus100/en_mr_opus.tsv      → english | marathi
  data/raw/opus100/en_kn_opus.tsv      → english | kannada
  data/raw/opus100/en_hi_opus.tsv      → english | hindi

Usage:
  python data/download_extra_parallel.py
  python data/download_extra_parallel.py --skip-pmindia
  python data/download_extra_parallel.py --skip-opus
"""

from __future__ import annotations
import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

# ── dependency check ───────────────────────────────────────────────────────────
def _check_deps():
    missing = []
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    if missing:
        print(f"[ERROR] Missing: pip install {' '.join(missing)}")
        sys.exit(1)

_check_deps()
import requests
from datasets import load_dataset

# ── config ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"

PMINDIA_URLS = {
    "marathi": "http://data.statmt.org/pmindia/v1/parallel/pmindia.v1.mr-en.tsv",
    "kannada": "http://data.statmt.org/pmindia/v1/parallel/pmindia.v1.kn-en.tsv",
    "hindi":   "http://data.statmt.org/pmindia/v1/parallel/pmindia.v1.hi-en.tsv",
}

OPUS100_CONFIGS = {
    "marathi": "en-mr",
    "kannada": "en-kn",
    "hindi":   "en-hi",
}

OPUS_HI_LIMIT = 100_000   # cap EN-HI at 100k (534k available, we already have Samanantar)

# ── helpers ────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())

def _quality_ok(en: str, tgt: str) -> bool:
    en  = _normalize(en)
    tgt = _normalize(tgt)
    if len(en) < 10 or len(tgt) < 5:
        return False
    if len(en) > 600 or len(tgt) > 800:
        return False
    # must have actual content (not just numbers/symbols)
    if not re.search(r"[A-Za-z]{3}", en):
        return False
    ratio = len(en) / max(len(tgt), 1)
    if not (0.25 <= ratio <= 6.0):
        return False
    return True

def _write_tsv(pairs: list[tuple[str, str]], path: Path, tgt_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str,str]] = set()
    deduped = []
    for en, tgt in pairs:
        key = (en.lower().strip(), tgt.strip())
        if key not in seen:
            seen.add(key)
            deduped.append((en.strip(), tgt.strip()))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["english", tgt_col])
        for en, tgt in deduped:
            w.writerow([en, tgt])

    print(f"  ✅ Saved {len(deduped):,} pairs → {path.relative_to(ROOT)}")
    return len(deduped)

# ── PMIndia downloader ─────────────────────────────────────────────────────────

def download_pmindia():
    """
    PMIndia TSV files have two tab-separated formats:
    Format A: EN\tTGT  (one sentence pair per line)
    Format B: alternating lines (EN on odd, TGT on even)
    We detect which and parse accordingly.
    """
    print("\n" + "="*60)
    print("PMIndia v1 — PM of India Speeches")
    print("="*60)

    tgt_col_map = {"marathi": "marathi", "kannada": "kannada", "hindi": "hindi"}
    out_map = {
        "marathi": RAW / "pmindia" / "en_mr_pmindia.tsv",
        "kannada": RAW / "pmindia" / "en_kn_pmindia.tsv",
        "hindi":   RAW / "pmindia" / "en_hi_pmindia.tsv",
    }

    total = 0
    for lang, url in PMINDIA_URLS.items():
        print(f"\n  Downloading {lang} ({url.split('/')[-1]})...")
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        lines = r.content.decode("utf-8", errors="replace").splitlines()
        print(f"  Downloaded {len(lines):,} lines")

        pairs: list[tuple[str, str]] = []

        # Detect format by checking first line
        if lines and "\t" in lines[0]:
            # Format A: EN\tTGT
            for line in lines:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    en, tgt = parts[0].strip(), parts[1].strip()
                    # Figure out which side is English
                    en_score  = len(re.findall(r"[A-Za-z]", en))
                    tgt_score = len(re.findall(r"[A-Za-z]", tgt))
                    if tgt_score > en_score:
                        en, tgt = tgt, en   # swap if needed
                    if _quality_ok(en, tgt):
                        pairs.append((_normalize(en), _normalize(tgt)))
        else:
            # Format B: alternating lines
            for i in range(0, len(lines) - 1, 2):
                en  = lines[i].strip()
                tgt = lines[i+1].strip()
                en_score  = len(re.findall(r"[A-Za-z]", en))
                tgt_score = len(re.findall(r"[A-Za-z]", tgt))
                if tgt_score > en_score:
                    en, tgt = tgt, en
                if _quality_ok(en, tgt):
                    pairs.append((_normalize(en), _normalize(tgt)))

        print(f"  Kept {len(pairs):,} quality pairs")
        n = _write_tsv(pairs, out_map[lang], tgt_col_map[lang])
        total += n

    print(f"\n  PMIndia total: {total:,} pairs")
    return total

# ── OPUS-100 downloader ────────────────────────────────────────────────────────

def download_opus100():
    print("\n" + "="*60)
    print("OPUS-100 — Curated Multilingual Parallel Sentences")
    print("="*60)

    out_map = {
        "marathi": RAW / "opus100" / "en_mr_opus.tsv",
        "kannada": RAW / "opus100" / "en_kn_opus.tsv",
        "hindi":   RAW / "opus100" / "en_hi_opus.tsv",
    }
    tgt_col_map = {"marathi": "marathi", "kannada": "kannada", "hindi": "hindi"}
    tgt_key_map = {"marathi": "mr", "kannada": "kn", "hindi": "hi"}

    total = 0
    for lang, config in OPUS100_CONFIGS.items():
        print(f"\n  Loading OPUS-100 {config}...")
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", config, split="train")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        tgt_key = tgt_key_map[lang]
        limit = OPUS_HI_LIMIT if lang == "hindi" else len(ds)

        pairs: list[tuple[str, str]] = []
        for row in ds.select(range(min(limit, len(ds)))):
            en  = row["translation"].get("en", "").strip()
            tgt = row["translation"].get(tgt_key, "").strip()
            if _quality_ok(en, tgt):
                pairs.append((_normalize(en), _normalize(tgt)))

        print(f"  Kept {len(pairs):,} / {min(limit, len(ds)):,} quality pairs")
        n = _write_tsv(pairs, out_map[lang], tgt_col_map[lang])
        total += n

    print(f"\n  OPUS-100 total: {total:,} pairs")
    return total

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download extra parallel corpora")
    parser.add_argument("--skip-pmindia", action="store_true")
    parser.add_argument("--skip-opus",    action="store_true")
    args = parser.parse_args()

    grand_total = 0

    if not args.skip_pmindia:
        grand_total += download_pmindia()

    if not args.skip_opus:
        grand_total += download_opus100()

    print("\n" + "="*60)
    print(f"GRAND TOTAL: {grand_total:,} new parallel pairs downloaded")
    print("="*60)
    print("\nNext steps:")
    print("  python data/prepare_legal_tsv.py --limit-samanantar 500000")
    print("  python data/build_quadruplets.py")

if __name__ == "__main__":
    main()
