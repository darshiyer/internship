"""
scrape_legislative.py
─────────────────────
Scrape legislative.gov.in for parallel EN↔HI, EN↔MR (Maharashtra),
and EN↔KN (Karnataka) sentence pairs extracted from official Act PDFs.

Sources:
  - Central Acts  → EN + HI  → https://legislative.gov.in/actsofparliamentfromtheyear/
  - Maharashtra   → EN + MR  → https://legislative.gov.in/state-acts-and-ordinances/maharashtra
  - Karnataka     → EN + KN  → https://legislative.gov.in/state-acts-and-ordinances/karnataka

Outputs (TSV files with columns auto-detected by prepare_legal_tsv.py):
  data/raw/legislative_gov_in/en_hi_acts.tsv   →  english | hindi
  data/raw/legislative_gov_in/en_mr_acts.tsv   →  english | marathi
  data/raw/legislative_gov_in/en_kn_acts.tsv   →  english | kannada

Usage:
  python scrape_legislative.py                        # all sources, max 50 acts each
  python scrape_legislative.py --max-acts 200         # more acts per source
  python scrape_legislative.py --source maharashtra   # only Maharashtra
  python scrape_legislative.py --output-dir /path/to/output

On Google Colab:
  !python scrape_legislative.py --output-dir /content/drive/MyDrive/legal_data
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Iterator

# ── dependency checks ─────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        import bs4
    except ImportError:
        missing.append("beautifulsoup4")
    try:
        import pdfplumber
    except ImportError:
        missing.append("pdfplumber")
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Install: pip install {' '.join(missing)}")
        sys.exit(1)

_check_deps()

import requests
from bs4 import BeautifulSoup
import pdfplumber

# ── constants ─────────────────────────────────────────────────────────────────

CENTRAL_URL    = "https://legislative.gov.in/actsofparliamentfromtheyear/"
MAHARASHTRA_URL = "https://legislative.gov.in/state-acts-and-ordinances/maharashtra"
KARNATAKA_URL   = "https://legislative.gov.in/state-acts-and-ordinances/karnataka"
BASE_URL        = "https://legislative.gov.in"

RATE_DELAY  = 2.5   # seconds between HTTP requests
PDF_TIMEOUT = 60    # seconds for PDF downloads
HTTP_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Language detection patterns (link text / anchor title)
LANG_PATTERNS = {
    "english": ["english", "eng", " en "],
    "hindi":   ["hindi", "हिन्दी", "हिंदी", " hi "],
    "marathi": ["marathi", "मराठी", " mr "],
    "kannada": ["kannada", "ಕನ್ನಡ", " kn "],
}

# ── HTTP session ──────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _get(url: str, stream: bool = False) -> requests.Response | None:
    """GET with retry (3 attempts) and polite delay."""
    for attempt in range(3):
        try:
            time.sleep(RATE_DELAY)
            r = SESSION.get(url, timeout=HTTP_TIMEOUT, stream=stream, allow_redirects=True)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  [WARN] Attempt {attempt+1}/3 failed for {url}: {e}")
            time.sleep(RATE_DELAY * 2)
    print(f"  [SKIP] Could not fetch: {url}")
    return None


def _soup(url: str) -> BeautifulSoup | None:
    r = _get(url)
    if r is None:
        return None
    return BeautifulSoup(r.text, "html.parser")


# ── act listing scraper ───────────────────────────────────────────────────────

def _find_lang_in_text(text: str) -> str | None:
    text_lower = text.lower().strip()
    for lang, patterns in LANG_PATTERNS.items():
        if any(p in text_lower for p in patterns):
            return lang
    return None


def _extract_pdf_links(soup: BeautifulSoup) -> dict[str, str]:
    """
    Find PDF download links per language on an act detail page.
    Returns {lang: absolute_url}.
    """
    links: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().endswith(".pdf"):
            continue
        # Build absolute URL
        if href.startswith("http"):
            abs_url = href
        else:
            abs_url = urljoin(BASE_URL, href)

        # Detect language from link text, title attr, or surrounding text
        candidates = [
            a.get_text(" ", strip=True),
            a.get("title", ""),
            a.get("aria-label", ""),
        ]
        parent = a.parent
        if parent:
            candidates.append(parent.get_text(" ", strip=True)[:80])

        combined = " ".join(candidates)
        lang = _find_lang_in_text(combined)
        if lang and lang not in links:
            links[lang] = abs_url

    return links


def scrape_central_act_links(max_acts: int) -> list[dict]:
    """
    Returns list of {title, url, target_lang, target_lang_url, en_url}
    for central acts (EN + HI).
    """
    print(f"\n[Central Acts] Scraping listing from {CENTRAL_URL}")
    soup = _soup(CENTRAL_URL)
    if soup is None:
        return []

    results = []
    # The page lists acts in tables or lists — find all links to individual act pages
    act_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        # Act detail pages typically contain "/acts-of-parliament/" or similar
        if (
            text
            and len(text) > 5
            and (
                "/acts-of-parliament/" in href
                or "/actsofparliament/" in href
                or re.search(r"/act[s]?/\d", href)
            )
        ):
            act_links.append({
                "title": text,
                "url": urljoin(BASE_URL, href),
            })

    # Remove duplicates
    seen_urls = set()
    unique_links = []
    for act in act_links:
        if act["url"] not in seen_urls:
            seen_urls.add(act["url"])
            unique_links.append(act)

    print(f"  Found {len(unique_links)} act detail pages")
    unique_links = unique_links[:max_acts]

    for i, act in enumerate(unique_links):
        print(f"  [{i+1}/{len(unique_links)}] {act['title'][:60]}")
        detail_soup = _soup(act["url"])
        if detail_soup is None:
            continue
        pdf_links = _extract_pdf_links(detail_soup)
        if "english" in pdf_links and "hindi" in pdf_links:
            results.append({
                "title": act["title"],
                "detail_url": act["url"],
                "en_url": pdf_links["english"],
                "target_lang": "hindi",
                "target_url": pdf_links["hindi"],
            })
            print(f"    ✅ Found EN+HI PDFs")
        else:
            print(f"    ⚠ Missing PDFs: found {list(pdf_links.keys())}")

    return results


def scrape_state_act_links(state_url: str, target_lang: str, max_acts: int) -> list[dict]:
    """
    Returns list of {title, en_url, target_lang, target_url} for a state.
    target_lang: "marathi" or "kannada"
    """
    print(f"\n[State Acts] Scraping {state_url}")
    soup = _soup(state_url)
    if soup is None:
        return []

    results = []
    act_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if text and len(text) > 5 and ".pdf" not in href.lower():
            # State acts may be listed as direct links or detail pages
            if any(kw in href for kw in ["/state-act", "/acts/", "/ordinance"]):
                act_links.append({"title": text, "url": urljoin(BASE_URL, href)})

    # Also check if PDFs are directly listed on the page
    direct_pdfs: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().endswith(".pdf"):
            continue
        text = a.get_text(" ", strip=True) + " " + a.get("title", "")
        lang = _find_lang_in_text(text)
        # Group by act name using surrounding context
        parent = a.find_parent(["tr", "li", "div"])
        act_key = parent.get_text(" ", strip=True)[:60] if parent else href
        if act_key not in direct_pdfs:
            direct_pdfs[act_key] = {}
        if lang:
            direct_pdfs[act_key][lang] = urljoin(BASE_URL, href)

    # Process direct PDFs first
    for act_key, lang_map in direct_pdfs.items():
        if "english" in lang_map and target_lang in lang_map:
            results.append({
                "title": act_key[:80],
                "detail_url": state_url,
                "en_url": lang_map["english"],
                "target_lang": target_lang,
                "target_url": lang_map[target_lang],
            })

    # Deduplicate act detail links
    seen_urls = set()
    unique_links = []
    for act in act_links:
        if act["url"] not in seen_urls:
            seen_urls.add(act["url"])
            unique_links.append(act)

    remaining = max_acts - len(results)
    if remaining > 0:
        unique_links = unique_links[:remaining]
        for i, act in enumerate(unique_links):
            print(f"  [{i+1}/{len(unique_links)}] {act['title'][:60]}")
            detail_soup = _soup(act["url"])
            if detail_soup is None:
                continue
            pdf_links = _extract_pdf_links(detail_soup)
            if "english" in pdf_links and target_lang in pdf_links:
                results.append({
                    "title": act["title"],
                    "detail_url": act["url"],
                    "en_url": pdf_links["english"],
                    "target_lang": target_lang,
                    "target_url": pdf_links[target_lang],
                })
                print(f"    ✅ Found EN+{target_lang.upper()[:2]} PDFs")
            else:
                print(f"    ⚠ Missing PDFs: {list(pdf_links.keys())}")

    print(f"  Total acts with EN+{target_lang} PDFs: {len(results)}")
    return results[:max_acts]


# ── PDF download & extraction ─────────────────────────────────────────────────

def _download_pdf(url: str, dest: Path) -> bool:
    """Download PDF to dest. Returns True on success."""
    try:
        time.sleep(RATE_DELAY)
        r = SESSION.get(url, timeout=PDF_TIMEOUT, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            print(f"    [WARN] Not a PDF (content-type: {content_type}): {url}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    [WARN] PDF download failed: {e}")
        return False


def _is_scanned(pages_text: list[str]) -> bool:
    """Return True if PDF is a scanned image (no extractable text)."""
    total_chars = sum(len(t) for t in pages_text)
    return total_chars < max(100 * len(pages_text), 200)


def _extract_pdf_text(pdf_path: Path) -> list[str]:
    """
    Extract text page-by-page. Returns list of page strings.
    Returns [] if scanned / unreadable.
    """
    try:
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        if _is_scanned(pages):
            return []
        return pages
    except Exception as e:
        print(f"    [WARN] PDF extraction failed: {e}")
        return []


# ── text cleaning ─────────────────────────────────────────────────────────────

# Patterns for headers/footers/page numbers to strip
_PAGE_NUM_RE  = re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE)
_HEADER_RE    = re.compile(
    r"^\s*(?:THE GAZETTE OF INDIA|MINISTRY OF|GOVERNMENT OF|KARNATAKA GAZETTE"
    r"|MAHARASHTRA GOVERNMENT GAZETTE|भारत का राजपत्र|ಕರ್ನಾಟಕ ರಾಜ್ಯಪತ್ರ"
    r"|महाराष्ट्र शासन राजपत्र)[^\n]*\n",
    re.MULTILINE | re.IGNORECASE,
)
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _clean_text(raw: str) -> str:
    """Strip page numbers, headers, excess blank lines."""
    text = _PAGE_NUM_RE.sub("", raw)
    text = _HEADER_RE.sub("", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _join_pages(pages: list[str]) -> str:
    return _clean_text("\n\n".join(pages))


# ── sentence splitting ────────────────────────────────────────────────────────

# Abbreviations common in Indian legal text — protect from sentence splitting
_LEGAL_ABBREVS = re.compile(
    r"\b(?:v|vs|Mr|Mrs|Dr|Hon|Smt|Shri|Govt|No|Art|Sec|Cl|Vol|Fig|viz|"
    r"i\.e|e\.g|etc|s\.t|w\.r\.t|w\.e\.f|u/s|u/a|r/w|ibid|op\.cit|"
    r"para|sub|sub-sec|sub-cl|proviso|expl|sch|appx|annex)\."
)


def _split_english(text: str) -> list[str]:
    """Split English legal text into sentences."""
    # Protect abbreviations
    protected = _LEGAL_ABBREVS.sub(lambda m: m.group().replace(".", "<DOT>"), text)
    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r'(?<=[.?!])\s+(?=[A-Z"("])', protected)
    sentences = []
    for p in parts:
        s = p.replace("<DOT>", ".").strip()
        # Merge very short fragments with previous sentence
        if sentences and len(s) < 15 and not s.endswith((".", "?", "!")):
            sentences[-1] = sentences[-1] + " " + s
        elif s:
            sentences.append(s)
    return sentences


def _split_indic(text: str, lang: str) -> list[str]:
    """
    Split Indic text into sentences.
    Uses indicnlp if available, falls back to Devanagari/Kannada danda (।) splitting.
    """
    try:
        from indicnlp.tokenize import sentence_tokenize
        return [s.strip() for s in sentence_tokenize.sentence_split(text, lang=lang) if s.strip()]
    except ImportError:
        pass
    # Fallback: split on danda (।), full stop, or newline
    parts = re.split(r"[।\n]+|(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 5]


# ── language / script detection ───────────────────────────────────────────────

def _dominant_script(text: str) -> str:
    """
    Returns 'devanagari', 'kannada', 'latin', or 'mixed'
    based on the most common Unicode block.
    """
    counts = {"devanagari": 0, "kannada": 0, "latin": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["kannada"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["latin"] += 1

    total = sum(counts.values())
    if total == 0:
        return "mixed"
    dominant = max(counts, key=counts.get)
    if counts[dominant] / total >= 0.40:
        return dominant
    return "mixed"


# ── section-level alignment ───────────────────────────────────────────────────

# Matches: "Section 3", "SECTION 3A", "SEC. 3.2.1", "3.", "3A."
_SECTION_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:(?:SECTION|Section|SEC\.?|Sec\.?|ಕಲಂ|ಅನುಚ್ಛೇದ|धारा|अनुच्छेद)\s+)?"
    r"(\d+[A-Z]?(?:[.\-]\d+[A-Z]?)*)"
    r"[.\-\s]",
    re.MULTILINE,
)


def _extract_sections(text: str) -> dict[str, str]:
    """
    Parse text into {section_id: section_text} by detecting section headers.
    Falls back to paragraph splitting if no section markers found.
    """
    matches = list(_SECTION_RE.finditer(text))
    if len(matches) < 3:
        # No section structure — return paragraphs keyed by index
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 30]
        return {str(i): p for i, p in enumerate(paras)}

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        sec_id = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[start:end].strip()
        if sec_text:
            sections[sec_id] = sec_text
    return sections


def _align_and_split(
    en_text: str,
    tgt_text: str,
    tgt_lang: str,
) -> list[tuple[str, str]]:
    """
    Align EN and target language texts at section level,
    then sentence-level within each aligned section.
    Returns list of (english_sentence, target_sentence) pairs.
    """
    en_sections  = _extract_sections(en_text)
    tgt_sections = _extract_sections(tgt_text)

    pairs: list[tuple[str, str]] = []
    matched_sections = set(en_sections.keys()) & set(tgt_sections.keys())

    if not matched_sections:
        # Fall back to positional alignment if no key overlap
        en_vals  = list(en_sections.values())
        tgt_vals = list(tgt_sections.values())
        matched_count = min(len(en_vals), len(tgt_vals))
        for i in range(matched_count):
            pairs.extend(_sentence_align(en_vals[i], tgt_vals[i], tgt_lang))
    else:
        for sec_id in sorted(matched_sections, key=lambda x: (len(x), x)):
            pairs.extend(
                _sentence_align(en_sections[sec_id], tgt_sections[sec_id], tgt_lang)
            )

    return pairs


def _sentence_align(
    en_para: str,
    tgt_para: str,
    tgt_lang: str,
) -> list[tuple[str, str]]:
    """
    Split both paragraphs into sentences and align by index.
    Apply quality filters before returning.
    """
    indicnlp_lang = {"hindi": "hi", "marathi": "mr", "kannada": "kn"}.get(tgt_lang, "hi")
    en_sents  = _split_english(en_para)
    tgt_sents = _split_indic(tgt_para, indicnlp_lang)

    pairs: list[tuple[str, str]] = []
    n = min(len(en_sents), len(tgt_sents))

    for i in range(n):
        en  = en_sents[i].strip()
        tgt = tgt_sents[i].strip()

        # Quality filters
        if len(en) < 20 or len(tgt) < 10:
            continue
        if len(en) > 600 or len(tgt) > 800:
            continue
        # English sentence should be mostly Latin script
        if _dominant_script(en) not in ("latin", "mixed"):
            continue
        # Target sentence should be in the right script
        expected_script = "devanagari" if tgt_lang in ("hindi", "marathi") else "kannada"
        if _dominant_script(tgt) not in (expected_script, "mixed"):
            continue
        # Length ratio check (Indic scripts are denser)
        ratio = len(en) / max(len(tgt), 1)
        if not (0.3 <= ratio <= 4.0):
            continue

        pairs.append((en, tgt))

    return pairs


# ── deduplication ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dedupe(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for en, tgt in pairs:
        key = (_normalize(en), _normalize(tgt))
        if key not in seen:
            seen.add(key)
            result.append((en, tgt))
    return result


# ── TSV writer ─────────────────────────────────────────────────────────────────

def _write_tsv(pairs: list[tuple[str, str]], path: Path, tgt_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["english", tgt_col])
        for en, tgt in pairs:
            w.writerow([en, tgt])
    print(f"  Saved {len(pairs)} pairs → {path}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def _process_act(act: dict, tmpdir: Path) -> list[tuple[str, str]]:
    """Download, extract, align one act. Returns sentence pairs."""
    title = act["title"][:60]
    en_url  = act["en_url"]
    tgt_url = act["target_url"]
    tgt_lang = act["target_lang"]

    en_pdf  = tmpdir / "en.pdf"
    tgt_pdf = tmpdir / "tgt.pdf"

    print(f"    Downloading EN PDF...")
    if not _download_pdf(en_url, en_pdf):
        return []
    print(f"    Downloading {tgt_lang.upper()[:2]} PDF...")
    if not _download_pdf(tgt_url, tgt_pdf):
        return []

    en_pages  = _extract_pdf_text(en_pdf)
    tgt_pages = _extract_pdf_text(tgt_pdf)

    if not en_pages:
        print(f"    [SKIP] Scanned/unreadable EN PDF: {title}")
        return []
    if not tgt_pages:
        print(f"    [SKIP] Scanned/unreadable {tgt_lang} PDF: {title}")
        return []

    en_text  = _join_pages(en_pages)
    tgt_text = _join_pages(tgt_pages)

    pairs = _align_and_split(en_text, tgt_text, tgt_lang)
    print(f"    ✅ Extracted {len(pairs)} sentence pairs")
    return pairs


def run(
    output_dir: Path,
    max_acts: int = 50,
    sources: list[str] | None = None,
):
    """
    Main entry point.
    sources: list of "central", "maharashtra", "karnataka" (default: all)
    """
    if sources is None:
        sources = ["central", "maharashtra", "karnataka"]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: dict[str, list[tuple[str, str]]] = {
        "hindi":   [],
        "marathi": [],
        "kannada": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Central Acts (EN + HI) ─────────────────────────────────────────
        if "central" in sources:
            acts = scrape_central_act_links(max_acts)
            print(f"\n[Central Acts] Processing {len(acts)} acts...")
            for i, act in enumerate(acts):
                print(f"\n  [{i+1}/{len(acts)}] {act['title'][:60]}")
                pairs = _process_act(act, tmp)
                all_pairs["hindi"].extend(pairs)

        # ── Maharashtra Acts (EN + MR) ─────────────────────────────────────
        if "maharashtra" in sources:
            acts = scrape_state_act_links(MAHARASHTRA_URL, "marathi", max_acts)
            print(f"\n[Maharashtra Acts] Processing {len(acts)} acts...")
            for i, act in enumerate(acts):
                print(f"\n  [{i+1}/{len(acts)}] {act['title'][:60]}")
                pairs = _process_act(act, tmp)
                all_pairs["marathi"].extend(pairs)

        # ── Karnataka Acts (EN + KN) ───────────────────────────────────────
        if "karnataka" in sources:
            acts = scrape_state_act_links(KARNATAKA_URL, "kannada", max_acts)
            print(f"\n[Karnataka Acts] Processing {len(acts)} acts...")
            for i, act in enumerate(acts):
                print(f"\n  [{i+1}/{len(acts)}] {act['title'][:60]}")
                pairs = _process_act(act, tmp)
                all_pairs["kannada"].extend(pairs)

    # ── Deduplicate + save ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    lang_to_col = {"hindi": "hindi", "marathi": "marathi", "kannada": "kannada"}
    lang_to_file = {
        "hindi":   output_dir / "en_hi_acts.tsv",
        "marathi": output_dir / "en_mr_acts.tsv",
        "kannada": output_dir / "en_kn_acts.tsv",
    }

    total = 0
    for lang, pairs in all_pairs.items():
        if not pairs:
            continue
        pairs = _dedupe(pairs)
        _write_tsv(pairs, lang_to_file[lang], lang_to_col[lang])
        total += len(pairs)
        print(f"  EN-{lang.upper()[:2]}: {len(pairs)} unique sentence pairs")

    print(f"\n  Total: {total} pairs across all languages")
    print(f"  Output directory: {output_dir}")
    print("\n✅ Done! Copy the TSV files to data/raw/legislative_gov_in/ and run:")
    print("   python data/prepare_legal_tsv.py")
    print("   python data/build_quadruplets.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape legislative.gov.in for parallel legal text.")
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "raw" / "legislative_gov_in",
        help="Directory to save TSV files (default: data/raw/legislative_gov_in/)",
    )
    parser.add_argument(
        "--max-acts", "-n",
        type=int,
        default=50,
        help="Max acts to scrape per source (default: 50)",
    )
    parser.add_argument(
        "--source", "-s",
        choices=["central", "maharashtra", "karnataka", "all"],
        default="all",
        help="Which source to scrape (default: all)",
    )
    args = parser.parse_args()

    sources = ["central", "maharashtra", "karnataka"] if args.source == "all" else [args.source]

    print("=" * 60)
    print("Legislative.gov.in Parallel Corpus Scraper")
    print("=" * 60)
    print(f"  Sources   : {', '.join(sources)}")
    print(f"  Max acts  : {args.max_acts} per source")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)

    run(output_dir=args.output_dir, max_acts=args.max_acts, sources=sources)


if __name__ == "__main__":
    main()
