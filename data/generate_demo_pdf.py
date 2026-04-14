"""
generate_demo_pdf.py
────────────────────
Generates a realistic-looking Maharashtra government Act PDF in Marathi
(with English headings) using sentences from the actual dataset.
Uses Playwright → Chromium for perfect Devanagari rendering.

Output: data/demo_pdfs/maharashtra_act_demo.pdf

Usage:
  python data/generate_demo_pdf.py
  python data/generate_demo_pdf.py --kannada    # Kannada version
"""

from __future__ import annotations
import argparse, asyncio, csv, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── pick sentences from dataset ───────────────────────────────────────────────

def load_sentences(lang: str, n: int = 35, seed: int = 42) -> list[tuple[str, str]]:
    """Load (english, indic) pairs from processed TSV."""
    col   = "marathi" if lang == "mr" else "kannada"
    tsv   = ROOT / "data" / "processed" / f"en_{lang}_legal.tsv"
    legal_kw = {
        "mr": ["कायदा","कलम","धारा","न्यायालय","सरकार","अधिनियम","नियम",
               "अधिकार","तरतूद","समिती","राज्य","केंद्र","संसद","मंत्रालय"],
        "kn": ["ಕಾನೂನು","ಸರ್ಕಾರ","ನಿಯಮ","ಕಲಂ","ಅಧಿಕಾರ","ರಾಜ್ಯ","ಕೇಂದ್ರ",
               "ಸಮಿತಿ","ಸಂಸತ್","ಮಂತ್ರಾಲಯ","ನ್ಯಾಯಾಲಯ","ಅಧಿನಿಯಮ"],
    }

    sentences = []
    try:
        with open(tsv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                tgt = row.get(col, "").strip()
                en  = row.get("english", "").strip()
                if 60 < len(tgt) < 280 and 60 < len(en) < 280:
                    if any(w in tgt for w in legal_kw[lang]):
                        sentences.append((en, tgt))
                if len(sentences) >= 500:
                    break
    except FileNotFoundError:
        print(f"[WARN] {tsv} not found — using placeholder sentences")
        return _placeholder_sentences(lang, n)

    if not sentences:
        return _placeholder_sentences(lang, n)

    random.seed(seed)
    return random.sample(sentences, min(n, len(sentences)))


def _placeholder_sentences(lang: str, n: int) -> list[tuple[str, str]]:
    if lang == "mr":
        return [
            ("No person shall be liable to pay any tax in excess of the amount prescribed.",
             "कोणत्याही व्यक्तीला विहित रकमेपेक्षा अधिक कर भरण्यास जबाबदार धरले जाणार नाही."),
        ] * n
    return [
        ("No person shall be liable to pay any tax in excess of the amount prescribed.",
         "ಯಾವುದೇ ವ್ಯಕ್ತಿಯು ನಿಗದಿಪಡಿಸಿದ ಮೊತ್ತಕ್ಕಿಂತ ಹೆಚ್ಚಿನ ತೆರಿಗೆ ಪಾವತಿಸಲು ಹೊಣೆಗಾರರಾಗಿರುವುದಿಲ್ಲ."),
    ] * n


# ── HTML template ─────────────────────────────────────────────────────────────

def build_html(sentences: list[tuple[str, str]], lang: str) -> str:
    is_mr = lang == "mr"

    state_en   = "Maharashtra"         if is_mr else "Karnataka"
    state_indic= "महाराष्ट्र"         if is_mr else "ಕರ್ನಾಟಕ"
    font_stack = "'ITFDevanagari', 'Devanagari Sangam MN', 'Kohinoor Devanagari', serif" \
                 if is_mr else \
                 "'Noto Sans Kannada', 'Kannada MN', 'Kannada Sangam MN', serif"
    act_no     = "XV"                  if is_mr else "XXII"
    year       = "2023"
    act_title_en   = f"THE {state_en.upper()} LEGAL SERVICES AND REGULATORY AUTHORITY ACT, {year}"
    act_title_indic= (
        f"महाराष्ट्र कायदेशीर सेवा आणि नियामक प्राधिकरण अधिनियम, {year}"
        if is_mr else
        f"ಕರ್ನಾಟಕ ಕಾನೂನು ಸೇವೆಗಳು ಮತ್ತು ನಿಯಂತ್ರಣ ಪ್ರಾಧಿಕಾರ ಅಧಿನಿಯಮ, {year}"
    )
    ministry_en    = "Law and Justice Department, Government of " + state_en
    ministry_indic = (
        "कायदे व न्याय विभाग, महाराष्ट्र शासन"
        if is_mr else
        "ಕಾನೂನು ಮತ್ತು ನ್ಯಾಯ ಇಲಾಖೆ, ಕರ್ನಾಟಕ ಸರ್ಕಾರ"
    )
    preamble = (
        "प्रस्तावना – हे महाराष्ट्र विधान मंडळाने संमत केलेले असून, "
        "या अधिनियमाद्वारे राज्यातील नागरिकांना न्यायिक सेवांचा लाभ देण्यासाठी "
        "तरतुदी केल्या जात आहेत. संविधानाच्या कलम २१ अंतर्गत प्रदत्त अधिकारांचा "
        "उपयोग करण्यासाठी हे अधिनियम अमलात आणण्यात येत आहे."
        if is_mr else
        "ಪ್ರಸ್ತಾವನೆ – ಈ ಅಧಿನಿಯಮವನ್ನು ಕರ್ನಾಟಕ ವಿಧಾನಮಂಡಲದಿಂದ ಅಂಗೀಕರಿಸಲಾಗಿದ್ದು, "
        "ರಾಜ್ಯದ ನಾಗರಿಕರಿಗೆ ನ್ಯಾಯಾಂಗ ಸೇವೆಗಳ ಲಾಭ ನೀಡಲು ನಿಬಂಧನೆಗಳನ್ನು ರಚಿಸಲಾಗಿದೆ."
    )

    # Build section rows
    section_html = ""
    section_titles_mr = [
        "प्राधिकरणाची स्थापना","सदस्यांचे अधिकार","नियामक तरतुदी",
        "दंड व शिक्षा","न्यायिक प्रक्रिया","अपील प्राधिकरण",
        "सरकारी अधिसूचना","अंतिम तरतुदी",
    ]
    section_titles_kn = [
        "ಪ್ರಾಧಿಕಾರದ ಸ್ಥಾಪನೆ","ಸದಸ್ಯರ ಅಧಿಕಾರಗಳು","ನಿಯಂತ್ರಣ ನಿಬಂಧನೆಗಳು",
        "ದಂಡ ಮತ್ತು ಶಿಕ್ಷೆ","ನ್ಯಾಯಿಕ ಪ್ರಕ್ರಿಯೆ","ಮೇಲ್ಮನವಿ ಪ್ರಾಧಿಕಾರ",
        "ಸರ್ಕಾರಿ ಅಧಿಸೂಚನೆ","ಅಂತಿಮ ನಿಬಂಧನೆಗಳು",
    ]
    section_titles = section_titles_mr if is_mr else section_titles_kn
    groups = [sentences[i:i+4] for i in range(0, len(sentences), 4)]

    for sec_idx, group in enumerate(groups[:8]):
        sec_num    = sec_idx + 1
        sec_title  = section_titles[sec_idx % len(section_titles)]
        clauses_html = ""
        for cl_idx, (en, indic) in enumerate(group):
            cl_label = f"({chr(97 + cl_idx)})"
            clauses_html += f"""
            <div class="clause">
              <div class="clause-indic">{cl_label}&nbsp; {indic}</div>
              <div class="clause-en">[{en}]</div>
            </div>"""

        section_html += f"""
        <div class="section">
          <div class="section-header">
            <span class="sec-num">कलम {sec_num}.</span>
            <span class="sec-title">{sec_title}</span>
          </div>
          {clauses_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="{'mr' if is_mr else 'kn'}">
<head>
<meta charset="UTF-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Devanagari:wght@400;700&family=Noto+Serif+Kannada:wght@400;700&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: {font_stack};
    font-size: 11.5pt;
    line-height: 1.75;
    color: #1a1a1a;
    background: white;
    padding: 0;
  }}

  .page {{
    width: 210mm;
    min-height: 297mm;
    padding: 22mm 25mm 22mm 30mm;
    margin: 0 auto;
  }}

  /* ── Official Header ── */
  .gov-header {{
    border-top: 4px solid #1a237e;
    border-bottom: 2px solid #1a237e;
    padding: 10px 0 8px;
    text-align: center;
    margin-bottom: 18px;
  }}
  .gov-header .ashoka {{
    font-size: 28pt;
    line-height: 1;
    margin-bottom: 4px;
  }}
  .gov-header .gov-name {{
    font-size: 13pt;
    font-weight: bold;
    letter-spacing: 0.5px;
    color: #1a237e;
  }}
  .gov-header .gov-name-indic {{
    font-size: 14pt;
    font-weight: bold;
    color: #1a237e;
  }}
  .gov-header .ministry {{
    font-size: 9.5pt;
    color: #444;
    margin-top: 3px;
  }}

  /* ── Gazette Band ── */
  .gazette-band {{
    background: #1a237e;
    color: white;
    text-align: center;
    padding: 5px 0;
    font-size: 9.5pt;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
  }}

  /* ── Act Meta ── */
  .act-meta {{
    text-align: center;
    margin-bottom: 20px;
  }}
  .act-no {{
    font-size: 10pt;
    color: #555;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .act-title-en {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 13.5pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
    color: #0d0d0d;
  }}
  .act-title-indic {{
    font-size: 15pt;
    font-weight: bold;
    color: #0d0d0d;
    margin-bottom: 10px;
  }}
  .act-date {{
    font-size: 9.5pt;
    color: #555;
    font-style: italic;
  }}

  /* ── Divider ── */
  .divider {{
    border: none;
    border-top: 1.5px solid #888;
    margin: 14px 0;
  }}
  .divider-thin {{
    border: none;
    border-top: 0.5px solid #bbb;
    margin: 10px 0;
  }}

  /* ── Preamble ── */
  .preamble {{
    font-size: 10.5pt;
    color: #333;
    margin-bottom: 20px;
    text-align: justify;
    padding: 10px 14px;
    background: #f9f9f9;
    border-left: 3px solid #1a237e;
    border-radius: 2px;
  }}
  .preamble-label {{
    font-weight: bold;
    display: block;
    margin-bottom: 4px;
    color: #1a237e;
    font-size: 9.5pt;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}

  /* ── Sections ── */
  .section {{
    margin-bottom: 18px;
    page-break-inside: avoid;
  }}
  .section-header {{
    background: #e8eaf6;
    padding: 5px 10px;
    margin-bottom: 8px;
    border-left: 4px solid #1a237e;
    border-radius: 2px;
  }}
  .sec-num {{
    font-family: 'Times New Roman', Times, serif;
    font-weight: bold;
    font-size: 11pt;
    color: #1a237e;
    margin-right: 8px;
  }}
  .sec-title {{
    font-weight: bold;
    font-size: 12pt;
    color: #1a237e;
  }}

  /* ── Clauses ── */
  .clause {{
    margin: 0 0 10px 18px;
  }}
  .clause-indic {{
    font-size: 11.5pt;
    color: #0d0d0d;
    text-align: justify;
    margin-bottom: 3px;
  }}
  .clause-en {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.5pt;
    color: #666;
    font-style: italic;
    margin-left: 18px;
    text-align: justify;
  }}

  /* ── Footer ── */
  .footer {{
    margin-top: 30px;
    border-top: 1.5px solid #888;
    padding-top: 8px;
    display: flex;
    justify-content: space-between;
    font-size: 8.5pt;
    color: #666;
    font-family: 'Times New Roman', Times, serif;
  }}
  .footer .seal {{
    font-size: 9pt;
    color: #1a237e;
    font-weight: bold;
  }}

  @media print {{
    body {{ padding: 0; }}
    .page {{ padding: 15mm 20mm 15mm 25mm; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- Government Header -->
  <div class="gov-header">
    <div class="ashoka">🔵</div>
    <div class="gov-name">GOVERNMENT OF {state_en.upper()}</div>
    <div class="gov-name-indic">{state_indic} शासन</div>
    <div class="ministry">{ministry_en} &nbsp;|&nbsp; {ministry_indic}</div>
  </div>

  <!-- Gazette Band -->
  <div class="gazette-band">
    {state_en} Government Gazette &nbsp;•&nbsp; Extraordinary &nbsp;•&nbsp; Part IV-B
  </div>

  <!-- Act Metadata -->
  <div class="act-meta">
    <div class="act-no">{state_en} Act No. {act_no} of {year}</div>
    <div class="act-title-en">{act_title_en}</div>
    <div class="act-title-indic">{act_title_indic}</div>
    <div class="act-date">
      Received the assent of the Governor on the 15th day of March, {year}.<br/>
      Published in the {state_en} Government Gazette, Extraordinary, dated 20th March {year}.
    </div>
  </div>

  <hr class="divider"/>

  <!-- Preamble -->
  <div class="preamble">
    <span class="preamble-label">Preamble / प्रस्तावना</span>
    {preamble}
  </div>

  <hr class="divider-thin"/>

  <!-- Sections -->
  {section_html}

  <!-- Footer -->
  <div class="footer">
    <div class="seal">✦ {state_en} Government Official Document</div>
    <div>Act No. {act_no}/{year} &nbsp;|&nbsp; Page 1 of 4</div>
    <div>Printed at Government Press, {state_en}</div>
  </div>

</div>
</body>
</html>"""
    return html


# ── generate PDF via Playwright ───────────────────────────────────────────────

async def generate_pdf(html: str, output_path: Path):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()

        await page.set_content(html, wait_until="networkidle")
        await asyncio.sleep(1)   # let fonts settle

        await page.pdf(
            path=str(output_path),
            format="A4",
            print_background=True,
            margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
        )
        await browser.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a realistic legal Act PDF")
    parser.add_argument("--kannada", action="store_true", help="Generate Kannada version")
    parser.add_argument("--both",    action="store_true", help="Generate both MR and KN")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "demo_pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = []
    if args.both:
        langs = ["mr", "kn"]
    elif args.kannada:
        langs = ["kn"]
    else:
        langs = ["mr"]

    for lang in langs:
        lang_name = "Marathi" if lang == "mr" else "Kannada"
        fname     = f"maharashtra_act_demo.pdf" if lang == "mr" else f"karnataka_act_demo.pdf"
        out_path  = out_dir / fname

        print(f"\n{'='*55}")
        print(f"Generating {lang_name} legal Act PDF...")
        print(f"{'='*55}")

        sentences = load_sentences(lang, n=35)
        print(f"  Loaded {len(sentences)} {lang_name} sentences from dataset")

        html = build_html(sentences, lang)
        print(f"  HTML built ({len(html):,} chars)")

        print(f"  Rendering PDF via Chromium...")
        asyncio.run(generate_pdf(html, out_path))

        size_kb = out_path.stat().st_size // 1024
        print(f"\n  ✅ PDF saved: {out_path}")
        print(f"     Size: {size_kb} KB")
        print(f"\n  Open with: open '{out_path}'")


if __name__ == "__main__":
    main()
