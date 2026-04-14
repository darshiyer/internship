"""
human_eval_generate.py

Generates a CSV of 60 machine-translated sentences for manual rating.
The three of you (Darsh / Malay / Aryan) each rate 20 sentences.

How to run:
    python experiments/human_eval_generate.py

Output:
    experiments/results/human_eval_sheet.csv   ← share via Google Sheets
    experiments/results/human_eval_blank.csv   ← blank template for raters

Rating scale (fill in columns F and G):
    Fluency  1-5:  1=incomprehensible  3=understandable  5=native-like
    Adequacy 1-5:  1=wrong meaning     3=mostly correct  5=perfect meaning

Paste results back as experiments/results/human_eval_filled.csv
then run: python experiments/human_eval_generate.py --analyse
"""

import json, csv, random, argparse
from pathlib import Path

THIS_DIR = Path(__file__).parent
ROOT     = THIS_DIR.parent if THIS_DIR.name == "experiments" else THIS_DIR
RESULTS  = ROOT / "experiments" / "results"
SPLITS   = RESULTS / "splits"

# Example translations — replace with actual NLLB output if you have it
# Format: (marathi_source, machine_translation_to_kannada, reference_kannada)
# If you ran translation_eval.py, load from there instead.
SAMPLE_TRANSLATIONS = [
    # These are real-looking examples — replace with your actual model output
    ("सदर कायद्याच्या तरतुदी खालीलप्रमाणे आहेत.",
     "ಈ ಕಾನೂನಿನ ನಿಬಂಧನೆಗಳು ಈ ಕೆಳಗಿನಂತಿವೆ.",
     "ಈ ಕಾನೂನಿನ ಷರತ್ತುಗಳು ಕೆಳಕಂಡಂತಿವೆ."),
    ("न्यायालयाने या प्रकरणात निकाल दिला.",
     "ನ್ಯಾಯಾಲಯವು ಈ ಪ್ರಕರಣದಲ್ಲಿ ತೀರ್ಪು ನೀಡಿತು.",
     "ನ್ಯಾಯಾಲಯ ಈ ಪ್ರಕರಣದಲ್ಲಿ ತೀರ್ಪು ಪ್ರಕಟಿಸಿತು."),
    ("सरकारने नवीन धोरण जाहीर केले.",
     "ಸರ್ಕಾರ ಹೊಸ ನೀತಿ ಘೋಷಿಸಿತು.",
     "ಸರ್ಕಾರ ಹೊಸ ಯೋಜನೆಯನ್ನು ಅಧಿಕೃತವಾಗಿ ಘೋಷಿಸಿತು."),
    ("या कलमाखाली दंडाची तरतूद आहे.",
     "ಈ ವಿಭಾಗದ ಅಡಿಯಲ್ಲಿ ದಂಡದ ನಿಬಂಧನೆ ಇದೆ.",
     "ಈ ಅಧ್ಯಾಯದ ಪ್ರಕಾರ ದಂಡ ವಿಧಿಸಲು ಅವಕಾಶ ಇದೆ."),
    ("संविधानाच्या अनुच्छेद ३७० नुसार विशेष दर्जा देण्यात आला होता.",
     "ಸಂವಿಧಾನದ ಅನುಚ್ಛೇದ 370 ರ ಪ್ರಕಾರ ವಿಶೇಷ ಸ್ಥಾನಮಾನ ನೀಡಲಾಗಿತ್ತು.",
     "ಸಂವಿಧಾನದ 370ನೇ ವಿಧಿ ಅನ್ವಯ ವಿಶೇಷ ಸ್ಥಾನಮಾನ ನೀಡಲಾಗಿತ್ತು."),
    ("हे विधेयक संसदेत मांडण्यात आले.",
     "ಈ ಮಸೂದೆ ಸಂಸತ್ತಿನಲ್ಲಿ ಮಂಡಿಸಲಾಯಿತು.",
     "ಈ ಮಸೂದೆಯನ್ನು ಸಂಸತ್ತಿನಲ್ಲಿ ಮಂಡಿಸಲಾಯಿತು."),
    ("मालमत्तेच्या हस्तांतरणासाठी नोंदणी आवश्यक आहे.",
     "ಆಸ್ತಿ ವರ್ಗಾವಣೆಗೆ ನೋಂದಣಿ ಅಗತ್ಯ.",
     "ಆಸ್ತಿ ವಹಿವಾಟಿಗೆ ನೋಂದಣಿ ಕಡ್ಡಾಯ."),
    ("पीडिताने न्यायालयात अर्ज दाखल केला.",
     "ಸಂತ್ರಸ್ತರು ನ್ಯಾಯಾಲಯದಲ್ಲಿ ಅರ್ಜಿ ಸಲ್ಲಿಸಿದರು.",
     "ಸಂತ್ರಸ್ತ ವ್ಯಕ್ತಿ ನ್ಯಾಯಾಲಯಕ್ಕೆ ಮನವಿ ಸಲ್ಲಿಸಿದರು."),
    ("या योजनेचा लाभ गरीब कुटुंबांना मिळेल.",
     "ಈ ಯೋಜನೆಯ ಪ್ರಯೋಜನ ಬಡ ಕುಟುಂಬಗಳಿಗೆ ಸಿಗುತ್ತದೆ.",
     "ಈ ಯೋಜನೆಯ ಲಾಭ ಬಡ ಕುಟುಂಬಗಳಿಗೆ ದೊರಕುತ್ತದೆ."),
    ("आरोपीला जामीन मंजूर करण्यात आला.",
     "ಆರೋಪಿಗೆ ಜಾಮೀನು ಮಂಜೂರು ಮಾಡಲಾಯಿತು.",
     "ಆರೋಪಿಗೆ ಜಾಮೀನು ನೀಡಲಾಯಿತು."),
    ("या प्रस्तावाला मंत्रिमंडळाने मंजुरी दिली.",
     "ಈ ಪ್ರಸ್ತಾಪಕ್ಕೆ ಸಂಪುಟ ಅನುಮೋದನೆ ನೀಡಿತು.",
     "ಮಂತ್ರಿ ಮಂಡಲ ಈ ಪ್ರಸ್ತಾಪಕ್ಕೆ ಒಪ್ಪಿಗೆ ಸೂಚಿಸಿತು."),
    ("सार्वजनिक आरोग्यविषयक कायदे कठोर असावेत.",
     "ಸಾರ್ವಜನಿಕ ಆರೋಗ್ಯ ಕಾನೂನುಗಳು ಕಟ್ಟುನಿಟ್ಟಾಗಿರಬೇಕು.",
     "ಸಾರ್ವಜನಿಕ ಆರೋಗ್ಯ ಸಂಬಂಧಿ ಕಾಯ್ದೆಗಳು ಕಠೋರವಾಗಿರಬೇಕು."),
    ("राज्य शासनाने अध्यादेश जारी केला.",
     "ರಾಜ್ಯ ಸರ್ಕಾರ ಅಧ್ಯಾದೇಶ ಹೊರಡಿಸಿತು.",
     "ರಾಜ್ಯ ಸರ್ಕಾರ ಸುಗ್ರೀವಾಜ್ಞೆ ಹೊರಡಿಸಿತು."),
    ("या व्यवहाराची नोंद सरकारी नोंदवहीत झाली.",
     "ಈ ವ್ಯವಹಾರ ಸರ್ಕಾರಿ ದಾಖಲೆಗಳಲ್ಲಿ ನೋಂದಾಯಿಸಲಾಗಿದೆ.",
     "ಈ ಮೊತ್ತ ಸರ್ಕಾರಿ ದಾಖಲೆಗಳಲ್ಲಿ ದಾಖಲಾಗಿದೆ."),
    ("उच्च न्यायालयाने आदेश रद्द केला.",
     "ಉಚ್ಚ ನ್ಯಾಯಾಲಯ ಆದೇಶ ರದ್ದು ಮಾಡಿತು.",
     "ಉಚ್ಚ ನ್ಯಾಯಾಲಯ ಆ ಆದೇಶ ರದ್ದುಗೊಳಿಸಿತು."),
    ("या कायद्यात दुरुस्ती करण्यात आली.",
     "ಈ ಕಾನೂನಿನಲ್ಲಿ ತಿದ್ದುಪಡಿ ಮಾಡಲಾಗಿದೆ.",
     "ಈ ಶಾಸನದಲ್ಲಿ ಮಾರ್ಪಾಡು ಮಾಡಲಾಯಿತು."),
    ("निवडणूक आयोगाने तारीख जाहीर केली.",
     "ಚುನಾವಣಾ ಆಯೋಗ ದಿನಾಂಕ ಘೋಷಿಸಿತು.",
     "ಚುನಾವಣಾ ಆಯೋಗ ಮತದಾನದ ದಿನಾಂಕ ಪ್ರಕಟಿಸಿತು."),
    ("याचिकाकर्त्याची मागणी न्यायालयाने फेटाळली.",
     "ಅರ್ಜಿದಾರರ ಮನವಿ ನ್ಯಾಯಾಲಯ ತಿರಸ್ಕರಿಸಿತು.",
     "ಅರ್ಜಿದಾರರ ಬೇಡಿಕೆಯನ್ನು ನ್ಯಾಯಾಲಯ ತಳ್ಳಿಹಾಕಿತು."),
    ("सरकारी कर्मचाऱ्यांना विशेष भत्ता मिळेल.",
     "ಸರ್ಕಾರಿ ನೌಕರರಿಗೆ ವಿಶೇಷ ಭತ್ಯೆ ಸಿಗಲಿದೆ.",
     "ಸರ್ಕಾರಿ ಉದ್ಯೋಗಿಗಳಿಗೆ ವಿಶೇಷ ಭತ್ತೆ ನೀಡಲಾಗುವುದು."),
    ("भूसंपादन प्रक्रिया कायद्यानुसार राबविण्यात येईल.",
     "ಭೂ ಸ್ವಾಧೀನ ಪ್ರಕ್ರಿಯೆ ಕಾನೂನಿನ ಪ್ರಕಾರ ನಡೆಯಲಿದೆ.",
     "ಭೂ ಸ್ವಾಧೀನ ಕಾರ್ಯವಿಧಾನ ಕಾಯ್ದೆ ಅನ್ವಯ ಜಾರಿಗೊಳಿಸಲಾಗುವುದು."),
]


def generate_sheets():
    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rater_names = ["Darsh Iyer", "Malay Thoria", "Aryan Khatu"]
    sentences_per_rater = len(SAMPLE_TRANSLATIONS) // 3

    for i, (src_mr, mt_kn, ref_kn) in enumerate(SAMPLE_TRANSLATIONS):
        rater_idx = i // sentences_per_rater
        rater = rater_names[min(rater_idx, 2)]
        rows.append({
            "ID": i + 1,
            "Assigned_Rater": rater,
            "Source_Marathi": src_mr,
            "MT_Output_Kannada": mt_kn,
            "Reference_Kannada": ref_kn,
            "Fluency_1to5": "",    # rater fills this
            "Adequacy_1to5": "",   # rater fills this
            "Comments": "",        # optional
        })

    # Full sheet (with references visible — for raters who know Kannada)
    with open(out_dir / "human_eval_sheet.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Blind sheet (without reference — for fluency-only evaluation)
    blind_rows = [{k: v for k, v in r.items() if k != "Reference_Kannada"} for r in rows]
    with open(out_dir / "human_eval_blind.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=blind_rows[0].keys())
        w.writeheader()
        w.writerows(blind_rows)

    print(f"Generated:")
    print(f"  {out_dir}/human_eval_sheet.csv   ← full sheet (with reference)")
    print(f"  {out_dir}/human_eval_blind.csv   ← blind sheet (fluency only)")
    print()
    print("Instructions for raters:")
    print("  1. Open human_eval_sheet.csv in Google Sheets or Excel")
    print("  2. Fill in columns F (Fluency 1–5) and G (Adequacy 1–5)")
    print("  3. Each rater handles their ~20 assigned rows")
    print()
    print("Rating guide:")
    print("  Fluency  5 = native-like Kannada   3 = understandable   1 = unreadable")
    print("  Adequacy 5 = perfect meaning        3 = mostly correct   1 = wrong meaning")


def analyse():
    filled_path = RESULTS / "human_eval_filled.csv"
    if not filled_path.exists():
        print(f"ERROR: {filled_path} not found.")
        print("Fill in the ratings first, then save as human_eval_filled.csv")
        return

    fluency, adequacy = [], []
    with open(filled_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fluency.append(float(row["Fluency_1to5"]))
                adequacy.append(float(row["Adequacy_1to5"]))
            except (ValueError, KeyError):
                pass

    if not fluency:
        print("No filled ratings found. Check column names in the CSV.")
        return

    n = len(fluency)
    avg_f = sum(fluency) / n
    avg_a = sum(adequacy) / n

    print(f"\nHuman Evaluation Results ({n} sentences)")
    print(f"  Mean Fluency  : {avg_f:.2f} / 5.0")
    print(f"  Mean Adequacy : {avg_a:.2f} / 5.0")
    print()
    print("Include these numbers in Section 7 of the paper to address")
    print("the 'absence of human evaluation' limitation.")

    # Save summary
    summary = {"n_sentences": n, "mean_fluency": round(avg_f, 3),
               "mean_adequacy": round(avg_a, 3)}
    with open(RESULTS / "human_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {RESULTS}/human_eval_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse", action="store_true",
                        help="Analyse filled ratings instead of generating sheets")
    args = parser.parse_args()
    if args.analyse:
        analyse()
    else:
        generate_sheets()
