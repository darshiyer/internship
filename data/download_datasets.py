"""
Download legal and fallback datasets for Marathi <-> Kannada MT via English pivot.

External references:
# MILPaC dataset: https://github.com/Law-AI/MILPaC
# English-Kannada legal corpus: https://www.futurebeeai.com/dataset/parallel-corpora/kannada-english-translated-parallel-corpus-for-legal-domain
# Samanantar: https://huggingface.co/datasets/ai4bharat/samanantar
# Kannada-Marathi benchmark: https://aikosh.indiaai.gov.in/home/datasets/details/kannada_to_marathi_translation_benchmark_dataset.html
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def clone_milpac() -> None:
    target = RAW / "MILPaC"
    if target.exists():
        print("MILPaC already exists, skipping clone.")
        return
    subprocess.run(["git", "clone", "https://github.com/Law-AI/MILPaC.git", str(target)], check=True)


def prepare_en_kn_legal_folder() -> None:
    target = RAW / "en_kn_legal"
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Download English-Kannada legal corpus from Futurebee and place files here.\n"
            "Source: https://www.futurebeeai.com/dataset/parallel-corpora/"
            "kannada-english-translated-parallel-corpus-for-legal-domain\n",
            encoding="utf-8",
        )


def download_samanantar(sample_size: int | None = None, force_refresh: bool = False) -> None:
    en_mr_path = RAW / "samanantar_en_mr"
    en_kn_path = RAW / "samanantar_en_kn"
    en_hi_path = RAW / "samanantar_en_hi"
    split = "train" if sample_size is None else f"train[:{sample_size}]"

    if force_refresh:
        for p in [en_mr_path, en_kn_path, en_hi_path]:
            if p.exists():
                print(f"Removing {p} for refresh…")
                subprocess.run(["rm", "-rf", str(p)], check=True)

    if not en_mr_path.exists():
        print(f"Downloading Samanantar EN-MR ({split})…")
        en_mr = load_dataset("ai4bharat/samanantar", "mr", split=split)
        en_mr.save_to_disk(str(en_mr_path))
        print(f"Saved Samanantar EN-MR → {en_mr_path}")
    else:
        print("Samanantar EN-MR already exists, skipping.")

    if not en_kn_path.exists():
        print(f"Downloading Samanantar EN-KN ({split})…")
        en_kn = load_dataset("ai4bharat/samanantar", "kn", split=split)
        en_kn.save_to_disk(str(en_kn_path))
        print(f"Saved Samanantar EN-KN → {en_kn_path}")
    else:
        print("Samanantar EN-KN already exists, skipping.")

    if not en_hi_path.exists():
        print(f"Downloading Samanantar EN-HI ({split})…")
        en_hi = load_dataset("ai4bharat/samanantar", "hi", split=split)
        en_hi.save_to_disk(str(en_hi_path))
        print(f"Saved Samanantar EN-HI → {en_hi_path}")
    else:
        print("Samanantar EN-HI already exists, skipping.")


def prepare_kn_mr_benchmark_folder() -> None:
    target = RAW / "kn_mr_benchmark"
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Download Kannada->Marathi benchmark from AIKosh and place files here.\n"
            "Source: https://aikosh.indiaai.gov.in/home/datasets/details/"
            "kannada_to_marathi_translation_benchmark_dataset.html\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download legal MT datasets.")
    parser.add_argument(
        "--samanantar-sample-size",
        type=int,
        default=5000,
        help="Number of Samanantar training rows per language to download (default: 5000).",
    )
    parser.add_argument(
        "--full-samanantar",
        action="store_true",
        help="Download full Samanantar train split (large).",
    )
    parser.add_argument(
        "--force-samanantar-refresh",
        action="store_true",
        help="Delete existing local Samanantar folders and re-download with current settings.",
    )
    parser.add_argument("--skip-milpac", action="store_true", help="Skip MILPaC git clone step.")
    parser.add_argument("--skip-samanantar", action="store_true", help="Skip Samanantar download step.")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if not args.skip_milpac:
        clone_milpac()
    prepare_en_kn_legal_folder()
    if not args.skip_samanantar:
        selected_size = None if args.full_samanantar else args.samanantar_sample_size
        download_samanantar(sample_size=selected_size, force_refresh=args.force_samanantar_refresh)
    prepare_kn_mr_benchmark_folder()
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
