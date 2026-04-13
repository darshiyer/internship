# Marathi ↔ Kannada Legal MT (English Pivot)

Full-stack prototype and research workspace for legal-domain machine translation:

- Pivot route: Marathi -> English -> Kannada (and reverse)
- Backend API: FastAPI
- Frontend: single-page HTML/CSS/JS dashboard
- Data scripts: dataset download, quadruplet building, glossary utilities
- Evaluation: BLEU, chrF, BERTScore

## Project Structure

- backend/main.py: FastAPI app with `/translate`, `/evaluate`, `/model_info`, `/dataset_stats`
- backend/translate.py: pivot translation and model runtime handling
- backend/align.py: sentence alignment helpers using embedding models
- backend/evaluate.py: BLEU/chrF/BERTScore functions
- backend/glossary.py: OCR extraction and unified glossary builder
- data/download_datasets.py: data bootstrap script
- data/prepare_legal_tsv.py: normalize raw corpora into processed TSVs
- data/build_quadruplets.py: English-pivot quadruplet dictionary builder
- frontend/dashboard.html: research analytics dashboard
- notebooks/: exploration, alignment, and evaluation notebooks

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

2. Download datasets and prepare folders:

```bash
python data/download_datasets.py --samanantar-sample-size 5000
```

For faster iteration during prototyping, reduce the sample size (for example 500 or 1000).

For full-corpus Samanantar ingestion (large):

```bash
python data/download_datasets.py --full-samanantar --force-samanantar-refresh
```

3. Start backend:

```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

Optional but recommended before starting backend:

```bash
python data/prepare_legal_tsv.py --limit-samanantar 5000
python data/build_quadruplets.py
```

For full local preprocessing from downloaded Samanantar disk datasets:

```bash
python data/prepare_legal_tsv.py --limit-samanantar 0
python data/build_quadruplets.py
```

These commands generate:

- data/processed/en_mr_legal.tsv
- data/processed/en_kn_legal.tsv
- data/processed/quadruplets.json

4. Open the dashboard:

- File: `frontend/dashboard.html`
- Backend URL used by frontend: `http://localhost:8000`

## One-command Start

```bash
bash start.sh
```

## One-command Full Ingestion

```bash
bash ingest_full_corpus.sh
```

## Data Notes

- MILPaC auto-clones into `data/raw/MILPaC`
- Samanantar EN-MR and EN-KN are downloaded via HuggingFace datasets
- Futurebee EN-KN legal corpus and AIKosh KN-MR benchmark require manual download and placement in their prepared folders
- The preprocessor scans raw files for common column names and also supports fallback from local Samanantar disk datasets

## External References

- IndicTrans2: https://github.com/ai4bharat/IndicTrans2
- InLegalTrans: https://huggingface.co/law-ai/InLegalTrans-En2Indic-1B
- MILPaC dataset: https://github.com/Law-AI/MILPaC
- English-Kannada legal corpus: https://www.futurebeeai.com/dataset/parallel-corpora/kannada-english-translated-parallel-corpus-for-legal-domain
- Samanantar: https://huggingface.co/datasets/ai4bharat/samanantar
- Kannada-Marathi benchmark: https://aikosh.indiaai.gov.in/home/datasets/details/kannada_to_marathi_translation_benchmark_dataset.html
- InLegalBERT: https://huggingface.co/law-ai/InLegalBERT
- sacrebleu: https://github.com/mjpost/sacrebleu
- indic-nlp-library: https://github.com/anoopkunchukuttan/indic_nlp_library
