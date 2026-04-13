#!/bin/bash
set -e

# Create and activate virtual environment for consistent interpreter behavior.
if [ ! -d ".venv" ]; then
	python3 -m venv .venv
fi
source .venv/bin/activate

# Install deps
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# Download datasets
python data/download_datasets.py --samanantar-sample-size 5000

# Build processed bilingual TSV files and aligned quadruplets.
python data/prepare_legal_tsv.py --limit-samanantar 5000
python data/build_quadruplets.py

# Start FastAPI backend
cd backend && python -m uvicorn main:app --reload --port 8000 &

# Open frontend guidance
echo "Backend running at http://localhost:8000"
echo "Open frontend/dashboard.html in your browser"
echo "API docs at http://localhost:8000/docs"
