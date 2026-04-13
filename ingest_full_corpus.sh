#!/bin/bash
set -e

# Full-scale ingestion pipeline (large download + full local preprocessing)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PY312="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
if [ -x "$PY312" ]; then
  PYTHON_BIN="$PY312"
else
  PYTHON_BIN="python3"
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# Force refresh to avoid stale sampled subsets.
python data/download_datasets.py --skip-milpac --full-samanantar --force-samanantar-refresh

# Use full local Samanantar disk datasets (0 => unlimited).
python data/prepare_legal_tsv.py --limit-samanantar 0
python data/build_quadruplets.py

echo "Full corpus ingestion complete."
