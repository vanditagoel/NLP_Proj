#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/app/applet/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

echo "1) Downloading mlburnham/Pol_NLI..."
"$PYTHON_BIN" -m ml.dataset.download --filter-climate

echo "2) Inspecting Pol_NLI splits and task compatibility..."
"$PYTHON_BIN" -m ml.dataset.inspect

echo "3) Extracting English 3-class climate stance source..."
"$PYTHON_BIN" -m ml.dataset.prepare --extract-source-only

echo "4) Translating half Hindi / half Bengali locally via HF MarianMT..."
"$PYTHON_BIN" -m ml.translation.translate --target both --limit 300 --backend hf --batch-size 4

echo "5) Preparing group-stratified splits (zero leakage)..."
"$PYTHON_BIN" -m ml.dataset.prepare
"$PYTHON_BIN" -m ml.dataset.prepare --mode pol_nli_auxiliary --max-rows 4000

echo "6) Generating dataset statistics and exploratory charts..."
"$PYTHON_BIN" -m ml.dataset.analyze

echo "7) Training classical TF-IDF & embedding baselines..."
"$PYTHON_BIN" -m ml.training.train_classical

echo "8) Running evaluation and benchmark report..."
"$PYTHON_BIN" -m ml.evaluation.evaluate
"$PYTHON_BIN" -m ml.benchmark
