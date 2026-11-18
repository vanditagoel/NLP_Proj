# Detecting Stance in Media on Global Warming for Indic Languages (CSET 346)

An end-to-end, reproducible multilingual NLP/ML research and benchmarking system for predicting climate stance (**Favour**, **Against**, **None**) across **Hindi**, **Bengali**, and **English** toward the claim:

> *"Climate change and global warming is a serious concern."*

## Architecture

- **`ml/`** — Modular Python ML pipeline:
  - `ml.dataset.download`: Downloads `mlburnham/Pol_NLI` from Hugging Face Datasets.
  - `ml.dataset.inspect`: Audits splits, columns, duplicates, missing values, and NLI vs 3-class stance compatibility.
  - `ml.translation.translate`: Deterministic 50/50 Hindi/Bengali translation pipeline with SHA-256 JSON cache, retries, resume, script verification, and local Hugging Face MarianMT (`Helsinki-NLP/opus-mt-en-hi`, `shhossain/opus-mt-en-to-bn`) plus API backends.
  - `ml.dataset.prepare`: Paired-hypothesis reconstruction (`mlburnham/global_warming_stance_entailment`) + leak-free `group_stratified_split` ensuring English originals and translations remain in the exact same split. Also supports `--mode pol_nli_auxiliary` for native binary NLI experiments.
  - `ml.dataset.analyze`: Computes dataset statistics JSON and 4 Matplotlib PNG exploratory charts.
  - `ml.models.classical`: `tfidf_logreg`, `tfidf_svm`, `tfidf_nb`, `word2vec_lr`, `fasttext_lr`.
  - `ml.models.transformer`: `muril` (`google/muril-base-cased`), `mbert` (`bert-base-multilingual-cased`), `xlmr` (`xlm-roberta-base`), `indicbert` (`ai4bharat/indic-bert`).
  - `ml.evaluation.evaluate` & `ml.benchmark`: Macro F1-prioritized evaluation, per-class F1, Hindi/Bengali/English Macro F1, confusion matrices, and Matplotlib charts.
- **`backend/`** — FastAPI REST server (`/api/project`, `/api/dataset/stats`, `/api/dataset/samples`, `/api/models`, `/api/evaluation`, `/api/train`, `/api/predict`, `/api/predict/compare`, `/api/predict/batch`, `/charts/*`).
- **`frontend/`** — Simple 4-page React + TypeScript + Vite prototype (`Home`, `Model Testing` + CSV batch upload, `Models`, `Dataset`).
- **`server.ts`** — Node/Express launcher spawning FastAPI on port `8000`, proxying `/api` and `/charts`, and serving Vite on port `3000`.

## Quickstart CLI Commands

```bash
# 1. Inspect & prepare dataset
python -m ml.dataset.download --filter-climate
python -m ml.dataset.inspect
python -m ml.dataset.prepare --extract-source-only
python -m ml.translation.translate --target both --limit 300 --backend hf --batch-size 4
python -m ml.dataset.prepare
python -m ml.dataset.analyze

# 2. Train classical & embedding baselines
python -m ml.training.train_classical

# 3. Fine-tune Hugging Face Indic / multilingual transformers (GPU recommended for full 4-model sweep)
python -m ml.training.train_transformer --model indicbert --epochs 2
python -m ml.training.train_transformer --model muril --epochs 3
python -m ml.training.train_transformer --model mbert --epochs 3
python -m ml.training.train_transformer --model xlmr --epochs 3

# 4. Evaluate, benchmark, and test
python -m ml.evaluation.evaluate
python -m ml.benchmark
pytest -v
```
