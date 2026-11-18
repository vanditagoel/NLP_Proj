"""Lightweight transformer metadata specs (avoids importing torch/transformers on API startup)."""
from __future__ import annotations

TRANSFORMER_MODEL_SPECS: dict[str, dict[str, str]] = {
    "muril": {
        "name": "MuRIL (google/muril-base-cased)",
        "hf_id": "google/muril-base-cased",
        "family": "Indic Transformer",
        "description": "Google Multilingual Representations for Indian Languages pre-trained on 17 Indic languages.",
    },
    "mbert": {
        "name": "mBERT (bert-base-multilingual-cased)",
        "hf_id": "bert-base-multilingual-cased",
        "family": "Multilingual Transformer",
        "description": "Multilingual BERT cased model covering 104 languages including Hindi, Bengali, and English.",
    },
    "xlmr": {
        "name": "XLM-RoBERTa (xlm-roberta-base)",
        "hf_id": "xlm-roberta-base",
        "family": "Multilingual Transformer",
        "description": "Cross-lingual RoBERTa trained on CommonCrawl across 100 languages including Hindi and Bengali.",
    },
    "indicbert": {
        "name": "IndicBERT (ai4bharat/indic-bert)",
        "hf_id": "ai4bharat/indic-bert",
        "family": "Indic Transformer",
        "description": "AI4Bharat ALBERT-based model trained specifically on 12 major Indian languages.",
    },
}
