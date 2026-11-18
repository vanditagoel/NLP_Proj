"""Shared path, config, seeding, and directory helpers."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "default_config.json"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRANSLATED_DIR = DATA_DIR / "translated"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = ROOT_DIR / "models"
TRAINED_CLASSICAL_DIR = MODELS_DIR / "trained"

RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CHARTS_DIR = RESULTS_DIR / "charts"
LOGS_DIR = RESULTS_DIR / "logs"

LABELS = ["Favour", "Against", "None"]


def ensure_directories() -> None:
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        TRANSLATED_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        TRAINED_CLASSICAL_DIR,
        METRICS_DIR,
        CHARTS_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
