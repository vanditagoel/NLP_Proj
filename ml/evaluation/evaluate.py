"""Evaluate all trained classical and transformer models on the held-out test split.

Usage:
    python -m ml.evaluation.evaluate
"""
from __future__ import annotations

import json
import time
from typing import Any

import joblib
import pandas as pd

from ml.evaluation.metrics import (
    compute_classification_metrics,
    save_confusion_matrix_chart,
    save_metrics_summary,
)
from ml.models.classical import CLASSICAL_MODEL_SPECS, predict_classical_batch
from ml.models.transformer import TRANSFORMER_MODEL_SPECS, predict_with_saved_transformer
from ml.utils import CHARTS_DIR, MODELS_DIR, SPLITS_DIR, TRAINED_CLASSICAL_DIR, ensure_directories


def evaluate_all_trained_models() -> dict[str, Any]:
    ensure_directories()
    test_path = SPLITS_DIR / "climate_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found at {test_path}. Run `python -m ml.dataset.prepare` first."
        )

    test_df = pd.read_csv(test_path, keep_default_na=False)
    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["label"].astype(str).tolist()
    langs = test_df["language"].astype(str).tolist() if "language" in test_df.columns else None
    labels = ["Favour", "Against", "None"]

    results: list[dict[str, Any]] = []

    for model_key, spec in CLASSICAL_MODEL_SPECS.items():
        art_path = TRAINED_CLASSICAL_DIR / f"{model_key}.joblib"
        if not art_path.exists():
            continue
        pipe = joblib.load(art_path)
        t0 = time.perf_counter()
        preds, _, _ = predict_classical_batch(pipe, texts)
        inf_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(texts))

        m = compute_classification_metrics(
            y_true=y_true,
            y_pred=preds,
            labels=labels,
            languages=langs,
            model_key=model_key,
            model_name=spec["name"],
            avg_inference_ms=inf_ms,
        )
        save_confusion_matrix_chart(
            m["confusion_matrix"],
            labels=labels,
            model_name=spec["name"],
            out_path=CHARTS_DIR / f"cm_{model_key}.png",
        )
        results.append(m)

    for model_key, spec in TRANSFORMER_MODEL_SPECS.items():
        model_dir = MODELS_DIR / model_key
        if not (model_dir / "config.json").exists():
            continue
        t0 = time.perf_counter()
        preds, _ = predict_with_saved_transformer(model_dir, texts)
        inf_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(texts))

        m = compute_classification_metrics(
            y_true=y_true,
            y_pred=preds,
            labels=labels,
            languages=langs,
            model_key=model_key,
            model_name=spec["name"],
            avg_inference_ms=inf_ms,
        )
        save_confusion_matrix_chart(
            m["confusion_matrix"],
            labels=labels,
            model_name=spec["name"],
            out_path=CHARTS_DIR / f"cm_{model_key}.png",
        )
        results.append(m)

    summary = save_metrics_summary(results)
    print(f"Evaluated {len(results)} trained model(s). Best Macro F1 model: {summary.get('best_model_name')}")
    return summary


def main() -> None:
    summary = evaluate_all_trained_models()
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
