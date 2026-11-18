"""Train all classical TF-IDF and dense Word2Vec/FastText baseline models.

Usage:
    python -m ml.training.train_classical
    python -m ml.training.train_classical --mode pol_nli_auxiliary
"""
from __future__ import annotations

import argparse
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
from ml.models.classical import (
    CLASSICAL_MODEL_SPECS,
    build_classical_pipeline,
    predict_classical_batch,
)
from ml.utils import (
    CHARTS_DIR,
    METRICS_DIR,
    SPLITS_DIR,
    TRAINED_CLASSICAL_DIR,
    ensure_directories,
    load_config,
    set_global_seed,
)


def train_classical_models(
    mode: str = "climate_stance",
    selected_keys: list[str] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    ensure_directories()
    set_global_seed(seed)

    prefix = "climate" if mode == "climate_stance" else "pol_nli"
    train_path = SPLITS_DIR / f"{prefix}_train.csv"
    val_path = SPLITS_DIR / f"{prefix}_validation.csv"
    test_path = SPLITS_DIR / f"{prefix}_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Prepared splits not found for mode={mode}. Run `python -m ml.dataset.prepare` first."
        )

    train_df = pd.read_csv(train_path, keep_default_na=False)
    val_df = pd.read_csv(val_path, keep_default_na=False) if val_path.exists() else pd.DataFrame()
    test_df = pd.read_csv(test_path, keep_default_na=False)

    labels = (
        ["Favour", "Against", "None"]
        if mode == "climate_stance"
        else sorted(train_df["label"].unique().tolist())
    )

    train_texts = train_df["text"].astype(str).tolist()
    train_y = train_df["label"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()
    test_y = test_df["label"].astype(str).tolist()
    test_langs = test_df["language"].astype(str).tolist() if "language" in test_df.columns else None

    keys = selected_keys or list(CLASSICAL_MODEL_SPECS.keys())
    all_metrics: list[dict[str, Any]] = []

    for model_key in keys:
        spec = CLASSICAL_MODEL_SPECS[model_key]
        t_train0 = time.perf_counter()
        pipe = build_classical_pipeline(model_key, seed=seed)
        pipe.fit(train_texts, train_y)
        train_sec = round(time.perf_counter() - t_train0, 3)

        out_dir = TRAINED_CLASSICAL_DIR if mode == "climate_stance" else (TRAINED_CLASSICAL_DIR / "pol_nli_auxiliary")
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / f"{model_key}.joblib")

        val_macro_f1 = None
        if len(val_df) > 0:
            v_preds, _, _ = predict_classical_batch(pipe, val_df["text"].astype(str).tolist())
            v_metrics = compute_classification_metrics(
                val_df["label"].astype(str).tolist(),
                v_preds,
                labels=labels,
                model_key=model_key,
                model_name=spec["name"],
            )
            val_macro_f1 = v_metrics["macro_f1"]

        t_inf0 = time.perf_counter()
        test_preds, _, _ = predict_classical_batch(pipe, test_texts)
        avg_inf_ms = ((time.perf_counter() - t_inf0) * 1000.0) / max(1, len(test_texts))

        metrics = compute_classification_metrics(
            y_true=test_y,
            y_pred=test_preds,
            labels=labels,
            languages=test_langs,
            model_key=model_key,
            model_name=spec["name"],
            avg_inference_ms=avg_inf_ms,
        )
        metrics["training_seconds"] = train_sec
        metrics["val_macro_f1"] = val_macro_f1
        all_metrics.append(metrics)

        if mode == "climate_stance":
            save_confusion_matrix_chart(
                metrics["confusion_matrix"],
                labels=labels,
                model_name=spec["name"],
                out_path=CHARTS_DIR / f"cm_{model_key}.png",
            )

        print(
            f"[{model_key}] Macro F1={metrics['macro_f1']:.4f} | Acc={metrics['accuracy']:.4f} "
            f"| Train={train_sec:.2f}s | Inf={avg_inf_ms:.3f}ms/sample"
        )

    if mode == "climate_stance":
        save_metrics_summary(all_metrics)
    else:
        with open(METRICS_DIR / "pol_nli_auxiliary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

    return all_metrics


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train classical TF-IDF and embedding baselines.")
    parser.add_argument(
        "--mode",
        choices=["climate_stance", "pol_nli_auxiliary"],
        default="climate_stance",
    )
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    train_classical_models(
        mode=args.mode,
        selected_keys=args.models,
        seed=int(cfg.get("random_seed", 42)),
    )


if __name__ == "__main__":
    main()
