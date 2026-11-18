"""Train a Hugging Face transformer (MuRIL, mBERT, XLM-R, or IndicBERT) on the climate stance splits.

Usage:
    python -m ml.training.train_transformer --model indicbert --epochs 2
    python -m ml.training.train_transformer --model muril --epochs 3
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd

from ml.evaluation.metrics import (
    compute_classification_metrics,
    save_confusion_matrix_chart,
)
from ml.models.transformer import (
    TRANSFORMER_MODEL_SPECS,
    check_transformer_compatibility,
    predict_with_saved_transformer,
    train_transformer_model,
)
from ml.utils import (
    CHARTS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    SPLITS_DIR,
    ensure_directories,
    load_config,
)


def run_transformer_training(
    model_key: str,
    mode: str = "climate_stance",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_length: int | None = None,
    weight_decay: float | None = None,
    gradient_accumulation_steps: int | None = None,
    early_stopping_patience: int | None = None,
    use_fp16: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    ensure_directories()
    cfg = load_config()
    t_cfg = cfg["training"]

    compat = check_transformer_compatibility(model_key)
    if not compat["compatible"]:
        raise RuntimeError(
            f"Transformer '{model_key}' ({compat.get('hf_id')}) compatibility check failed: {compat.get('error')}"
        )

    prefix = "climate" if mode == "climate_stance" else "pol_nli"
    train_df = pd.read_csv(SPLITS_DIR / f"{prefix}_train.csv", keep_default_na=False)
    val_df = pd.read_csv(SPLITS_DIR / f"{prefix}_validation.csv", keep_default_na=False)
    test_df = pd.read_csv(SPLITS_DIR / f"{prefix}_test.csv", keep_default_na=False)

    labels = (
        ["Favour", "Against", "None"]
        if mode == "climate_stance"
        else sorted(train_df["label"].unique().tolist())
    )
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}

    train_texts = train_df["text"].astype(str).tolist()
    train_labels = [label2id[lbl] for lbl in train_df["label"].astype(str)]
    val_texts = val_df["text"].astype(str).tolist()
    val_labels = [label2id[lbl] for lbl in val_df["label"].astype(str)]
    test_texts = test_df["text"].astype(str).tolist()
    test_y_true = test_df["label"].astype(str).tolist()
    test_langs = test_df["language"].astype(str).tolist() if "language" in test_df.columns else None

    out_dir = (MODELS_DIR / model_key) if mode == "climate_stance" else (MODELS_DIR / "pol_nli_auxiliary" / model_key)

    train_meta = train_transformer_model(
        model_key=model_key,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        label_names=labels,
        output_dir=out_dir,
        epochs=epochs if epochs is not None else int(t_cfg["epochs"]),
        batch_size=batch_size if batch_size is not None else int(t_cfg["batch_size"]),
        learning_rate=learning_rate if learning_rate is not None else float(t_cfg["learning_rate"]),
        max_length=max_length if max_length is not None else int(t_cfg["max_length"]),
        weight_decay=weight_decay if weight_decay is not None else float(t_cfg["weight_decay"]),
        gradient_accumulation_steps=(
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else int(t_cfg["gradient_accumulation_steps"])
        ),
        early_stopping_patience=(
            early_stopping_patience
            if early_stopping_patience is not None
            else int(t_cfg["early_stopping_patience"])
        ),
        use_fp16=use_fp16,
        seed=seed if seed is not None else int(cfg["random_seed"]),
    )

    t0 = time.perf_counter()
    test_preds, _ = predict_with_saved_transformer(
        out_dir,
        test_texts,
        max_length=max_length if max_length is not None else int(t_cfg["max_length"]),
    )
    inf_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(test_texts))

    spec = TRANSFORMER_MODEL_SPECS[model_key]
    test_metrics = compute_classification_metrics(
        y_true=test_y_true,
        y_pred=test_preds,
        labels=labels,
        languages=test_langs,
        model_key=model_key,
        model_name=spec["name"],
        avg_inference_ms=inf_ms,
    )
    test_metrics["training_seconds"] = train_meta["training_seconds"]
    test_metrics["best_val_macro_f1"] = train_meta["best_val_macro_f1"]

    with open(METRICS_DIR / f"transformer_{model_key}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    save_confusion_matrix_chart(
        test_metrics["confusion_matrix"],
        labels=labels,
        model_name=spec["name"],
        out_path=CHARTS_DIR / f"cm_{model_key}.png",
    )

    print(
        f"\n[{model_key}] Test Macro F1={test_metrics['macro_f1']:.4f} "
        f"| Test Accuracy={test_metrics['accuracy']:.4f}"
    )
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for climate stance.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(TRANSFORMER_MODEL_SPECS.keys()),
    )
    parser.add_argument(
        "--mode",
        choices=["climate_stance", "pol_nli_auxiliary"],
        default="climate_stance",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_transformer_training(
        model_key=args.model,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
        early_stopping_patience=args.early_stopping,
        use_fp16=not args.no_fp16,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
