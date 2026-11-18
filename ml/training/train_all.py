"""Run classical baselines + optional transformers sequentially and produce unified benchmark results.

Usage:
    python -m ml.training.train_all --skip-transformers
    python -m ml.training.train_all --transformers indicbert
"""
from __future__ import annotations

import argparse

from ml.evaluation.evaluate import evaluate_all_trained_models
from ml.models.transformer import TRANSFORMER_MODEL_SPECS
from ml.training.train_classical import train_classical_models
from ml.training.train_transformer import run_transformer_training
from ml.utils import load_config


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train classical and transformer models sequentially.")
    parser.add_argument("--skip-transformers", action="store_true")
    parser.add_argument(
        "--transformers",
        nargs="*",
        default=list(TRANSFORMER_MODEL_SPECS.keys()),
        choices=list(TRANSFORMER_MODEL_SPECS.keys()),
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_classical_models(mode="climate_stance", seed=int(cfg.get("random_seed", 42)))

    if not args.skip_transformers:
        for t_key in args.transformers:
            run_transformer_training(model_key=t_key, epochs=args.epochs)

    evaluate_all_trained_models()


if __name__ == "__main__":
    main()
