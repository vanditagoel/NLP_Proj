"""Generate formatted comparison benchmark table across all trained models.

Usage:
    python -m ml.benchmark
"""
from __future__ import annotations

import json

from ml.evaluation.evaluate import evaluate_all_trained_models
from ml.utils import METRICS_DIR


def main() -> None:
    summary_path = METRICS_DIR / "evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = evaluate_all_trained_models()

    models = summary.get("models", [])
    if not models:
        print("No trained models found yet.")
        return

    print("\n=== Indic Climate Stance Benchmark (Sorted by Macro F1) ===")
    header = f"{'Model':<38} | {'Acc':>6} | {'MacroF1':>7} | {'HindiF1':>7} | {'BenF1':>7} | {'EngF1':>7} | {'Inf(ms)':>7}"
    print(header)
    print("-" * len(header))
    for m in models:
        lm = m.get("language_metrics", {})
        hi_f1 = lm.get("Hindi", {}).get("macro_f1", 0.0)
        bn_f1 = lm.get("Bengali", {}).get("macro_f1", 0.0)
        en_f1 = lm.get("English", {}).get("macro_f1", 0.0)
        print(
            f"{m['model_name'][:38]:<38} | "
            f"{m['accuracy']:>6.4f} | "
            f"{m['macro_f1']:>7.4f} | "
            f"{hi_f1:>7.4f} | "
            f"{bn_f1:>7.4f} | "
            f"{en_f1:>7.4f} | "
            f"{m.get('avg_inference_ms', 0.0):>7.3f}"
        )


if __name__ == "__main__":
    main()
