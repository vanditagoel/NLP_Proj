"""Evaluation metrics prioritizing Macro F1 plus per-class, per-language, confusion matrix, and PNG charts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.utils import CHARTS_DIR, METRICS_DIR, ensure_directories


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    languages: list[str] | None = None,
    model_key: str = "",
    model_name: str = "",
    avg_inference_ms: float = 0.0,
) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        lbl: {
            "precision": round(float(rep.get(lbl, {}).get("precision", 0.0)), 4),
            "recall": round(float(rep.get(lbl, {}).get("recall", 0.0)), 4),
            "f1": round(float(rep.get(lbl, {}).get("f1-score", 0.0)), 4),
            "support": int(rep.get(lbl, {}).get("support", 0)),
        }
        for lbl in labels
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    lang_metrics: dict[str, dict[str, Any]] = {}
    if languages is not None and len(languages) == len(y_true):
        for lang_name in sorted(set(languages)):
            idxs = [i for i, l in enumerate(languages) if l == lang_name]
            if not idxs:
                continue
            yt_l = [y_true[i] for i in idxs]
            yp_l = [y_pred[i] for i in idxs]
            lang_metrics[lang_name] = {
                "samples": len(idxs),
                "accuracy": round(float(accuracy_score(yt_l, yp_l)), 4),
                "macro_f1": round(
                    float(f1_score(yt_l, yp_l, labels=labels, average="macro", zero_division=0)),
                    4,
                ),
            }

    return {
        "model_key": model_key,
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "precision_macro": round(macro_precision, 4),
        "recall_macro": round(macro_recall, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
        "language_metrics": lang_metrics,
        "avg_inference_ms": round(float(avg_inference_ms), 4),
        "test_samples": len(y_true),
    }


def save_confusion_matrix_chart(
    cm: list[list[int]],
    labels: list[str],
    model_name: str,
    out_path: Path,
) -> None:
    ensure_directories()
    arr = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    im = ax.imshow(arr, cmap="Teal" if "Teal" in plt.colormaps() else "Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    thresh = arr.max() / 2.0 if arr.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                str(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "#0f172a",
                fontsize=11,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_model_comparison_chart(models_metrics: list[dict[str, Any]], out_path: Path) -> None:
    if not models_metrics:
        return
    ensure_directories()
    sorted_m = sorted(models_metrics, key=lambda m: m.get("macro_f1", 0.0), reverse=True)
    names = [m["model_name"] for m in sorted_m]
    macro_f1s = [m["macro_f1"] * 100.0 for m in sorted_m]
    accs = [m["accuracy"] * 100.0 for m in sorted_m]

    y = np.arange(len(names))
    height = 0.36
    fig, ax = plt.subplots(figsize=(8.2, max(3.6, 0.75 * len(names) + 1.2)), dpi=150)
    ax.barh(y + height / 2, macro_f1s, height, label="Macro F1 (%)", color="#0f766e")
    ax.barh(y - height / 2, accs, height, label="Accuracy (%)", color="#64748b")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (%)")
    ax.set_title("Model Performance Comparison (Sorted by Macro F1)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_language_f1_chart(models_metrics: list[dict[str, Any]], out_path: Path) -> None:
    if not models_metrics:
        return
    ensure_directories()
    langs = ["English", "Hindi", "Bengali"]
    active_langs = [l for l in langs if any(l in m.get("language_metrics", {}) for m in models_metrics)]
    if not active_langs:
        return

    names = [m["model_name"] for m in models_metrics]
    x = np.arange(len(names))
    width = 0.25
    colors = {"English": "#334155", "Hindi": "#0f766e", "Bengali": "#0284c7"}

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(names)), 4.6), dpi=150)
    for idx, lang in enumerate(active_langs):
        vals = [m.get("language_metrics", {}).get(lang, {}).get("macro_f1", 0.0) * 100.0 for m in models_metrics]
        ax.bar(x + (idx - (len(active_langs) - 1) / 2) * width, vals, width, label=lang, color=colors.get(lang, "#0f766e"))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Language-Wise Macro F1 (English vs Hindi vs Bengali)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_metrics_summary(models_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_directories()
    sorted_models = sorted(models_metrics, key=lambda m: m.get("macro_f1", 0.0), reverse=True)
    best_model = sorted_models[0] if sorted_models else None

    payload = {
        "best_model_key": best_model["model_key"] if best_model else None,
        "best_model_name": best_model["model_name"] if best_model else None,
        "models": sorted_models,
    }
    json_path = METRICS_DIR / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if sorted_models:
        rows = []
        for m in sorted_models:
            lm = m.get("language_metrics", {})
            rows.append(
                {
                    "model_key": m["model_key"],
                    "model_name": m["model_name"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "weighted_f1": m["weighted_f1"],
                    "precision_macro": m["precision_macro"],
                    "recall_macro": m["recall_macro"],
                    "hindi_macro_f1": lm.get("Hindi", {}).get("macro_f1"),
                    "bengali_macro_f1": lm.get("Bengali", {}).get("macro_f1"),
                    "english_macro_f1": lm.get("English", {}).get("macro_f1"),
                    "avg_inference_ms": m.get("avg_inference_ms"),
                }
            )
        pd.DataFrame(rows).to_csv(METRICS_DIR / "evaluation_summary.csv", index=False)

    save_model_comparison_chart(sorted_models, CHARTS_DIR / "model_comparison.png")
    save_language_f1_chart(sorted_models, CHARTS_DIR / "language_f1_comparison.png")
    return payload
