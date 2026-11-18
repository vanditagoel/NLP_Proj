"""Generate dataset statistics and visualizations saved to results/metrics and results/charts.

Usage:
    python -m ml.dataset.analyze
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml.utils import CHARTS_DIR, METRICS_DIR, PROCESSED_DIR, ensure_directories


def analyze_dataset(dataset_csv: Path | None = None) -> dict[str, Any]:
    ensure_directories()
    csv_path = dataset_csv or (PROCESSED_DIR / "climate_stance_dataset.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {csv_path}. Run `python -m ml.dataset.prepare` first."
        )

    df = pd.read_csv(csv_path, keep_default_na=False)
    df["char_length"] = df["text"].astype(str).str.len()
    df["word_length"] = df["text"].astype(str).str.split().str.len()

    lang_x_class = (
        pd.crosstab(df["language"], df["label"]).to_dict(orient="index")
        if {"language", "label"}.issubset(df.columns)
        else {}
    )

    stats: dict[str, Any] = {
        "dataset_path": str(csv_path),
        "total_samples": int(len(df)),
        "split_distribution": (
            {str(k): int(v) for k, v in df["split"].value_counts().items()}
            if "split" in df.columns
            else {}
        ),
        "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "language_distribution": {str(k): int(v) for k, v in df["language"].value_counts().items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "text_length_chars": {
            "mean": round(float(df["char_length"].mean()), 2),
            "median": round(float(df["char_length"].median()), 2),
            "min": int(df["char_length"].min()),
            "max": int(df["char_length"].max()),
        },
        "text_length_words": {
            "mean": round(float(df["word_length"].mean()), 2),
            "median": round(float(df["word_length"].median()), 2),
            "min": int(df["word_length"].min()),
            "max": int(df["word_length"].max()),
        },
        "language_by_class": lang_x_class,
        "translation_status": (
            {str(k): int(v) for k, v in df["translation_status"].value_counts().items()}
            if "translation_status" in df.columns
            else {}
        ),
    }

    stats_path = METRICS_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    cls_counts = df["label"].value_counts()
    colors = ["#059669" if c == "Favour" else "#e11d48" if c == "Against" else "#64748b" for c in cls_counts.index]
    ax.bar(cls_counts.index, cls_counts.values, color=colors)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "class_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    lang_counts = df["language"].value_counts()
    ax.bar(lang_counts.index, lang_counts.values, color="#0f766e")
    ax.set_title("Language Distribution")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "language_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(df["word_length"], bins=15, color="#334155", edgecolor="white")
    ax.set_title("Text Length Distribution (Words)")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "text_length_distribution.png")
    plt.close(fig)

    ct = pd.crosstab(df["language"], df["label"])
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ct.plot(
        kind="bar",
        ax=ax,
        color=["#e11d48", "#059669", "#64748b"] if set(ct.columns) == {"Against", "Favour", "None"} else None,
    )
    ax.set_title("Language x Class Distribution")
    ax.set_ylabel("Samples")
    ax.set_xlabel("Language")
    plt.xticks(rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "language_x_class_distribution.png")
    plt.close(fig)

    print(f"Dataset statistics saved -> {stats_path}")
    print(f"Charts saved to -> {CHARTS_DIR}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prepared climate stance dataset and save charts.")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    analyze_dataset(args.input)


if __name__ == "__main__":
    main()
