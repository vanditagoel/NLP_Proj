"""Prepare clean Train/Validation/Test splits with strict group-based leakage prevention.

Usage:
    python -m ml.dataset.prepare
    python -m ml.dataset.prepare --extract-source-only
    python -m ml.dataset.prepare --mode pol_nli_auxiliary --max-rows 4000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.dataset.seed_climate_data import (
    build_seed_multilingual_dataframe,
    build_seed_source_dataframe,
)
from ml.preprocessing.cleaner import clean_text
from ml.utils import (
    METRICS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS_DIR,
    TRANSLATED_DIR,
    ensure_directories,
    load_config,
)

HYP_POS = "Climate change is a serious concern."
HYP_NEG = "Climate change is not a concern."


def extract_pol_nli_climate_source(raw_dir: Path | None = None) -> pd.DataFrame | None:
    """Reconstruct exact 3-class (Favour/Against/None) unique premises from Pol_NLI's paired hypotheses."""
    base_dir = raw_dir or RAW_DIR
    parts = []
    for split in ("train", "validation", "test"):
        p = base_dir / f"pol_nli_{split}.parquet"
        if p.exists():
            df_split = pd.read_parquet(p)
            gw = df_split[
                df_split["dataset"].astype(str) == "mlburnham/global_warming_stance_entailment"
            ]
            if len(gw) > 0:
                parts.append(gw)
    if not parts:
        return None

    gw_all = pd.concat(parts, ignore_index=True)
    piv = gw_all.pivot_table(
        index="premise",
        columns="hypothesis",
        values="entailment",
        aggfunc=lambda s: list(s.unique()),
    )

    rows: list[dict[str, Any]] = []
    idx_counter = 1
    for prem, row in piv.iterrows():
        p_vals = row.get(HYP_POS)
        n_vals = row.get(HYP_NEG)
        if (
            not isinstance(p_vals, list)
            or not isinstance(n_vals, list)
            or len(p_vals) != 1
            or len(n_vals) != 1
        ):
            continue

        pv, nv = int(p_vals[0]), int(n_vals[0])
        if pv == 0 and nv == 1:
            label = "Favour"
            orig_label = "pos_entail=0,neg_entail=1"
        elif pv == 1 and nv == 0:
            label = "Against"
            orig_label = "pos_entail=1,neg_entail=0"
        elif pv == 1 and nv == 1:
            label = "None"
            orig_label = "pos_entail=1,neg_entail=1"
        else:
            continue

        cleaned = clean_text(str(prem))
        if not cleaned:
            continue

        gid = f"pol_gw_{idx_counter:04d}"
        idx_counter += 1
        rows.append(
            {
                "id": gid,
                "group_id": gid,
                "original_text": str(prem),
                "text": cleaned,
                "language": "English",
                "language_code": "en",
                "label": label,
                "original_label": orig_label,
                "original_source": "mlburnham/Pol_NLI::global_warming_stance_entailment",
                "translation_status": "Original",
                "is_machine_translated": False,
            }
        )

    return pd.DataFrame(rows) if rows else None


def group_stratified_split(
    df: pd.DataFrame,
    group_col: str = "group_id",
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each group_id deterministically to Train, Validation, or Test stratified by label."""
    out = df.copy()
    rng = np.random.default_rng(seed)

    group_labels = (
        out.groupby(group_col)[label_col]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.iloc[0])
        .reset_index()
    )

    group_to_split: dict[Any, str] = {}
    for _, grp_df in group_labels.groupby(label_col):
        groups = sorted(grp_df[group_col].tolist())
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n and n >= 3:
            n_train = max(1, n - 2)
            n_val = 1

        for idx, g in enumerate(groups):
            if idx < n_train:
                group_to_split[g] = "Train"
            elif idx < n_train + n_val:
                group_to_split[g] = "Validation"
            else:
                group_to_split[g] = "Test"

    out["split"] = out[group_col].map(group_to_split)

    train_g = set(out[out["split"] == "Train"][group_col])
    val_g = set(out[out["split"] == "Validation"][group_col])
    test_g = set(out[out["split"] == "Test"][group_col])
    assert train_g.isdisjoint(val_g), "Group leakage detected between Train and Validation!"
    assert train_g.isdisjoint(test_g), "Group leakage detected between Train and Test!"
    assert val_g.isdisjoint(test_g), "Group leakage detected between Validation and Test!"

    return out


def prepare_climate_stance_dataset(
    custom_csv: Path | None = None,
    use_translated: bool = True,
    extract_source_only: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    ensure_directories()
    cfg = load_config()
    ratios = cfg["split_ratios"]

    pol_source = extract_pol_nli_climate_source()
    source_df = pol_source if pol_source is not None else build_seed_source_dataframe()
    source_path = PROCESSED_DIR / "climate_stance_source.csv"
    source_df.to_csv(source_path, index=False, encoding="utf-8")
    print(f"Exported English 3-class source ({len(source_df):,} unique premises) -> {source_path}")

    if extract_source_only:
        return {"source_rows": int(len(source_df)), "source_path": str(source_path)}

    default_custom = RAW_DIR / "climate_stance_custom.csv"
    translated_csv = TRANSLATED_DIR / "climate_stance_translated.csv"

    if custom_csv is not None and custom_csv.exists():
        df = pd.read_csv(custom_csv, keep_default_na=False)
        dataset_origin = f"imported_custom:{custom_csv.name}"
    elif default_custom.exists():
        df = pd.read_csv(default_custom, keep_default_na=False)
        dataset_origin = f"imported_custom:{default_custom.name}"
    elif use_translated and translated_csv.exists():
        df = pd.read_csv(translated_csv, keep_default_na=False)
        dataset_origin = "pol_nli_paired_hypotheses + translated_hi_bn"
    elif pol_source is not None:
        df = pol_source.copy()
        dataset_origin = "pol_nli_paired_hypotheses_english_only"
    else:
        df = build_seed_multilingual_dataframe()
        dataset_origin = "curated_multilingual_seed"

    if "group_id" not in df.columns:
        df["group_id"] = df["original_id"] if "original_id" in df.columns else df["id"]
    if "original_text" not in df.columns:
        df["original_text"] = df["text"]
    if "original_label" not in df.columns:
        df["original_label"] = df["label"]
    if "translation_status" not in df.columns:
        df["translation_status"] = "Original"
    if "language_code" not in df.columns:
        lang_map = {"English": "en", "Hindi": "hi", "Bengali": "bn"}
        df["language_code"] = df["language"].map(lang_map).fillna("en")

    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()

    valid_labels = {"Favour", "Against", "None"}
    df = df[df["label"].isin(valid_labels)].copy()

    df = group_stratified_split(
        df,
        group_col="group_id",
        label_col="label",
        train_ratio=float(ratios["train"]),
        val_ratio=float(ratios["validation"]),
        seed=seed,
    )

    full_out = PROCESSED_DIR / "climate_stance_dataset.csv"
    df.to_csv(full_out, index=False, encoding="utf-8")

    for split_name in ("Train", "Validation", "Test"):
        split_df = df[df["split"] == split_name].copy()
        split_df.to_csv(
            SPLITS_DIR / f"climate_{split_name.lower()}.csv",
            index=False,
            encoding="utf-8",
        )

    meta = {
        "mode": "climate_stance",
        "dataset_origin": dataset_origin,
        "total_rows": int(len(df)),
        "unique_groups": int(df["group_id"].nunique()),
        "splits": {s: int((df["split"] == s).sum()) for s in ("Train", "Validation", "Test")},
        "languages": {str(k): int(v) for k, v in df["language"].value_counts().items()},
        "labels": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "translation_status": {
            str(k): int(v) for k, v in df["translation_status"].value_counts().items()
        },
        "leakage_verified": True,
    }
    with open(METRICS_DIR / "dataset_prepare_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Prepared 3-class climate stance dataset ({len(df)} rows, origin={dataset_origin}).")
    print(f"Splits: {meta['splits']} | Languages: {meta['languages']} | Labels: {meta['labels']}")
    return meta


def prepare_pol_nli_auxiliary(max_rows: int = 4000, seed: int = 42) -> dict[str, Any]:
    """Prepare auxiliary Pol_NLI dataset keeping authentic binary NLI labels (entailment / not_entailment)."""
    ensure_directories()
    dfs: list[pd.DataFrame] = []
    for split_name, std_name in [("train", "Train"), ("validation", "Validation"), ("test", "Test")]:
        p = RAW_DIR / f"pol_nli_{split_name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run `python -m ml.dataset.download` before preparing Pol_NLI auxiliary data."
            )
        part = pd.read_parquet(p)
        cap = max_rows if split_name == "train" else max(400, max_rows // 4)
        sampled = part.sample(n=min(cap, len(part)), random_state=seed).copy()
        sampled["split"] = std_name
        dfs.append(sampled)

    full = pd.concat(dfs, ignore_index=True)
    full["id"] = [f"pol_nli_{i}" for i in range(len(full))]
    full["original_text"] = full["premise"].astype(str)
    full["text"] = (
        full["premise"].astype(str).apply(clean_text)
        + " [SEP] "
        + full["hypothesis"].astype(str).apply(clean_text)
    )
    full["language"] = "English"
    full["language_code"] = "en"
    full["original_label"] = full["entailment"].astype(str)
    full["label"] = full["entailment"].map({0: "entailment", 1: "not_entailment"}).fillna("not_entailment")
    full["translation_status"] = "Original"

    out_path = PROCESSED_DIR / "pol_nli_auxiliary.csv"
    full.to_csv(out_path, index=False, encoding="utf-8")
    for s in ("Train", "Validation", "Test"):
        full[full["split"] == s].to_csv(
            SPLITS_DIR / f"pol_nli_{s.lower()}.csv", index=False, encoding="utf-8"
        )

    meta = {
        "mode": "pol_nli_auxiliary",
        "dataset_origin": "mlburnham/Pol_NLI",
        "total_rows": int(len(full)),
        "splits": {s: int((full["split"] == s).sum()) for s in ("Train", "Validation", "Test")},
        "labels": {str(k): int(v) for k, v in full["label"].value_counts().items()},
        "note": "Preserves native Pol_NLI binary entailment task; not relabeled as 3-class climate stance.",
    }
    with open(METRICS_DIR / "pol_nli_prepare_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Prepared auxiliary Pol_NLI dataset ({len(full)} rows) -> {out_path}")
    return meta


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Prepare train/validation/test splits without leakage.")
    parser.add_argument(
        "--mode",
        choices=["climate_stance", "pol_nli_auxiliary"],
        default="climate_stance",
    )
    parser.add_argument("--custom-csv", type=Path, default=None)
    parser.add_argument("--extract-source-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=4000)
    args = parser.parse_args()

    seed = int(cfg.get("random_seed", 42))
    if args.mode == "pol_nli_auxiliary":
        prepare_pol_nli_auxiliary(max_rows=args.max_rows, seed=seed)
    else:
        prepare_climate_stance_dataset(
            custom_csv=args.custom_csv,
            extract_source_only=args.extract_source_only,
            seed=seed,
        )


if __name__ == "__main__":
    main()
