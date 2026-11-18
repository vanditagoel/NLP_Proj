"""Download real mlburnham/Pol_NLI dataset splits via Hugging Face datasets.

Usage:
    python -m ml.dataset.download
    python -m ml.dataset.download --filter-climate
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from ml.utils import RAW_DIR, ensure_directories


def download_pol_nli(out_dir: Path = RAW_DIR, filter_climate: bool = True) -> dict[str, int]:
    ensure_directories()
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("mlburnham/Pol_NLI")

    counts: dict[str, int] = {}
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        counts[split_name] = int(len(df))
        out_path = out_dir / f"pol_nli_{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {split_name}: {len(df):,} rows -> {out_path}")

        if filter_climate:
            gw_df = df[df["dataset"].astype(str) == "mlburnham/global_warming_stance_entailment"]
            gw_path = out_dir / f"pol_nli_climate_{split_name}.parquet"
            gw_df.to_parquet(gw_path, index=False)
            print(f"  Climate subset ({split_name}): {len(gw_df):,} rows -> {gw_path}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download mlburnham/Pol_NLI dataset from Hugging Face.")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--filter-climate", action="store_true", help="Also export global_warming_stance_entailment subset.")
    args = parser.parse_args()
    download_pol_nli(out_dir=args.out_dir, filter_climate=args.filter_climate)


if __name__ == "__main__":
    main()
