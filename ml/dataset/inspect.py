"""Inspect mlburnham/Pol_NLI splits, schema, label distributions, duplicates, missing values, and task compatibility.

Usage:
    python -m ml.dataset.inspect
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.utils import METRICS_DIR, RAW_DIR, ensure_directories


def inspect_pol_nli(raw_dir: Path = RAW_DIR) -> dict[str, Any]:
    ensure_directories()
    report: dict[str, Any] = {
        "dataset_id": "mlburnham/Pol_NLI",
        "splits": {},
        "total_rows": 0,
        "global_warming_subset": {},
        "compatibility_analysis": {},
    }

    gw_parts: list[pd.DataFrame] = []
    for split_name in ("train", "validation", "test"):
        p = raw_dir / f"pol_nli_{split_name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run `python -m ml.dataset.download --filter-climate` first."
            )
        df = pd.read_parquet(p)
        report["total_rows"] += int(len(df))
        report["splits"][split_name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_premises": int(df["premise"].duplicated().sum()) if "premise" in df.columns else 0,
            "entailment_distribution": (
                {str(k): int(v) for k, v in df["entailment"].value_counts().items()}
                if "entailment" in df.columns
                else {}
            ),
            "top_subdatasets": (
                {str(k): int(v) for k, v in df["dataset"].value_counts().head(10).items()}
                if "dataset" in df.columns
                else {}
            ),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
        gw = df[df["dataset"].astype(str) == "mlburnham/global_warming_stance_entailment"]
        if len(gw) > 0:
            gw_parts.append(gw)

    if gw_parts:
        gw_all = pd.concat(gw_parts, ignore_index=True)
        hyp_counts = {str(k): int(v) for k, v in gw_all["hypothesis"].value_counts().items()}
        piv = gw_all.pivot_table(
            index="premise",
            columns="hypothesis",
            values="entailment",
            aggfunc=lambda s: list(s.unique()),
        )
        valid_3class = {"Favour": 0, "Against": 0, "None": 0}
        excluded = 0
        h_pos = "Climate change is a serious concern."
        h_neg = "Climate change is not a concern."
        for _, row in piv.iterrows():
            pv = row.get(h_pos)
            nv = row.get(h_neg)
            if isinstance(pv, list) and isinstance(nv, list) and len(pv) == 1 and len(nv) == 1:
                p0, n0 = int(pv[0]), int(nv[0])
                if p0 == 0 and n0 == 1:
                    valid_3class["Favour"] += 1
                elif p0 == 1 and n0 == 0:
                    valid_3class["Against"] += 1
                elif p0 == 1 and n0 == 1:
                    valid_3class["None"] += 1
                else:
                    excluded += 1
            else:
                excluded += 1

        report["global_warming_subset"] = {
            "total_nli_rows": int(len(gw_all)),
            "unique_premises": int(gw_all["premise"].nunique()),
            "hypotheses": hyp_counts,
            "reconstructed_3class_counts": valid_3class,
            "reconstructed_total": sum(valid_3class.values()),
            "excluded_ambiguous_or_partial_premises": excluded,
        }

    report["compatibility_analysis"] = {
        "task_formulation": "Binary NLI entailment (premise + hypothesis -> entailment 0/1) across multiple political corpora.",
        "directly_compatible_raw_rows": False,
        "reason": (
            "Raw Pol_NLI rows are binary NLI pairs (entailment=0 or not_entailment=1) spanning many political topics "
            "(elections, partisanship, COVID, guns, global warming). Blindly mapping 0/1 to Favour/Against/None would be invalid."
        ),
        "legitimate_resolution": (
            "1) Isolate the embedded `mlburnham/global_warming_stance_entailment` subset where each premise is paired with "
            "`Climate change is a serious concern.` and `Climate change is not a concern.`; reconstruct exact 3-class stance "
            "(Favour / Against / None) only for unambiguous paired annotations and exclude contradictory/partial premises. "
            "2) Preserve full Pol_NLI separately with native binary labels for auxiliary NLI experiments."
        ),
    }

    out_path = METRICS_DIR / "pol_nli_inspection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"Pol_NLI inspection report saved -> {out_path}")
    print(f"Total Pol_NLI rows: {report['total_rows']:,}")
    if report["global_warming_subset"]:
        gw_info = report["global_warming_subset"]
        print(
            f"Global Warming subset: {gw_info['total_nli_rows']:,} NLI rows -> "
            f"{gw_info['reconstructed_total']:,} verified 3-class unique premises "
            f"({gw_info['reconstructed_3class_counts']}, excluded={gw_info['excluded_ambiguous_or_partial_premises']})"
        )
    return report


def main() -> None:
    inspect_pol_nli()


if __name__ == "__main__":
    main()
