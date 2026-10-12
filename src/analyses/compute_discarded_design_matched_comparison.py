"""Recompute the discarded-design-vs-originals comparison (paper Section 6.2,
Figure 9) using a genuinely matched pair of resamples: the baseline's own
published per-resample results restricted to resamples 0-4, against our own
5 resamples -- both drawn from the same canonical UCR resample-index scheme
(PythonResampleIndices), per Johann's confirmation that the splits match.

This supersedes the old `comparison_with_originals.csv` (30-resample
baseline mean vs our 5-resample mean, flagged in the paper as "not a matched
paired test"), whose own generating script was never committed. This script
is the new, committed generator, run from raw per-unit manifests on both
sides so the whole pipeline is reproducible from results/ alone.

Usage (from the repo root):
    python3 -m src.analyses.compute_discarded_design_matched_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, ".")
from src.experiments.common import dev_subset_datasets  # noqa: E402

RESULTS_DIR = Path("results")
FEATURE_COUNTS_UNITS = RESULTS_DIR / "evaluate_reduced_feature_counts" / "units"
COMPARISONS_DIR = Path("comparisons")

FAMILY_ORDER = ["convolution", "interval", "shapelet", "dictionary"]
BEST_TAG = {"convolution": "default", "interval": "n10", "shapelet": "n50", "dictionary": "n100"}
# Baseline source per family: (csv_path, dataset_col_is_plain)
BASELINE_CSV = {
    "convolution": COMPARISONS_DIR / "KG-MTP" / "KG-MTP_accuracy.csv",
    "interval": COMPARISONS_DIR / "bake-off-redux-2023" / "QUANT_accuracy.csv",
    "shapelet": COMPARISONS_DIR / "bake-off-redux-2023" / "RDST_accuracy.csv",
    "dictionary": COMPARISONS_DIR / "bake-off-redux-2023" / "WEASEL-2_accuracy.csv",
}
BASELINE_ALGO = {"convolution": "KG-MTP", "interval": "QUANT", "shapelet": "RDST", "dictionary": "WEASEL-2"}

MATCHED_RESAMPLES = [0, 1, 2, 3, 4]

datasets = dev_subset_datasets()
assert len(datasets) == 20


def load_our_per_dataset_means() -> dict:
    """family -> {dataset: mean accuracy over resamples 0-4 at that family's best tag, tabiclv2}"""
    out = {}
    for family in FAMILY_ORDER:
        tag = BEST_TAG[family]
        per_dataset = {}
        for dataset in datasets:
            accs = []
            for r in MATCHED_RESAMPLES:
                path = FEATURE_COUNTS_UNITS / dataset / f"resample{r}" / family / tag / "manifest.csv"
                df = pd.read_csv(path)
                row = df[df["Model"] == "tabiclv2"]
                assert len(row) == 1, f"{path}: expected 1 tabiclv2 row, got {len(row)}"
                accs.append(float(row["Accuracy"].iloc[0]))
            assert len(accs) == 5
            per_dataset[dataset] = float(np.mean(accs))
        out[family] = per_dataset
    return out


def load_baseline_per_dataset_means() -> dict:
    """family -> {dataset: mean accuracy over baseline's own resamples 0-4}"""
    out = {}
    for family in FAMILY_ORDER:
        path = BASELINE_CSV[family]
        df = pd.read_csv(path)
        # bake-off-redux-2023 files have a "Resamples:" header cell in the first column;
        # KG-MTP files use "Dataset" -- normalize both to a Dataset-indexed frame.
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Dataset"}).set_index("Dataset")
        df.columns = [c.strip() for c in df.columns]
        # strip the "_eq" suffix some bake-off-redux-2023 names carry (per
        # feature_count_study_analysis.md's own reconciliation note)
        df.index = [str(i).replace("_eq", "") for i in df.index]
        resample_cols = [str(r) for r in MATCHED_RESAMPLES]
        missing_cols = [c for c in resample_cols if c not in df.columns]
        assert not missing_cols, f"{path}: missing resample columns {missing_cols}"
        per_dataset = {}
        for dataset in datasets:
            assert dataset in df.index, f"{path}: dataset {dataset} not found (available: {list(df.index)[:5]}...)"
            row = df.loc[dataset, resample_cols]
            if isinstance(row, pd.DataFrame):
                # duplicate index entries (shouldn't happen after _eq stripping, but guard anyway)
                row = row.iloc[0]
            vals = row.astype(float).to_numpy()
            assert len(vals) == 5
            per_dataset[dataset] = float(np.mean(vals))
        out[family] = per_dataset
    return out


def main() -> None:
    our = load_our_per_dataset_means()
    baseline = load_baseline_per_dataset_means()

    rows = []
    for family in FAMILY_ORDER:
        our_vals = np.array([our[family][d] for d in datasets])
        base_vals = np.array([baseline[family][d] for d in datasets])
        deltas = our_vals - base_vals
        wins = int(np.sum(deltas > 0))
        losses = int(np.sum(deltas < 0))
        ties = int(np.sum(deltas == 0))
        stat, p = wilcoxon(deltas)
        rows.append({
            "Family": family,
            "BaselineAlgo": BASELINE_ALGO[family],
            "Tag": BEST_TAG[family],
            "OurAcc": our_vals.mean(),
            "BaselineAcc": base_vals.mean(),
            "DeltaAcc": (our_vals - base_vals).mean(),
            "WinsAcc": wins, "TiesAcc": ties, "LossesAcc": losses,
            "WilcoxonP_Acc": p,
        })

    out_df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(out_df.round(5).to_string(index=False))

    out_path = RESULTS_DIR / "evaluate_reduced_feature_counts" / "analysis" / "comparison_with_originals_matched.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
