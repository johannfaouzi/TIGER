"""Compute what fraction of each development dataset's samples are used
in TRAINING across the 5 resamples used for meta-learner-selection
development.

Motivation: no resample's *test* set is ever used to make development
decisions (see ``build_stacking_features.py``'s "On not using the test
scores to make development decisions" docstring) -- but a sample held out
in one resample's test set can easily fall into a *different* resample's
training set, since each resample re-splits the same pooled dataset with
a different membership. So in practice, most of each dataset's pool of
samples ends up touched by *some* resample's training process, even
though no single resample's training set alone sees them all, and the
held-out-test discipline is still honored resample-by-resample.

For each of the 20 development datasets: pools all of that dataset's
samples (train + test, exactly as ``common.py``'s own ``load_resample``
does), reconstructs each of the 5 resamples' (0..4) training-set
membership over that pool using the same convention ``load_resample``
uses (resample 0 is the archive's original train split; resamples 1-4
come from ``PythonResampleIndices/<dataset>/resample{k}Indices_TRAIN.txt``),
and reports what fraction of the pooled samples appear in at least one
resample's training set -- both per dataset and pooled/averaged overall.

Usage (from the repo root):
    python3 -m src.experiments.compute_resample_train_coverage
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.common import (
    RESAMPLE_INDICES_DIR,
    dev_subset_datasets,
    load_classification_with_fallback,
)

NUM_DEV_RESAMPLES = 5
OUT_PATH = Path("results") / "meta_model_predictions" / "dev_subset_resample_train_coverage.csv"


def _train_indices_for_resample(
    dataset: str, resample: int, n_train_full: int,
    resample_indices_dir: Path = RESAMPLE_INDICES_DIR,
) -> np.ndarray:
    """Which pooled-array positions are this resample's training set --
    same membership convention as ``common.py``'s own ``load_resample``,
    without materializing X/y (only indices are needed here).
    """
    if resample == 0:
        return np.arange(n_train_full)
    path = resample_indices_dir / dataset / f"resample{resample}Indices_TRAIN.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find resample indices file {path}.")
    return np.loadtxt(path, dtype=np.int64)


def main() -> None:
    datasets = dev_subset_datasets()
    rows = []
    for dataset in datasets:
        _, y_train = load_classification_with_fallback(dataset, split="train")
        _, y_test = load_classification_with_fallback(dataset, split="test")
        n_train_full, n_test_full = len(y_train), len(y_test)
        n_pooled = n_train_full + n_test_full

        covered = np.zeros(n_pooled, dtype=bool)
        for resample in range(NUM_DEV_RESAMPLES):
            train_idx = _train_indices_for_resample(dataset, resample, n_train_full)
            covered[train_idx] = True

        n_covered = int(covered.sum())
        proportion = n_covered / n_pooled
        rows.append({
            "Dataset": dataset,
            "NTrainOriginal": n_train_full,
            "NTestOriginal": n_test_full,
            "NPooled": n_pooled,
            "NCoveredByAnyResampleTrain": n_covered,
            "ProportionCoveredByAnyResampleTrain": proportion,
        })
        print(f"  {dataset}: {n_covered}/{n_pooled} = {proportion:.4f}")

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    macro_mean = df["ProportionCoveredByAnyResampleTrain"].mean()
    micro = df["NCoveredByAnyResampleTrain"].sum() / df["NPooled"].sum()
    print(f"\nSaved {len(df)} rows -> {OUT_PATH}")
    print(f"Macro-average (mean of per-dataset proportions):        {macro_mean:.4f}")
    print(f"Micro-average (pooled samples covered / pooled total):  {micro:.4f}")
    print(f"Min / max per-dataset proportion: "
          f"{df['ProportionCoveredByAnyResampleTrain'].min():.4f} / "
          f"{df['ProportionCoveredByAnyResampleTrain'].max():.4f}")


if __name__ == "__main__":
    main()
