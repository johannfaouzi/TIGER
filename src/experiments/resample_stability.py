"""How many resamples does a *development-stage* run need, for the
proposed dev subset (``results/family_strengths/dev_subset.csv``), to
give a trustworthy read on which family wins each dataset?

``run_142_ucr_datasets.py``'s ``NUM_RESAMPLES = 3`` was picked
arbitrarily. This answers it empirically instead of by convention: the
``comparisons/`` CSVs already give each family's real algorithm 30
independent resamples per dataset (the standard UCR-benchmark resample
count), so we can measure directly how much using only ``n < 30`` of them
would have changed our read of "which family wins this dataset" versus
using all 30 -- for exactly the 24 datasets the dev subset covers, since
that's what a smaller development-stage resample count needs to stay
faithful to.

Method (subsampling, not a closed-form CI, so it captures win-*ranking*
stability directly rather than just per-family estimate noise):
    for each dev-subset dataset, for each candidate n:
        repeat ``N_TRIALS`` times: draw n of the 30 resamples (without
        replacement), compute each family's mean BalAcc from just those
        n, and check whether the argmax family matches the argmax family
        from the *full* 30-resample mean (the best estimate we have of
        "ground truth").
    report, per n, the fraction of trials where the winner matches
    (averaged first within each dataset's own trials, then across the 24
    datasets) -- and separately, the standard error of the per-family
    mean estimate as a function of n (finite-population-corrected, since
    we're sampling without replacement from a pool of exactly 30).

Usage
-----
    python -m src.experiments.resample_stability
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMPARISONS_DIR = Path("comparisons")
RESULTS_DIR = Path("results") / "family_strengths"
DEV_SUBSET_PATH = RESULTS_DIR / "dev_subset.csv"

FAMILY_SOURCES = {
    "convolution": ("KG-MTP", COMPARISONS_DIR / "KG-MTP"),
    "interval": ("QUANT", COMPARISONS_DIR / "bake-off-redux-2023"),
    "shapelet": ("RDST", COMPARISONS_DIR / "bake-off-redux-2023"),
    "dictionary": ("WEASEL-2", COMPARISONS_DIR / "bake-off-redux-2023"),
}
METRIC = "balacc"
N_RESAMPLES_AVAILABLE = 30
CANDIDATE_N = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]
N_TRIALS = 2000
RNG_SEED = 0


def _normalize_dataset_name(name: str) -> str:
    for suffix in ("_eq", "_nmv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_all_resamples(family: str) -> pd.DataFrame:
    """Every resample's raw score for ``family``'s algorithm, indexed by
    (normalized) dataset name -- unlike ``analyze_family_strengths.py``'s
    ``load_family_metric``, this keeps every column instead of collapsing
    to the mean, since the per-resample values are exactly what this
    script needs.
    """
    algo_name, directory = FAMILY_SOURCES[family]
    path = directory / f"{algo_name}_{METRIC}.csv"
    df = pd.read_csv(path)
    id_col = df.columns[0]
    df = df.set_index(df[id_col].map(_normalize_dataset_name)).drop(columns=[id_col])
    return df.astype(float)


def main() -> None:
    dev_subset = pd.read_csv(DEV_SUBSET_PATH)
    datasets = dev_subset["Dataset"].tolist()
    families = list(FAMILY_SOURCES)

    # scores[family] -> DataFrame (dataset x resample), restricted to the
    # dev-subset datasets and put in a consistent resample-column order.
    scores = {}
    for family in families:
        df = load_all_resamples(family)
        df = df.loc[datasets]
        assert df.shape[1] == N_RESAMPLES_AVAILABLE, (
            f"{family}: expected {N_RESAMPLES_AVAILABLE} resample columns, got {df.shape[1]}"
        )
        scores[family] = df.to_numpy()  # (24, 30)

    # (24, 4) stacked family scores, and the "ground truth" winner per
    # dataset from the full 30-resample mean (matches
    # analyze_family_strengths.py's own winner definition).
    full_mean = np.stack([scores[f].mean(axis=1) for f in families], axis=1)  # (24, 4)
    true_winner = np.argmax(full_mean, axis=1)  # (24,)
    true_margin = np.sort(full_mean, axis=1)[:, -1] - np.sort(full_mean, axis=1)[:, -2]

    rng = np.random.RandomState(RNG_SEED)
    rows = []
    for n in CANDIDATE_N:
        # Per-dataset win-agreement rate and per-family SE, averaged over
        # N_TRIALS random n-of-30 draws (same draw reused across families
        # so the comparison is apples-to-apples within a trial, exactly
        # mirroring what a real n-resample run would do: one shared set
        # of n resamples, not independently chosen per family).
        agreements = np.zeros((len(datasets), N_TRIALS), dtype=bool)
        se_estimates = np.zeros((len(datasets), N_TRIALS, len(families)))
        for t in range(N_TRIALS):
            idx = rng.choice(N_RESAMPLES_AVAILABLE, size=n, replace=False)
            subset_mean = np.stack([scores[f][:, idx].mean(axis=1) for f in families], axis=1)  # (24, 4)
            winner = np.argmax(subset_mean, axis=1)
            agreements[:, t] = winner == true_winner
            se_estimates[:, t, :] = subset_mean

        win_agreement_per_dataset = agreements.mean(axis=1)  # (24,)
        # Empirical std of the *subsampled n-resample mean* across trials
        # -- this already reflects sampling without replacement from a
        # pool of 30 (no separate finite-population-correction formula
        # needed, since it's measured directly from the same draws).
        se_per_dataset_family = se_estimates.std(axis=1)  # (24, 4)

        rows.append({
            "n_resamples": n,
            "mean_win_agreement": win_agreement_per_dataset.mean(),
            "min_win_agreement": win_agreement_per_dataset.min(),
            "mean_se_balacc": se_per_dataset_family.mean(),
            "max_se_balacc": se_per_dataset_family.max(),
        })

    curve = pd.DataFrame(rows)
    curve.to_csv(RESULTS_DIR / "resample_stability_curve.csv", index=False)

    # Per-dataset detail at every candidate n (not just a couple), so the
    # curve can be re-sliced afterward by e.g. excluding the dev subset's
    # intentionally-near-tied "close_call_all_families" datasets (whose
    # true_margin is close to 0 by construction -- no resample count
    # will make their "winner" stable, and that's expected, not a
    # resample-count problem).
    detail_rows = []
    for n in CANDIDATE_N:
        agreements = np.zeros((len(datasets), N_TRIALS), dtype=bool)
        for t in range(N_TRIALS):
            idx = rng.choice(N_RESAMPLES_AVAILABLE, size=n, replace=False)
            subset_mean = np.stack([scores[f][:, idx].mean(axis=1) for f in families], axis=1)
            winner = np.argmax(subset_mean, axis=1)
            agreements[:, t] = winner == true_winner
        for i, ds in enumerate(datasets):
            detail_rows.append({
                "Dataset": ds,
                "n_resamples": n,
                "true_winner": families[true_winner[i]],
                "true_margin": true_margin[i],
                "win_agreement": agreements[i].mean(),
            })
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(RESULTS_DIR / "resample_stability_detail.csv", index=False)

    print(f"Dev subset: {len(datasets)} datasets, {len(families)} families, "
          f"{N_TRIALS} trials per n, metric={METRIC}\n")
    print(curve.to_string(index=False))
    print(f"\nFull curve -> {RESULTS_DIR / 'resample_stability_curve.csv'}")
    print(f"Per-dataset detail (n=3,5,10) -> {RESULTS_DIR / 'resample_stability_detail.csv'}")


if __name__ == "__main__":
    main()
