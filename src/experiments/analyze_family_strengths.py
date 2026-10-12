"""Which of this package's four families would be expected to win on which
dataset, and a curated subset of datasets for fast development iteration.

Each of ``tsfm_ensemble``'s four families reimplements (feature extractor +
matched reducer around a shared TabICLv2 backbone) a specific published
algorithm -- see ``pipeline.DEFAULT_FAMILY_KWARGS``'s comment and each
family class's docstring in ``families.py``:

    family        published algorithm   comparisons/ source
    ------------  ---------------------  -------------------------------
    convolution   KG-MTP                 comparisons/KG-MTP/
    interval      QUANT                  comparisons/bake-off-redux-2023/
    shapelet      RDST                   comparisons/bake-off-redux-2023/
    dictionary    WEASEL 2.0             comparisons/bake-off-redux-2023/

Those ``comparisons/`` CSVs are the published algorithms' own per-resample
scores on the 142-dataset UCR benchmark (one row per dataset, one column
per resample; KG-MTP additionally ships accuracy/balacc/logloss/auroc/f1,
the bake-off-redux-2023 algorithms ship accuracy/balacc/logloss/auroc). We
treat each family's matching algorithm's published score as a proxy for
"how well would this family's own reimplementation be expected to do on
this dataset" -- not a guarantee (our own extractor/reducer/backbone
combination isn't literally the same code), but the closest thing to
ground truth available without re-running all four families n=142 times
(which is exactly the expensive thing this analysis exists to help avoid
during day-to-day development).

Usage
-----
    python -m src.experiments.analyze_family_strengths

Writes, under ``results/family_strengths/``:
    family_strengths_full.csv  -- one row per dataset: every family's mean
                                   BalAcc/Accuracy/LogLoss, the winning
                                   family under each metric, and the
                                   winning margin.
    family_win_counts.csv      -- how many datasets each family "wins" on,
                                   under each metric.
    dev_subset.csv             -- the proposed development subset (see
                                   ``build_dev_subset``'s docstring).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

COMPARISONS_DIR = Path("comparisons")
RESULTS_DIR = Path("results") / "family_strengths"

# family name (as used throughout tsfm_ensemble) -> (published algorithm
# name, directory under comparisons/ its CSVs live in).
FAMILY_SOURCES = {
    "convolution": ("KG-MTP", COMPARISONS_DIR / "KG-MTP"),
    "interval": ("QUANT", COMPARISONS_DIR / "bake-off-redux-2023"),
    "shapelet": ("RDST", COMPARISONS_DIR / "bake-off-redux-2023"),
    "dictionary": ("WEASEL-2", COMPARISONS_DIR / "bake-off-redux-2023"),
}

METRICS = ("balacc", "accuracy", "logloss")
# Whether higher is better for each metric -- used both for picking each
# dataset's per-metric winner and for the dev-subset margin calculation.
HIGHER_IS_BETTER = {"balacc": True, "accuracy": True, "logloss": False}


def _normalize_dataset_name(name: str) -> str:
    """Strip the "_eq"/"_nmv" suffixes some algorithms' result files use
    (e.g. QUANT/RDST/WEASEL-2's "AconityMINIPrinterLarge_eq",
    "DodgerLoopDay_nmv") so all four families' tables key on the same
    142 dataset names KG-MTP's own file already uses unsuffixed.
    """
    for suffix in ("_eq", "_nmv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_family_metric(family: str, metric: str) -> pd.Series:
    """Mean ``metric`` across all resamples, indexed by (normalized)
    dataset name, for the algorithm ``family`` is standing in for.
    """
    algo_name, directory = FAMILY_SOURCES[family]
    path = directory / f"{algo_name}_{metric}.csv"
    df = pd.read_csv(path)
    id_col = df.columns[0]  # "Dataset" (KG-MTP) or "Resamples:" (the rest)
    df = df.set_index(df[id_col].map(_normalize_dataset_name)).drop(columns=[id_col])
    resample_cols = [c for c in df.columns]
    return df[resample_cols].astype(float).mean(axis=1).rename(family)


def build_full_table() -> pd.DataFrame:
    """One row per dataset: every family's mean score under every metric,
    plus the winning family and winning margin under each metric.
    """
    per_metric_frames: Dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        family_series = [load_family_metric(family, metric) for family in FAMILY_SOURCES]
        frame = pd.concat(family_series, axis=1)
        assert not frame.isna().any().any(), (
            f"metric={metric!r}: missing values after aligning families on dataset "
            "name -- a family's comparisons/ file is missing a dataset the others "
            "have, or normalize_dataset_name needs another suffix case."
        )
        per_metric_frames[metric] = frame

    out = pd.DataFrame(index=per_metric_frames[METRICS[0]].index)
    out.index.name = "Dataset"
    for metric in METRICS:
        frame = per_metric_frames[metric]
        for family in FAMILY_SOURCES:
            out[f"{metric}_{family}"] = frame[family]

        ascending = not HIGHER_IS_BETTER[metric]
        ranked = frame.apply(lambda row: row.sort_values(ascending=ascending), axis=1)
        out[f"{metric}_winner"] = frame.idxmax(axis=1) if HIGHER_IS_BETTER[metric] else frame.idxmin(axis=1)
        best = ranked.iloc[:, 0]
        runner_up = ranked.iloc[:, 1]
        out[f"{metric}_margin"] = (best - runner_up).abs()

    return out.reset_index().sort_values("Dataset").reset_index(drop=True)


def win_counts(full_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        counts = full_table[f"{metric}_winner"].value_counts()
        for family in FAMILY_SOURCES:
            rows.append({"Metric": metric, "Family": family, "NumDatasetsWon": int(counts.get(family, 0))})
    return pd.DataFrame(rows)


def build_dev_subset(
    full_table: pd.DataFrame, primary_metric: str = "balacc", n_per_family: int = 5, n_close_calls: int = 0
) -> pd.DataFrame:
    """A development-iteration subset spanning all four families' regimes.

    For each family, the ``n_per_family`` datasets where it beats the
    runner-up family by the largest margin under ``primary_metric``.
    These are the closest thing to a guarantee that, on this dataset,
    that family's real strength is being exercised -- useful for checking
    that a change to (say) the convolution family's reducer doesn't
    quietly regress it on exactly the datasets where it's supposed to
    shine.

    ``n_close_calls`` (default 0, i.e. off) optionally adds the
    ``n_close_calls`` datasets with the *smallest* margin between the
    best and worst family (all four roughly tied) -- these stress the
    ensemble's CAWPE fusion/weighting logic instead of any one family,
    which the "decisive" group barely exercises. Left off by default: a
    follow-up resample-count analysis (``resample_stability.py``) found
    that these near-four-way-tied datasets' "winning family" doesn't
    stabilize even at the full 30 resamples available in ``comparisons/``
    (their true margin is close to noise by construction), which makes
    them a poor fit for the kind of per-family regression-checking this
    subset is for during development -- kept as an opt-in for anyone who
    specifically wants fusion-stress-test datasets instead.
    """
    reason_frames = []

    for family in FAMILY_SOURCES:
        is_winner = full_table[f"{primary_metric}_winner"] == family
        decisive = full_table[is_winner].nlargest(n_per_family, f"{primary_metric}_margin").copy()
        decisive["SubsetReason"] = f"decisive_win_{family}"
        reason_frames.append(decisive)

    if n_close_calls > 0:
        family_cols = [f"{primary_metric}_{family}" for family in FAMILY_SOURCES]
        spread = full_table[family_cols].max(axis=1) - full_table[family_cols].min(axis=1)
        # A small spread is only an interesting "all four families
        # genuinely contend" case if the dataset isn't just easy for
        # everyone (e.g. all four at ~1.0 balacc, which is a tie for
        # trivial reasons and exercises nothing about CAWPE's fusion
        # weighting). Restrict to datasets where the mean family score is
        # below near-ceiling before ranking by spread.
        non_trivial = full_table[full_table[family_cols].mean(axis=1) < 0.95]
        close_calls = non_trivial.loc[spread.loc[non_trivial.index].nsmallest(n_close_calls).index].copy()
        close_calls["SubsetReason"] = "close_call_all_families"
        reason_frames.append(close_calls)

    subset = pd.concat(reason_frames, ignore_index=True)
    # A dataset can legitimately qualify more than once (e.g. a decisive
    # win that's also, separately, near the close-call cutoff); keep the
    # first reason it was added under and drop the duplicate row rather
    # than silently deduplicating reasons away.
    subset = subset.drop_duplicates(subset="Dataset", keep="first")
    return subset.sort_values(["SubsetReason", "Dataset"]).reset_index(drop=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_table = build_full_table()
    full_table.to_csv(RESULTS_DIR / "family_strengths_full.csv", index=False)

    counts = win_counts(full_table)
    counts.to_csv(RESULTS_DIR / "family_win_counts.csv", index=False)

    subset = build_dev_subset(full_table)
    subset.to_csv(RESULTS_DIR / "dev_subset.csv", index=False)

    print(f"{len(full_table)} datasets analyzed.\n")
    print("Win counts (BalAcc):")
    print(counts[counts["Metric"] == "balacc"].to_string(index=False))
    print(f"\nAgreement BalAcc vs Accuracy winner: {(full_table['balacc_winner'] == full_table['accuracy_winner']).mean():.1%}")
    print(f"Agreement BalAcc vs LogLoss winner: {(full_table['balacc_winner'] == full_table['logloss_winner']).mean():.1%}")
    print(f"\nDev subset: {len(subset)} datasets -> {RESULTS_DIR / 'dev_subset.csv'}")
    print(subset["SubsetReason"].value_counts().to_string())
    print(f"\nFull results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
