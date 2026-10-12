"""Derivation script for ``tsfm_ensemble.meta_learner_selection``: scans
every candidate samples-per-class threshold for a single-threshold
(``cawpe_hard_vote`` below / ``tabiclv2`` at-or-above) rule and reports the
best one, alongside a comparison against the six real methods' own fixed
mean accuracies and a 6-method hindsight oracle -- all computed on
``results/meta_model_predictions/dev_subset_characteristics_and_winners.csv``
(the 20 development datasets x 5 resamples produced by
``build_meta_model_predictions.py`` + ``patch_tabiclv2_dev_subset.py``; see
``meta_learner_selection``'s own module docstring for the full derivation
writeup, and why this was NOT validated via leave-one-dataset-out).

Usage (from the repo root):
    python3 -m src.experiments.analyze_meta_learner_selection
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REAL_METHODS = (
    "hard_majority_vote", "logistic_regression", "lightgbm",
    "cawpe", "cawpe_hard_vote", "tabiclv2",
)
CSV_PATH = Path("results") / "meta_model_predictions" / "dev_subset_characteristics_and_winners.csv"
LOW_SPC_METHOD = "cawpe_hard_vote"
HIGH_SPC_METHOD = "tabiclv2"


def samples_per_class_threshold_scan(df: pd.DataFrame) -> tuple[float, float]:
    """Scan every midpoint between the 20 datasets' own (dataset-level,
    deduplicated) samples-per-class values for the threshold maximizing
    ``rule_accuracy.mean()`` across all 100 (dataset, resample) rows.
    Returns ``(best_threshold, best_mean_accuracy)``.
    """
    spc = df["SamplesPerClassIfBalanced"].to_numpy()
    unique_vals = sorted(df["SamplesPerClassIfBalanced"].unique())
    midpoints = [(unique_vals[i] + unique_vals[i + 1]) / 2 for i in range(len(unique_vals) - 1)]

    best_threshold, best_mean_acc = None, -1.0
    for threshold in midpoints:
        rule_acc = np.where(spc < threshold, df[f"Acc__{LOW_SPC_METHOD}"], df[f"Acc__{HIGH_SPC_METHOD}"])
        mean_acc = rule_acc.mean()
        print(f"  T={threshold:8.2f}  rule mean acc={mean_acc:.4f}")
        if mean_acc > best_mean_acc:
            best_threshold, best_mean_acc = threshold, mean_acc
    return best_threshold, best_mean_acc


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    print("=== Single-threshold rule scan: spc < T -> "
          f"{LOW_SPC_METHOD}, else {HIGH_SPC_METHOD} ===")
    threshold, rule_mean_acc = samples_per_class_threshold_scan(df)
    print(f"\nBest threshold: T={threshold:.2f}, mean acc={rule_mean_acc:.4f}")

    spc = df["SamplesPerClassIfBalanced"].to_numpy()
    rule_acc = np.where(spc < threshold, df[f"Acc__{LOW_SPC_METHOD}"], df[f"Acc__{HIGH_SPC_METHOD}"])

    print("\n=== Comparison: rule vs fixed single-method baselines vs oracle-over-6 ===")
    print(f"  {'single-threshold rule (T=' + f'{threshold:.1f}' + ')':32s} mean acc = {rule_acc.mean():.4f}")
    for method in REAL_METHODS:
        print(f"  {'always ' + method:32s} mean acc = {df[f'Acc__{method}'].mean():.4f}")
    oracle_over_six = df[[f"Acc__{m}" for m in REAL_METHODS]].max(axis=1)
    print(f"  {'oracle (best of 6, hindsight)':32s} mean acc = {oracle_over_six.mean():.4f}")

    best_fixed_method = max(REAL_METHODS, key=lambda m: df[f"Acc__{m}"].mean())
    stat, p = wilcoxon(rule_acc, df[f"Acc__{best_fixed_method}"])
    wins = int((rule_acc > df[f"Acc__{best_fixed_method}"]).sum())
    losses = int((rule_acc < df[f"Acc__{best_fixed_method}"]).sum())
    ties = int((rule_acc == df[f"Acc__{best_fixed_method}"]).sum())
    print(f"\nrule vs always-{best_fixed_method} (best fixed method): "
          f"wins={wins} losses={losses} ties={ties} Wilcoxon p={p:.4f}")

    headroom = oracle_over_six.mean() - df[f"Acc__{best_fixed_method}"].mean()
    captured = rule_mean_acc - df[f"Acc__{best_fixed_method}"].mean()
    print(f"\nHeadroom over best fixed method captured by the rule: "
          f"{captured:.4f} / {headroom:.4f} = {100 * captured / headroom:.1f}%")


if __name__ == "__main__":
    main()
