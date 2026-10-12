"""Meta-oracle headroom (paper Section 5.4.3/5.5) broken down by data set
characteristics: number of classes and training set size.

Reads ``results/meta_model_predictions_142/manifest.csv`` directly -- the
same manifest ``table_oracle_headroom.py`` already uses for the pooled
142-data-set number, which already carries ``NClasses`` and ``NTrain`` per
row (both constant within a data set across its 30 resamples, confirmed
here). This script reuses that table's own aggregation
(``common.per_dataset_mean``: each data set's resamples averaged first,
see ``common.py``'s "Aggregation convention" docstring section) and its
same reference methods, but grouped into bins of ``NClasses`` and
``NTrain`` instead of one pooled row, to see whether TIGER's adaptive rule
leaves more headroom on many-class or small-training-set data sets than
on the benchmark as a whole.

Uses ``common.six_method_oracle`` for the oracle column -- NOT the
manifest's own "oracle" column, which is a different, higher-scoring
diagnostic (the best of TIGER's twelve base learners by test hindsight,
no meta-classification stage at all) that an earlier version of
``table_oracle_headroom.py`` conflated with the six-method meta-oracle.
See ``common.py``'s own docstring next to ``SIX_META_METHODS`` for the
full story; that table now reports both, correctly labeled, and this
script's own bins follow the corrected definition throughout.

Usage (from the repo root):
    python3 -m src.analyses.table_oracle_headroom_by_characteristics
"""
from __future__ import annotations

import pandas as pd

from src.analyses import common

MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"
ROW_ORDER = ["oracle_cv", "auto_selected", "two_method_oracle", "six_method_oracle"]

NCLASSES_BINS = [1, 2, 5, 10, 1000]
NCLASSES_LABELS = ["2 (binary)", "3-5", "6-10", "11+"]

NTRAIN_BINS = [0, 61, 218, 467, 100000]
NTRAIN_LABELS = ["<=60", "61-217", "218-466", ">=467"]


def _headroom_by_bin(wide: pd.DataFrame, ds_char: pd.DataFrame, bin_col: str, bins, labels) -> pd.DataFrame:
    groups = pd.cut(ds_char[bin_col], bins=bins, labels=labels, right=True)
    rows = []
    for label in labels:
        datasets = ds_char.index[groups == label]
        sub = wide.loc[wide.index.intersection(datasets), ROW_ORDER]
        means = sub.mean()
        rows.append({
            "Bin": label,
            "N data sets": len(sub),
            "Best base learner (CV, hindsight)": means["oracle_cv"],
            "TIGER's adaptive rule (deployed)": means["auto_selected"],
            "2-method oracle (hindsight)": means["two_method_oracle"],
            "6-method oracle (hindsight)": means["six_method_oracle"],
            "Gap to 6-method oracle": means["six_method_oracle"] - means["auto_selected"],
        })
    out = pd.DataFrame(rows).set_index("Bin")
    return out


def main() -> None:
    df = common.load_manifest(MANIFEST_PATH, methods=common.ALL_OWN_METHODS)

    ds_char = df.groupby("Dataset")[["NClasses", "NTrain"]].mean()
    assert (df.groupby("Dataset")["NClasses"].nunique() == 1).all(), "NClasses varies within a data set across resamples."
    assert (df.groupby("Dataset")["NTrain"].nunique() == 1).all(), "NTrain varies within a data set across resamples."

    wide = common.per_dataset_mean(df, "Accuracy")
    wide["two_method_oracle"] = wide[["cawpe_hard_vote", "tabiclv2"]].max(axis=1)
    wide["six_method_oracle"] = common.six_method_oracle(wide)

    # Sanity check: the pooled (unbinned) six-method oracle mean here must
    # match table_oracle_headroom.py's own 0.8872.
    assert abs(wide["six_method_oracle"].mean() - 0.8872) < 0.001, (
        f"pooled six_method_oracle mean ({wide['six_method_oracle'].mean():.4f}) no longer matches "
        "table_oracle_headroom.py's reported 0.8872 -- re-check both before trusting either."
    )

    by_nclasses = _headroom_by_bin(wide, ds_char, "NClasses", NCLASSES_BINS, NCLASSES_LABELS)
    by_ntrain = _headroom_by_bin(wide, ds_char, "NTrain", NTRAIN_BINS, NTRAIN_LABELS)

    assert by_nclasses["N data sets"].sum() == 142
    assert by_ntrain["N data sets"].sum() == 142

    print("By number of classes:")
    print(by_nclasses.round(4).to_string())
    print()
    print("By training-set size:")
    print(by_ntrain.round(4).to_string())

    common.save_table(
        by_nclasses, "oracle_headroom_by_nclasses",
        float_format="%.4f",
        caption=(
            "Meta-oracle headroom (as in Table~\\ref{tab:oracle-headroom}'s six-method-oracle row), "
            "broken down by the number of classes in the data set (per-data-set means over 30 resamples, "
            "averaged within each bin; 142 data sets total)."
        ),
        label="tab:oracle-headroom-by-nclasses",
    )
    common.save_table(
        by_ntrain, "oracle_headroom_by_ntrain",
        float_format="%.4f",
        caption=(
            "Meta-oracle headroom (as in Table~\\ref{tab:oracle-headroom}'s six-method-oracle row), "
            "broken down by training set size (per-data-set means over 30 resamples, averaged within each "
            "bin; 142 data sets total; bin edges at the training-set-size quartiles)."
        ),
        label="tab:oracle-headroom-by-ntrain",
    )


if __name__ == "__main__":
    main()
