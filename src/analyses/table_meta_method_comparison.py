"""Headline table: every meta-method's mean/median/std accuracy and mean
rank on the full 142-dataset x 30-resample benchmark
(``results/meta_model_predictions_142/manifest.csv``).

This is ``results/meta_model_predictions_142/analysis_report.md``
Section 2's table, reproduced as re-runnable code and extended: that
report's table only covers the 8 methods present when it was written.
This run's manifest carries 10 -- ``cawpe_hard_vote`` and
``auto_selected`` (this paper's dev-subset-derived automatic
meta-learner-selection rule, applied to the 142-dataset benchmark for the
first time) were patched in afterward (see
``apply_meta_learner_selection_142.py``) -- so mean ranks here differ
slightly from the report's own 8-method numbers even for methods whose
own mean accuracy is unchanged, simply because there are now 10
competitors to rank among instead of 8.

Usage (from the repo root):
    python3 -m src.analyses.table_meta_method_comparison
"""
from __future__ import annotations

from src.analyses import common

MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"


def main() -> None:
    df = common.load_manifest(MANIFEST_PATH, methods=common.ALL_OWN_METHODS)
    wide = common.per_dataset_mean(df, "Accuracy")[list(common.ALL_OWN_METHODS)]

    ranks = common.mean_rank(wide)
    table = wide.agg(["mean", "median", "std"]).T
    table.columns = ["Mean accuracy", "Median accuracy", "Std"]
    table["Mean rank"] = ranks
    table["Deployable"] = [common.is_deployable(m) for m in table.index]
    table = table.sort_values("Mean rank")
    table.index = [common.METHOD_DISPLAY_NAMES.get(m, m) for m in table.index]
    table.index.name = "Method"

    print(table.round(4).to_string())

    common.save_table(
        table,
        "meta_method_comparison",
        caption=(
            "Meta-method comparison on the 142-dataset x 30-resample benchmark. "
            "Oracle and meta-oracle require test-time hindsight and are not deployable."
        ),
        label="tab:meta_method_comparison",
    )

    # Sanity checks against results/meta_model_predictions_142/analysis_report.md
    # Section 2's own (pre-cawpe_hard_vote/auto_selected) reported numbers, for
    # the 8 methods that report already covers -- catches a silent regression
    # in the aggregation convention itself, not just a data change.
    reported_mean_acc = {
        "oracle": 0.8922, "meta_oracle": 0.8916, "tabiclv2": 0.8789, "oracle_cv": 0.8778,
        "cawpe": 0.8737, "logistic_regression": 0.8721, "hard_majority_vote": 0.8680, "lightgbm": 0.8655,
    }
    for method, expected in reported_mean_acc.items():
        actual = wide[method].mean()
        assert abs(actual - expected) < 0.0005, (
            f"{method}: mean accuracy {actual:.4f} does not match analysis_report.md's {expected:.4f} "
            "-- has the underlying manifest changed since that report was written?"
        )
    print("\nSanity check passed: all 8 pre-existing methods' mean accuracy matches "
          "analysis_report.md Section 2 to within 0.0005.")


if __name__ == "__main__":
    main()
