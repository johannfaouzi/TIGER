"""Held-out validation of ``tsfm_ensemble.meta_learner_selection``'s
automatic meta-learner-selection rule on the 142-dataset x 30-resample
benchmark -- the first time this rule has been evaluated on data it
never touched during derivation (see that module's own docstring:
derived entirely from the 20-dataset dev subset, "The 142-dataset
benchmark ... was never consulted while designing this rule").

Two tables, two different granularities -- deliberately not mixed into
one:

1. ``auto_selected_vs_fixed_methods``: per-dataset-mean paired comparison
   (``common.paired_comparison``'s convention -- see that function's
   docstring) of ``auto_selected`` against the two methods it chooses
   between (``cawpe_hard_vote``, ``tabiclv2``) and against
   ``meta_oracle`` (the achievable ceiling). This is the hypothesis-test
   granularity: N=142.
2. ``auto_selected_selection_behavior``: per-*unit* (one row per
   (dataset, resample), N=4260) statistics on the rule's actual
   choices -- how often it picked each method, and how often that choice
   matched or beat the better of the two fixed methods, or matched
   ``meta_oracle``'s own pick outright. This describes the rule's
   mechanical behavior, not a hypothesis test, so it is intentionally
   computed at the same granularity ``select_meta_learner`` was actually
   invoked at (see ``apply_meta_learner_selection_142.py``), not
   collapsed to per-dataset means first.

Usage (from the repo root):
    python3 -m src.analyses.table_auto_selected_holdout
"""
from __future__ import annotations

import pandas as pd

from src.analyses import common

MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"
METHODS = ("auto_selected", "cawpe_hard_vote", "tabiclv2", "meta_oracle")


def main() -> None:
    df = common.load_manifest(MANIFEST_PATH, methods=METHODS)

    # --- Table 1: per-dataset-mean paired comparison (N=142) -----------
    wide = common.per_dataset_mean(df, "Accuracy")[list(METHODS)]
    summary = wide.agg(["mean", "median", "std"]).T
    summary.columns = ["Mean accuracy", "Median accuracy", "Std"]
    summary["Mean rank"] = common.mean_rank(wide)
    comparisons = common.paired_comparison_table(wide, "auto_selected", ["cawpe_hard_vote", "tabiclv2", "meta_oracle"])
    comparisons.columns = [
        "Wins (beats auto_selected)", "Losses (loses to auto_selected)", "Ties",
        "Mean diff (method - auto_selected)", "Wilcoxon p",
    ]
    table1 = summary.join(comparisons, how="left")
    table1 = table1.sort_values("Mean rank")
    table1.index = [common.METHOD_DISPLAY_NAMES.get(m, m) for m in table1.index]
    table1.index.name = "Method"
    print(table1.round(4).to_string())
    common.save_table(
        table1, "auto_selected_vs_fixed_methods",
        caption=(
            "Held-out evaluation of the dev-subset-derived automatic meta-learner-selection rule "
            "on the 142-dataset benchmark. Wins/losses/ties and Wilcoxon p are against auto\\_selected, "
            "computed over per-dataset means (N=142)."
        ),
        label="tab:auto_selected_vs_fixed_methods",
    )

    # --- Table 2: per-unit selection behavior (N=4260) ------------------
    auto = df[df["Method"] == "auto_selected"].set_index(["Dataset", "Resample"])
    chv = df[df["Method"] == "cawpe_hard_vote"].set_index(["Dataset", "Resample"])["Accuracy"]
    tcl = df[df["Method"] == "tabiclv2"].set_index(["Dataset", "Resample"])["Accuracy"]
    oracle = df[df["Method"] == "meta_oracle"].set_index(["Dataset", "Resample"])["Accuracy"]

    n_units = len(auto)
    selection_counts = auto["SelectedMethod"].value_counts()
    best_of_fixed = pd.concat([chv, tcl], axis=1).max(axis=1)
    matches_or_beats_max = (auto["Accuracy"] >= best_of_fixed).mean()
    matches_oracle = (auto["Accuracy"] == oracle).mean()

    behavior = pd.DataFrame(
        {
            "Value": [
                n_units,
                int(selection_counts.get("cawpe_hard_vote", 0)),
                int(selection_counts.get("tabiclv2", 0)),
                round(float(matches_or_beats_max), 4),
                round(float(matches_oracle), 4),
            ]
        },
        index=[
            "Total (dataset, resample) units",
            "Units where cawpe_hard_vote selected",
            "Units where tabiclv2 selected",
            "Fraction matching or beating max(cawpe_hard_vote, tabiclv2)",
            "Fraction matching meta_oracle's own pick",
        ],
    )
    behavior.index.name = "Statistic"
    print()
    print(behavior.to_string())
    common.save_table(
        behavior, "auto_selected_selection_behavior", float_format="%.4f",
        caption=(
            "Per-unit (dataset, resample) behavior of the automatic meta-learner-selection rule "
            "on the 142-dataset benchmark (N=4260)."
        ),
        label="tab:auto_selected_selection_behavior",
    )


if __name__ == "__main__":
    main()
