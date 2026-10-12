"""Reports the "wide selection" experiment (paper Section 6.2, discarded
design alternatives): for each family, its own original classifier (Ridge
for KG-MTP/RDST/WEASEL 2.0, Extra Trees for QUANT -- matching the rest of
this paper's base-learner design, see ``base_learners.py``) fit on (1) the
family's full, un-reduced raw features and (2) up to 1,000 selected raw
features (group size 100, the "wide selection" tag), against TabICLv2 fit
on that same selected feature set. Every runtime is end to end -- feature
selection's own cost (fit the reducer + transform both splits) is included
wherever selection is actually used, not just the final model's own fit.

Runtime aggregation: for each of the 20 development-subset data sets, we
first average that data set's elapsed time across its 5 resamples, then sum
those 20 per-data-set averages. This reports the total wall-clock
cost of running the design once across the whole development subset (a
mean resample's cost per data set, summed over data sets), rather than a
flat per-(data set, resample) mean. Accuracy is unaffected by this choice
and stays the plain mean over all 100 (data set, resample) units, as it
always was.

Reads three files:
  - ``results/evaluate_wide_selection_with_originals/manifest.csv`` (raw,
    per-unit -- produced by ``evaluate_wide_selection_with_originals.py``,
    on the server): ``FullElapsedSeconds``, ``ReductionElapsedSeconds``, and
    ``WideTotalElapsedSeconds`` per (data set, resample, family) unit, used
    for the new per-data-set-then-summed runtime aggregation described
    above.
  - ``results/evaluate_wide_selection_with_originals/summary.csv``: the
    "full features" and "wide selection, own classifier" *accuracy* columns
    only (``FullAccuracy_mean``, ``WideAccuracy_mean`` -- unaffected by the
    runtime-aggregation change).
  - ``results/evaluate_wide_selection_group_size/manifest.csv`` (raw,
    per-unit -- produced by the now-removed
    ``evaluate_wide_selection_group_size.py``, see ``_removed_from_repo/``
    -- the results themselves were kept, also measured on the server):
    TabICLv2's own accuracy (``summary.csv``'s ``Accuracy_mean``) and
    model-only fit+predict time, aggregated the same per-data-set-then-summed
    way as the other two runtime columns and then added to the matching
    ``ReductionElapsedSeconds`` sum, so TabICLv2's own reported runtime is
    end-to-end too, without refitting it (not refit here -- ~1 hour/unit,
    see the module's own docstring -- just combined with the newly-computed
    reduction-time sum).

Usage (from the repo root):
    python3 -m src.analyses.table_discarded_design_wide_selection
"""
from __future__ import annotations

import pandas as pd

from src.analyses import common

NEW_SUMMARY_PATH = common.RESULTS_DIR / "evaluate_wide_selection_with_originals" / "summary.csv"
NEW_MANIFEST_PATH = common.RESULTS_DIR / "evaluate_wide_selection_with_originals" / "manifest.csv"
OLD_SUMMARY_PATH = common.RESULTS_DIR / "evaluate_wide_selection_group_size" / "summary.csv"
OLD_MANIFEST_PATH = common.RESULTS_DIR / "evaluate_wide_selection_group_size" / "manifest.csv"

FAMILY_ORDER = ["convolution", "interval", "shapelet", "dictionary"]
# Row labels use each family's own original literature algorithm (the one whose
# transformation step this family reuses, see sn-article.tex Section 3.1) rather
# than the generic family name, consistent with Figures 8 and 9.
FAMILY_DISPLAY = {"convolution": "KG-MTP", "interval": "QUANT", "shapelet": "RDST", "dictionary": "WEASEL 2.0"}


def _per_dataset_mean_then_summed(df: pd.DataFrame, metric: str) -> pd.Series:
    """For each Family, average ``metric`` across each Dataset's 5 resamples,
    then sum those per-data-set averages across the 20 data sets. Returns a
    Family-indexed Series.
    """
    per_dataset = df.groupby(["Family", "Dataset"])[metric].mean()
    n_datasets = per_dataset.groupby("Family").size()
    assert (n_datasets == 20).all(), f"expected 20 data sets per family, got:\n{n_datasets}"
    n_resamples = df.groupby(["Family", "Dataset"]).size()
    assert (n_resamples == 5).all(), "expected exactly 5 resamples per (family, dataset) unit"
    return per_dataset.groupby("Family").sum()


def main() -> None:
    acc = pd.read_csv(NEW_SUMMARY_PATH).set_index("Family")
    new_manifest = pd.read_csv(NEW_MANIFEST_PATH)
    old_manifest = pd.read_csv(OLD_MANIFEST_PATH)
    old_manifest_tcl = old_manifest[old_manifest["Model"] == "tabiclv2"]

    full_runtime = _per_dataset_mean_then_summed(new_manifest, "FullElapsedSeconds")
    wide_runtime = _per_dataset_mean_then_summed(new_manifest, "WideTotalElapsedSeconds")
    reduction_runtime = _per_dataset_mean_then_summed(new_manifest, "ReductionElapsedSeconds")
    tabiclv2_model_runtime = _per_dataset_mean_then_summed(old_manifest_tcl, "ElapsedSeconds")

    acc_old = pd.read_csv(OLD_SUMMARY_PATH)
    tcl_acc = acc_old[acc_old["Model"] == "tabiclv2"].set_index("Family")["Accuracy_mean"]

    rows = []
    for family in FAMILY_ORDER:
        rows.append({
            "Algorithm": FAMILY_DISPLAY[family],
            "Full accuracy": acc.loc[family, "FullAccuracy_mean"],
            "Full runtime (min)": full_runtime.loc[family] / 60.0,
            "Wide-selected accuracy": acc.loc[family, "WideAccuracy_mean"],
            "Wide-selected runtime (min)": wide_runtime.loc[family] / 60.0,
            "TabICLv2 accuracy": tcl_acc.loc[family],
            "TabICLv2 runtime (min)": (reduction_runtime.loc[family] + tabiclv2_model_runtime.loc[family]) / 60.0,
        })
    out = pd.DataFrame(rows).set_index("Algorithm")
    print(out.round(4).to_string())

    common.save_table(
        out, "discarded_design_wide_selection",
        float_format="%.3f",
        caption=(
            "The discarded ``wide selection'' design: each family's original classifier on full vs.\\ "
            "wide-selected raw features, against TabICLv2 on the selected features. Accuracy is the mean "
            "over the 20 data sets' 5 resamples each; runtime (minutes) is the total cost of "
            "running each design once across the whole development subset (see text)."
        ),
        label="tab:discarded-design-wide-selection",
    )


if __name__ == "__main__":
    main()
