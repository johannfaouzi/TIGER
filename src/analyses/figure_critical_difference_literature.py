"""Critical-difference diagram for the paper's main six-algorithm
comparison (Tables~\\ref{tab:multi-metric-142}/\\ref{tab:multi-metric-122-heldout}):
TIGER (this project's deployed rule, the manifest's ``auto_selected``
column) against the five literature algorithms -- RDST, QUANT, WEASEL~2.0,
KG-MTP (all four run locally on the same 142 datasets/30 resamples,
``results/run_142_original_algorithms/manifest.csv``) and HC2 (published
numbers from ``comparisons/bake-off-redux-2023/``, exactly as in
``table_literature_comparison.py``, which this script's data loading
mirrors).

Built with aeon's own ``aeon.visualisation.plot_critical_difference``
(pairwise Wilcoxon signed-rank + Holm correction for clique-finding), same
family of test as the pairwise comparisons already quoted in the paper's
prose immediately above Table~\\ref{tab:multi-metric-142} -- this figure is
the standard TSC-venue companion picture for that paragraph, not a new
statistical claim.

Usage (from the repo root):
    python3 -m src.analyses.figure_critical_difference_literature
"""
from __future__ import annotations

import pandas as pd
from aeon.visualisation import plot_critical_difference

from src.analyses import common
from src.analyses.table_literature_comparison import (
    HC2_PUBLISHED_PATH,
    LOCAL_EXTERNAL_ALGORITHMS,
    ORIGINAL_ALGOS_MANIFEST_PATH,
    OWN_MANIFEST_PATH,
    _load_published_wide,
)

# TIGER's own column in the manifest is "auto_selected" (common.METHOD_DISPLAY_NAMES);
# display name here matches how the six algorithms are named in Tables 7-9, not
# table_literature_comparison.py's own (differently-worded) EXTERNAL_DISPLAY_NAMES.
DISPLAY_NAMES = {
    "auto_selected": "TIGER",
    "RDST": "RDST",
    "QUANT": "QUANT",
    "WEASEL2.0": "WEASEL 2.0",
    "KG-MTP": "KG-MTP",
    "HC2": "HC2",
}
HIGHLIGHT = {"TIGER": "#DD8452"}  # same accent color as figure_auto_selected_threshold.py's COLOR_ABOVE


def main() -> None:
    own_df = common.load_manifest(OWN_MANIFEST_PATH, methods=common.ALL_OWN_METHODS)
    own_wide = common.per_dataset_mean(own_df, "Accuracy")[["auto_selected"]]

    local_df = common.load_manifest(ORIGINAL_ALGOS_MANIFEST_PATH, method_col="Algorithm", methods=LOCAL_EXTERNAL_ALGORITHMS)
    local_wide = common.per_dataset_mean(local_df, "Accuracy", method_col="Algorithm")[list(LOCAL_EXTERNAL_ALGORITHMS)]

    hc2_published = _load_published_wide(HC2_PUBLISHED_PATH)

    wide = own_wide.join(local_wide, how="inner")
    wide["HC2"] = hc2_published
    assert wide["HC2"].notna().all(), "HC2 join left NaN rows -- dataset-name alignment failed for some dataset."

    methods = ["auto_selected"] + list(LOCAL_EXTERNAL_ALGORITHMS) + ["HC2"]
    labels = [DISPLAY_NAMES[m] for m in methods]

    fig, ax = plot_critical_difference(wide[methods].values, labels, highlight=HIGHLIGHT, lower_better=False)
    common.save_figure(fig, "critical_difference_literature")


if __name__ == "__main__":
    main()
