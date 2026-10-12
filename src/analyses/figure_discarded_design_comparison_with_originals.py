"""Compares the discarded reduced-feature design (paper Section 6.2) against
each representation family's own original literature algorithm, fit on raw
features: for each family, TabICLv2 fit on that family's best reduced-feature
tag (identified in ``results/evaluate_reduced_feature_counts/analysis/
feature_count_study_analysis.md``, Section 5) vs. the corresponding original
algorithm's own accuracy, on the same twenty-data-set development subset.

Family -> original-algorithm mapping (also documented in that report's
Section 7): convolution -> KG-MTP (this project's own raw-feature
implementation, ``comparisons/KG-MTP/``), dictionary -> WEASEL~2.0, interval
-> QUANT, shapelet -> RDST (the latter three via
``comparisons/bake-off-redux-2023/``).

This is a genuinely matched comparison: both sides are averaged over the
same five resamples (0-4) of the same twenty data sets, drawn from the same
canonical UCR resample-index scheme (``PythonResampleIndices``) that this
project's own resampling (``src/experiments/common.py``'s ``load_resample``)
uses -- each baseline file's "resample k" column is the identical train/test
split as this project's own resample k. The baseline files (both
``comparisons/bake-off-redux-2023/`` and ``comparisons/KG-MTP/``) report 30
resamples; only the first 5 are used here, matching our own resample count
exactly rather than comparing against their full 30-resample mean.

Reads ``results/evaluate_reduced_feature_counts/analysis/
comparison_with_originals_matched.csv``, produced by
``compute_discarded_design_matched_comparison.py`` directly from raw
per-unit manifests on both sides (our own ``evaluate_reduced_feature_counts``
units at each family's best tag, and the baseline CSVs' own resample-0-4
columns) -- exactly four rows, one per family.

Usage (from the repo root):
    python3 -m src.analyses.figure_discarded_design_comparison_with_originals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.analyses import common

COMPARISON_PATH = (
    common.RESULTS_DIR / "evaluate_reduced_feature_counts" / "analysis" / "comparison_with_originals_matched.csv"
)

FAMILY_ORDER = ["convolution", "interval", "shapelet", "dictionary"]
BASELINE_DISPLAY = {"KG-MTP": "KG-MTP", "QUANT": "QUANT", "RDST": "RDST", "WEASEL-2": "WEASEL 2.0"}

# Validated two-slot categorical palette (dataviz skill, references/palette.md
# slots 1-2; node scripts/validate_palette.js "#eb6834,#2a78d6" --mode light
# passes all six checks). Orange for "ours" keeps this project's own
# established convention (orange = tabiclv2) from
# figure_auto_selected_threshold.py and feature_count_study_analysis.md's own
# figures; blue marks the original, raw-feature literature algorithm.
COLOR_OURS = "#eb6834"
COLOR_BASELINE = "#2a78d6"

BAR_WIDTH = 0.36


def main() -> None:
    df = pd.read_csv(COMPARISON_PATH)
    df = df.set_index("Family").reindex(FAMILY_ORDER)
    assert df["BaselineAlgo"].notna().all(), "expected exactly one row per family in the matched comparison CSV."

    x = np.arange(len(FAMILY_ORDER))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - BAR_WIDTH / 2, df["OurAcc"], width=BAR_WIDTH, color=COLOR_OURS,
           label="TabICLv2 on reduced features (ours)", zorder=3)
    ax.bar(x + BAR_WIDTH / 2, df["BaselineAcc"], width=BAR_WIDTH, color=COLOR_BASELINE,
           label="Original algorithm on raw features", zorder=3)

    xtick_labels = [BASELINE_DISPLAY[df.loc[family, "BaselineAlgo"]] for family in FAMILY_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", color="0.9", linewidth=0.6, zorder=0)
    fig.tight_layout()

    common.save_figure(fig, "discarded_design_comparison_with_originals")


if __name__ == "__main__":
    main()
