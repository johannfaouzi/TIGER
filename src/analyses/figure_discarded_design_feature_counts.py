"""Illustrates the reduced/selected-feature TabICLv2-per-family study
(paper Section 6.2, discarded design alternatives): for each of the four
representation families, TabICLv2's own mean accuracy across the tested
feature-count grid (10/25/50/100/150 reduced components, plus each
family's own unconstrained default reduction).

Reads ``results/evaluate_reduced_feature_counts/summary.csv`` directly
(produced by the now-removed ``evaluate_reduced_feature_counts.py``,
see ``_removed_from_repo/`` -- the results themselves were kept). Shows
only the ``tabiclv2`` rows, since the point of this figure is TabICLv2's
own accuracy as a function of feature count, not a comparison against the
reference_extratrees baseline (that comparison is instead reported as a
table for the separate wide_selection experiment,
``table_discarded_design_wide_selection.py``).

Usage (from the repo root):
    python3 -m src.analyses.figure_discarded_design_feature_counts
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.analyses import common

SUMMARY_PATH = common.RESULTS_DIR / "evaluate_reduced_feature_counts" / "summary.csv"

TAG_ORDER = ["n10", "n25", "n50", "n100", "n150", "default"]
TAG_DISPLAY = {"n10": "10", "n25": "25", "n50": "50", "n100": "100", "n150": "150", "default": "default"}

FAMILY_ORDER = ["convolution", "interval", "shapelet", "dictionary"]
# Legend labels use each family's own original literature algorithm (the one
# whose transformation step this family reuses, see sn-article.tex Section 3.1)
# rather than the generic family name, so the legend reads the same way as
# Figure 9's x-tick labels (figure_discarded_design_comparison_with_originals.py).
FAMILY_DISPLAY = {"convolution": "KG-MTP", "interval": "QUANT", "shapelet": "RDST", "dictionary": "WEASEL 2.0"}

# Brand-neutral, colorblind-distinguishable qualitative palette, consistent
# in style with figure_auto_selected_threshold.py's blue/orange pair.
FAMILY_COLOR = {
    "convolution": "#4C72B0",  # blue
    "interval": "#DD8452",     # orange
    "shapelet": "#55A868",     # green
    "dictionary": "#8172B2",   # purple
}
FAMILY_MARKER = {
    "convolution": "o",
    "interval": "s",
    "shapelet": "^",
    "dictionary": "D",
}


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)
    df = df[df["Model"] == "tabiclv2"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(TAG_ORDER))

    for family in FAMILY_ORDER:
        sub = df[df["Family"] == family].set_index("MethodTag").reindex(TAG_ORDER)
        ax.plot(
            x, sub["Accuracy_mean"],
            color=FAMILY_COLOR[family], marker=FAMILY_MARKER[family],
            markersize=6, linewidth=1.8, label=FAMILY_DISPLAY[family], zorder=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([TAG_DISPLAY[t] for t in TAG_ORDER])
    ax.set_xlabel("Reduced feature count per family")
    ax.set_ylabel("TabICLv2 mean accuracy")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", color="0.9", linewidth=0.6, zorder=0)
    fig.tight_layout()

    common.save_figure(fig, "discarded_design_feature_counts")


if __name__ == "__main__":
    main()
