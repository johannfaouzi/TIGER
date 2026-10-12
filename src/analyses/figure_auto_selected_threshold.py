"""Illustrates the automatic meta-learner-selection rule's core mechanism
and its held-out result in one picture: for every (dataset, resample) unit
in the 142-dataset benchmark, the accuracy gap between the two methods
the rule picks between (tabiclv2 minus cawpe_hard_vote) plotted against
that unit's own training samples-per-class (the rule's only input --
``select_meta_learner``), with the derived threshold (60.83, dev-subset
only -- see ``tsfm_ensemble.meta_learner_selection``'s module docstring)
drawn in.

If the rule generalizes, the picture should show the accuracy gap mostly
negative (cawpe_hard_vote ahead) to the left of the threshold line and
mostly positive (tabiclv2 ahead) to the right -- this is the 142-dataset
benchmark's own version of the trend the threshold was originally read
off of, but on data the rule never saw while being derived. A binned mean
trend line is overlaid since 4260 raw points are too dense to read a
trend from by eye alone.

Usage (from the repo root):
    python3 -m src.analyses.figure_auto_selected_threshold
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analyses import common
from src.tsfm_ensemble.meta_learner_selection import SAMPLES_PER_CLASS_THRESHOLD

MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"

# Brand-neutral, colorblind-distinguishable qualitative pair (blue / orange),
# consistent low-saturation style suitable for a print figure.
COLOR_BELOW = "#4C72B0"  # samples/class < threshold -> rule picks weighted hard majority vote
COLOR_ABOVE = "#DD8452"  # samples/class >= threshold -> rule picks TabICLv2
N_BINS = 24


def main() -> None:
    df = common.load_manifest(MANIFEST_PATH, methods=["tabiclv2", "cawpe_hard_vote"])
    tcl = df[df["Method"] == "tabiclv2"].set_index(["Dataset", "Resample"])
    chv = df[df["Method"] == "cawpe_hard_vote"].set_index(["Dataset", "Resample"])

    samples_per_class = tcl["NTrain"] / tcl["NClasses"]
    delta = tcl["Accuracy"] - chv["Accuracy"]
    below = samples_per_class < SAMPLES_PER_CLASS_THRESHOLD

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="0.6", linewidth=1, zorder=1)
    ax.axvline(SAMPLES_PER_CLASS_THRESHOLD, color="0.3", linewidth=1.2, linestyle="--", zorder=1,
               label=f"derived threshold = {SAMPLES_PER_CLASS_THRESHOLD:.2f}")

    ax.scatter(samples_per_class[below], delta[below], s=10, alpha=0.25, color=COLOR_BELOW,
               edgecolors="none", label="below threshold (rule picks weighted hard majority vote)", zorder=2)
    ax.scatter(samples_per_class[~below], delta[~below], s=10, alpha=0.25, color=COLOR_ABOVE,
               edgecolors="none", label="at/above threshold (rule picks TabICLv2)", zorder=2)

    # Binned mean trend line (equal-count bins over log(samples/class)).
    log_spc = np.log10(samples_per_class.clip(lower=1e-6))
    bin_edges = np.quantile(log_spc, np.linspace(0, 1, N_BINS + 1))
    bin_edges = np.unique(bin_edges)
    bin_idx = np.digitize(log_spc, bin_edges[1:-1])
    trend = pd.DataFrame({"bin": bin_idx, "x": samples_per_class, "y": delta}).groupby("bin").mean()
    ax.plot(trend["x"], trend["y"], color="black", linewidth=1.8, zorder=3, label=f"binned mean ({len(trend)} bins)")

    ax.set_xscale("log")
    ax.set_xlabel("Mean training samples per class (log scale)")
    ax.set_ylabel("Accuracy gap: TabICLv2 − WHMV")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", axis="both", color="0.9", linewidth=0.6, zorder=0)
    fig.tight_layout()

    common.save_figure(fig, "auto_selected_threshold")


if __name__ == "__main__":
    main()
