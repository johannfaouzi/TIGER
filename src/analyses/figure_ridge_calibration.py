"""Calibration quality of the Platt-scaled Ridge base learners' meta-features
(paper Section 5.5): for each representation family, how well the
sigmoid-calibrated Ridge probabilities (the ones CAWPE actually feeds into
the meta-classifier, via ``calibrate_ridge_family_from_scores``) track
empirical accuracy on held-out test rows.

Reads ``results/cv_ablation_142/fragments/<dataset>/resample<k>/<family>.npz``
(the same cache ``table_family_standalone_strength.py`` uses), taking each
family's ridge block's already-cached OOF ``decision_function`` scores
(``ridge__cv__oof_train``/``ridge__cv__test``, the "cv" arm actually
deployed -- see ``base_learners.py``), calibrating with
``tsfm_ensemble.meta_learners.calibrate_ridge_family_from_scores`` (Platt/
sigmoid, fit per class on the OOF scores, applied to the test scores --
exactly what CAWPE does at meta-classification time), then pooling the
resulting (confidence, correctness) pairs across the full 142-data-set x
30-resample benchmark into a standard reliability diagram and an Expected
Calibration Error (ECE) per family. As with
``table_family_standalone_strength.py``, a (family, dataset, resample) job
missing the ridge "cv" arm is skipped and reported -- see that script's
docstring for why a handful of jobs lack it.

Pooling convention
--------------------------------------------------------------------------
Calibration diagnostics are pooled directly over every test row across all
data sets and resamples (the standard reliability-diagram/ECE definition,
e.g. ``sklearn.calibration.calibration_curve``), NOT per-data-set-then-
across-data-sets like this project's cross-method accuracy comparisons
(see ``src/analyses/common.py``'s module docstring) -- a calibration curve
is a property of the whole predictive distribution, not a per-data-set
metric to be averaged.

Runs one family per invocation (``--family``) and appends bin-level sums to
a shared per-family checkpoint CSV, for the same wall-clock reasons as
``table_family_standalone_strength.py``; rerunning with the same
``--family`` recomputes that family's bins from scratch (cheap, no partial
resume needed since nothing here is required to be idempotent-by-job the
way that other script's per-job CSV is).

Usage (from the repo root, once per family):
    python3 -m src.analyses.figure_ridge_calibration --family convolution
    python3 -m src.analyses.figure_ridge_calibration --family interval
    python3 -m src.analyses.figure_ridge_calibration --family shapelet
    python3 -m src.analyses.figure_ridge_calibration --family dictionary
Then, once all four have run:
    python3 -m src.analyses.figure_ridge_calibration --finalize
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.analyses import common
from src.tsfm_ensemble.base_learners import FAMILIES
from src.tsfm_ensemble.meta_learners import calibrate_ridge_family_from_scores

FRAGMENTS_DIR = common.RESULTS_DIR / "cv_ablation_142" / "fragments"
MANIFEST_142_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"
BINS_CSV = common.RESULTS_DIR / "ridge_calibration_bins.csv"
N_BINS = 15
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


def _bin_family(family: str) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_142_PATH, usecols=["Dataset", "Resample"]).drop_duplicates()
    jobs = list(manifest.itertuples(index=False))

    bin_count = np.zeros(N_BINS)
    bin_conf_sum = np.zeros(N_BINS)
    bin_correct_sum = np.zeros(N_BINS)
    n_done = n_skipped = 0

    for job in jobs:
        path = FRAGMENTS_DIR / job.Dataset / f"resample{job.Resample}" / f"{family}.npz"
        try:
            npz = np.load(path, allow_pickle=True)
            oof_key, test_key = "ridge__cv__oof_train", "ridge__cv__test"
            if oof_key not in npz.files or test_key not in npz.files:
                raise KeyError(f"{oof_key}/{test_key} not in this fragment (missing 'cv' arm for ridge)")
            y_train_enc = np.asarray(npz["y_train_enc"])
            y_test_enc = np.asarray(npz["y_test_enc"])
            classes_ = np.asarray(npz["classes_"])
            n_classes = len(classes_)
            oof_scores = np.asarray(npz[oof_key])
            test_scores = np.asarray(npz[test_key])

            calibrated = calibrate_ridge_family_from_scores(oof_scores, y_train_enc, test_scores, n_classes)
            confidence = calibrated.max(axis=1)
            predicted = calibrated.argmax(axis=1)
            correct = (predicted == y_test_enc).astype(float)

            bin_idx = np.clip(np.digitize(confidence, BIN_EDGES[1:-1]), 0, N_BINS - 1)
            for b in range(N_BINS):
                mask = bin_idx == b
                if mask.any():
                    bin_count[b] += mask.sum()
                    bin_conf_sum[b] += confidence[mask].sum()
                    bin_correct_sum[b] += correct[mask].sum()
            n_done += 1
        except Exception as exc:  # noqa: BLE001 - one bad job must not sink the whole family run
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  skipped {family}/{job.Dataset}/resample{job.Resample}: {exc!r}")

    print(f"family={family}: {n_done} jobs pooled, {n_skipped} skipped.")
    return pd.DataFrame({
        "Family": family,
        "Bin": range(N_BINS),
        "BinLo": BIN_EDGES[:-1],
        "BinHi": BIN_EDGES[1:],
        "Count": bin_count,
        "ConfidenceSum": bin_conf_sum,
        "CorrectSum": bin_correct_sum,
    })


def run_family(family: str) -> None:
    new_rows = _bin_family(family)
    if BINS_CSV.exists():
        existing = pd.read_csv(BINS_CSV)
        existing = existing[existing["Family"] != family]
        out = pd.concat([existing, new_rows], ignore_index=True)
    else:
        out = new_rows
    out.to_csv(BINS_CSV, index=False)
    print(f"wrote {BINS_CSV} ({len(out)} total bin-rows)")


def finalize() -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(BINS_CSV)
    present = set(df["Family"].unique())
    for family in FAMILIES:
        if family not in present:
            raise ValueError(f"No bins for family={family!r} yet -- run that family first.")

    FAMILY_DISPLAY = {"convolution": "KG-MTP", "interval": "QUANT", "shapelet": "RDST", "dictionary": "WEASEL 2.0"}
    FAMILY_COLOR = {"convolution": "#4C72B0", "interval": "#DD8452", "shapelet": "#55A868", "dictionary": "#8172B2"}
    FAMILY_MARKER = {"convolution": "o", "interval": "s", "shapelet": "^", "dictionary": "D"}

    ece_rows = []
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1.2, zorder=1, label="Perfect calibration")

    for family in FAMILIES:
        sub = df[df["Family"] == family].sort_values("Bin")
        total = sub["Count"].sum()
        ece = float((sub["Count"] * (sub["ConfidenceSum"] / sub["Count"].replace(0, np.nan) -
                                      sub["CorrectSum"] / sub["Count"].replace(0, np.nan)).abs()).sum() / total)
        ece_rows.append({"Family": family, "ECE": ece, "N pooled test predictions": int(total)})

        populated = sub[sub["Count"] > 0]
        mean_conf = populated["ConfidenceSum"] / populated["Count"]
        mean_acc = populated["CorrectSum"] / populated["Count"]
        ax.plot(
            mean_conf, mean_acc, color=FAMILY_COLOR[family], marker=FAMILY_MARKER[family],
            markersize=5, linewidth=1.6, label=FAMILY_DISPLAY[family], zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence (calibrated Ridge, per bin)")
    ax.set_ylabel("Empirical accuracy (per bin)")
    ax.set_title("Reliability of the Platt-calibrated Ridge meta-features")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, color="0.9", linewidth=0.6, zorder=0)
    fig.tight_layout()
    common.save_figure(fig, "ridge_calibration")

    ece_df = pd.DataFrame(ece_rows).set_index("Family").loc[list(FAMILIES)]
    print(ece_df.round(4).to_string())
    common.save_table(
        ece_df, "ridge_calibration_ece",
        float_format="%.4f",
        caption=(
            "Expected Calibration Error (15 equal-width confidence bins) of the Platt-scaled Ridge "
            "meta-features, per representation family, pooled over every test row across the full "
            "142-data-set benchmark (30 resamples each)."
        ),
        label="tab:ridge-calibration-ece",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", choices=FAMILIES)
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        finalize()
    else:
        run_family(args.family)


if __name__ == "__main__":
    main()
