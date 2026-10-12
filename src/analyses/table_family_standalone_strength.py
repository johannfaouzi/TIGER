"""Each representation family's own standalone predictive strength before
stacking (paper Section 5.5): TIGER's own three base learners (ridge,
extra_trees, naive_bayes, all on the "cv" out-of-fold arm -- the one
actually deployed, see ``base_learners.py``) combined by a
within-family hard majority vote (reusing
``tsfm_ensemble.meta_learners.hard_majority_vote`` verbatim, the same
combinator TIGER's own meta-classification stage uses across all twelve
base learners -- here restricted to one family's three), evaluated on
the full 142-data-set benchmark.

Deliberately NOT ``results/family_strengths/family_strengths_full.csv``
--------------------------------------------------------------------------
That file is a proxy built from the published KG-MTP/QUANT/RDST/WEASEL 2.0
papers' own reported numbers (see ``analyze_family_strengths.py``'s own
docstring) -- useful for fast dev-subset curation, but not this project's
own reimplementations' actual performance. This script instead reads
``results/cv_ablation_142/fragments/<dataset>/resample<k>/<family>.npz``,
already-cached OOF/test ``decision_function``/``predict_proba`` scores
from TIGER's own pipeline at the full 142-dataset x 30-resample scale,
so every number here is what TIGER's own code actually produces.

Why hard majority vote, not probability averaging
--------------------------------------------------------------------------
Ridge's ``decision_function`` margins are not on a comparable scale to
the other two learners' genuine probabilities (the same reason TIGER's
own meta-classification stage uses hard majority vote as its
non-learned baseline rather than probability averaging -- see
``hard_majority_vote``'s own docstring). Reusing that exact function
here (rather than a new probability-averaging combinator) keeps this
analysis on the same footing.

A (family, dataset, resample) job whose "cv" arm is missing for one of
the three learners (cv_ablation_142's own fragment occasionally only has
the bagging arms for a pathological dataset -- see that harness's own
docstring) is skipped and reported, the same "one bad job must not sink
the whole analysis" convention this project uses throughout.

Runs one family per invocation (``--family``) and appends to a shared
per-job CSV, since one full pass over all four families' ~17,000 jobs
comfortably exceeds a single shell call's time budget on this machine's
fuse-mounted results/ mount -- rerunning with the same ``--family``
resumes by skipping (dataset, resample) pairs already present in the
output CSV.

Usage (from the repo root, once per family):
    python3 -m src.analyses.table_family_standalone_strength --family convolution
    python3 -m src.analyses.table_family_standalone_strength --family interval
    python3 -m src.analyses.table_family_standalone_strength --family shapelet
    python3 -m src.analyses.table_family_standalone_strength --family dictionary
Then, once all four have run:
    python3 -m src.analyses.table_family_standalone_strength --finalize
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.analyses import common
from src.tsfm_ensemble.base_learners import FAMILIES, LEARNER_METHOD
from src.tsfm_ensemble.meta_learners import ColumnBlock, MetaFeatures, hard_majority_vote

FRAGMENTS_DIR = common.RESULTS_DIR / "cv_ablation_142" / "fragments"
MANIFEST_142_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"
PER_JOB_CSV = common.RESULTS_DIR / "family_standalone_strength_per_job.csv"
LEARNERS = ("ridge", "extra_trees", "naive_bayes")
ARM = "cv"  # the out-of-fold arm actually deployed for all twelve (family, learner) pairs -- base_learners.py
CSV_FIELDS = ["Family", "Dataset", "Resample", "Accuracy", "BalancedAccuracy"]


def _family_meta(npz, family: str) -> MetaFeatures:
    classes_ = np.asarray(npz["classes_"])
    y_train_enc = np.asarray(npz["y_train_enc"])
    y_test_enc = np.asarray(npz["y_test_enc"])
    n_classes = len(classes_)

    train_blocks, test_blocks, blocks = [], [], []
    col = 0
    for learner in LEARNERS:
        oof_key = f"{learner}__{ARM}__oof_train"
        test_key = f"{learner}__{ARM}__test"
        if oof_key not in npz.files or test_key not in npz.files:
            raise KeyError(f"{oof_key}/{test_key} not in this fragment (missing '{ARM}' arm for {learner})")
        train_blocks.append(np.asarray(npz[oof_key]))
        test_blocks.append(np.asarray(npz[test_key]))
        blocks.append(ColumnBlock(family, learner, LEARNER_METHOD[learner], col, col + n_classes))
        col += n_classes

    X_train = np.concatenate(train_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)
    return MetaFeatures(X_train, X_test, y_train_enc, y_test_enc, classes_, blocks)


def run_family(family: str) -> None:
    manifest = pd.read_csv(MANIFEST_142_PATH, usecols=["Dataset", "Resample"]).drop_duplicates()
    jobs = list(manifest.itertuples(index=False))

    already_done = set()
    if PER_JOB_CSV.exists():
        existing = pd.read_csv(PER_JOB_CSV)
        already_done = set(
            existing.loc[existing["Family"] == family, ["Dataset", "Resample"]].itertuples(index=False, name=None)
        )

    write_header = not PER_JOB_CSV.exists()
    n_done = n_skipped = 0
    with open(PER_JOB_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for job in jobs:
            if (job.Dataset, job.Resample) in already_done:
                continue
            path = FRAGMENTS_DIR / job.Dataset / f"resample{job.Resample}" / f"{family}.npz"
            try:
                npz = np.load(path, allow_pickle=True)
                meta = _family_meta(npz, family)
                preds = hard_majority_vote(meta)
                row = {
                    "Family": family,
                    "Dataset": job.Dataset,
                    "Resample": job.Resample,
                    "Accuracy": accuracy_score(meta.y_test_enc, preds),
                    "BalancedAccuracy": balanced_accuracy_score(meta.y_test_enc, preds),
                }
                writer.writerow(row)
                n_done += 1
            except Exception as exc:  # noqa: BLE001 - one bad job must not sink the whole family run
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  skipped {family}/{job.Dataset}/resample{job.Resample}: {exc!r}")
    print(f"family={family}: {n_done} new rows written, {n_skipped} skipped, {len(already_done)} already done.")


def finalize() -> None:
    df = pd.read_csv(PER_JOB_CSV)
    counts = df.groupby("Family").size()
    print("Rows per family:")
    print(counts.to_string())
    for family in FAMILIES:
        if counts.get(family, 0) == 0:
            raise ValueError(f"No rows for family={family!r} yet -- run that family first.")

    per_dataset = df.groupby(["Family", "Dataset"], as_index=False)[["Accuracy", "BalancedAccuracy"]].mean()
    wide_acc = per_dataset.pivot(index="Dataset", columns="Family", values="Accuracy")
    wide_balacc = per_dataset.pivot(index="Dataset", columns="Family", values="BalancedAccuracy")

    pooled = pd.DataFrame({
        "Mean accuracy": wide_acc.mean(),
        "Mean balanced accuracy": wide_balacc.mean(),
        "Data sets won (accuracy)": wide_acc.eq(wide_acc.max(axis=1), axis=0).sum(),
    }).loc[list(FAMILIES)]
    pooled.index.name = "Family"

    print()
    print(pooled.round(4).to_string())

    common.save_table(
        pooled, "family_standalone_strength",
        float_format="%.4f",
        caption=(
            "Each representation family's own standalone predictive strength before stacking: TIGER's "
            "own three base learners (ridge, extra trees, naive Bayes, all on the deployed ``cv'' "
            "out-of-fold arm) combined by a within-family hard majority vote, on the full 142-data-set "
            "benchmark (per-data-set means over 30 resamples, averaged across data sets). "
            "``Data sets won'' counts the data sets on which that family's own accuracy is highest "
            "among the four (ties counted for every tied family)."
        ),
        label="tab:family-standalone-strength",
    )

    per_dataset_out = per_dataset.pivot(index="Dataset", columns="Family", values="Accuracy")[list(FAMILIES)]
    per_dataset_out.to_csv(common.RESULTS_DIR / "family_standalone_strength_per_dataset.csv")
    print("\nPer-data-set accuracy also written to results/family_standalone_strength_per_dataset.csv")


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
