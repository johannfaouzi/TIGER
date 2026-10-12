"""
Combine three already-measured manifests into TIGER's own training-time /
inference-time split, per (Dataset, Resample), over the full 142-data-set
benchmark (142 x 30 resamples), matching the conventions already used for
Table~\ref{tab:runtimes} (median of per-data-set medians, same hardware).

Training time, per (Dataset, Resample):
    + feature extraction on the training set, once per family (4 families;
      NOT once per (family, learner) -- the raw per-row manifest duplicates
      this value across a family's 3 learners, so it must be summed only
      over the 4 distinct families, not all 12 rows)
    + out-of-fold / cross-validation fitting, for all 12 (family, learner)
      base learners -- this is what actually builds the meta-feature matrix
      the meta-classifier is trained on
    + the final fit on the *entire* training set, for all 12 base learners
      -- this is the single fitted model actually used at inference, and
      preparing it is part of training, not inference
    + the selected meta-classifier's own fit time (weighted hard majority
      vote -- "cawpe_hard_vote" in the code -- or TabICLv2, chosen via the
      same tau = 60.83 threshold on mean training samples per class used by
      the deployed pipeline)

Inference time, per (Dataset, Resample):
    + feature extraction on the test set, once per family
    + the final fitted model's own predict time, for all 12 base learners
    + the selected meta-classifier's own predict time

Sources (all already-measured, nothing re-run here):
    results/measure_base_learner_final_split_142/manifest.csv
        -- ExtractionTrainSeconds, ExtractionTestSeconds,
           FinalFitSeconds, FinalPredictSeconds
    results/cv_ablation_142/manifest.csv
        -- CvOOFSeconds (the OOF/CV fitting time missing from the manifest
           above by design; see that script's own docstring). 360 of its
           51,120 rows are NaN (ridge on 3 singleton-class data sets x 30
           resamples x 4 families) and are treated here as a zero
           contribution, per the same convention these rows already get
           elsewhere in the paper (excluded from the accuracy numbers too).
    results/measure_meta_classification_split_142/manifest.csv
        -- FitSeconds, PredictSeconds for both candidate methods
           (cawpe_hard_vote, tabiclv2) at every (Dataset, Resample); the
           deployed one is picked out here via the tau rule.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path.home() / "mnt" / "tsc-with-tfm" / "results"
TAU = 60.83

BASE_MANIFEST = RESULTS_DIR / "measure_base_learner_final_split_142" / "manifest.csv"
CV_MANIFEST = RESULTS_DIR / "cv_ablation_142" / "manifest.csv"
META_MANIFEST = RESULTS_DIR / "measure_meta_classification_split_142" / "manifest.csv"

OUT_DIR = RESULTS_DIR / "tiger_train_inference_split_142"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize(per_dataset_medians: pd.Series, label: str) -> None:
    print(
        f"{label}: {per_dataset_medians.median():.1f} "
        f"[{per_dataset_medians.min():.1f}, {per_dataset_medians.max():.1f}] "
        f"(s, median [min, max] over {len(per_dataset_medians)} data sets)"
    )


def main() -> None:
    base = pd.read_csv(BASE_MANIFEST)
    cv = pd.read_csv(
        CV_MANIFEST,
        usecols=["Dataset", "Resample", "Family", "Learner", "CvOOFSeconds"],
    )
    meta = pd.read_csv(META_MANIFEST)

    # --- base learners: join in the OOF/CV time the final-split manifest
    # deliberately excluded, then fill the 360 documented ridge-on-
    # singleton-class NaNs with zero, per Johann's instruction.
    merged = base.merge(
        cv, on=["Dataset", "Resample", "Family", "Learner"], how="left", validate="one_to_one"
    )
    n_nan = merged["CvOOFSeconds"].isna().sum()
    assert n_nan == 360, f"expected exactly 360 NaN CvOOFSeconds rows, got {n_nan}"
    merged["CvOOFSeconds"] = merged["CvOOFSeconds"].fillna(0.0)

    # Extraction time is duplicated across a family's 3 learner rows in the
    # raw manifest (confirmed: at most 1 distinct value per (Dataset,
    # Resample, Family)), so it must be summed once per family, not once
    # per (family, learner) row.
    extraction = (
        merged.drop_duplicates(["Dataset", "Resample", "Family"])
        .groupby(["Dataset", "Resample"])[["ExtractionTrainSeconds", "ExtractionTestSeconds"]]
        .sum()
    )

    base_learner_totals = merged.groupby(["Dataset", "Resample"])[
        ["CvOOFSeconds", "FinalFitSeconds", "FinalPredictSeconds"]
    ].sum()

    per_unit = extraction.join(base_learner_totals)

    # --- meta-classification: pick the method the deployed pipeline would
    # actually select for each (Dataset, Resample), via the same tau rule
    # used in Section 3.2.
    meta = meta.copy()
    meta["MeanSamplesPerClass"] = meta["NTrain"] / meta["NClasses"]
    meta["SelectedMethod"] = np.where(
        meta["MeanSamplesPerClass"] >= TAU, "tabiclv2", "cawpe_hard_vote"
    )
    selected = meta[meta["Method"] == meta["SelectedMethod"]].set_index(
        ["Dataset", "Resample"]
    )[["FitSeconds", "PredictSeconds", "Method"]]
    selected = selected.rename(
        columns={
            "FitSeconds": "MetaFitSeconds",
            "PredictSeconds": "MetaPredictSeconds",
            "Method": "SelectedMetaMethod",
        }
    )

    combined = per_unit.join(selected, how="inner")
    assert len(combined) == len(per_unit) == len(selected), (
        "(Dataset, Resample) coverage mismatch between the base-learner/CV "
        "join and the meta-classification manifest"
    )

    combined["TrainingSeconds"] = (
        combined["ExtractionTrainSeconds"]
        + combined["CvOOFSeconds"]
        + combined["FinalFitSeconds"]
        + combined["MetaFitSeconds"]
    )
    combined["InferenceSeconds"] = (
        combined["ExtractionTestSeconds"]
        + combined["FinalPredictSeconds"]
        + combined["MetaPredictSeconds"]
    )
    combined["TotalSeconds"] = combined["TrainingSeconds"] + combined["InferenceSeconds"]

    combined = combined.reset_index()
    out_path = OUT_DIR / "manifest.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows to {out_path}\n")

    # --- collapse resamples to each data set's own median, then report
    # median/min/max across the 142 per-data-set medians, exactly as
    # Table~\ref{tab:runtimes} already does for the combined number.
    per_dataset = combined.groupby("Dataset")[
        ["TrainingSeconds", "InferenceSeconds", "TotalSeconds"]
    ].median()
    assert len(per_dataset) == 142, f"expected 142 data sets, got {len(per_dataset)}"

    summarize(per_dataset["TrainingSeconds"], "Training")
    summarize(per_dataset["InferenceSeconds"], "Inference")
    summarize(per_dataset["TotalSeconds"], "Total (train+test, this script)")
    print(
        "\nFor comparison, Table~\\ref{tab:runtimes}'s existing TIGER row: "
        "157.0 [20.7, 13602.8]"
    )

    per_dataset_path = OUT_DIR / "per_dataset_medians.csv"
    per_dataset.to_csv(per_dataset_path)
    print(f"\nWrote per-data-set medians to {per_dataset_path}")


if __name__ == "__main__":
    main()
