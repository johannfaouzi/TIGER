"""TabICLv2 native vs. ECOC many-class handling (paper Section 5.4.2).

Reads ``results/compare_tabiclv2_multiclass_handling/manifest.csv``
directly -- already fully computed by
``compare_tabiclv2_multiclass_handling.py`` on the dev subset's own
>10-class data sets (see that module's docstring). This script does not
recompute anything, it only reshapes that manifest into the paper's table
and picks the resulting default the way
``build_meta_model_predictions.py``'s own ``--tabiclv2-ecoc auto`` mode
does (whichever mode has the higher pooled mean accuracy).

Usage (from the repo root):
    python3 -m src.analyses.table_multiclass_handling
"""
from __future__ import annotations

import pandas as pd

from src.analyses import common

MANIFEST_PATH = common.RESULTS_DIR / "compare_tabiclv2_multiclass_handling" / "manifest.csv"


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)
    if "Error" in df.columns:
        errored = df[df["Error"].notna()]
        if len(errored):
            raise ValueError(f"{MANIFEST_PATH}: {len(errored)} row(s) carry a recorded Error -- resolve before analyzing.")

    df["Mode"] = df["Ecoc"].map({False: "Native", True: "ECOC"})

    per_dataset = (
        df.groupby(["Dataset", "NClasses", "Mode"])
        .agg(Accuracy=("Accuracy", "mean"), Seconds=("FitPredictSeconds", "mean"))
        .unstack("Mode")
    )
    per_dataset.columns = [f"{metric} ({mode})" for metric, mode in per_dataset.columns]
    per_dataset = per_dataset.sort_values("NClasses")

    pooled = df.groupby("Mode")["Accuracy"].mean()
    pooled_seconds = df.groupby("Mode")["Seconds" if "Seconds" in df.columns else "FitPredictSeconds"].mean() \
        if "Seconds" in df.columns else df.groupby("Mode")["FitPredictSeconds"].mean()
    default_mode = pooled.idxmax()

    print(per_dataset.round(4).to_string())
    print()
    print("Pooled mean accuracy:")
    print(pooled.round(4).to_string())
    print()
    print("Pooled mean fit+predict seconds:")
    print(pooled_seconds.round(1).to_string())
    print(f"\nResulting default (higher pooled accuracy): {default_mode}")

    common.save_table(
        per_dataset, "multiclass_handling",
        float_format="%.4f",
        caption=(
            "TabICLv2's native many-class handling against its ECOC-wrapped variant "
            "(``many\\_class'' backend), on the six development-subset data sets with more than ten "
            "classes, averaged over 5 resamples each. Seconds is the mean fit-and-predict wall-clock "
            "time per (data set, resample) job."
        ),
        label="tab:multiclass-handling",
    )


if __name__ == "__main__":
    main()
