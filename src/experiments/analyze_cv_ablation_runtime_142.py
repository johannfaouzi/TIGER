"""Runtime companion to analyze_cv_ablation_142.py: for each of the twelve
(family, learner) pairs and each candidate OOF estimator arm (cv,
bagB15, bagB25 for ridge/naive_bayes; cv, nativeoob for extra_trees),
report the mean wall-clock seconds per unit, over the same qualifying
row population analyze_cv_ablation_142.py's honesty comparison uses
(that arm's own Error column NaN, and its OOFAcc/TestAcc both present).
Answers: is cross-validation, already the more honest OOF estimator for
11 of 12 pairs (see cv_ablation_142/chosen_arms.json), also the cheaper
one?
"""
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path.home() / "mnt" / "tsc-with-tfm" / "results" / "cv_ablation_142"
MANIFEST = RESULTS_DIR / "manifest.csv"

FAMILIES = ["convolution", "interval", "shapelet", "dictionary"]
LEARNER_NAMES = ["ridge", "extra_trees", "naive_bayes"]
ARM_COLUMN_PREFIX = {"cv": "Cv", "bagB15": "Bag15", "bagB25": "Bag25", "nativeoob": "NativeOOB"}
CANDIDATE_ARMS = {
    "ridge": ("cv", "bagB15", "bagB25"),
    "naive_bayes": ("cv", "bagB15", "bagB25"),
    "extra_trees": ("cv", "nativeoob"),
}

usecols = (
    ["Dataset", "Resample", "Family", "Learner"]
    + [f"{p}{suf}" for p in ["Cv", "Bag15", "Bag25", "NativeOOB"] for suf in ["OOFAcc", "TestAcc", "Seconds", "Error"]]
)
manifest = pd.read_csv(MANIFEST, usecols=lambda c: c in usecols)


def arm_runtime(family, learner, arm):
    prefix = ARM_COLUMN_PREFIX[arm]
    sub = manifest[(manifest["Family"] == family) & (manifest["Learner"] == learner)]
    error_col, oof_col, test_col, sec_col = f"{prefix}Error", f"{prefix}OOFAcc", f"{prefix}TestAcc", f"{prefix}Seconds"
    if error_col not in sub.columns:
        sub = sub.assign(**{error_col: np.nan})
    if oof_col not in sub.columns or test_col not in sub.columns:
        return {"mean": float("nan"), "n_rows": 0}
    valid = sub[sub[error_col].isna() & sub[oof_col].notna() & sub[test_col].notna()]
    if valid.empty or sec_col not in valid.columns:
        return {"mean": float("nan"), "n_rows": 0}
    return {"mean": float(valid[sec_col].mean()), "n_rows": int(len(valid))}


print(f"{'Family':12s} {'Learner':12s} {'Arm':10s} {'Mean s/unit':>12s} {'n':>6s}")
for family in FAMILIES:
    for learner in LEARNER_NAMES:
        for arm in CANDIDATE_ARMS[learner]:
            r = arm_runtime(family, learner, arm)
            print(f"{family:12s} {learner:12s} {arm:10s} {r['mean']:12.4f} {r['n_rows']:6d}")
    print()

# Speed ratio: is cv faster than every other candidate, per (family, learner)?
print("=== CV vs. alternatives: speed ratio (alternative / cv) ===")
for family in FAMILIES:
    for learner in LEARNER_NAMES:
        cv = arm_runtime(family, learner, "cv")
        for arm in CANDIDATE_ARMS[learner]:
            if arm == "cv":
                continue
            alt = arm_runtime(family, learner, arm)
            ratio = alt["mean"] / cv["mean"] if cv["mean"] else float("nan")
            print(f"  {family:12s} {learner:12s} cv={cv['mean']:.4f}s  {arm}={alt['mean']:.4f}s  ratio={ratio:.2f}x")
