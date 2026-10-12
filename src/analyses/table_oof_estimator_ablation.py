"""Ablation: cross-validation vs. bagging-out-of-bag vs. native-OOB as
TIGER's per-(family, learner) out-of-fold train-score estimator (paper
Section 5.4.1).

Reads ``results/cv_ablation_142/chosen_arms.json`` directly -- already
computed by ``cv_ablation_142.py``'s own finalize step: for every (family,
learner) pair, the mean absolute gap between each candidate arm's own
out-of-fold (OOF) accuracy and the true held-out test accuracy, over every
qualifying (data set, resample) row of the 142-data-set benchmark (an arm
"qualifies" a row only if it has >=100 such rows overall -- see that
script's own selection logic) -- the smaller this gap, the more honest
(less optimistic) that arm's own OOF accuracy is as a predictor of
test-time performance. This script does not recompute anything; it just
reshapes that JSON into the paper's table. See this repo's own
``base_learners.py`` for why the *deployed* pipeline nonetheless uses "cv"
uniformly for every one of the twelve pairs (a known, flagged discrepancy
with this ablation's own per-pair recommendation for convolution/naive_bayes
-- see this script's printed warning below).

Usage (from the repo root):
    python3 -m src.analyses.table_oof_estimator_ablation
"""
from __future__ import annotations

import json

import pandas as pd

from src.analyses import common

CHOSEN_ARMS_PATH = common.RESULTS_DIR / "cv_ablation_142" / "chosen_arms.json"

ARM_ORDER = ["cv", "bagB15", "bagB25", "nativeoob"]
ARM_DISPLAY = {"cv": "CV", "bagB15": "Bag(15)", "bagB25": "Bag(25)", "nativeoob": "Native OOB"}
FAMILY_ORDER = ["convolution", "interval", "shapelet", "dictionary"]
LEARNER_ORDER = ["ridge", "extra_trees", "naive_bayes"]
FAMILY_DISPLAY = {"convolution": "KG-MTP", "interval": "QUANT", "shapelet": "RDST", "dictionary": "WEASEL 2.0"}
LEARNER_DISPLAY = {"ridge": "Ridge", "extra_trees": "Extra Trees", "naive_bayes": "Naive Bayes"}

# base_learners.py always uses "cv" for every one of the twelve pairs (see
# that module's own docstring/deployed logic) -- this ablation's own
# recommendation differs for this one pair. Flagged, not silently
# swallowed: see the printed warning in main() below.
DEPLOYED_ARM = "cv"


def main() -> None:
    with open(CHOSEN_ARMS_PATH) as f:
        chosen_arms = json.load(f)

    rows = []
    mismatches = []
    for family in FAMILY_ORDER:
        for learner in LEARNER_ORDER:
            entry = chosen_arms[family][learner]
            row = {"Family": FAMILY_DISPLAY[family], "Learner": LEARNER_DISPLAY[learner]}
            for arm in ARM_ORDER:
                cand = entry["candidates"].get(arm)
                row[ARM_DISPLAY[arm]] = cand["mean"] if cand is not None else float("nan")
            row["Chosen"] = ARM_DISPLAY[entry["chosen_arm"]]
            rows.append(row)
            if entry["chosen_arm"] != DEPLOYED_ARM:
                mismatches.append(
                    f"  {family}/{learner}: ablation recommends {entry['chosen_arm']!r}, "
                    f"deployed base_learners.py uses {DEPLOYED_ARM!r}"
                )

    df = pd.DataFrame(rows).set_index(["Family", "Learner"])
    print(df.round(4).to_string())
    print()
    print(f"CV chosen for {sum(1 for r in rows if r['Chosen'] == 'CV')}/12 pairs.")
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} pair(s) where this ablation's own recommendation "
              f"differs from what the deployed pipeline actually uses:")
        print("\n".join(mismatches))

    common.save_table(
        df.drop(columns="Chosen"), "oof_estimator_ablation",
        caption=(
            "Mean absolute gap between each candidate out-of-fold estimator's own OOF accuracy and the "
            "corresponding held-out test accuracy, over every qualifying (data set, resample) row of the "
            "142-data-set benchmark, for each of TIGER's twelve (family, learner) pairs (smaller is more "
            "honest). CV: k-fold cross-validation. Bag(15)/Bag(25): 15-/25-model bagging out-of-bag. "
            "Native OOB: Extra Trees' own built-in out-of-bag estimate."
        ),
        label="tab:oof-estimator-ablation",
    )


if __name__ == "__main__":
    main()
