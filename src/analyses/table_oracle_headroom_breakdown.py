"""Per-method breakdown of the six-method oracle's tiny headroom gain over
the two-method (weighted hard majority vote / TabICLv2) oracle (paper
Section 5.5.1): for each of the four meta-classification methods TIGER
does not deploy (hard majority vote, logistic regression, LightGBM,
CAWPE), how many of the 142 data sets it is the outright best of all six
candidates on, and by how much it then beats the two-method oracle.

Motivated by a factual check on the paper's own text: an earlier draft
claimed these four methods are "each, individually, essentially never
the uniquely best choice" once weighted hard majority vote and TabICLv2
are available. That is not what the per-data-set win counts below show
(each of the four wins outright on a meaningful number of data sets) --
what IS true, and is what table_oracle_headroom.py's own 0.0003
six-vs-two-method gap actually reflects, is that each win's margin over
the two-method oracle is tiny. This script makes that mechanism explicit
and table-backed instead of leaving it as an unverified inference drawn
from the aggregate gap alone.

Reads ``results/meta_model_predictions_142/manifest.csv`` directly (same
source as ``table_oracle_headroom.py``), aggregated the same way (each
data set's own resamples averaged first, see ``common.py``'s own
"Aggregation convention" docstring section).

"Data sets won": per data set, whichever of the six candidate methods
(from Section 5.1) has the highest mean accuracy; ties counted for every
tied method (same convention as ``table_family_standalone_strength.py``).
"Mean/max margin": that method's own accuracy minus the two-method
(weighted hard majority vote / TabICLv2) oracle's accuracy, on the data
sets where this method wins -- i.e. how much the six-method oracle gains
over the two-method oracle specifically because of this method's wins.

Usage (from the repo root):
    python3 -m src.analyses.table_oracle_headroom_breakdown
"""
from __future__ import annotations

import pandas as pd

from src.analyses import common

MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"

SIX_METHODS = common.SIX_META_METHODS
TWO_METHODS = ("cawpe_hard_vote", "tabiclv2")
OTHER_FOUR = ("cawpe", "hard_majority_vote", "lightgbm", "logistic_regression")

DISPLAY_NAMES = {
    "cawpe": "CAWPE",
    "hard_majority_vote": "Hard majority vote",
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic regression",
}


def main() -> None:
    df = common.load_manifest(MANIFEST_PATH, methods=common.ALL_OWN_METHODS)
    wide = common.per_dataset_mean(df, "Accuracy")[list(SIX_METHODS)]

    six_max = wide.max(axis=1)
    two_max = wide[list(TWO_METHODS)].max(axis=1)
    margin = six_max - two_max
    is_won = wide.eq(six_max, axis=0)

    # Sanity check: must match table_oracle_headroom.py's own reported gap
    # (six-method oracle mean accuracy minus two-method oracle mean accuracy).
    total_gain = margin.clip(lower=0).mean()
    assert abs(total_gain - 0.0003) < 0.0001, (
        f"six-vs-two-method oracle gain ({total_gain:.4f}) no longer matches "
        "table_oracle_headroom.py's reported 0.0003 -- re-check both before trusting either."
    )

    rows = []
    for method in OTHER_FOUR:
        won = is_won[method]
        n = int(won.sum())
        rows.append({
            "Method": DISPLAY_NAMES[method],
            "Data sets won": n,
            "Mean margin": margin[won].mean() if n else float("nan"),
            "Max margin": margin[won].max() if n else float("nan"),
        })
    table = pd.DataFrame(rows).set_index("Method").sort_values("Data sets won", ascending=False)

    print(table.round(4).to_string())
    print(f"\n{(margin > 1e-12).sum()} of 142 data sets have margin > 0 "
          "(six-method oracle beats two-method oracle).")

    common.save_table(
        table, "oracle_headroom_breakdown",
        float_format="%.4f",
        caption=(
            "Per-data-set breakdown of the six-method oracle's own gain over the two-method (weighted hard "
            "majority vote / TabICLv2) oracle of Table~\\ref{tab:oracle-headroom}, for the four "
            "meta-classification methods TIGER does not deploy. ``Data sets won'' counts the 142 data sets "
            "on which that method's own mean accuracy is the highest of all six candidates (ties counted for "
            "every tied method). ``Mean/max margin'' is that method's own accuracy minus the two-method "
            "oracle's, on the data sets it wins."
        ),
        label="tab:oracle-headroom-breakdown",
    )


if __name__ == "__main__":
    main()
