"""Oracle headroom above TIGER's own adaptive meta-classification rule
(paper Section 5.4.3).

Reads ``results/meta_model_predictions_142/manifest.csv`` directly (the
same manifest ``table_literature_comparison.py`` and
``figure_critical_difference.py`` already use) and reports, aggregated the
same way as every other cross-method comparison in this project (see
``common.py``'s own "Aggregation convention" docstring section: each
data set's own resamples are averaged first, then that per-data-set
series is averaged across the 142 data sets), the mean accuracy of:

    oracle_cv           : the single best base learner, chosen
                           independently per (data set, resample) by
                           cross-validated OOF accuracy -- a hindsight
                           oracle restricted to TIGER's twelve base
                           learners, not its meta-classification stage
                           at all.
    auto_selected        : TIGER's own deployed adaptive rule -- the
                           threshold tau (Section 3.2) choosing, per
                           (data set, resample), between weighted hard
                           majority vote and TabICLv2.
    two_method_oracle    : a hindsight oracle restricted to exactly
                           those same two candidates
                           (max(cawpe_hard_vote, tabiclv2) per row) --
                           the best auto_selected *could* do if tau were
                           replaced by a perfectly-informed rule but the
                           two candidates stayed fixed. Not itself a
                           column in the manifest (computed here);
                           included as a diagnostic, splitting the gap
                           between TIGER's rule and the six-method oracle
                           below into "tau's own imperfection" vs. "only
                           2 of 6 candidates are ever considered".
    six_method_oracle    : the full hindsight oracle across all six
                           candidate meta-classification methods from
                           Section 5.1 (hard_majority_vote,
                           logistic_regression, lightgbm, cawpe,
                           cawpe_hard_vote, tabiclv2), computed here via
                           ``common.six_method_oracle`` -- a row-wise max
                           over those six methods' own accuracy. NOT the
                           manifest's own "oracle" column: that column is
                           actually ``oracle_best_single_learner``, the
                           best of TIGER's twelve *base learners* by test
                           hindsight, a different and, as it turns out,
                           higher-scoring diagnostic (see
                           ``best_single_base_learner`` below and
                           ``common.py``'s own caveat next to
                           ``SIX_META_METHODS``). An earlier version of
                           this script conflated the two; this version
                           reports both, correctly labeled.
    best_single_base_learner : kept for reference/continuity with that
                           earlier, mislabeled version -- this IS what the
                           manifest's "oracle" column contains: the best
                           of TIGER's twelve base learners, chosen by
                           test-set hindsight, with no meta-classification
                           stage (no stacking) at all. It happens to score
                           higher than the six-method meta-oracle above,
                           which is itself worth noting in the text: on
                           this benchmark, none of the six realistic
                           combiners is a perfect oracle over the twelve
                           base learners either, so a single, very strong
                           base learner picked with hindsight can and does
                           occasionally beat the best any of the six
                           combiners manages on the same data set.

Usage (from the repo root):
    python3 -m src.analyses.table_oracle_headroom
"""
from __future__ import annotations

import pandas as pd

from src.analyses import common

MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"

DISPLAY_NAMES = {
    "oracle_cv": "Best base learner (CV-picked, hindsight)",
    "auto_selected": "TIGER's adaptive rule (deployed)",
    "two_method_oracle": "Weighted hard majority vote / TabICLv2 oracle (hindsight)",
    "six_method_oracle": "Six-method oracle (hindsight)",
    "best_single_base_learner": "Best single base learner (test-set hindsight, no stacking)",
}
ROW_ORDER = ["oracle_cv", "auto_selected", "two_method_oracle", "six_method_oracle", "best_single_base_learner"]


def main() -> None:
    df = common.load_manifest(MANIFEST_PATH, methods=common.ALL_OWN_METHODS)
    wide = common.per_dataset_mean(df, "Accuracy")

    wide["two_method_oracle"] = wide[["cawpe_hard_vote", "tabiclv2"]].max(axis=1)
    wide["six_method_oracle"] = common.six_method_oracle(wide)
    wide["best_single_base_learner"] = wide["oracle"]  # manifest's "oracle" column -- see docstring above

    means = wide[ROW_ORDER].mean()
    six_oracle_mean = means["six_method_oracle"]

    summary = pd.DataFrame({
        "Mean accuracy": means,
        "Gap to six-method oracle": six_oracle_mean - means,
    })
    summary.index = [DISPLAY_NAMES[m] for m in summary.index]
    summary.index.name = "Method"

    print(summary.round(4).to_string())

    # Sanity check: auto_selected's mean accuracy here must match the
    # 0.884 already reported for TIGER/auto_selected in Table 6
    # (tab:multi-metric-142) -- same manifest, same aggregation.
    assert abs(means["auto_selected"] - 0.884) < 0.001, (
        f"auto_selected mean accuracy ({means['auto_selected']:.4f}) no longer matches "
        "Table 6's reported 0.884 -- re-check both before trusting either."
    )

    common.save_table(
        summary, "oracle_headroom",
        caption=(
            "Mean accuracy (per-data-set means over 30 resamples, averaged across the 142 data sets) of "
            "TIGER's deployed adaptive meta-classification rule against four hindsight references. The "
            "first is the best of its own two constituent methods (weighted hard majority vote, TabICLv2) "
            "chosen per data set after the fact. The second is the best of all six candidate "
            "meta-classification methods from Section~5.1, chosen the same way. The third is the best "
            "single base learner, chosen per data set by cross-validated out-of-fold accuracy alone (no "
            "meta-classification stage). The fourth is the best single base learner chosen by test-set "
            "hindsight instead, an unrealizable upper bound included for reference. ``Gap to six-method "
            "oracle'' is the six-method oracle's own mean accuracy minus each row's."
        ),
        label="tab:oracle-headroom",
    )


if __name__ == "__main__":
    main()
