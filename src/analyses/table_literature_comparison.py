"""Compare every one of this project's own meta-methods against five
external time series classifiers -- RDST, QUANT, WEASEL 2.0, KG-MTP (all
four run locally, same 142 datasets/30 resamples:
``results/run_142_original_algorithms/manifest.csv``) and HC2 (never run
in this repo -- published numbers from ``comparisons/bake-off-redux-2023/``,
since aeon 1.6.0 left HC2's own file logic-unchanged from 1.5.0 but sped
up its component algorithms; see this repo's own aeon-version-impact
investigation).

This is ``results/meta_model_predictions_142/analysis_report.md``
Section 5, reproduced as re-runnable code -- that comparison previously
existed only as hand-written prose; the ``_eq``/``_nmv`` dataset-name
alignment it depended on was scattered across several experiment scripts
and never assembled into a standalone, checked comparison. This script
also reproduces that section's own sanity check: the three algorithms
present in both the local run and the published bake-off (QUANT,
WEASEL2.0, RDST) should agree with the published numbers to within a few
thousandths of mean accuracy -- if aeon's own RDST/WEASEL2.0 implementation
had drifted since the bake-off, or this repo's own runs disagreed with
another team's run of the same published algorithms, this is where it
would show up.

``best_existing`` is the *hindsight* per-dataset best of the five external
classifiers -- not deployable, but the honest ceiling for "just run the
best off-the-shelf classifier, picked after seeing test accuracy".

Usage (from the repo root):
    python3 -m src.analyses.table_literature_comparison
"""
from __future__ import annotations

import pandas as pd

from src.analyses import common

OWN_MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"
ORIGINAL_ALGOS_MANIFEST_PATH = common.RESULTS_DIR / "run_142_original_algorithms" / "manifest.csv"
HC2_PUBLISHED_PATH = common.COMPARISONS_DIR / "bake-off-redux-2023" / "HC2_accuracy.csv"

LOCAL_EXTERNAL_ALGORITHMS = ("RDST", "QUANT", "WEASEL2.0", "KG-MTP")
CROSS_CHECK_ALGORITHMS = ("RDST", "QUANT", "WEASEL2.0")  # also published in bake-off-redux-2023
# This project's own Algorithm name -> bake-off-redux-2023's own published-file
# prefix, where they differ (WEASEL 2.0 only -- see comparisons/bake-off-redux-2023/'s
# own "WEASEL-2_*.csv" naming).
PUBLISHED_FILE_PREFIX = {"WEASEL2.0": "WEASEL-2"}
EXTERNAL_DISPLAY_NAMES = {
    "RDST": "RDST", "QUANT": "QUANT", "WEASEL2.0": "WEASEL 2.0", "KG-MTP": "KG-MTP", "HC2": "HC2 (published)",
}


def _load_published_wide(path) -> pd.Series:
    """A ``comparisons/*_accuracy.csv`` (Dataset x 30 resample columns,
    published numbers) collapsed to one per-dataset mean, dataset names
    normalized via ``common.normalize_dataset_name`` (the ``_eq``/``_nmv``
    suffix strip -- see this module's own docstring).
    """
    wide = pd.read_csv(path, index_col=0)
    wide.index = [common.normalize_dataset_name(n) for n in wide.index]
    return wide.mean(axis=1)


def main() -> None:
    own_df = common.load_manifest(OWN_MANIFEST_PATH, methods=common.ALL_OWN_METHODS)
    own_wide = common.per_dataset_mean(own_df, "Accuracy")[list(common.ALL_OWN_METHODS)]

    local_df = common.load_manifest(ORIGINAL_ALGOS_MANIFEST_PATH, method_col="Algorithm", methods=LOCAL_EXTERNAL_ALGORITHMS)
    local_wide = common.per_dataset_mean(local_df, "Accuracy", method_col="Algorithm")[list(LOCAL_EXTERNAL_ALGORITHMS)]

    hc2_published = _load_published_wide(HC2_PUBLISHED_PATH)

    # --- Sanity check: local runs of RDST/QUANT/WEASEL2.0 vs published ---
    print("Cross-check: locally-run accuracy vs published bake-off-redux-2023 (mean over 142 datasets)")
    for algo in CROSS_CHECK_ALGORITHMS:
        prefix = PUBLISHED_FILE_PREFIX.get(algo, algo)
        published = _load_published_wide(common.COMPARISONS_DIR / "bake-off-redux-2023" / f"{prefix}_accuracy.csv")
        aligned = local_wide[algo].align(published)
        diff = (aligned[0] - aligned[1]).abs().mean()
        print(f"  {algo}: local mean={local_wide[algo].mean():.4f}, published mean={published.mean():.4f}, "
              f"mean |per-dataset diff|={diff:.4f}")
        assert diff < 0.01, f"{algo}: local vs published diverge by more than expected (mean |diff|={diff:.4f})"
    print()

    wide = own_wide.join(local_wide, how="inner")
    wide["HC2"] = hc2_published
    assert wide["HC2"].notna().all(), "HC2 join left NaN rows -- dataset-name alignment failed for some dataset."

    external_cols = list(LOCAL_EXTERNAL_ALGORITHMS) + ["HC2"]
    wide["best_existing"] = wide[external_cols].max(axis=1)

    all_methods = list(common.ALL_OWN_METHODS) + external_cols + ["best_existing"]
    summary = wide[all_methods].agg(["mean", "median", "std"]).T
    summary.columns = ["Mean accuracy", "Median accuracy", "Std"]
    summary["Mean rank"] = common.mean_rank(wide[all_methods])
    summary = summary.sort_values("Mean rank")

    display_names = dict(common.METHOD_DISPLAY_NAMES)
    display_names.update(EXTERNAL_DISPLAY_NAMES)
    display_names["best_existing"] = "Best existing (per-dataset best of 5, hindsight)"
    summary.index = [display_names.get(m, m) for m in summary.index]
    summary.index.name = "Method"

    print(summary.round(4).to_string())
    common.save_table(
        summary, "literature_comparison",
        caption=(
            "Comparison against five external time series classifiers (RDST, QUANT, WEASEL 2.0, KG-MTP run "
            "locally on the same 142 datasets/30 resamples; HC2 from the published bake-off-redux-2023 benchmark) "
            "alongside every method from this project's own stacking ensemble."
        ),
        label="tab:literature_comparison",
    )

    # tabiclv2 vs best_existing -- analysis_report.md Section 5's headline caveat.
    r = common.paired_comparison(wide, "tabiclv2", "best_existing")
    print(f"\ntabiclv2 vs best_existing: wins={r['wins']}, losses={r['losses']}, ties={r['ties']}, "
          f"mean_diff={r['mean_diff']:.4f}, wilcoxon_p={r['wilcoxon_p']:.2g}")
    r2 = common.paired_comparison(wide, "auto_selected", "best_existing")
    print(f"auto_selected vs best_existing: wins={r2['wins']}, losses={r2['losses']}, ties={r2['ties']}, "
          f"mean_diff={r2['mean_diff']:.4f}, wilcoxon_p={r2['wilcoxon_p']:.2g}")


if __name__ == "__main__":
    main()
