"""Critical-difference diagram across every meta-method on the full
142-dataset x 30-resample benchmark -- the standard TSC-venue figure for
"are these accuracy differences real", built with aeon's own
``aeon.visualisation.plot_critical_difference`` (pairwise Wilcoxon
signed-rank + Holm correction for clique-finding, the same test family
``common.paired_comparison`` uses elsewhere in this subpackage, so this
figure and the paired-comparison tables tell a consistent story).

Two panels, matching this project's own deployable/hindsight distinction
(see ``common.DEPLOYABLE_METHODS``/``HINDSIGHT_METHODS``): one CD diagram
over all 10 methods (oracle/meta_oracle included, as an honest upper
bound), one restricted to the 8 deployable methods only -- the second is
the one that actually matters for "which method should ship", since a
hindsight oracle trivially wins the first and would otherwise dominate
every clique.

Usage (from the repo root):
    python3 -m src.analyses.figure_critical_difference
"""
from __future__ import annotations

from aeon.visualisation import plot_critical_difference

from src.analyses import common

MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"


def _plot(wide, methods, name) -> None:
    labels = [common.METHOD_DISPLAY_NAMES.get(m, m) for m in methods]
    fig, ax = plot_critical_difference(wide[list(methods)].values, labels, lower_better=False)
    common.save_figure(fig, name)


def main() -> None:
    df = common.load_manifest(MANIFEST_PATH, methods=common.ALL_OWN_METHODS)
    wide = common.per_dataset_mean(df, "Accuracy")[list(common.ALL_OWN_METHODS)]

    _plot(wide, common.ALL_OWN_METHODS, "critical_difference_all_methods")
    _plot(wide, common.DEPLOYABLE_METHODS, "critical_difference_deployable_methods")


if __name__ == "__main__":
    main()
