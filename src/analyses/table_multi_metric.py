"""Summary performance measures (ACC, BALACC, AUROC, NLL, F1) of TIGER
(``auto_selected``) against five literature algorithms, on the full
142-dataset benchmark (the paper's Table 4, ``tab:multi-metric-142``) and
restricted to the 57 datasets where the adaptive rule always selects
TabICLv2 (Appendix Table A1, ``tab:multi-metric-tabiclv2-subset``).

Why this exists
----------------
No script previously in this repository computed AUROC/NLL for TIGER's own
``auto_selected`` predictions at all (see
``compute_auto_selected_metrics_142.py``'s own module docstring for the
full account of that gap and how it was found); nor was there any
``src/analyses/`` script left behind that assembled the five-metric
comparison table the paper actually reports -- ``results/paper/tables/
multi_metric_142.csv`` existed as an orphaned artifact with no generating
``.py`` file anywhere in the repo (tracked or not), numbers that did not
match the paper's own published table, and a caption that contradicted
its own non-empty AUROC/NLL cells. This script replaces that gap with a
single, checked, re-runnable source for both tables.

Per this subpackage's own convention (see ``common.py``'s module
docstring), this script only *reads* already-computed manifests/published
baselines -- it does not fit anything or touch raw predictions.npz files
directly. ``compute_auto_selected_metrics_142.py`` (an ``src/experiments/``
script) must be run first to populate ``auto_selected``'s
BalancedAccuracy/LogLoss/ROCAUC/F1 columns in
``results/meta_model_predictions_142/manifest.csv``.

Per-algorithm data sources
-----------------------------
    TIGER (``auto_selected``): ``results/meta_model_predictions_142/
        manifest.csv`` -- locally computed, one-hot-when-WHMV convention
        (see ``compute_auto_selected_metrics_142.py``).
    KG-MTP, RDST, WEASEL 2.0, QUANT: ``results/run_142_original_algorithms/
        manifest.csv`` -- locally run and locally computed
        (``run_142_original_algorithms.compute_classification_metrics``,
        the same formula ``compute_auto_selected_metrics_142.py``
        reproduces verbatim for TIGER).
    HC2: published ``comparisons/bake-off-redux-2023/HC2_{accuracy,
        balacc,auroc,logloss}.csv`` (never run locally for accuracy --
        only for runtime, see Section 5.3/``run_142_hc2.py``). No F1 is
        published for HC2, so its F1 cell is left blank ("--" in the
        rendered table), matching the paper's own convention.

Aggregation: per-dataset mean over resamples, then mean (and rank, 1 =
best, independently per metric column) across datasets -- this
subpackage's standard convention (see ``common.py``'s own docstring).
NLL is reported as-is (lower is better) with ranks computed accordingly
(ascending); every other metric is ranked descending (higher is better).

Table A1's own restriction (57 datasets where the adaptive rule always
selects TabICLv2 across all 30 resamples) is derived directly from
``meta_model_predictions_142/manifest.csv``'s own ``SelectedMethod``
column, not hand-copied from anywhere -- 57 "always tabiclv2" / 85
"always cawpe_hard_vote" / 0 mixed, matching the paper's own prose counts.

Usage (from the repo root, after compute_auto_selected_metrics_142.py):
    python3 -m src.analyses.table_multi_metric
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analyses import common

OWN_MANIFEST_PATH = common.RESULTS_DIR / "meta_model_predictions_142" / "manifest.csv"
ORIGINAL_ALGOS_MANIFEST_PATH = common.RESULTS_DIR / "run_142_original_algorithms" / "manifest.csv"

LOCAL_EXTERNAL_ALGORITHMS = ("RDST", "QUANT", "WEASEL2.0", "KG-MTP")
EXTERNAL_DISPLAY_NAMES = {
    "RDST": "RDST", "QUANT": "QUANT", "WEASEL2.0": "WEASEL 2.0", "KG-MTP": "KG-MTP",
    "auto_selected": "TIGER", "HC2": "HC2",
}

# (paper column, manifest column, higher-is-better)
METRICS = (
    ("ACC", "Accuracy", True),
    ("BALACC", "BalancedAccuracy", True),
    ("AUROC", "ROCAUC", True),
    ("NLL", "LogLoss", False),
    ("F1", "F1", True),
)

# Paper's own currently-published numbers (Table 4 / Table A1), kept here
# purely as a cross-check: this script should reproduce them, not the
# other way around -- if a future rerun of the underlying experiments
# changes these, this assertion is the thing that should be relaxed, not
# silently ignored.
_PUBLISHED_142 = {
    "auto_selected": dict(ACC=0.884, BALACC=0.863, AUROC=0.933, NLL=2.775, F1=0.863),
}
_PUBLISHED_TABICLV2_SUBSET = {
    "auto_selected": dict(ACC=0.895, BALACC=0.862, AUROC=0.955, NLL=0.265, F1=0.860),
}


def _load_published_wide(path) -> pd.Series:
    """A ``comparisons/*_<metric>.csv`` (Dataset x 30 resample columns,
    published numbers) collapsed to one per-dataset mean, dataset names
    normalized via ``common.normalize_dataset_name``.
    """
    wide = pd.read_csv(path, index_col=0)
    wide.index = [common.normalize_dataset_name(n) for n in wide.index]
    return wide.mean(axis=1)


def _load_hc2_per_dataset_means() -> pd.DataFrame:
    base = common.COMPARISONS_DIR / "bake-off-redux-2023"
    out = pd.DataFrame(
        {
            "ACC": _load_published_wide(base / "HC2_accuracy.csv"),
            "BALACC": _load_published_wide(base / "HC2_balacc.csv"),
            "AUROC": _load_published_wide(base / "HC2_auroc.csv"),
            "NLL": _load_published_wide(base / "HC2_logloss.csv"),
        }
    )
    out["F1"] = np.nan  # not published for HC2 -- "--" in the rendered table.
    return out


def _always_tabiclv2_datasets(own_df: pd.DataFrame) -> list[str]:
    auto = own_df[own_df["Method"] == "auto_selected"]
    per_dataset_selection = auto.groupby("Dataset")["SelectedMethod"].agg(lambda s: set(s))
    always_tabiclv2 = per_dataset_selection[per_dataset_selection.apply(lambda s: s == {"tabiclv2"})]
    always_whmv = per_dataset_selection[per_dataset_selection.apply(lambda s: s == {"cawpe_hard_vote"})]
    mixed = per_dataset_selection[per_dataset_selection.apply(lambda s: len(s) > 1)]
    print(
        f"Adaptive-rule selection consistency across resamples: "
        f"{len(always_tabiclv2)} datasets always tabiclv2, {len(always_whmv)} always cawpe_hard_vote "
        f"(WHMV), {len(mixed)} mixed (out of {len(per_dataset_selection)} total)."
    )
    assert len(mixed) == 0, "Expected every dataset to select the same method on all 30 resamples."
    return sorted(always_tabiclv2.index)


def _per_dataset_means(own_df: pd.DataFrame, local_df: pd.DataFrame, hc2_wide: pd.DataFrame, metric_col: str,
                        own_col: str) -> pd.DataFrame:
    own_pd = common.per_dataset_mean(own_df, own_col)[["auto_selected"]]
    local_pd = common.per_dataset_mean(local_df, own_col, method_col="Algorithm")[list(LOCAL_EXTERNAL_ALGORITHMS)]
    wide = own_pd.join(local_pd, how="inner")
    wide["HC2"] = hc2_wide[metric_col]
    if metric_col != "F1":
        # F1 is intentionally left NaN for HC2 (not published, see module docstring) --
        # every other metric must align cleanly against every requested dataset.
        assert wide["HC2"].notna().all(), f"HC2 join left NaN rows for {metric_col} -- dataset-name alignment failed."
    return wide


def _format_table(wide_by_metric: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per algorithm, one column per paper metric name, each cell
    ``"value (rank)"`` (``\\textbf{...}`` applied downstream by the caller
    for the best cell per column), ``"--"`` for a metric this algorithm
    has no value for (HC2's F1).
    """
    algorithms = list(wide_by_metric["ACC"].columns)
    rows = {}
    bold_mask = {}
    for paper_name, _, higher_better in METRICS:
        wide = wide_by_metric[paper_name]
        means = wide.mean(axis=0)
        ranks = means.rank(ascending=not higher_better, method="average")
        best_algo = means.idxmax() if higher_better else means.idxmin()
        cells = {}
        for algo in algorithms:
            if pd.isna(means[algo]):
                cells[algo] = "--"
            else:
                cells[algo] = f"{means[algo]:.3f} ({int(round(ranks[algo]))})"
        rows[paper_name] = cells
        bold_mask[paper_name] = best_algo
    table = pd.DataFrame(rows)
    table = table.reindex(wide_by_metric["ACC"].mean(axis=0).sort_values(ascending=False).index)
    return table, bold_mask


def _write_latex(table: pd.DataFrame, bold_mask: dict, caption: str, label: str, path) -> None:
    """Hand-rolled booktabs .tex in the paper's own Table-4 style ("value
    (rank)" cells, best cell per column in bold, "--" left as-is) --
    written directly rather than through ``common.to_latex_booktabs``,
    whose automatic ``_latex_escape`` would otherwise mangle the
    intentional ``\\textbf{...}`` markup added here.
    """
    cols = list(table.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule"]
    lines.append("Algorithm & " + " & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for algo, row in table.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if value == "--":
                cells.append("--")
            elif bold_mask.get(col) == algo:
                cells.append(r"\textbf{" + value + "}")
            else:
                cells.append(value)
        lines.append(f"{algo} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines))


def _check_against_published(wide_by_metric: dict[str, pd.DataFrame], published: dict, label: str) -> None:
    for algo, expected in published.items():
        for paper_name, expected_value in expected.items():
            actual = float(wide_by_metric[paper_name][algo].mean())
            if not np.isclose(actual, expected_value, atol=5e-4):
                print(
                    f"WARNING [{label}] {algo} {paper_name}: recomputed {actual:.4f} vs. "
                    f"published {expected_value:.4f} -- more than rounding drift, investigate."
                )
            else:
                print(f"OK [{label}] {algo} {paper_name}: recomputed {actual:.4f} matches published {expected_value:.4f}.")


def _build_table(own_df: pd.DataFrame, local_df: pd.DataFrame, hc2_wide: pd.DataFrame,
                  datasets: list[str] | None, published: dict, label: str, name: str, caption: str, latex_label: str) -> None:
    own_sub = own_df if datasets is None else own_df[own_df["Dataset"].isin(datasets)]
    local_sub = local_df if datasets is None else local_df[local_df["Dataset"].isin(datasets)]
    hc2_sub = hc2_wide if datasets is None else hc2_wide.loc[hc2_wide.index.isin(datasets)]

    wide_by_metric = {}
    for paper_name, manifest_col, _ in METRICS:
        wide_by_metric[paper_name] = _per_dataset_means(own_sub, local_sub, hc2_sub, paper_name, manifest_col)

    n_datasets = len(wide_by_metric["ACC"])
    print(f"\n=== {label} ({n_datasets} datasets) ===")
    _check_against_published(wide_by_metric, published, label)

    table, bold_mask = _format_table(wide_by_metric)
    table.index = [EXTERNAL_DISPLAY_NAMES.get(a, a) for a in table.index]
    bold_mask = {k: EXTERNAL_DISPLAY_NAMES.get(v, v) for k, v in bold_mask.items()}
    table.index.name = "Algorithm"

    print(table.to_string(), "\n(bold/best per column:", bold_mask, ")")

    common.ensure_output_dirs()
    csv_path = common.TABLES_DIR / f"{name}.csv"
    md_path = common.TABLES_DIR / f"{name}.md"
    tex_path = common.TABLES_DIR / f"{name}.tex"
    table.to_csv(csv_path)
    md_path.write_text(table.to_markdown())
    _write_latex(table, bold_mask, caption, latex_label, tex_path)
    print(f"wrote {csv_path}, {md_path}, and {tex_path}")


def main() -> None:
    own_df = common.load_manifest(OWN_MANIFEST_PATH, methods=["auto_selected"])
    local_df = common.load_manifest(ORIGINAL_ALGOS_MANIFEST_PATH, method_col="Algorithm",
                                     methods=list(LOCAL_EXTERNAL_ALGORITHMS))
    hc2_wide = _load_hc2_per_dataset_means()

    _build_table(
        own_df, local_df, hc2_wide, datasets=None,
        published=_PUBLISHED_142, label="Table 4 (full 142-dataset benchmark)",
        name="multi_metric_142",
        caption=(
            "Summary performance measures of TIGER and five literature algorithms, over 30 resamples of all "
            "142 datasets. TIGER's AUROC/NLL use a one-hot probability vector (1 for the predicted class, 0 "
            "elsewhere) whenever the adaptive rule selects weighted hard majority vote -- see "
            "compute_auto_selected_metrics_142.py."
        ),
        latex_label="tab:multi_metric_142",
    )

    # Appendix A's own restriction -- derived fresh, not hand-copied.
    selection_df = common.load_manifest(OWN_MANIFEST_PATH, methods=["auto_selected"])
    always_tabiclv2 = _always_tabiclv2_datasets(selection_df)

    _build_table(
        own_df, local_df, hc2_wide, datasets=always_tabiclv2,
        published=_PUBLISHED_TABICLV2_SUBSET, label="Table A1 (57 datasets, adaptive rule always selects TabICLv2)",
        name="multi_metric_tabiclv2_subset",
        caption=(
            "Summary performance measures, restricted to the 57 datasets where TIGER's adaptive rule always "
            "selects TabICLv2 (derived from meta_model_predictions_142/manifest.csv's own SelectedMethod "
            "column). TIGER's AUROC/NLL use TabICLv2's own genuine predicted class probabilities throughout "
            "-- no one-hot fallback is reached on this subset by construction."
        ),
        latex_label="tab:multi_metric_tabiclv2_subset",
    )


if __name__ == "__main__":
    main()
