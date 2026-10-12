# `src/analyses/`

Paper-facing tables and figures, built strictly from already-computed
`results/**/manifest.csv` files and published `comparisons/` baselines.
Nothing here fits a model or runs a benchmark -- that's `src/experiments/`'s
job. If a table needs a number that isn't on disk yet, that's a sign the
number belongs in `src/experiments/` first.

- `common.py` -- shared paths, method bookkeeping, dataset-name alignment,
  the per-dataset-mean aggregation + paired-comparison + ranking helpers
  every table/figure script should use (see its own module docstring for
  why the aggregation convention is load-bearing), and `save_table`/
  `save_figure`.
- One script per deliverable: `table_<name>.py` / `figure_<name>.py`,
  each runnable as `python3 -m src.analyses.table_<name>` from the repo
  root, each writing its output under `results/paper/tables/` or
  `results/paper/figures/` (csv+md for tables, pdf+png for figures) via
  `common.save_table`/`common.save_figure`.

Verified against `results/meta_model_predictions_142/analysis_report.md`:
`common.py`'s `paired_comparison`/`mean_rank`/`friedman_omnibus` reproduce
that report's tabiclv2-vs-oracle_cv numbers (92/47/3, +0.0010, p=0.00033),
its 8-method mean-rank ordering, and its Friedman statistic exactly.
