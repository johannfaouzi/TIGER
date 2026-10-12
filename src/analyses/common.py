"""Shared plumbing for ``src/analyses/`` -- the paper's tables and figures.

``src/experiments/`` runs things and caches raw results under ``results/``;
``src/analyses/`` is a strict step downstream of that -- every script here
only *reads* an already-computed ``results/**/manifest.csv`` (or a
``comparisons/`` published baseline) and *writes* a camera-ready table or
figure under ``results/paper/``. Nothing in this subpackage fits a model,
runs a benchmark, or touches a dataset directly; if a script here needs a
number that doesn't already exist on disk, that's a sign the number
belongs in ``src/experiments/`` first, not that this subpackage should go
compute it.

One script per table or figure (mirrors ``src/experiments/``'s flat,
self-contained-script convention -- see that package's own scripts for the
precedent), each runnable directly:

    python3 -m src.analyses.table_<name>
    python3 -m src.analyses.figure_<name>

and each writing its output(s) under ``TABLES_DIR``/``FIGURES_DIR`` below,
named after the script itself, so ``results/paper/`` stays a flat,
browsable mirror of this directory. A script that reproduces a specific
number quoted in an already-written report (e.g.
``results/meta_model_predictions_142/analysis_report.md``) should say so
in its docstring and assert the reproduced number still matches -- turning
prose claims in that report into checked, re-runnable code rather than
letting them silently drift from the underlying manifest as new rows are
added.

Aggregation convention (load-bearing -- get this wrong and every p-value
and win/loss count in the paper is wrong)
---------------------------------------------------------------------------
Every cross-method comparison in this project, starting with
``analysis_report.md``, is computed **per dataset, not per (dataset,
resample) row**: each dataset's score is first averaged over its own
resamples (30 for the 142-dataset benchmark, 5 for the dev subset), and
*then* wins/losses/ties, mean differences, and the Wilcoxon signed-rank
test are computed across those N per-dataset means (N=142 or N=20) -- not
across the N x NumResamples individual rows. This is *why*, e.g., "tabiclv2
vs oracle_cv" in that report reads "92 wins, 47 losses, 3 ties" (92+47+3 =
142, the dataset count, not 4260). ``paired_comparison`` and
``per_dataset_mean`` below exist specifically so every script in this
subpackage gets this right by construction instead of each one
re-deriving (or, worse, silently getting wrong) the same aggregation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# --------------------------------------------------------------------------- #
# Paths -- same "run from the repo root" convention as every
# src/experiments/ script (see e.g. run_142_hc2.py's own usage docstring);
# PAPER_DIR is overridable for smoke-testing into a scratch location, same
# pattern as RUN_142_HC2_RESULTS_DIR / META_MODEL_PREDICTIONS_142_RESULTS_DIR.
# --------------------------------------------------------------------------- #
RESULTS_DIR = Path("results")
COMPARISONS_DIR = Path("comparisons")
PAPER_DIR = Path(os.environ.get("ANALYSES_PAPER_DIR", str(RESULTS_DIR / "paper")))
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Method bookkeeping -- shared across every table/figure that lists this
# project's own meta-methods, so "is this one deployable" and "what order
# do these print in" are answered once, not per-script. See
# analysis_report.md Section 2's own correction re: oracle_cv, and
# meta_learners.py's module docstring for the method list itself.
# --------------------------------------------------------------------------- #
DEPLOYABLE_METHODS = (
    "tabiclv2",
    "oracle_cv",
    "cawpe",
    "cawpe_hard_vote",
    "logistic_regression",
    "hard_majority_vote",
    "lightgbm",
    "auto_selected",
)
HINDSIGHT_METHODS = ("oracle", "meta_oracle")  # test-time / not deployable -- upper bounds only
ALL_OWN_METHODS = DEPLOYABLE_METHODS + HINDSIGHT_METHODS

# The six candidate meta-classification methods from Section 5.1 of the
# paper (hard_majority_vote, logistic_regression, lightgbm, cawpe,
# cawpe_hard_vote, tabiclv2) -- NOT the same thing as the manifest's own
# "oracle" method column, despite what that column's name suggests.
#
# "oracle" in the manifest is ``oracle_best_single_learner``
# (meta_learners.py): the single best of TIGER's *twelve base learners*
# (four families x three learners), picked by test-set hindsight -- a
# diagnostic upper bound one level below the meta-classification stage
# entirely, not an oracle over these six methods. "meta_oracle" comes
# closer (it does pick among meta-methods) but only searches five of the
# six (it omits cawpe_hard_vote), so it undercounts the true headroom by a
# small amount too. Discovered while building
# table_oracle_headroom_by_characteristics.py: the pooled "oracle" number
# already in the paper (0.892) does not fall between auto_selected and
# any per-row combination of these six methods for every data set, which
# is only possible if it is not in fact their oracle.
#
# ``six_method_oracle`` below is the correct, if more expensive, way to
# get this quantity: pick per (data set, resample) row via a row-wise max
# over these six methods' own already-scored test accuracy, since neither
# manifest column can be trusted for it as-is.
SIX_META_METHODS = (
    "hard_majority_vote", "logistic_regression", "lightgbm",
    "cawpe", "cawpe_hard_vote", "tabiclv2",
)


def six_method_oracle(wide: pd.DataFrame) -> pd.Series:
    """The true hindsight oracle across ``SIX_META_METHODS``: per
    (data set) row of an already ``per_dataset_mean``-collapsed wide
    frame, the row-wise max over those six methods' own mean accuracy
    columns. See this module's own "oracle"/"meta_oracle" caveat above
    for why neither existing manifest column already gives this.
    """
    missing = set(SIX_META_METHODS) - set(wide.columns)
    if missing:
        raise ValueError(f"six_method_oracle: wide frame is missing column(s) {sorted(missing)}.")
    return wide[list(SIX_META_METHODS)].max(axis=1)


METHOD_DISPLAY_NAMES = {
    "tabiclv2": "TabICLv2 (stacked)",
    "oracle_cv": "Best base learner (CV-picked)",
    "cawpe": "CAWPE",
    "cawpe_hard_vote": "Weighted hard majority vote",
    "logistic_regression": "Logistic regression (stacked)",
    "hard_majority_vote": "Hard majority vote",
    "lightgbm": "LightGBM (stacked)",
    "auto_selected": "Auto-selected (this paper's rule)",
    "oracle": "Oracle (test-time, upper bound)",
    "meta_oracle": "Meta-oracle (test-time, upper bound)",
}


def is_deployable(method: str) -> bool:
    return method in DEPLOYABLE_METHODS


# --------------------------------------------------------------------------- #
# Dataset-name alignment -- canonical copy of
# analyze_family_strengths.py's own ``_normalize_dataset_name`` (QUANT/RDST/
# WEASEL-2's published result files suffix some names with "_eq"/"_nmv";
# KG-MTP's and this project's own 142-dataset names never do). Duplicated
# here rather than imported from that experiments-side script, matching
# this project's small-self-contained-helper convention, but now the one
# copy every analyses/ script should use when joining against
# comparisons/bake-off-redux-2023/ or comparisons/KG-MTP/.
# --------------------------------------------------------------------------- #
_DATASET_SUFFIXES = ("_eq", "_nmv")


def normalize_dataset_name(name: str) -> str:
    for suffix in _DATASET_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_manifest(path: Path, method_col: str = "Method", methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a ``manifest.csv`` (this project's standard per-job row schema --
    see e.g. ``build_meta_model_predictions_142.py``'s own ``_record``) and,
    if ``methods`` is given, filter + assert every one of them is present
    with zero recorded errors -- fail loudly here rather than let a script
    silently compute a headline number from a manifest that's missing rows
    or still carries an unresolved error, the exact failure mode
    ``analysis_report.md`` Section 1 checked for by hand before trusting
    anything downstream of it.
    """
    df = pd.read_csv(path)
    if methods is not None:
        missing = set(methods) - set(df[method_col].unique())
        if missing:
            raise ValueError(f"{path}: missing method(s) {sorted(missing)}.")
        df = df[df[method_col].isin(methods)].copy()
        if "Error" in df.columns:
            errored = df[df["Error"].notna()]
            if len(errored):
                raise ValueError(
                    f"{path}: {len(errored)} row(s) carry a recorded Error for "
                    f"method(s) {sorted(errored[method_col].unique())} -- resolve before analyzing."
                )
    return df


# --------------------------------------------------------------------------- #
# Aggregation -- see this module's own docstring's "Aggregation convention"
# section for why this two-step (resample mean, then per-dataset series) is
# load-bearing and not just a convenience.
# --------------------------------------------------------------------------- #
def per_dataset_mean(
    df: pd.DataFrame, value_col: str, group_col: str = "Dataset", method_col: str = "Method"
) -> pd.DataFrame:
    """Collapse (Dataset, Resample, Method, ...) rows to one row per
    (Dataset, Method), each dataset's ``value_col`` averaged over its own
    resamples first. Returns a wide ``Dataset x Method`` frame (one column
    per method) -- the shape every function below expects.
    """
    long = df.groupby([group_col, method_col], as_index=False)[value_col].mean()
    return long.pivot(index=group_col, columns=method_col, values=value_col)


def mean_rank(wide: pd.DataFrame, methods: Optional[Sequence[str]] = None) -> pd.Series:
    """Per-dataset ranks (1 = best) averaged across datasets -- the "Mean
    rank" column in every method-comparison table in ``analysis_report.md``.
    """
    cols = list(methods) if methods is not None else list(wide.columns)
    ranks = wide[cols].rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()


def paired_comparison(wide: pd.DataFrame, method_a: str, method_b: str) -> dict:
    """Wins/losses/ties + mean difference + Wilcoxon signed-rank p-value for
    ``method_a`` vs ``method_b``, computed across per-dataset means (see
    module docstring) -- one row of the "Comparison / Wins / Losses / Ties /
    Mean diff / Wilcoxon p" tables throughout ``analysis_report.md``.
    """
    a, b = wide[method_a], wide[method_b]
    diff = a - b
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    nonzero = diff[diff != 0]
    if len(nonzero) >= 1:
        _, p = wilcoxon(nonzero)
    else:
        p = float("nan")
    return {
        "method_a": method_a,
        "method_b": method_b,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_diff": float(diff.mean()),
        "wilcoxon_p": float(p),
    }


def paired_comparison_table(wide: pd.DataFrame, reference: str, others: Iterable[str]) -> pd.DataFrame:
    """``paired_comparison`` of every method in ``others`` against
    ``reference``, stacked into one table -- the exact shape of
    ``analysis_report.md``'s repeated "X vs Y" tables.
    """
    rows = [paired_comparison(wide, other, reference) for other in others]
    return pd.DataFrame(rows).set_index("method_a").drop(columns="method_b")


def friedman_omnibus(wide: pd.DataFrame, methods: Optional[Sequence[str]] = None) -> dict:
    """Omnibus Friedman test across >=3 methods' per-dataset means --
    ``analysis_report.md`` Section 2's "chi^2 = 498.9, p ~ 1.4e-103" line."""
    cols = list(methods) if methods is not None else list(wide.columns)
    stat, p = friedmanchisquare(*[wide[c].dropna() for c in cols])
    return {"statistic": float(stat), "p_value": float(p), "n_methods": len(cols)}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
_LATEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def _latex_escape(value) -> str:
    """Escape LaTeX special characters in a cell/header/index value. Method
    and dataset names in this project are full of underscores
    (``cawpe_hard_vote``, ``hard_majority_vote``), which LaTeX treats as
    subscript-start -- every string that ends up in a ``.tex`` table needs
    this, not just a few risky-looking ones.
    """
    text = str(value)
    # Backslash must be replaced first so later replacements' own
    # backslashes don't get re-escaped.
    text = text.replace("\\", _LATEX_SPECIAL["\\"])
    for char, escaped in _LATEX_SPECIAL.items():
        if char == "\\":
            continue
        text = text.replace(char, escaped)
    return text


def to_latex_booktabs(
    df: pd.DataFrame, index: bool = True, float_format: str = "%.4f",
    caption: Optional[str] = None, label: Optional[str] = None,
) -> str:
    """Hand-rolled booktabs table (no ``jinja2``/``Styler`` dependency --
    not installed in this project's environment, see ``requirements.txt``)
    -- ``\\usepackage{booktabs}`` is assumed in the paper's own preamble,
    matching every modern venue's own style file. Every string cell
    (method names, dataset names -- this project's own names are full of
    underscores) is LaTeX-escaped via ``_latex_escape``; every float cell
    is formatted with ``float_format`` first.
    """
    header_cells = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    header = " & ".join(_latex_escape(c) for c in header_cells) + r" \\"

    body_lines = []
    for idx, row in df.iterrows():
        cells = []
        if index:
            cells.append(_latex_escape(idx))
        for value in row:
            if isinstance(value, (int, np.integer)):
                cells.append(_latex_escape(value))
            elif isinstance(value, (float, np.floating)):
                cells.append(_latex_escape(float_format % value) if pd.notna(value) else "--")
            else:
                cells.append(_latex_escape(value))
        body_lines.append(" & ".join(cells) + r" \\")

    n_cols = len(header_cells)
    col_spec = ("l" if index else "") + "r" * (n_cols - (1 if index else 0))

    lines = [r"\begin{table}[t]", r"\centering"]
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.extend(body_lines)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(r"\caption{" + _latex_escape(caption) + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_table(
    df: pd.DataFrame, name: str, index: bool = True, float_format: str = "%.4f",
    caption: Optional[str] = None, label: Optional[str] = None,
) -> None:
    """Write ``df`` under ``TABLES_DIR`` as ``.csv`` (source of truth,
    diffable, re-loadable), ``.md`` (paste straight into a draft), and
    ``.tex`` (booktabs, ``\\input``-able directly into the paper) -- every
    table script should call this exactly once, at the end, so
    ``results/paper/tables/`` stays a complete, current mirror of whatever
    this subpackage can currently produce. ``caption``/``label`` are
    LaTeX-only (used in the ``.tex`` output; ignored by csv/md) -- default
    to ``name`` when omitted so the ``.tex`` file is never left without a
    ``\\label``.
    """
    ensure_output_dirs()
    csv_path = TABLES_DIR / f"{name}.csv"
    md_path = TABLES_DIR / f"{name}.md"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=index, float_format=float_format)
    md_path.write_text(df.to_markdown(index=index, floatfmt=float_format.replace("%", "").replace("f", "f")))
    tex_path.write_text(
        to_latex_booktabs(
            df, index=index, float_format=float_format,
            caption=caption or name.replace("_", " "), label=label or f"tab:{name}",
        )
    )
    print(f"wrote {csv_path}, {md_path}, and {tex_path} ({df.shape[0]} row(s) x {df.shape[1]} col(s))")


def save_figure(fig, name: str, dpi: int = 300) -> None:
    """Save ``fig`` under ``FIGURES_DIR`` as both ``.pdf`` (vector, for the
    camera-ready) and ``.png`` (quick preview) -- every figure script
    should call this exactly once, at the end.
    """
    ensure_output_dirs()
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"wrote {pdf_path} and {png_path}")
