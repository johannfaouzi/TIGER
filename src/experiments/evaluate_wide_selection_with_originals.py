"""Evaluate each family's own original classifier (the one
``tsfm_ensemble.base_learners`` already uses to match "that family's own
reference implementation" -- see ``sn-article.tex`` Section 5.2 and
``base_learners.RIDGE_ALPHAS``/``EXTRA_TREES_MAX_FEATURES``) on three
feature budgets, with END-TO-END wall time for every step:

1. "full": that learner fit on the family's full, un-reduced raw
   features (``results/dev_subset_transforms/raw/``, already cached --
   no re-run of the family extractor itself needed). Timed as fit+predict
   only (there is no reduction step at this budget).
2. "wide_selection": the same learner fit on a *selected* (not
   projected) subset of at most ``WIDE_SELECTION_MAX_FEATURES`` (1,000)
   raw columns -- restores the reducer classes this project moved to
   ``_removed_from_repo/reduction.py`` during the pre-publication
   cleanup (DetachRocketSelector / FDRUnivariateSelector /
   ChiSquareSelector, imported here, not copy-pasted, so this script and
   that module can never silently drift) and the exact per-family
   reducer configuration this project used for the (now-removed)
   ``wide_selection`` ``REDUCER_VARIANTS`` tag. Timed as reduction
   (fit the reducer + transform both splits) *plus* fit+predict of the
   final classifier -- i.e. the complete cost of using this feature
   budget, not just the final model's own fit.
3. TabICLv2 at this same ``wide_selection`` budget is NOT refit here --
   ``results/evaluate_wide_selection_group_size/summary.csv`` already
   has it, measured on this same server. This script's own
   ``ReductionElapsedSeconds`` (mean per family) is meant to be ADDED to
   that existing TabICLv2 ``ElapsedSeconds_mean`` by the downstream
   table script, so its reported runtime is end-to-end too, without
   paying TabICLv2's ~1-hour-per-unit cost a second time over 100 more
   units.

Motivation
----------
Table 18 in the paper (``table_discarded_design_wide_selection.py``)
previously reported ``reference_extratrees`` for every family, which is
only actually "that family's own original classifier" for QUANT
(interval) -- KG-MTP/RDST/WEASEL~2.0 all use Ridge in the rest of this
paper's own base-learner design (Section 5.2: "...matched, where a
literature precedent exists, to that family's own reference
implementation (KG-MTP, RDST, WEASEL~2.0)..."). This script adds the
correct per-family learner, at both feature budgets, with timing that
actually accounts for every step -- selection included -- so Table 18's
runtime column is comparable across all three approaches it reports.

Why this has to run on the server, not a laptop
-------------------------------------------------
Every other runtime already in the paper (including the TabICLv2
numbers this script's own output gets added to) was measured on the
project's usual server, not the author's Mac. Running this script's
timing on different hardware would make its numbers incomparable to
those already in ``results/evaluate_wide_selection_group_size/summary.csv``
and to the rest of the paper's runtime tables -- so this is meant to be
run on the server, exactly like the TabICLv2-bearing scripts, even
though nothing here needs a GPU.

Output layout, under ``RESULTS_DIR``
(``results/evaluate_wide_selection_with_originals/``)
------------------------------------------------------
    units/<dataset>/resample<k>/<family>/
        Per-job bookkeeping: ``DONE`` marker (or ``ERROR.txt``) and a
        one-row ``manifest.csv`` fragment -- same per-job isolation
        pattern as ``save_dev_subset_transforms.py`` /
        ``evaluate_wide_selection_group_size.py``.
    manifest.csv / summary.csv
        Aggregated by ``finalize()`` / ``summarize()`` -- one row per
        family in ``summary.csv`` (mean/std across the 20 datasets x 5
        resamples = 100 units for that family).

As a side effect, this script also (re-)writes the ``wide_selection``
reduced train/test split to
``results/dev_subset_transforms/reduced/<dataset>/resample<k>/<family>/
wide_selection_{train,test}.npz`` -- the exact path the (removed)
``evaluate_wide_selection_group_size.py`` used to read from, in the same
on-disk format ``common.load_features_npz`` already reads (``is_sparse``
+ dense ``X`` + ``y``) -- so this is a true regeneration of what that
pipeline always produced and never actually kept on disk, not a
parallel, incompatible cache.

Usage
-----
    python -m src.experiments.evaluate_wide_selection_with_originals
    # --max-workers defaults to near-every detected core (these are cheap
    # CPU-only fits, nothing like TabICLv2's memory footprint -- see
    # evaluate_wide_selection_group_size.py's own "Parallelism" section
    # for why that script's near-full-core default is safe here too).
    # Pass --family to restrict to one family (e.g. while sanity-checking
    # a first run), or --max-workers 1 for a sequential run.
"""

import argparse
import gc
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Must be set before numpy (or anything importing it) is ever imported in
# this process -- see the sibling scripts' own identical blocks.
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse as sp  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from src.experiments.common import detect_cpu_count, load_raw_family_features  # noqa: E402
from src.experiments.save_dev_subset_transforms import (  # noqa: E402
    RESULTS_DIR as TRANSFORMS_RESULTS_DIR,
    dev_subset_datasets,
    NUM_RESAMPLES,
)
from src.tsfm_ensemble.base_learners import FAMILIES, _fit_and_score_final  # noqa: E402
from src.tsfm_ensemble.reduction import (  # noqa: E402
    ChiSquareSelector,
    DetachRocketSelector,
    FDRUnivariateSelector,
)

RESULTS_DIR = Path(
    os.environ.get(
        "EVALUATE_WIDE_SELECTION_WITH_ORIGINALS_RESULTS_DIR",
        str(Path("results") / "evaluate_wide_selection_with_originals"),
    )
)

# WIDE_SELECTION_MAX_FEATURES = 1,000, matched exactly to this project's own
# (removed) REDUCER_VARIANTS["wide_selection"] entries -- see
# save_dev_subset_transforms.py's git history (commit e591929^ or earlier)
# for the original, or _removed_from_repo/evaluate_wide_selection_group_size.py's
# own module docstring for the full motivation. Reusing this exact cap (not
# re-deriving it) is what keeps this script's "wide_selection" comparable to
# the already-published Figure 8/9 and the existing TabICLv2 numbers in
# results/evaluate_wide_selection_group_size/summary.csv, which used the same
# cached reduced features this script regenerates.
WIDE_SELECTION_MAX_FEATURES = 1_000

# (reducer_class, reducer_kwargs, learner_name) per family. ``learner_name``
# is this family's own original classifier -- Ridge for every family whose
# literature reference implementation uses Ridge (KG-MTP, RDST, WEASEL 2.0),
# ExtraTrees for QUANT (interval) -- see sn-article.tex Section 5.2 and
# base_learners.py's own module docstring ("per-family literature signal...
# shows the runner-up algorithm...").
FAMILY_CONFIG = {
    "convolution": (
        DetachRocketSelector,
        {"mode": "fixed", "fixed_fraction": 1.0, "min_features": 32, "max_features": WIDE_SELECTION_MAX_FEATURES},
        "ridge",
    ),
    "interval": (
        FDRUnivariateSelector,
        {"alpha": 1.0, "min_features": 16, "max_features": WIDE_SELECTION_MAX_FEATURES},
        "extra_trees",
    ),
    "shapelet": (
        FDRUnivariateSelector,
        {"alpha": 1.0, "group_size": 3, "min_features": 16, "max_features": WIDE_SELECTION_MAX_FEATURES},
        "ridge",
    ),
    "dictionary": (
        ChiSquareSelector,
        {"chi2_alpha": 1.0, "min_features": 16, "max_features": WIDE_SELECTION_MAX_FEATURES},
        "ridge",
    ),
}

RANDOM_STATE = 0


def _discover_jobs(only_family: Optional[str] = None) -> List[Tuple[str, int, str]]:
    datasets = dev_subset_datasets()
    families = [only_family] if only_family else list(FAMILIES)
    jobs = []
    for dataset in datasets:
        for resample in range(NUM_RESAMPLES):
            for family in families:
                jobs.append((dataset, resample, family))
    return jobs


def _job_dir(dataset: str, resample: int, family: str) -> Path:
    return RESULTS_DIR / "units" / dataset / f"resample{resample}" / family


def _reduced_paths(dataset: str, resample: int, family: str) -> Tuple[Path, Path]:
    reduced_family_dir = TRANSFORMS_RESULTS_DIR / "reduced" / dataset / f"resample{resample}" / family
    return reduced_family_dir / "wide_selection_train.npz", reduced_family_dir / "wide_selection_test.npz"


def _is_done(dataset: str, resample: int, family: str) -> bool:
    return (_job_dir(dataset, resample, family) / "DONE").is_file()


def _maybe_densify(X_train, X_test):
    if not sp.issparse(X_train):
        return X_train, X_test
    return np.asarray(X_train.todense()), np.asarray(X_test.todense())


def _save_reduced_split(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, is_sparse=np.array(False), X=X, y=y)


def _process_job(dataset: str, resample: int, family: str) -> dict:
    job_dir = _job_dir(dataset, resample, family)
    job_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        reducer_class, reducer_kwargs, learner_name = FAMILY_CONFIG[family]

        X_train_raw, y_train = load_raw_family_features(dataset, resample, family, "train")
        X_test_raw, y_test = load_raw_family_features(dataset, resample, family, "test")
        X_train_raw, X_test_raw = _maybe_densify(X_train_raw, X_test_raw)

        le = LabelEncoder().fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        # --- 1. Full, un-reduced features ---------------------------------
        full_scores, full_fit_s, full_predict_s = _fit_and_score_final(
            family, learner_name, X_train_raw, y_train_enc, X_test_raw, random_state=RANDOM_STATE,
        )
        full_elapsed = full_fit_s + full_predict_s
        full_accuracy = accuracy_score(y_test_enc, np.argmax(full_scores, axis=1))
        num_features_raw = X_train_raw.shape[1]

        # --- 2. wide_selection: reduce (timed), then fit+predict (timed) --
        reduction_start = time.perf_counter()
        reducer = reducer_class(**reducer_kwargs)
        reducer.fit(X_train_raw, y_train_enc)
        X_train_reduced = reducer.transform(X_train_raw)
        X_test_reduced = reducer.transform(X_test_raw)
        reduction_elapsed = time.perf_counter() - reduction_start
        num_features_reduced = X_train_reduced.shape[1]

        train_path, test_path = _reduced_paths(dataset, resample, family)
        _save_reduced_split(train_path, X_train_reduced, y_train)
        _save_reduced_split(test_path, X_test_reduced, y_test)

        wide_scores, wide_fit_s, wide_predict_s = _fit_and_score_final(
            family, learner_name, X_train_reduced, y_train_enc, X_test_reduced, random_state=RANDOM_STATE,
        )
        wide_model_elapsed = wide_fit_s + wide_predict_s
        wide_total_elapsed = reduction_elapsed + wide_model_elapsed
        wide_accuracy = accuracy_score(y_test_enc, np.argmax(wide_scores, axis=1))

        row = {
            "Dataset": dataset, "Resample": resample, "Family": family, "LearnerName": learner_name,
            "NumFeaturesRaw": num_features_raw, "NumFeaturesReduced": num_features_reduced,
            "FullAccuracy": full_accuracy, "FullElapsedSeconds": full_elapsed,
            "ReductionElapsedSeconds": reduction_elapsed,
            "WideAccuracy": wide_accuracy,
            "WideModelElapsedSeconds": wide_model_elapsed,
            "WideTotalElapsedSeconds": wide_total_elapsed,
        }
        pd.DataFrame([row]).to_csv(job_dir / "manifest.csv", index=False)
        (job_dir / "DONE").touch()
        del X_train_raw, X_test_raw, X_train_reduced, X_test_reduced, full_scores, wide_scores
        gc.collect()
        return {**row, "elapsed": time.perf_counter() - start}
    except Exception as exc:  # noqa: BLE001
        (job_dir / "ERROR.txt").write_text(traceback.format_exc())
        return {
            "Dataset": dataset, "Resample": resample, "Family": family,
            "elapsed": time.perf_counter() - start, "error": repr(exc),
        }


def finalize() -> None:
    rows = []
    for manifest_path in sorted((RESULTS_DIR / "units").glob("*/*/*/manifest.csv")):
        rows.append(pd.read_csv(manifest_path))
    if not rows:
        print("No completed jobs found -- nothing to finalize.")
        return
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'manifest.csv'} ({len(df)} rows).")


def summarize() -> Optional[pd.DataFrame]:
    manifest_path = RESULTS_DIR / "manifest.csv"
    if not manifest_path.is_file():
        print(f"{manifest_path} doesn't exist yet -- run finalize() first.")
        return None
    df = pd.read_csv(manifest_path)
    metric_cols = [
        "FullAccuracy", "FullElapsedSeconds", "ReductionElapsedSeconds",
        "WideAccuracy", "WideModelElapsedSeconds", "WideTotalElapsedSeconds",
    ]
    grouped = df.groupby("Family")[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped["NumDatasets"] = df.groupby("Family")["Dataset"].nunique()
    grouped["LearnerName"] = df.groupby("Family")["LearnerName"].first()
    grouped = grouped.reset_index()
    grouped.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'summary.csv'}:")
    print(grouped.to_string())
    return grouped


@dataclass
class Job:
    dataset: str
    resample: int
    family: str


def build_jobs(only_family: Optional[str] = None) -> List[Job]:
    return [Job(d, r, f) for d, r, f in _discover_jobs(only_family) if not _is_done(d, r, f)]


def _worker_init() -> None:
    for _env_var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(_env_var, "1")


def main(argv: Optional[List[str]] = None) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--family", type=str, default=None, choices=list(FAMILIES))
    args = parser.parse_args(argv)

    jobs = build_jobs(only_family=args.family)
    print(f"{len(jobs)} job(s) remaining (already-DONE jobs are skipped).")
    if not jobs:
        finalize()
        summarize()
        return

    max_workers = args.max_workers if args.max_workers is not None else max(1, detect_cpu_count() - 1)
    print(f"Running with max_workers={max_workers}.")

    ctx = multiprocessing.get_context("spawn")
    completed, errors = 0, 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init) as executor:
        futures = {executor.submit(_process_job, j.dataset, j.resample, j.family): j for j in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            completed += 1
            if "error" in result:
                errors += 1
                print(f"[{completed}/{len(jobs)}] ERROR {job.dataset}/resample{job.resample}/{job.family}: {result['error']}")
            else:
                print(
                    f"[{completed}/{len(jobs)}] OK {job.dataset}/resample{job.resample}/{job.family} "
                    f"(full_acc={result['FullAccuracy']:.3f}, wide_acc={result['WideAccuracy']:.3f}, "
                    f"elapsed={result['elapsed']:.1f}s)"
                )
    print(f"Done: {completed - errors} succeeded, {errors} failed.")
    finalize()
    summarize()


if __name__ == "__main__":
    main()
