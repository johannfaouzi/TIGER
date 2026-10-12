"""Build and cache the full stacking feature matrices for every
(dataset, resample) in the development subset, from each family's
already-cached raw features (``save_dev_subset_transforms.py``'s
``raw/`` output) -- one ``scores.npz`` per (dataset, resample) holding
every (family, learner) base learner's out-of-fold train scores and
test-set scores (``base_learners.run_family_base_learners``), saved
individually (not pre-concatenated into one wide matrix) so later
ablation experiments can pick any subset of the 12 (family, learner)
blocks without needing to know column offsets into a combined matrix.

On not using the test scores to make development decisions
-------------------------------------------------------------
This module computes and saves BOTH out-of-fold (OOF) train scores and
test-set scores for every learner -- the test scores are necessary
input to any eventual ablation experiment's *evaluation* (you need them
to measure an ablated meta-model's held-out accuracy at all) and are
cheap to compute alongside the OOF scores (same fitted-on-full-train
estimator, one extra ``.predict_proba``/``.decision_function`` call).

But looking at test-set accuracy *while choosing* which architecture,
hyperparameters, or (family, learner) combinations to keep is a
different thing from merely computing it, and is exactly the kind of
repeated-testing/benchmark-overfitting risk this project already
guards against at the 142-vs-20-dataset level (see
``analyze_family_strengths.py``'s dev-subset rationale) -- just at a
finer grain: the 5 resamples' train/test indices are the same fixed,
deterministic files (``PythonResampleIndices/``) every time this script
(or the eventual 142-dataset final run) is invoked, so an architecture
choice made by eyeballing test accuracy on these 20 datasets' resamples
during development would, when the same fixed splits are reused for the
"final" 142-dataset evaluation, report an inflated number for exactly
these 20 datasets relative to genuine generalization.

Checked directly (not assumed) how much this actually matters here:
across the resampling scheme's 5 resamples (resample 0 = the official
split; resamples 1-4 pool train+test and re-split via the same fixed
index files), the *median* dev-subset dataset has only ~3% of its
pooled samples that never appear in ANY of the 5 resamples' training
sets -- for most datasets, "the test set" is a different ~50% each
resample, so the specific rows any one resample holds out are, in
aggregate across resamples, mostly also trained on at some point. But
this is not true for every dataset: MoteStrain (official split
n_train=20/n_test=1252), CinCECGTorso (40/1380), and FacesUCR
(200/2050) all have such small official training sets that 63-92% of
their pooled samples are *never* in any resample's training set at
all -- for these three specifically, "the test set" is essentially the
same ~90%+ of samples every resample, so peeking at their test-set
accuracy during development is NOT diluted the way it is for the other
17 datasets.

Practical rule going forward: use OOF-train metrics (never test-set
metrics) as the signal for any development-stage decision -- which
learners to keep per family, hyperparameter choices, architecture
changes -- for every dataset, and treat that discipline as non-optional
specifically for MoteStrain/CinCECGTorso/FacesUCR. Test-set scores
saved by this module are for the eventual real evaluation pass (once
the design is stable) and for computing an ablation's *final* reported
number, not for choosing which ablation looks promising.

Output layout, under RESULTS_DIR (results/stacking_features/)
---------------------------------------------------------------
    <dataset>/resample<k>/scores.npz
        For every family in base_learners.FAMILIES and learner in
        base_learners.LEARNER_NAMES that scored successfully (see
        base_learners.run_family_base_learners's docstring for the one
        way a learner can fail -- a genuinely singleton training
        class breaking RidgeClassifierCV's OOF scoring specifically):
            "{family}__{learner}__oof_train"  (n_train, n_classes)
            "{family}__{learner}__test"       (n_test, n_classes)
        A learner that errored has neither key -- check manifest.csv's
        ``Error`` column (or this file's own ``"{family}__{learner}__error"``
        string key) rather than assuming every one of the 12 combinations
        is present. Also holds, shared across every family (asserted
        equal when this job runs -- see ``_process_dataset_resample_job``):
            "y_train", "y_test"      raw (un-encoded) labels
            "y_train_enc", "y_test_enc"   integer-encoded, matching
                                           "classes_"'s order
            "classes_"                the label encoder's classes, i.e.
                                       column c of every *_train/*_test
                                       array above is the score for
                                       class classes_[c]
    units/<dataset>/resample<k>/
        Per-job bookkeeping (this script's job granularity is one
        (dataset, resample) -- all 4 families/12 learners together, not
        split further the way save_dev_subset_transforms.py's raw
        extraction is: these learners are cheap enough, and similar
        enough in cost to each other, that the finer split isn't needed
        for good load balancing -- see that module's own "Parallelism"
        section for why it split further, which doesn't apply here): a
        ``DONE`` marker (resumable reruns) and a ``manifest.csv``
        fragment (one row per (family, learner): NClasses, NTrain,
        NTest, FitSeconds, Error). ``ERROR.txt`` instead of ``DONE`` if
        the whole job raised (distinct from one *learner* erroring
        inside a job, which is recorded per-row in the manifest and
        does not fail the job -- see base_learners's own docstring).
    manifest.csv
        All completed jobs' manifest fragments, concatenated by
        ``finalize()`` (safe to re-run standalone).

Usage
-----
    python -m src.experiments.build_stacking_features
    python -m src.experiments.build_stacking_features --max-workers 8
    python -m src.experiments.build_stacking_features --dry-run
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Same reasoning as save_dev_subset_transforms.py's identical block: must
# happen before numpy (or anything that transitively imports it) is
# imported, so every worker process -- each a fresh "spawn"-started
# interpreter that re-imports this module -- picks these up before BLAS
# initializes. RidgeClassifierCV/ExtraTreesClassifier both use BLAS/
# OpenMP internally, so without this, N worker processes each also
# spawning their own BLAS thread pool would oversubscribe the very core
# count this script detects and schedules against.
# --------------------------------------------------------------------------- #
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

from src.experiments.common import detect_cpu_count, dev_subset_datasets, load_raw_family_features  # noqa: E402
from src.tsfm_ensemble.base_learners import FAMILIES, LEARNER_NAMES, run_family_base_learners  # noqa: E402

RESULTS_DIR = Path(os.environ.get("STACKING_FEATURES_RESULTS_DIR", str(Path("results") / "stacking_features")))

# Matches save_dev_subset_transforms.py's own NUM_RESAMPLES -- the raw/
# cache this script reads only has these 5 resamples saved.
NUM_RESAMPLES = 5
RANDOM_STATE = 0
RESERVED_CORES = 1


def _paths(dataset: str, resample: int):
    job_dir = RESULTS_DIR / "units" / dataset / f"resample{resample}"
    scores_path = RESULTS_DIR / dataset / f"resample{resample}" / "scores.npz"
    return job_dir, scores_path


def _is_done(dataset: str, resample: int) -> bool:
    job_dir, scores_path = _paths(dataset, resample)
    return (job_dir / "DONE").is_file() and scores_path.is_file()


def _process_dataset_resample_job(dataset: str, resample: int) -> dict:
    """Worker entry point: run every family's base-learner portfolio for
    one (dataset, resample), assemble and save ``scores.npz``, and write
    this job's manifest fragment + DONE marker.

    Any exception that escapes the whole job (as opposed to one learner
    failing inside ``run_family_base_learners``, which that function
    already contains -- see its own docstring) is caught and recorded to
    ``ERROR.txt`` rather than propagated, matching this project's
    existing "one bad job must not abort the whole run" convention.
    """
    job_dir, scores_path = _paths(dataset, resample)
    job_dir.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        npz_payload = {}
        manifest_rows = []
        shared_classes = None
        shared_y_train_enc = shared_y_test_enc = None
        shared_y_train = shared_y_test = None

        for family in FAMILIES:
            X_train, y_train = load_raw_family_features(dataset, resample, family, "train")
            X_test, y_test = load_raw_family_features(dataset, resample, family, "test")

            family_results = run_family_base_learners(
                family, X_train, y_train, X_test, random_state=RANDOM_STATE
            )

            for learner_name in LEARNER_NAMES:
                out = family_results[learner_name]
                if "error" in out:
                    npz_payload[f"{family}__{learner_name}__error"] = np.asarray(out["error"])
                    manifest_rows.append({
                        "Dataset": dataset, "Resample": resample, "Family": family,
                        "Learner": learner_name, "NClasses": None,
                        "NTrain": X_train.shape[0], "NTest": X_test.shape[0],
                        "FitSeconds": None, "Error": out["error"],
                    })
                    continue

                npz_payload[f"{family}__{learner_name}__oof_train"] = out["oof_train"]
                npz_payload[f"{family}__{learner_name}__test"] = out["test"]
                manifest_rows.append({
                    "Dataset": dataset, "Resample": resample, "Family": family,
                    "Learner": learner_name, "NClasses": len(out["classes_"]),
                    "NTrain": X_train.shape[0], "NTest": X_test.shape[0],
                    "FitSeconds": out["fit_seconds"], "Error": None,
                })

                # Every family fits its own LabelEncoder independently
                # inside run_family_base_learners (on the same y_train
                # array every family receives, since it's the same
                # dataset/resample) -- deterministic sorting means they
                # should always agree; assert that instead of silently
                # trusting it, so a real data-loading inconsistency
                # between families surfaces immediately as a loud
                # per-job error rather than a silently-misaligned
                # stacking matrix downstream.
                if shared_classes is None:
                    shared_classes = out["classes_"]
                else:
                    if not np.array_equal(shared_classes, out["classes_"]):
                        raise RuntimeError(
                            f"{dataset}/resample{resample}: class ordering disagrees between "
                            f"families ({shared_classes!r} vs {family}'s {out['classes_']!r}) -- "
                            "this should be impossible for the same (dataset, resample)."
                        )

            if shared_classes is None:
                raise RuntimeError(
                    f"{dataset}/resample{resample}: every learner for family {family!r} failed -- "
                    "no classes_ available to encode labels against. Check this job's manifest "
                    "fragment / ERROR.txt for the individual learner errors."
                )

            if shared_y_train is None:
                shared_y_train, shared_y_test = y_train, y_test
                shared_y_train_enc = np.searchsorted(shared_classes, y_train)
                shared_y_test_enc = np.searchsorted(shared_classes, y_test)
            elif not (np.array_equal(shared_y_train, y_train) and np.array_equal(shared_y_test, y_test)):
                raise RuntimeError(
                    f"{dataset}/resample{resample}: raw labels disagree between families for "
                    f"{family} -- every family should be reading the same cached (dataset, resample)."
                )

        npz_payload["classes_"] = shared_classes
        npz_payload["y_train"] = shared_y_train
        npz_payload["y_test"] = shared_y_test
        npz_payload["y_train_enc"] = shared_y_train_enc
        npz_payload["y_test_enc"] = shared_y_test_enc
        np.savez_compressed(scores_path, **npz_payload)

        pd.DataFrame(manifest_rows).to_csv(job_dir / "manifest.csv", index=False)
        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        n_errors = sum(1 for row in manifest_rows if row["Error"] is not None)
        return {
            "dataset": dataset, "resample": resample, "status": "done",
            "elapsed": time.perf_counter() - start, "n_learner_errors": n_errors,
        }
    except Exception as exc:  # noqa: BLE001 - one bad job must not abort the whole run
        (job_dir / "ERROR.txt").write_text(f"{exc!r}\n\n{traceback.format_exc()}")
        return {
            "dataset": dataset, "resample": resample, "status": "error",
            "elapsed": time.perf_counter() - start, "error": repr(exc),
        }


def finalize() -> None:
    """Aggregate every completed job's manifest fragment into one
    top-level ``manifest.csv``. Safe to call standalone.
    """
    fragments = sorted(RESULTS_DIR.glob("units/*/resample*/manifest.csv"))
    if not fragments:
        print("No completed jobs found; nothing to aggregate.")
        return
    combined = pd.concat([pd.read_csv(p) for p in fragments], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    print(f"Aggregated {len(fragments)} completed job(s) -> {RESULTS_DIR / 'manifest.csv'}")


@dataclass
class Job:
    dataset: str
    resample: int


def build_jobs(datasets: List[str]) -> List[Job]:
    return [
        Job(dataset=dataset, resample=resample)
        for dataset in datasets
        for resample in range(NUM_RESAMPLES)
        if not _is_done(dataset, resample)
    ]


def _worker_init() -> None:
    for env_var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ[env_var] = "1"
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=None, help="Override detected CPU-based worker count.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned job list, then exit without running anything.",
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cpu_count = detect_cpu_count()
    n_workers = args.max_workers or max(1, cpu_count - RESERVED_CORES)

    datasets = dev_subset_datasets()
    jobs = build_jobs(datasets)
    n_total = len(datasets) * NUM_RESAMPLES
    print(
        f"Detected {cpu_count} CPU core(s) -> {n_workers} worker(s).\n"
        f"{len(jobs)} job(s) to run ({n_total - len(jobs)} already done, resuming) "
        f"across {len(datasets)} dataset(s) x {NUM_RESAMPLES} resample(s)."
    )
    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        for job in jobs:
            print(f"  {job.dataset} resample{job.resample}")
        print("--dry-run: exiting without running anything.")
        return

    import concurrent.futures as cf
    import multiprocessing

    mp_context = multiprocessing.get_context("spawn")
    completed, errored, total_learner_errors = 0, 0, 0

    with cf.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp_context, initializer=_worker_init
    ) as executor:
        start_time = time.perf_counter()
        futures = {
            executor.submit(_process_dataset_resample_job, job.dataset, job.resample): job
            for job in jobs
        }
        for future in cf.as_completed(futures):
            job = futures[future]
            result = future.result()
            if result["status"] == "done":
                completed += 1
                total_learner_errors += result["n_learner_errors"]
                extra = f" ({result['n_learner_errors']} learner error(s))" if result["n_learner_errors"] else ""
                print(
                    f"[{completed + errored}/{len(jobs)}] OK   {job.dataset} resample{job.resample} "
                    f"({result['elapsed']:.1f}s){extra}"
                )
            else:
                errored += 1
                print(
                    f"[{completed + errored}/{len(jobs)}] FAIL {job.dataset} resample{job.resample}: "
                    f"{result.get('error')}"
                )
        elapsed = time.perf_counter() - start_time
        print(
            f"\nDone in {elapsed / 60:.1f} min: {completed} succeeded ({total_learner_errors} total "
            f"learner-level error(s) among them), {errored} job(s) failed outright."
        )

    finalize()


if __name__ == "__main__":
    main()
