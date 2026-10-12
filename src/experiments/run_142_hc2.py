"""Run aeon's HIVECOTEV2 (HC2) across the full 142-dataset x 30-resample
UCR archive, measuring training and inference wall-clock runtime for each
(dataset, resample) unit.

Why this exists: HC2 isn't part of this project's own 12-learner stacking
portfolio, and its underlying components (DrCIF/BaseIntervalForest,
Arsenal, TDE) got real performance changes in aeon v1.6.0 (see the
project's own v1.5.0->v1.6.0 audit of RDST/WEASEL2.0/HC2 -- RDST and
WEASEL2.0 came back numerically unchanged, but HC2's component algorithms
did speed up), so HC2's own runtime needs to be measured fresh under
whatever aeon version is actually installed here rather than reused from
an older run or a published number.

Configuration: ``HIVECOTEV2(random_state=RANDOM_STATE, n_jobs=1)`` --
default (unbudgeted, ``time_limit_in_minutes=0``) component sizes, i.e.
the same literature-faithful, paper-matching configuration this project's
sibling ``run_142_original_algorithms.py`` uses for its 4 algorithms and
that ``comparisons/bake-off-redux-2023/HC2_*.csv``'s published numbers
were produced with -- not a custom time-budgeted variant. ``n_jobs=1``
because parallelism here is at the process level (one unit per worker,
see "Parallelism" below); HC2's own internal threading would both
contend for cores and contaminate the runtime numbers this script exists
to collect.

Per-unit timing
----------------
Exactly the two numbers asked for, each a single ``time.perf_counter()``
span:

    TrainSeconds     : ``hc2.fit(X_train, y_train_enc)``
    InferenceSeconds : ``hc2.predict_proba(X_test)`` -- NOT
                        ``hc2.predict(X_test)``: aeon's own
                        ``HIVECOTEV2._predict`` calls
                        ``self.predict_proba(X)`` internally and just
                        argmaxes the result (confirmed by reading aeon's
                        source directly), so calling ``.predict()``
                        afterwards would silently re-run the entire
                        ensemble's inference a second time. Predicted
                        labels for the accuracy sanity-check below are
                        derived from the already-computed ``y_proba`` via
                        a plain (untimed) ``argmax``, exactly mirroring
                        what aeon's own ``_predict`` does internally
                        (Structural tie-breaking is not reproduced --
                        irrelevant for a wall-clock measurement, and
                        ties are rare enough not to matter for the
                        Accuracy sanity check).

A 3rd number, ``UnitTotalSeconds``, is also recorded -- the total
wall-clock for the whole unit (dataset/resample load+split, label
encoding, the 2 timed steps above, accuracy computation, manifest I/O),
measured directly rather than reconstructed from the 2 step timings
after the fact, same reasoning as the sibling 142-scale scripts'
``UnitTotalSeconds``: this is meant to run once, so anything worth
having later needs to be captured now.

``Accuracy``, ``BalancedAccuracy``, ``LogLoss``, ``ROCAUC``, and ``F1``
are also recorded, via ``run_142_original_algorithms.py``'s own
``compute_classification_metrics`` (imported, not duplicated, so HC2's
numbers use exactly the same formulas -- weighted-OvR ROCAUC, minority-
class-or-weighted F1, etc. -- as RDST/QUANT/WEASEL2.0/KG-MTP's). This
was originally accuracy-only, on the reasoning that a full evaluation
wasn't this script's purpose and wasn't asked for; that reasoning didn't
hold up, since HC2 is never re-run for accuracy alone (the published
bake-off-redux-2023 numbers cover that -- see
``table_literature_comparison.py``) but F1 specifically isn't among the
metrics that release published, and this script is already paying for
HC2's fit + predict_proba regardless. All four extra metrics are
computed from the already-in-memory ``y_proba``/``y_pred`` from the
timed inference step above, so they add negligible untimed cost per
unit on top of what this script otherwise does.

Dataset loading, parallelism, thread pinning, resumability
-------------------------------------------------------------
Identical design to ``run_142_original_algorithms.py`` (see that
module's own docstring for the full reasoning, only summarized here):
all 142 datasets' full train+test arrays are loaded once in the parent
process into a module-level global (``_DATASETS``), shared with every
``multiprocessing.get_context("fork")`` worker via copy-on-write --
"spawn" would require re-pickling that whole dict per job instead of
once per worker. Each worker is pinned to 1 BLAS/numba thread via
``threadpoolctl.threadpool_limits(1)`` and ``numba.set_num_threads(1)``
in ``_worker_init`` (env vars alone are not reliable under fork if the
parent's native thread pool already initialized before the fork -- same
reasoning as the sibling script). ``NUMBA_THREADING_LAYER=workqueue`` is
set before aeon is ever imported, belt-and-braces against the same
fork-vs-GNU-OpenMP crash the sibling script hit from ``kg_mtp.py``'s
eager-compiled numba code (not confirmed to affect aeon's own numba use,
but the fix is free and this project has been burned by it once
already). Jobs are scheduled at (dataset, resample) granularity -- no
algorithm dimension, unlike the sibling script -- dataset-interleaved
(shuffled, round-robin across datasets) so a burst of ``n_workers`` jobs
never shares one (large) dataset's peak memory. Each unit writes its own
``units/<dataset>/resample<k>/`` directory (``DONE`` marker +
``manifest.csv`` fragment on success, ``ERROR.txt`` on failure) --
resumable, one bad unit never aborts the run. ``finalize()`` aggregates
into ``RESULTS_DIR/manifest.csv``.

Expect this to run for a long time, unattended, on the user's own server
(NOT the linked development Mac, NOT this sandbox) -- HC2's default,
unbudgeted component sizes (500 DrCIF trees, an unbounded-by-default STC
shapelet search, etc.) are the same literature-faithful choice that made
RDST/KG-MTP slow in the sibling script, and HC2 is the single slowest of
the "original literature algorithm" family by a wide margin.

Usage
-----
    python -m src.experiments.run_142_hc2 --dry-run
    python -m src.experiments.run_142_hc2 --max-workers 40
    python -m src.experiments.run_142_hc2 --datasets Computers,OliveOil --resamples 3
    python -m src.experiments.run_142_hc2 --finalize-only
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Belt-and-braces, same convention/caveats as run_142_original_algorithms.py's
# own top-of-file block -- see that module's "Parallelism and thread pinning
# under fork" docstring section. Not confirmed load-bearing for aeon's own
# numba use (unlike kg_mtp.py's confirmed-crashing eager compile), but free
# insurance against the same class of fork-vs-GNU-OpenMP crash, set before
# aeon is ever imported below.
# --------------------------------------------------------------------------- #
import os  # noqa: E402

for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse  # noqa: E402
import multiprocessing  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

# Blanket suppression, not targeted per-message filtering (superseding the
# earlier targeted filters this block used to hold -- kept here only as a
# record of what was seen: joblib's SequentialBackend-timeout notice,
# sklearn.utils.parallel's delayed/Parallel notice, aeon's
# near-constant-input warning, and scipy.linalg's LinAlgWarning +
# sklearn's own "singular matrix, falling back to least-squares" warning
# out of RidgeClassifierCV inside Arsenal). A fixed-config sweep across
# 142 datasets x 30 resamples will keep surfacing new benign warnings from
# components whose defaults were never tuned for degenerate/ill-conditioned
# per-dataset data, so this ignores every warning outright rather than
# adding filters one at a time. Registered before HC2/aeon/sklearn are
# imported and forked (see the module docstring's fork section) so every
# worker process inherits it via copy-on-write -- it does not need to be
# (and is not) re-registered in ``_worker_init``.
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from aeon.classification.hybrid import HIVECOTEV2  # noqa: E402

from src.experiments.common import (  # noqa: E402
    RESAMPLE_INDICES_DIR,
    load_classification_with_fallback,
    load_resample,
)
from src.experiments.run_142_original_algorithms import (  # noqa: E402
    compute_classification_metrics,
)
from src.tsfm_ensemble.base_learners import RANDOM_STATE  # noqa: E402


def list_142_datasets(resample_indices_dir: Path = RESAMPLE_INDICES_DIR) -> List[str]:
    """The 142-dataset list, taken from ``PythonResampleIndices/``'s own
    subdirectory names -- same function, same reasoning, as every other
    142-scale script in this project (see
    ``run_142_original_algorithms.py``'s own copy's docstring for why
    aeon's own dataset registry isn't used instead). Duplicated here
    rather than imported, matching this project's established convention
    of small, self-contained helpers per 142-scale script.
    """
    return sorted(p.name for p in resample_indices_dir.iterdir() if p.is_dir())


RESULTS_DIR = Path(os.environ.get("RUN_142_HC2_RESULTS_DIR", str(Path("results") / "run_142_hc2")))
UNITS_DIR = RESULTS_DIR / "units"

NUM_RESAMPLES = 30

# Scheduling-only randomness -- same constant as cv_ablation_142.py's and
# run_142_original_algorithms.py's own JOB_SHUFFLE_SEED, for consistency
# across sibling scripts (not load-bearing that they match).
JOB_SHUFFLE_SEED = 20260915


# --------------------------------------------------------------------------- #
# Physical-core detection -- identical to run_142_original_algorithms.py's
# own copy (see that module's "Physical vs. logical cores" docstring
# section).
# --------------------------------------------------------------------------- #
def detect_physical_cpu_count() -> Optional[int]:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    try:
        pairs = set()
        physical_id, core_id = None, None
        with open("/proc/cpuinfo") as f:
            for line in f:
                line = line.strip()
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    if physical_id is not None and core_id is not None:
                        pairs.add((physical_id, core_id))
                elif line == "":
                    physical_id, core_id = None, None
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------- #
# Dataset loading -- load once in the parent, shared with every forked
# worker via copy-on-write. Identical design to
# run_142_original_algorithms.py's own ``_DATASETS``/``_load_all_datasets``.
# --------------------------------------------------------------------------- #
_DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _load_all_datasets(datasets: List[str]) -> List[Tuple[str, str]]:
    """Populate the module-level ``_DATASETS`` global in place. Returns a
    list of ``(dataset, error_repr)`` for any dataset that failed to load
    at all -- those datasets contribute zero jobs rather than aborting
    the whole run.
    """
    failed = []
    for i, dataset in enumerate(datasets):
        start = time.perf_counter()
        try:
            X_train_full, y_train_full = load_classification_with_fallback(dataset, split="train")
            X_test_full, y_test_full = load_classification_with_fallback(dataset, split="test")
            X_train_full = np.ascontiguousarray(X_train_full, dtype=np.float64)
            X_test_full = np.ascontiguousarray(X_test_full, dtype=np.float64)
            _DATASETS[dataset] = (X_train_full, y_train_full, X_test_full, y_test_full)
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not abort the load
            print(f"  [{i + 1:>3}/{len(datasets)}] FAILED to load {dataset}: {exc!r}")
            failed.append((dataset, repr(exc)))
        else:
            elapsed = time.perf_counter() - start
            print(
                f"  [{i + 1:>3}/{len(datasets)}] loaded {dataset} "
                f"(train {X_train_full.shape}, test {X_test_full.shape}, {elapsed:.1f}s)"
            )
    return failed


# --------------------------------------------------------------------------- #
# Per-unit work
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    dataset: str
    resample: int


def _unit_dir(dataset: str, resample: int) -> Path:
    return UNITS_DIR / dataset / f"resample{resample}"


def _worker_init() -> None:
    """Runs once per worker process at startup (any start method,
    including "fork") -- same mechanism, same reasoning, as
    run_142_original_algorithms.py's own ``_worker_init``.
    """
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
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass


def _run_unit(dataset: str, resample: int) -> Dict:
    """Run one (dataset, resample) unit against the shared, already-loaded
    ``_DATASETS`` global: fit HC2, time training, time a single
    ``predict_proba`` inference pass (see module docstring for why not
    ``.predict()``), and record an untimed accuracy sanity check. Any
    exception is caught and recorded to ``ERROR.txt`` rather than
    propagated -- one bad unit must not take down the whole run.
    """
    job_dir = _unit_dir(dataset, resample)
    job_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        X_train_full, y_train_full, X_test_full, y_test_full = _DATASETS[dataset]
        X_train, y_train, X_test, y_test = load_resample(
            dataset, resample, X_train_full, y_train_full, X_test_full, y_test_full
        )

        le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        n_classes = len(le.classes_)

        hc2 = HIVECOTEV2(random_state=RANDOM_STATE, n_jobs=1)

        t0 = time.perf_counter()
        hc2.fit(X_train, y_train_enc)
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_proba = hc2.predict_proba(X_test)
        inference_s = time.perf_counter() - t0

        y_pred = np.argmax(y_proba, axis=1)
        # Untimed -- see module docstring's "Metrics" section: y_proba is
        # already sitting in memory from the timed predict_proba call above,
        # so BalancedAccuracy/LogLoss/ROCAUC/F1 cost only a few cheap sklearn
        # calls on top of it, identical formulas to
        # run_142_original_algorithms.py's own compute_classification_metrics
        # (imported, not duplicated, so HC2's numbers are computed exactly
        # the same way as RDST/QUANT/WEASEL2.0/KG-MTP's).
        metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba, n_classes)

        row = {
            "Dataset": dataset, "Resample": resample,
            "NumTrainInstances": X_train.shape[0], "NumTestInstances": X_test.shape[0],
            "NumClasses": n_classes,
            "TrainSeconds": train_s, "InferenceSeconds": inference_s,
            "UnitTotalSeconds": time.perf_counter() - start,
            "Accuracy": metrics["Accuracy"],
            "BalancedAccuracy": metrics["BalancedAccuracy"],
            "LogLoss": metrics["LogLoss"],
            "ROCAUC": metrics["ROCAUC"],
            "F1": metrics["F1"],
        }
        pd.DataFrame([row]).to_csv(job_dir / "manifest.csv", index=False)

        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        return {"dataset": dataset, "resample": resample, "status": "done",
                "elapsed": time.perf_counter() - start}
    except Exception:  # noqa: BLE001 - one bad unit must not abort the whole run
        (job_dir / "ERROR.txt").write_text(traceback.format_exc())
        return {"dataset": dataset, "resample": resample, "status": "error",
                "elapsed": time.perf_counter() - start}


# --------------------------------------------------------------------------- #
# Job building / resumability / finalize
# --------------------------------------------------------------------------- #
def build_jobs(datasets: List[str], resamples: int) -> List[Job]:
    """(dataset, resample) units still missing their ``DONE`` marker,
    dataset-interleaved (shuffled dataset order, shuffled per-dataset job
    order, round-robin across datasets) so a burst of ``n_workers`` jobs
    dispatched at once never shares one (large) dataset's peak memory --
    same fix, same reasoning, as ``cv_ablation_142.py``'s and
    ``run_142_original_algorithms.py``'s own ``build_jobs``.
    """
    by_dataset: Dict[str, List[Job]] = {}
    for dataset in datasets:
        for resample in range(resamples):
            if not (_unit_dir(dataset, resample) / "DONE").is_file():
                by_dataset.setdefault(dataset, []).append(Job(dataset, resample))

    scheduler_rng = random.Random(JOB_SHUFFLE_SEED)
    dataset_order = list(by_dataset)
    scheduler_rng.shuffle(dataset_order)
    for dataset in dataset_order:
        scheduler_rng.shuffle(by_dataset[dataset])

    queues = [by_dataset[dataset] for dataset in dataset_order]
    jobs: List[Job] = []
    while queues:
        next_round = []
        for queue in queues:
            jobs.append(queue.pop(0))
            if queue:
                next_round.append(queue)
        queues = next_round
    return jobs


def finalize() -> pd.DataFrame:
    """Concatenate every completed unit's manifest fragment into
    ``RESULTS_DIR/manifest.csv``. Safe to call at any time, including
    while a run is still in progress elsewhere (``--finalize-only``).
    """
    rows = []
    for manifest_path in UNITS_DIR.glob("*/*/manifest.csv"):
        if (manifest_path.parent / "DONE").is_file():
            rows.append(pd.read_csv(manifest_path))
    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame(
            columns=[
                "Dataset", "Resample", "NumTrainInstances", "NumTestInstances", "NumClasses",
                "TrainSeconds", "InferenceSeconds", "UnitTotalSeconds", "Accuracy",
                "BalancedAccuracy", "LogLoss", "ROCAUC", "F1",
            ]
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Number of worker processes (should be the server's PHYSICAL core count). "
             "Defaults to detect_physical_cpu_count(), falling back to 40 with a warning if detection fails.",
    )
    parser.add_argument("--resamples", type=int, default=NUM_RESAMPLES, help=f"Number of resamples (default {NUM_RESAMPLES}).")
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset subset to run (default: all 142). Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--finalize-every", type=int, default=50,
        help="Re-run finalize() every N completed units, so results/manifest.csv stays current during a long run "
             "(default 50 -- lower than run_142_original_algorithms.py's 200 since HC2 units are much slower/rarer; "
             "0 disables periodic finalization, still finalizes once at the end).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the detected resources and the planned job schedule, then exit without running anything.",
    )
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="Skip loading datasets / running anything; just (re)build results/manifest.csv from whatever units "
             "are already DONE on disk, then exit.",
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UNITS_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        df = finalize()
        print(f"Finalized {len(df)} completed unit(s) -> {RESULTS_DIR / 'manifest.csv'}")
        return

    datasets = args.datasets.split(",") if args.datasets else list_142_datasets()

    physical = detect_physical_cpu_count()
    if args.max_workers is not None:
        n_workers = args.max_workers
    elif physical:
        n_workers = physical
    else:
        n_workers = 40
        print("WARNING: could not detect physical core count (no psutil, no /proc/cpuinfo); "
              "defaulting --max-workers to 40 -- pass --max-workers explicitly to override.")
    print(f"Detected physical core count: {physical!r}. Using {n_workers} worker process(es).")

    print(f"\nLoading {len(datasets)} dataset(s) once, in the main process...")
    load_start = time.perf_counter()
    failed_datasets = _load_all_datasets(datasets)
    print(f"Done loading in {time.perf_counter() - load_start:.1f}s "
          f"({len(datasets) - len(failed_datasets)}/{len(datasets)} loaded successfully).")
    if failed_datasets:
        pd.DataFrame(failed_datasets, columns=["Dataset", "Error"]).to_csv(RESULTS_DIR / "load_errors.csv", index=False)
        print(f"WARNING: {len(failed_datasets)} dataset(s) failed to load and contribute 0 jobs "
              f"(see {RESULTS_DIR / 'load_errors.csv'}).")

    loadable_datasets = [d for d in datasets if d in _DATASETS]
    jobs = build_jobs(loadable_datasets, args.resamples)
    n_total_possible = len(loadable_datasets) * args.resamples
    n_already_done = n_total_possible - len(jobs)
    print(f"\n{len(jobs)} job(s) to run ({n_already_done} already done, resuming) across "
          f"{len(loadable_datasets)} loadable dataset(s) x {args.resamples} resample(s) "
          f"= {n_total_possible} possible unit(s).")

    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    import concurrent.futures as cf

    # "fork", not "spawn" -- required for the load-once/copy-on-write
    # design. Only available on Linux/macOS, which is fine -- this is
    # meant to run on the user's Linux server.
    mp_context = multiprocessing.get_context("fork")

    completed, errored = 0, 0
    start_time = time.perf_counter()

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context, initializer=_worker_init) as executor:
        in_flight = {}
        remaining = list(jobs)

        while remaining or in_flight:
            while remaining and len(in_flight) < n_workers:
                job = remaining.pop(0)
                future = executor.submit(_run_unit, job.dataset, job.resample)
                in_flight[future] = job

            done_futures, _ = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            for future in done_futures:
                job = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker-process-level crash must not abort the run
                    print(f"WORKER CRASH on {job.dataset} resample{job.resample}: {exc!r}")
                    errored += 1
                    continue
                if result["status"] == "done":
                    completed += 1
                else:
                    errored += 1
                n_finished = completed + errored
                elapsed = time.perf_counter() - start_time
                rate = n_finished / elapsed if elapsed > 0 else 0.0
                eta = (len(jobs) - n_finished) / rate if rate > 0 else float("nan")
                print(
                    f"[{n_finished}/{len(jobs)}] {completed} done, {errored} errored "
                    f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min remaining) -- "
                    f"last: {job.dataset} resample{job.resample} ({result['status']}, {result['elapsed']:.1f}s)"
                )
                if args.finalize_every and n_finished % args.finalize_every == 0:
                    finalize()

    print(f"\nRun complete: {completed} done, {errored} errored, "
          f"{time.perf_counter() - start_time:.1f}s total.")
    df = finalize()
    print(f"Finalized {len(df)} completed unit(s) -> {RESULTS_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
