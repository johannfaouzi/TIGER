"""Save every family's raw (pre-reduction) transform output across the
full 142-dataset x 30-resample archive -- the 142-scale sibling of
``save_dev_subset_transforms.py`` (which only covers the 20-dataset dev
subset x 5 resamples).

NOT required for CAWPE / build_meta_model_predictions_142.py
------------------------------------------------------------------
This was originally written because ``build_meta_model_predictions.py``'s
dev-subset CAWPE (``meta_learners.calibrate_ridge_family``) refits a
fresh ridge classifier on raw features to calibrate it. Turns out that
refit is unnecessary: calibration only needs each ridge block's already-
saved OOF/test decision_function scores plus the true labels (see
``meta_learners.calibrate_ridge_family_from_scores``'s docstring for the
full reasoning), which ``cv_ablation_142.py``'s fragments already carry.
``build_meta_model_predictions_142.py`` uses that scores-only calibration
and never reads this script's output -- confirmed by running it with no
raw cache present at all and getting identical results. This script is
kept as a standalone utility (e.g. for inspecting/debugging raw
per-family features at 142 scale, or some future experiment that
genuinely needs them) but is NOT part of the 142-scale meta-learner
pipeline and does not need to be run for it -- skip it, it only costs
disk space for no benefit to that pipeline.

Output layout matches ``save_dev_subset_transforms.py``'s exactly (same
dense-always convention, same is_sparse-flagged .npz format via
``common.load_features_npz``/``_save_features`` -- see that module's
docstring for the measurements behind always densifying), just under a
different ``RESULTS_DIR`` and over all 142 datasets x (``--resamples``,
default 30) resamples instead of the dev subset:
    raw/<dataset>/resample<k>/<family>_{train,test}.npz
    units/<dataset>/resample<k>/<family>/{DONE,ERROR.txt,manifest.csv}
    manifest.csv (written by finalize())

Scheduling: reused directly from cv_ablation_142.py, not from
save_dev_subset_transforms.py
-----------------------------------------------------------------
save_dev_subset_transforms.py's own scheduler (a custom memory-budget
admission loop, "spawn" workers, per-job dataset (re)loading) was built
for a 20-dataset run where memory-aware admission control mattered more
than dataset-load cost. At 142-dataset scale, cv_ablation_142.py already
solved the harder version of this exact problem -- same 4 families' same
extractors, same full archive including the same largest datasets (e.g.
Crop, ~24,000 instances, which cv_ablation_142.py's own build_jobs
docstring documents as having caused a real dataset-major-ordering OOM
crash) -- via load-once-in-the-parent + fork (copy-on-write dataset
sharing, no per-job reload cost) + dataset-interleaved round-robin job
ordering (guarantees no two of the next N dispatched jobs share a
dataset, which is what actually fixed that crash -- see
cv_ablation_142.py's build_jobs docstring). This script's per-unit
memory footprint is a strict subset of cv_ablation_142.py's own (pure
extraction, no base-learner CV/bagging fitting on top), so the same
scheduling approach is reused as-is (duplicated, not imported -- matching
this project's convention of sibling experiment scripts not depending on
each other's import-time side effects) rather than reinventing
save_dev_subset_transforms.py's memory-budget machinery for a workload
already proven safe at this exact scale.

Usage
-----
    python -m src.experiments.save_142_transforms --dry-run
    python -m src.experiments.save_142_transforms --max-workers 40
    python -m src.experiments.save_142_transforms --datasets OliveOil,PigCVP --resamples 2
    python -m src.experiments.save_142_transforms --finalize-only
"""
from __future__ import annotations

import os  # noqa: E402

for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")
# Load-bearing under fork, not belt-and-braces -- see
# run_142_original_algorithms.py's and cv_ablation_142.py's own module
# docstrings: kg_mtp.py's eager-compiled parallel=True numba function
# initializes GNU OpenMP at import time, which aborts on a later fork()
# unless numba is forced onto its plain-pthreads "workqueue" layer first.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse  # noqa: E402
import inspect  # noqa: E402
import multiprocessing  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse as sp  # noqa: E402

from src.experiments.common import (  # noqa: E402
    RESAMPLE_INDICES_DIR,
    load_classification_with_fallback,
    load_features_npz,
    load_resample,
)
from src.tsfm_ensemble.base_learners import FAMILIES, RANDOM_STATE  # noqa: E402
from src.tsfm_ensemble.families import FAMILY_REGISTRY  # noqa: E402

RESULTS_DIR = Path(os.environ.get("TRANSFORMS_142_RESULTS_DIR", str(Path("results") / "transforms_142")))
NUM_RESAMPLES = 30
# Scheduling-only randomness -- see cv_ablation_142.py's own
# JOB_SHUFFLE_SEED comment: distinct from RANDOM_STATE (the scientific
# seed) on purpose, since shuffling here has zero effect on any saved
# result, only on which unit runs when.
JOB_SHUFFLE_SEED = 20260917

_load_raw_features = load_features_npz  # same convention as save_dev_subset_transforms.py's own alias


def list_142_datasets(resample_indices_dir: Path = RESAMPLE_INDICES_DIR) -> List[str]:
    """Duplicated from cv_ablation_142.py / run_142_original_algorithms.py
    -- see either module's docstring for why this reads
    PythonResampleIndices/ directly instead of aeon's tsc_datasets
    registry.
    """
    return sorted(p.name for p in resample_indices_dir.iterdir() if p.is_dir())


def detect_physical_cpu_count() -> Optional[int]:
    """Duplicated from cv_ablation_142.py -- see that module for the
    psutil-then-/proc/cpuinfo-then-None fallback chain.
    """
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


# --------------------------------------------------------------------------- #
# Dataset loading -- load once in the parent, shared with every forked
# worker via copy-on-write. Duplicated from cv_ablation_142.py.
# --------------------------------------------------------------------------- #
_DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _load_all_datasets(datasets: List[str]) -> List[Tuple[str, str]]:
    failed = []
    for i, dataset in enumerate(datasets):
        start = time.perf_counter()
        try:
            X_train_full, y_train_full = load_classification_with_fallback(dataset, split="train")
            X_test_full, y_test_full = load_classification_with_fallback(dataset, split="test")
            X_train_full = np.ascontiguousarray(X_train_full, dtype=np.float64)
            X_test_full = np.ascontiguousarray(X_test_full, dtype=np.float64)
            _DATASETS[dataset] = (X_train_full, y_train_full, X_test_full, y_test_full)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i + 1:>3}/{len(datasets)}] FAILED to load {dataset}: {exc!r}")
            failed.append((dataset, repr(exc)))
        else:
            elapsed = time.perf_counter() - start
            print(f"  [{i + 1:>3}/{len(datasets)}] loaded {dataset} "
                  f"(train {X_train_full.shape}, test {X_test_full.shape}, {elapsed:.1f}s)")
    return failed


# --------------------------------------------------------------------------- #
# Per-unit work -- same dense-always convention as save_dev_subset_transforms.py
# --------------------------------------------------------------------------- #
def _paths(dataset: str, resample: int, family: str) -> Tuple[Path, Path]:
    job_dir = RESULTS_DIR / "units" / dataset / f"resample{resample}" / family
    raw_dir = RESULTS_DIR / "raw" / dataset / f"resample{resample}"
    return job_dir, raw_dir


def _is_done(dataset: str, resample: int, family: str) -> bool:
    job_dir, _ = _paths(dataset, resample, family)
    return (job_dir / "DONE").is_file()


def _to_dense_output(X_train, X_test):
    """Same convention as save_dev_subset_transforms.py's own
    ``_to_dense_output`` -- see that module's docstring's "Output layout"
    section for the measurements (RidgeClassifierCV 38-62x slower on
    sparse input, ExtraTreesClassifier ~2.5x slower) behind always
    densifying rather than keeping sparse storage as a hedge.
    """
    if not sp.issparse(X_train):
        return X_train, X_test
    return np.asarray(X_train.todense()), np.asarray(X_test.todense())


def _save_features(path: Path, X, y) -> int:
    """Identical convention to save_dev_subset_transforms.py's own
    ``_save_features`` (duplicated, not imported -- see this module's
    docstring on why sibling experiment scripts aren't cross-imported).
    """
    if sp.issparse(X):
        X_csr = X.tocsr()
        np.savez_compressed(
            path, data=X_csr.data, indices=X_csr.indices, indptr=X_csr.indptr,
            shape=np.array(X_csr.shape), y=np.asarray(y), is_sparse=True,
        )
    else:
        np.savez_compressed(path, X=np.asarray(X), y=np.asarray(y), is_sparse=False)
    return path.stat().st_size


def _make_extractor(family: str):
    cls = FAMILY_REGISTRY[family]
    accepted = inspect.signature(cls.__init__).parameters
    return cls(random_state=RANDOM_STATE) if "random_state" in accepted else cls()


def _process_family_job(dataset: str, resample: int, family: str) -> Dict:
    """Worker entry point: run one family's extractor for one
    (dataset, resample, family) job, save its raw output, write this
    job's manifest fragment + DONE marker. Any exception is caught and
    recorded to ERROR.txt rather than propagated -- one bad job must not
    take down the whole run (same convention as every other per-unit
    worker in this codebase).
    """
    job_dir, raw_dir = _paths(dataset, resample, family)
    job_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        raw_train_path = raw_dir / f"{family}_train.npz"
        raw_test_path = raw_dir / f"{family}_test.npz"

        if raw_train_path.is_file() and raw_test_path.is_file():
            X_raw_train, _y_train = _load_raw_features(raw_train_path)
            raw_train_bytes = raw_train_path.stat().st_size
            raw_test_bytes = raw_test_path.stat().st_size
        else:
            X_train_full, y_train_full, X_test_full, y_test_full = _DATASETS[dataset]
            X_train, y_train, X_test, y_test = load_resample(
                dataset, resample, X_train_full, y_train_full, X_test_full, y_test_full
            )

            extractor = _make_extractor(family)
            X_raw_train = extractor.fit_transform(X_train, y_train)
            X_raw_test = extractor.transform(X_test)
            X_raw_train, X_raw_test = _to_dense_output(X_raw_train, X_raw_test)
            raw_train_bytes = _save_features(raw_train_path, X_raw_train, y_train)
            raw_test_bytes = _save_features(raw_test_path, X_raw_test, y_test)
            del extractor, X_raw_test

        row = {
            "Dataset": dataset, "Resample": resample, "Family": family,
            "NumFeaturesRaw": X_raw_train.shape[1],
            "IsRawSparse": bool(sp.issparse(X_raw_train)),
            "RawTrainBytes": raw_train_bytes, "RawTestBytes": raw_test_bytes,
            "ElapsedSeconds": time.perf_counter() - start,
        }
        pd.DataFrame([row]).to_csv(job_dir / "manifest.csv", index=False)
        del X_raw_train

        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        return {"dataset": dataset, "resample": resample, "family": family, "status": "done",
                "elapsed": time.perf_counter() - start}
    except Exception:  # noqa: BLE001 - one bad job must not abort the whole run
        (job_dir / "ERROR.txt").write_text(traceback.format_exc())
        return {"dataset": dataset, "resample": resample, "family": family, "status": "error",
                "elapsed": time.perf_counter() - start}


# --------------------------------------------------------------------------- #
# Job building / resumability / finalize
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    dataset: str
    resample: int
    family: str


def build_jobs(datasets: List[str], resamples: int, families: List[str]) -> List[Job]:
    """Dataset-interleaved round-robin job order -- duplicated from
    cv_ablation_142.py's own build_jobs (see that function's docstring
    for the full "why": dataset-major ordering let several workers
    extract the same huge dataset simultaneously and crashed on Crop).
    """
    by_dataset: Dict[str, List[Job]] = {}
    for dataset in datasets:
        for resample in range(resamples):
            for family in families:
                if not _is_done(dataset, resample, family):
                    by_dataset.setdefault(dataset, []).append(Job(dataset, resample, family))

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
    fragments = sorted(RESULTS_DIR.glob("units/*/resample*/*/manifest.csv"))
    rows = [pd.read_csv(p) for p in fragments if (p.parent / "DONE").is_file()]
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-workers", type=int, default=None,
                         help="Should be the server's PHYSICAL core count. Defaults to detection, falling back to 40.")
    parser.add_argument("--resamples", type=int, default=NUM_RESAMPLES)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated subset (default: all 142).")
    parser.add_argument("--families", type=str, default=None, help=f"Comma-separated subset (default: all of {list(FAMILIES)}).")
    parser.add_argument("--finalize-every", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "units").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "raw").mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        df = finalize()
        print(f"Finalized {len(df)} completed unit(s) -> {RESULTS_DIR / 'manifest.csv'}")
        return

    datasets = args.datasets.split(",") if args.datasets else list_142_datasets()
    families = args.families.split(",") if args.families else list(FAMILIES)
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}; expected one of {list(FAMILIES)}.")

    physical = detect_physical_cpu_count()
    if args.max_workers is not None:
        n_workers = args.max_workers
    elif physical:
        n_workers = physical
    else:
        n_workers = 40
        print("WARNING: could not detect physical core count; defaulting --max-workers to 40.")
    print(f"Detected physical core count: {physical!r}. Using {n_workers} worker process(es).")

    print(f"\nLoading {len(datasets)} dataset(s) once, in the main process...")
    load_start = time.perf_counter()
    failed_datasets = _load_all_datasets(datasets)
    print(f"Done loading in {time.perf_counter() - load_start:.1f}s "
          f"({len(datasets) - len(failed_datasets)}/{len(datasets)} loaded successfully).")
    if failed_datasets:
        pd.DataFrame(failed_datasets, columns=["Dataset", "Error"]).to_csv(RESULTS_DIR / "load_errors.csv", index=False)
        print(f"WARNING: {len(failed_datasets)} dataset(s) failed to load (see {RESULTS_DIR / 'load_errors.csv'}).")

    loadable_datasets = [d for d in datasets if d in _DATASETS]
    jobs = build_jobs(loadable_datasets, args.resamples, families)
    n_total_possible = len(loadable_datasets) * args.resamples * len(families)
    print(f"\n{len(jobs)} job(s) to run ({n_total_possible - len(jobs)} already done, resuming) across "
          f"{len(loadable_datasets)} dataset(s) x {args.resamples} resample(s) x {len(families)} famil(y/ies) "
          f"= {n_total_possible} possible unit(s).")

    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    import concurrent.futures as cf

    mp_context = multiprocessing.get_context("fork")
    completed, errored = 0, 0
    start_time = time.perf_counter()

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context, initializer=_worker_init) as executor:
        in_flight = {}
        remaining = list(jobs)

        while remaining or in_flight:
            while remaining and len(in_flight) < n_workers:
                job = remaining.pop(0)
                future = executor.submit(_process_family_job, job.dataset, job.resample, job.family)
                in_flight[future] = job

            done_futures, _ = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            for future in done_futures:
                job = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"WORKER CRASH on {job.dataset} resample{job.resample} {job.family}: {exc!r}")
                    errored += 1
                    continue
                completed += result["status"] == "done"
                errored += result["status"] != "done"
                n_finished = completed + errored
                if n_finished % 10 == 0 or not (remaining or in_flight):
                    elapsed = time.perf_counter() - start_time
                    rate = n_finished / elapsed if elapsed > 0 else 0.0
                    eta = (len(jobs) - n_finished) / rate if rate > 0 else float("nan")
                    print(f"[{n_finished}/{len(jobs)}] {completed} done, {errored} errored "
                          f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min remaining) -- "
                          f"last: {job.dataset} resample{job.resample} {job.family} "
                          f"({result['status']}, {result['elapsed']:.1f}s)")
                if args.finalize_every and n_finished % args.finalize_every == 0:
                    finalize()

    print(f"\nRun complete: {completed} done, {errored} errored, {time.perf_counter() - start_time:.1f}s total.")
    df = finalize()
    print(f"Finalized {len(df)} completed unit(s) -> {RESULTS_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
