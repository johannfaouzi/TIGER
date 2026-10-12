"""Split TIGER's own base-learner "final fit + predict" block
(``base_learners._fit_and_score_final``, née the inline step-2 block
inside ``run_family_base_learners``) into its two separately-timed
halves, at full 142-dataset x 30-resample scale -- the piece missing
from ``results/cv_ablation_142/manifest.csv``'s ``CvFinalFitSeconds``
(fit and predict-on-test timed together as one block there), needed to
report an exact training-only / inference-only / both split for TIGER
in sn-article.tex Table~\\ref{tab:runtimes}.

Deliberately NOT a rerun of ``cv_ablation_142.py``'s full job: that
script's dominant cost is the 5-fold OOF cross-validation
(``CvOOFSeconds``, already correctly cached and purely train-side, no
need to re-measure) and the bagging/native-OOB ablation arms (not the
deployed strategy -- ``base_learners.run_family_base_learners`` only
ever uses "cv", confirmed directly against the deployed module; see
sn-article.tex Section 5.3/5.4's own text and Figure 1). This script
redoes only: (1) feature extraction (fit_transform on train, transform
on test -- unavoidable, since raw per-family features are not cached
anywhere on disk for the 142-dataset archive; see
``save_142_transforms.py``'s own docstring for why not) and (2) the
final fit + predict step, now timed as two blocks instead of one.
Expected total cost ~= (results/cv_ablation_142/manifest.csv's own
ExtractionTrainSeconds + ExtractionTestSeconds + CvFinalFitSeconds
column totals) -- i.e. the same underlying computation, just
re-instrumented, not new work on top of what that script already did
(skipping CvOOFSeconds and the bagging/native-OOB arms is exactly what
keeps this fast -- see the runtime estimate in this project's own
paper-drafting notes: ~434 core-hours serial, ~11 hours at 40 workers).

Scheduling, dataset (pre)loading, resumability, and the dataset-
interleaved job order are all copied from ``cv_ablation_142.py``
(not imported -- matching this project's convention of sibling
experiment scripts not depending on each other's import-time side
effects; see that module's own docstring). Only the per-unit work
inside ``_run_unit`` differs: extraction, then
``base_learners._fit_and_score_final`` per learner, no OOF, no bagging.

Usage (from the repo root):
    python3 -m src.experiments.measure_base_learner_final_split_142 --dry-run
    python3 -m src.experiments.measure_base_learner_final_split_142 --max-workers 40
"""
from __future__ import annotations

import os  # noqa: E402

for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")
# Load-bearing, not belt-and-braces -- see cv_ablation_142.py's own
# module-level comment (same reasoning, copied verbatim): kg_mtp.py's
# eager-compiled parallel=True numba function initializes GNU OpenMP at
# import time, which aborts on a later fork() unless numba is forced
# onto its plain-pthreads "workqueue" layer first.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse as sp  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=r"The backend class 'SequentialBackend' does not support timeout\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"`sklearn\.utils\.parallel\.delayed` should be used with "
        r"`sklearn\.utils\.parallel\.Parallel` to make it possible to propagate "
        r"the scikit-learn configuration of the current thread to the joblib workers\."
    ),
    category=UserWarning,
)

from src.experiments.common import (  # noqa: E402
    RESAMPLE_INDICES_DIR,
    load_classification_with_fallback,
    load_resample,
)
from src.tsfm_ensemble.base_learners import (  # noqa: E402
    FAMILIES,
    LEARNER_NAMES,
    RANDOM_STATE,
    _fit_and_score_final,
)
from src.tsfm_ensemble.families import FAMILY_REGISTRY  # noqa: E402

RESULTS_DIR = Path(os.environ.get(
    "MEASURE_BASE_LEARNER_FINAL_SPLIT_142_RESULTS_DIR", str(Path("results") / "measure_base_learner_final_split_142")
))
UNITS_DIR = RESULTS_DIR / "units"

NUM_RESAMPLES = 30
# Scheduling-only randomness (see cv_ablation_142.py's build_jobs docstring
# for the full reasoning) -- a distinct seed from both RANDOM_STATE and
# cv_ablation_142.py's own JOB_SHUFFLE_SEED, since this is a different job
# list; has zero effect on any result, only on which unit runs when.
JOB_SHUFFLE_SEED = 20260929


def list_142_datasets(resample_indices_dir: Path = RESAMPLE_INDICES_DIR) -> List[str]:
    """Identical to cv_ablation_142.py's own version -- see that module's
    docstring for why this reads PythonResampleIndices/ directly."""
    return sorted(p.name for p in resample_indices_dir.iterdir() if p.is_dir())


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


def _maybe_densify(Xt):
    if sp.issparse(Xt):
        return np.asarray(Xt.todense())
    return np.asarray(Xt)


# --------------------------------------------------------------------------- #
# Dataset loading -- load once in the parent, shared with every forked
# worker via copy-on-write. Identical to cv_ablation_142.py's own.
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


def _unit_dir(dataset: str, resample: int, family: str) -> Path:
    return UNITS_DIR / dataset / f"resample{resample}" / family


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
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass


def _run_unit(dataset: str, resample: int, family: str) -> Dict:
    job_dir = _unit_dir(dataset, resample, family)
    job_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest_rows: List[Dict] = []

    try:
        X_train_full, y_train_full, X_test_full, y_test_full = _DATASETS[dataset]
        X_train, y_train, X_test, y_test = load_resample(
            dataset, resample, X_train_full, y_train_full, X_test_full, y_test_full
        )

        le = LabelEncoder().fit(y_train)
        y_train_enc = le.transform(y_train)
        try:
            y_test_enc = le.transform(y_test)
        except ValueError:
            # Same rare case cv_ablation_142.py's _run_unit already
            # documents: a test-split label never seen in training.
            y_test_enc = None
        n_classes = len(le.classes_)

        cls = FAMILY_REGISTRY[family]
        import inspect

        accepted = inspect.signature(cls.__init__).parameters
        extractor = cls(random_state=RANDOM_STATE) if "random_state" in accepted else cls()

        t0 = time.perf_counter()
        Xt_train = _maybe_densify(extractor.fit_transform(X_train, y_train))
        extraction_train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        Xt_test = _maybe_densify(extractor.transform(X_test))
        extraction_test_s = time.perf_counter() - t0

        for learner_name in LEARNER_NAMES:
            row = {
                "Dataset": dataset, "Resample": resample, "Family": family, "Learner": learner_name,
                "NTrain": Xt_train.shape[0], "NTest": Xt_test.shape[0], "NClasses": n_classes,
                "ExtractionTrainSeconds": extraction_train_s, "ExtractionTestSeconds": extraction_test_s,
            }
            try:
                test_scores, fit_s, predict_s = _fit_and_score_final(
                    family, learner_name, Xt_train, y_train_enc, Xt_test, random_state=RANDOM_STATE
                )
            except Exception as exc:  # noqa: BLE001 - one learner's failure must not sink the whole unit
                row["Error"] = repr(exc)
            else:
                row["FinalFitSeconds"] = fit_s
                row["FinalPredictSeconds"] = predict_s
                if y_test_enc is not None:
                    row["TestAcc"] = accuracy_score(y_test_enc, test_scores.argmax(axis=1))
            manifest_rows.append(row)

        unit_total_s = time.perf_counter() - start
        for row in manifest_rows:
            row["UnitTotalSeconds"] = unit_total_s

        pd.DataFrame(manifest_rows).to_csv(job_dir / "manifest.csv", index=False)
        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        return {"dataset": dataset, "resample": resample, "family": family, "status": "done",
                "elapsed": time.perf_counter() - start}
    except Exception:  # noqa: BLE001 - one bad unit must not abort the whole run
        (job_dir / "ERROR.txt").write_text(traceback.format_exc())
        return {"dataset": dataset, "resample": resample, "family": family, "status": "error",
                "elapsed": time.perf_counter() - start}


# --------------------------------------------------------------------------- #
# Job building / resumability / finalize -- identical shape to
# cv_ablation_142.py's own (dataset-interleaved round-robin order; see
# that module's build_jobs docstring for why dataset-major order once
# caused a real OOM crash on Crop).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Job:
    dataset: str
    resample: int
    family: str


def build_jobs(datasets: List[str], resamples: int, families: List[str]) -> List[Job]:
    by_dataset: Dict[str, List[Job]] = {}
    for dataset in datasets:
        for resample in range(resamples):
            for family in families:
                if not (_unit_dir(dataset, resample, family) / "DONE").is_file():
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
    rows = []
    for manifest_path in UNITS_DIR.glob("*/*/*/manifest.csv"):
        if (manifest_path.parent / "DONE").is_file():
            rows.append(pd.read_csv(manifest_path))
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
    UNITS_DIR.mkdir(parents=True, exist_ok=True)

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
                future = executor.submit(_run_unit, job.dataset, job.resample, job.family)
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
