"""Save every family's raw (pre-reduction) transform output for the
development subset, so trying a different downstream classification
algorithm on top of a given family's features doesn't need to re-run
the (expensive) extractor from scratch.

Dataset list
------------
The 20 datasets in ``results/family_strengths/dev_subset.csv`` (see
``analyze_family_strengths.py``) -- each is where one specific family
"decisively" beats the other three on its published-algorithm proxy, per
that script's methodology. The four "close call" datasets that script can
also produce are deliberately excluded here (as of the module default
``n_close_calls=0``): a follow-up analysis (``resample_stability.py``)
found their winning family doesn't stabilize even at 30 resamples, so
they're not useful for the kind of per-family comparison this transform
cache is for.

Resample count
--------------
``NUM_RESAMPLES = 5`` -- ``resample_stability.py`` measured, directly
against the 30 real resamples in ``comparisons/``, that mean
win-agreement with the full-30-resample "ground truth" reaches ~92% at
n=5 on the (decisive-only) dev-subset datasets, with sharply diminishing
returns afterward (n=10 only buys another ~4 points for double the
cost). See that script's output for the full curve.

Output layout, under ``RESULTS_DIR`` (``results/dev_subset_transforms/``)
---------------------------------------------------------------------
    raw/<dataset>/resample<k>/<family>_{train,test}.npz
        Every family's extractor output. Dense families
        (convolution/interval/shapelet) save ``X`` (float64) and ``y``.
        The dictionary family's bag-of-words output comes from its
        extractor as sparse CSR, but is always *saved* dense (see
        ``_to_dense_output``): real dev-subset measurements found scipy
        sparse input forces ``RidgeClassifierCV`` onto a much slower
        internal code path than a plain dense array (measured 38-62x
        slower, e.g. 37.5s vs 0.6s on one dataset), ``ExtraTreesClassifier``
        similarly ~2.5x slower, and every one of this dev subset's 20
        datasets' dictionary output measured at least 18% dense -- WEASEL's
        bag-of-words counts are not reliably sparse in the performance
        sense here, despite the sparse *type* -- so this project prefers
        paying dense's larger memory footprint for the speedup rather than
        keeping sparse storage as a hedge. ``{family}_{train,test}.npz``
        therefore saves ``X``/``y`` like the dense families above for
        every family in practice; ``_save_features`` still supports
        saving a still-sparse ``X`` (CSR components plus ``y``) for any
        future family/case that isn't densified, and
        ``common.load_features_npz`` reads either back via its
        ``is_sparse`` flag with no caller-side branching needed.
    units/<dataset>/resample<k>/<family>/
        Per-*job* bookkeeping, one directory per (dataset, resample,
        family) -- matching the scheduling granularity (see
        "Parallelism" below): a ``DONE`` marker (for resuming an
        interrupted run) and a ``manifest.csv`` fragment (one row for
        this job: raw feature count, byte sizes, elapsed time,
        sparsity). ``ERROR.txt`` instead of ``DONE`` if this
        (dataset, resample, family) job failed -- since jobs are this
        fine-grained, one family failing on one dataset no longer forces
        the other 3 (already-successful) families to be redone on a
        rerun; only the failed one is retried.
    manifest.csv
        All completed jobs' manifest fragments, concatenated, written by
        ``finalize()`` at the end of the run (safe to re-run; also
        callable standalone to re-aggregate after an interrupted run).
        Columns: ``Dataset``, ``Resample``, ``Family``, ``NumFeaturesRaw``,
        ``IsRawSparse``, ``RawTrainBytes``, ``RawTestBytes``,
        ``ElapsedSeconds``.

Parallelism
-----------
This is meant to run unattended on a server whose core count and RAM are
unknown ahead of time, so both are detected at runtime (see
``detect_cpu_count``/``detect_available_memory_bytes``) rather than
hardcoded, and jobs are scheduled under a custom memory-aware admission
loop (not a plain ``ProcessPoolExecutor(max_workers=N)``): the four
families have very different peak memory footprints per dataset --
convolution's is set by its ~50-60k-wide dense raw feature matrix almost
irrespective of dataset size, empirically 60-500+ MB per job even on the
smallest datasets here (see ``estimate_family_peak_memory_bytes``'s
docstring for the measurements this is based on) -- so a fixed worker
count sized for the CPU alone risks OOM-killing the whole run on a
memory-constrained box, while sizing it conservatively enough to always
be safe would leave most cores idle on a well-provisioned one. See
``main()`` for the scheduling loop itself.

Jobs are scheduled at (dataset, resample, family) granularity, not
(dataset, resample) -- i.e. each of the 4 families for a given
dataset/resample is its own independently-schedulable unit of work, in
its own worker process. This matters concretely on a high-core-count
box: with the 20-dataset dev subset x 5 resamples = 100
(dataset, resample) units, an 80-core server would admit nearly all of
them at once under the coarser granularity, and the *whole run* would
then wait on however long the single slowest unit takes, working
through its 4 families serially inside one worker the entire time --
worst case, one slow dataset (e.g. a long-series, many-instance one
where convolution's and shapelet's extractors both run long) blocks a
worker slot for 4 families' worth of sequential work while every other
slot has long since gone idle. Splitting to (dataset, resample, family)
turns that same run into up to 400 independently-schedulable jobs: a
worker that finishes a fast family (e.g. dictionary on a short dataset)
immediately picks up the next-largest-remaining job instead of being
locked into that one dataset's other 3 families first, which drains and
refills worker slots at 4x the resolution and meaningfully shortens the
tail. The cost is real but small: raw feature extraction is now
duplicated across independent processes instead of shared within one
(each (dataset, resample, family) job reloads and re-resamples the
dataset itself, rather than 4 families sharing one load) -- empirically
well under a second per dataset (see ``_probe_dataset_shape``'s own
cache-warming pass, timed at 0.08-0.2s on the datasets checked during
development) -- and the per-unit memory model became a per-family one
(``estimate_family_peak_memory_bytes``, no longer a max-over-families
``estimate_unit_peak_memory_bytes``), which if anything is a more
accurate estimate per job, not a less accurate one.

Usage
-----
    python -m src.experiments.save_dev_subset_transforms
    python -m src.experiments.save_dev_subset_transforms --max-workers 8
    python -m src.experiments.save_dev_subset_transforms --memory-budget-gb 32
    python -m src.experiments.save_dev_subset_transforms --dry-run
"""
from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Limit each worker process's own internal BLAS/OpenMP thread pools to 1
# thread. We're already parallelizing across (dataset, resample) units at
# the process level (see main()'s scheduler); letting numpy/scipy/sklearn
# *also* spawn multiple threads per worker would oversubscribe the very
# core count this script goes out of its way to detect and respect, and
# make the "N worker processes" accounting below meaningless. Must happen
# before numpy (or anything that transitively imports it) is imported --
# hence this sits above every other import in the file, including
# stdlib-adjacent ones that pull numpy in.
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
from scipy import sparse as sp  # noqa: E402

from src.experiments.common import (  # noqa: E402
    detect_cpu_count,
    dev_subset_datasets,
    load_classification_with_fallback,
    load_features_npz,
    load_resample,
)

# DEV_SUBSET_TRANSFORMS_RESULTS_DIR lets this be overridden without
# editing the file -- e.g. for a throwaway test run pointed at a scratch
# directory. An env var (rather than a module attribute a caller might
# try to monkeypatch) is what actually works here: every worker process
# re-imports this module fresh (see main()'s "spawn" comment), so a
# plain `save_dev_subset_transforms.RESULTS_DIR = ...` reassignment in
# the parent process would silently *not* apply inside the workers --
# environment variables, unlike Python object state, are inherited
# across that boundary.
RESULTS_DIR = Path(os.environ.get("DEV_SUBSET_TRANSFORMS_RESULTS_DIR", str(Path("results") / "dev_subset_transforms")))

# See module docstring's "Resample count" section for why 5.
NUM_RESAMPLES = 5
FAMILIES = ("convolution", "interval", "shapelet", "dictionary")
RANDOM_STATE = 0

# --------------------------------------------------------------------------- #
# Memory estimation
# --------------------------------------------------------------------------- #
# Raw (pre-reduction) feature-column-count model per family, used only to
# *estimate* peak memory before running anything -- not used anywhere in
# the actual extraction (the real extractors decide their own output
# width; this is a planning-time approximation of it). Grounded in real
# measurements taken across 6 UCR datasets spanning series length 24-1460
# (see the session's own investigation; kept here as inline constants
# rather than re-deriving them at runtime, since that would mean running
# the expensive extractors just to plan how to run them):
#   convolution (KGMTP, num_features=50_000 default): empirically 50768
#   (length 24) to 59328 (length 1460) raw columns -- roughly flat in
#   series length, since it's a fixed feature-count budget. 60_000 covers
#   the observed range with margin.
#   shapelet (RDST, max_shapelets=10_000 default => <=30_000 raw columns
#   at 3 features/shapelet): empirically reaches exactly the 30_000 cap
#   for long-enough series (e.g. OSULeaf, length 427); use the cap.
#   dictionary (WEASEL 2.0, max_feature_count=30_000 default): empirically
#   12945-20108 raw columns (sparse output, but WEASELTransformerV2 always
#   materializes dense internally first -- see DictionaryFamily's
#   docstring -- so plan for dense-sized transient memory); 20_000 covers
#   the observed range with margin.
#   interval (QUANT, depth=6 default): the only one of the four whose raw
#   width scales with series length in this sample -- observed ratios of
#   raw_columns/length were consistently ~9-11x across all 6 lengths
#   tested (e.g. 13021/1460=8.9, 3997/427=9.4, 227/24=9.5); 10x length is
#   the fitted approximation.
EST_RAW_FEATURES = {
    "convolution": lambda length: 60_000,
    "shapelet": lambda length: 30_000,
    "dictionary": lambda length: 20_000,
    "interval": lambda length: max(200, round(10 * length)),
}

# Empirically, peak RSS while running one family's extractor ran
# 2.6x-3.2x the *nominal* size of its raw (train+test) dense array
# (measured via resource.getrusage before/after, on real UCR data) --
# real transformers hold multiple intermediate representations (per-
# dilation buffers, distance matrices, etc.) alive at once, on top of the
# final output array. 4x gives headroom over the observed range; the
# additive term covers baseline Python/numpy/aeon/sklearn process
# overhead (observed ~230 MB at the same measurement points), doubled for
# margin.
PEAK_OVER_NOMINAL_SAFETY_MULTIPLIER = 4.0
BASELINE_PROCESS_OVERHEAD_BYTES = 600 * 1024**2

# Only schedule up to this fraction of detected available RAM -- leaves
# headroom for the OS, the orchestrating main process itself, and any
# other processes sharing the machine (a shared server is exactly the
# case this script is meant to run on).
MEMORY_BUDGET_FRACTION = 0.8
# Leave one core free for the orchestrating main process / OS scheduling
# noise, rather than claiming every detected core for workers.
RESERVED_CORES = 1


def detect_available_memory_bytes() -> int:
    """Best-effort available-RAM detection, in bytes.

    Tries, in order: ``psutil`` (most accurate -- accounts for reclaimable
    cache/buffers correctly across platforms; not a hard dependency of
    this package, so this degrades gracefully if it isn't installed),
    then ``/proc/meminfo``'s ``MemAvailable`` (Linux; already accounts
    for reclaimable memory, unlike naively summing free+cached), then a
    conservative 4 GB fallback with a loud warning -- pass
    ``--memory-budget-gb`` explicitly instead of trusting that fallback.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        for line in meminfo_path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb * 1024

    fallback = 4 * 1024**3
    print(
        "WARNING: could not detect available memory (psutil not installed, "
        "and /proc/meminfo unavailable -- not on Linux?). Falling back to a "
        f"conservative {fallback / 1024**3:.0f} GB assumption; pass "
        "--memory-budget-gb to override with an accurate value."
    )
    return fallback


def estimate_family_peak_memory_bytes(family: str, n_train: int, n_test: int, length: int) -> int:
    """Estimated peak RSS for processing one (dataset, resample, family)
    job -- jobs are scheduled at this granularity (see the module
    docstring's "Parallelism" section for why), each running exactly one
    family's extractor in its own worker process, so this is a per-family
    estimate, not a max-over-families one. See the module-level
    ``EST_RAW_FEATURES``/safety-multiplier comments for where the model
    and constants come from.
    """
    n_instances = n_train + n_test
    nominal_bytes = EST_RAW_FEATURES[family](length) * n_instances * 8
    return int(nominal_bytes * PEAK_OVER_NOMINAL_SAFETY_MULTIPLIER + BASELINE_PROCESS_OVERHEAD_BYTES)


# --------------------------------------------------------------------------- #
# Per-unit work
# --------------------------------------------------------------------------- #
def _paths(dataset: str, resample: int, family: str) -> Tuple[Path, Path]:
    """Every path one (dataset, resample, family) job needs: its own
    bookkeeping dir (DONE/ERROR.txt/manifest fragment -- job-granular,
    matching the scheduling granularity) and the shared raw/ dir for this
    (dataset, resample) (raw output isn't split by family in its own
    directory, just by the ``{family}_{train,test}.npz`` filename prefix
    -- other families' concurrently-running jobs write different
    filenames in the same dir, which is safe).
    """
    job_dir = RESULTS_DIR / "units" / dataset / f"resample{resample}" / family
    raw_dir = RESULTS_DIR / "raw" / dataset / f"resample{resample}"
    return job_dir, raw_dir


def _is_done(dataset: str, resample: int, family: str) -> bool:
    """A job is done once its DONE marker is on disk."""
    job_dir, _ = _paths(dataset, resample, family)
    return (job_dir / "DONE").is_file()


# Always densify a sparse family extractor's output before saving --
# see the module docstring's "Output layout" section for the
# measurements behind this (RidgeClassifierCV 38-62x slower on sparse
# input than dense at the same data, e.g. 37.5s -> 0.6s on
# AsphaltPavementTypeUni's chi2-selected dictionary features;
# ExtraTreesClassifier ~2.5x slower). An earlier version of this
# function only densified once nonzero density passed a threshold
# (0.15), to protect a hypothetical genuinely sparse, high-n dataset
# outside this dev subset from paying dense's larger memory cost for no
# speed benefit -- replaced with an unconditional conversion per this
# project's own stated preference for a faster algorithm at more RAM
# usage over a slower one that uses less. Every one of the 20 dev-subset
# datasets' dictionary output already measured at least 18% dense, and
# no measured case here showed sparse storage winning on speed at any
# density, so this is expected to be a strict improvement in practice,
# not just a preference-driven simplification.
def _to_dense_output(X_train, X_test):
    """Convert both splits of one sparse family job's output to dense
    ``np.ndarray``s -- a no-op when ``X_train`` isn't sparse to begin
    with (the dense families never call this).

    Preserves the sparse matrix's own dtype when converting (e.g.
    WEASEL's ``uint32`` counts) rather than upcasting to ``float64`` here
    -- the same reasoning as ``DictionaryFamily._to_sparse``'s docstring:
    downstream consumers (this project's base learners, via sklearn) that
    actually need float64 already promote internally where they need to.
    """
    if not sp.issparse(X_train):
        return X_train, X_test
    return np.asarray(X_train.todense()), np.asarray(X_test.todense())


def _save_features(path: Path, X, y) -> int:
    """Save one feature matrix (dense or sparse) plus its labels to
    ``path`` (compressed .npz); returns the written file's byte size.
    Sparse input is saved via its CSR components rather than densified
    (see ``_densify_if_dense_enough`` for the density-aware decision of
    *which* family jobs' output ever reaches this function still sparse).
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


# Reverse of _save_features (same is_sparse-flagged .npz convention) --
# shared with any later reader of this raw/ cache (e.g.
# tsfm_ensemble.base_learners, via common.load_raw_family_features) so
# both stay in exact agreement about the file format; see
# common.load_features_npz's own docstring.
_load_raw_features = load_features_npz


def _process_family_job(dataset: str, resample: int, family: str) -> dict:
    """Worker entry point: run *one* family's extractor for one
    (dataset, resample, family) job, save its raw (pre-reduction) output,
    and write this job's own manifest fragment + DONE marker. Returns a
    small status dict for the main process to log.

    Jobs are scheduled at (dataset, resample, family) granularity, not
    (dataset, resample) -- see the module docstring's "Parallelism"
    section for why (in short: on an 80-core server, ~100
    (dataset, resample) units means almost every unit starts at once,
    and the whole run then waits on however long the single slowest
    *unit* takes, serially through all 4 of its families; splitting to
    (dataset, resample, family) means the scheduler drains and refills
    worker slots at 4x the resolution instead of leaving each slot
    pinned to one dataset's full family sequence).

    A job whose raw/ output is already saved on disk (e.g. an earlier
    attempt that was interrupted after saving but before writing its
    DONE marker/manifest fragment) reuses that saved output instead of
    re-running the extractor.

    Any exception is caught and recorded to ``ERROR.txt`` in this job's
    directory rather than propagated -- one bad (dataset, resample,
    family) combination must not take down the whole run.
    """
    # Imported lazily (not at module top-level) so that, combined with
    # this script's env vars being set before any worker touches numpy
    # (they're set at *import* time of this module, which every worker
    # re-imports fresh under the "spawn" start method main() uses --
    # see main()'s own comment), thread-pool-sensitive libraries never
    # get a chance to initialize with more than 1 thread in a worker
    # process.
    import inspect

    from src.tsfm_ensemble.families import FAMILY_REGISTRY

    def _make(cls, **kwargs):
        # Not every family constructor accepts every kwarg (e.g.
        # IntervalFamily's QUANT wrapper takes no random_state, being
        # deterministic given input shape) -- inspect the actual class's
        # __init__ rather than hardcoding which ones do.
        accepted = inspect.signature(cls.__init__).parameters
        return cls(**{k: v for k, v in kwargs.items() if k in accepted})

    job_dir, raw_dir = _paths(dataset, resample, family)
    job_dir.mkdir(parents=True, exist_ok=True)
    # raw_dir is shared with this (dataset, resample)'s other 3 family
    # jobs (they each only ever write their own {family}_{train,test}.npz
    # filename into it, so concurrent mkdir/writes from sibling jobs are
    # safe -- exist_ok=True handles the directory itself already existing).
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        raw_train_path = raw_dir / f"{family}_train.npz"
        raw_test_path = raw_dir / f"{family}_test.npz"

        if raw_train_path.is_file() and raw_test_path.is_file():
            X_raw_train, y_train = _load_raw_features(raw_train_path)
            X_raw_test, _y_test = _load_raw_features(raw_test_path)
            raw_train_bytes = raw_train_path.stat().st_size
            raw_test_bytes = raw_test_path.stat().st_size
        else:
            X_train_full, y_train_full = load_classification_with_fallback(dataset, split="train")
            X_test_full, y_test_full = load_classification_with_fallback(dataset, split="test")
            X_train, y_train, X_test, y_test = load_resample(
                dataset, resample, X_train_full, y_train_full, X_test_full, y_test_full
            )
            X_train = np.ascontiguousarray(X_train, dtype=np.float64)
            X_test = np.ascontiguousarray(X_test, dtype=np.float64)

            extractor = _make(FAMILY_REGISTRY[family], random_state=RANDOM_STATE)
            X_raw_train = extractor.fit_transform(X_train, y_train)
            X_raw_test = extractor.transform(X_test)
            X_raw_train, X_raw_test = _to_dense_output(X_raw_train, X_raw_test)
            raw_train_bytes = _save_features(raw_train_path, X_raw_train, y_train)
            raw_test_bytes = _save_features(raw_test_path, X_raw_test, y_test)
            del extractor

        row = {
            "Dataset": dataset, "Resample": resample, "Family": family,
            "NumFeaturesRaw": X_raw_train.shape[1],
            "IsRawSparse": bool(sp.issparse(X_raw_train)),
            "RawTrainBytes": raw_train_bytes, "RawTestBytes": raw_test_bytes,
            "ElapsedSeconds": time.perf_counter() - start,
        }
        pd.DataFrame([row]).to_csv(job_dir / "manifest.csv", index=False)

        # Free this job's raw arrays -- not to bound peak memory *within*
        # this job (there's no "next family" to protect within a single
        # (dataset, resample, family) job any more), but because
        # ProcessPoolExecutor workers are reused across many submitted
        # jobs: dropping references and collecting here means the next
        # job this same worker process picks up isn't competing with
        # memory this job's arrays would otherwise still be holding.
        del X_raw_train, X_raw_test
        gc.collect()

        (job_dir / "DONE").touch()
        # Clear out a stale failure marker from an earlier, unsuccessful
        # attempt at this same job -- otherwise a later successful rerun
        # would leave both DONE and ERROR.txt present, which would wrongly
        # read as still-failing to anything scanning for ERROR.txt.
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        return {
            "dataset": dataset, "resample": resample, "family": family, "status": "done",
            "elapsed": time.perf_counter() - start,
        }
    except Exception as exc:  # noqa: BLE001 - one bad job must not abort the whole run
        (job_dir / "ERROR.txt").write_text(f"{exc!r}\n\n{traceback.format_exc()}")
        return {
            "dataset": dataset, "resample": resample, "family": family, "status": "error",
            "elapsed": time.perf_counter() - start, "error": repr(exc),
        }


def finalize() -> None:
    """Aggregate every completed (dataset, resample, family) job's
    manifest fragment into one top-level ``manifest.csv``. Safe to call
    standalone (e.g. after an interrupted run) -- it only reads what's
    already on disk.
    """
    fragments = sorted(RESULTS_DIR.glob("units/*/resample*/*/manifest.csv"))
    if not fragments:
        print("No completed jobs found; nothing to aggregate.")
        return
    combined = pd.concat([pd.read_csv(p) for p in fragments], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    print(f"Aggregated {len(fragments)} completed job(s) -> {RESULTS_DIR / 'manifest.csv'}")


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    dataset: str
    resample: int
    family: str
    mem_estimate: int


def _probe_dataset_shape(dataset: str):
    """(n_train, n_test, series_length) for ``dataset``'s original split
    -- used only to plan memory estimates before any (dataset, resample)
    unit is dispatched. Also warms aeon's on-disk dataset cache (a real
    fetch, if not already cached), so the parallel workers that load this
    same dataset again per-resample all hit a warm cache instead of
    racing each other on a cold one.
    """
    X_train, _ = load_classification_with_fallback(dataset, split="train")
    X_test, _ = load_classification_with_fallback(dataset, split="test")
    return X_train.shape[0], X_test.shape[0], X_train.shape[2]


def build_jobs(datasets: List[str]) -> List[Job]:
    """Probe every dataset's shape and build the full (dataset, resample,
    family) job list.

    One dataset failing to load here (corrupted cache, a genuinely bad
    file, transient network issue, ...) must not take down the other 19
    -- the same "one bad job doesn't abort the run" principle
    ``_process_family_job`` already follows once jobs are actually
    running, but this probing pass happens *before* that, in the main
    process, so it needs its own try/except rather than inheriting
    theirs. A dataset that fails here is skipped entirely (no jobs are
    created for it, for any family) and reported in the returned
    ``failed_datasets`` list rather than only printed, so ``main()`` can
    surface it in the final summary too.
    """
    jobs = []
    failed_datasets = []
    print(f"Probing {len(datasets)} dataset shape(s) (also warms aeon's cache for the workers)...")
    for dataset in datasets:
        try:
            n_train, n_test, length = _probe_dataset_shape(dataset)
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not abort the whole run
            print(f"  {dataset:32s} FAILED TO LOAD: {exc!r} -- skipping this dataset entirely.")
            failed_datasets.append((dataset, repr(exc)))
            continue
        family_estimates = {
            family: estimate_family_peak_memory_bytes(family, n_train, n_test, length)
            for family in FAMILIES
        }
        estimates_str = ", ".join(
            f"{family}={est / 1024**2:.0f}MB" for family, est in family_estimates.items()
        )
        print(
            f"  {dataset:32s} n_train={n_train:5d} n_test={n_test:5d} length={length:5d} "
            f"-> est. peak/family: {estimates_str}"
        )
        for resample in range(NUM_RESAMPLES):
            for family in FAMILIES:
                if _is_done(dataset, resample, family):
                    continue
                jobs.append(Job(
                    dataset=dataset, resample=resample, family=family,
                    mem_estimate=family_estimates[family],
                ))
    # Largest-memory-first: a standard bin-packing heuristic that reduces
    # fragmentation/tail latency versus e.g. dataset order or FIFO --
    # otherwise a handful of large jobs left for last, after small ones
    # already filled every worker slot, forces them to run near-serially
    # at the very end. (This is on top of, not instead of, the
    # (dataset, resample, family)-level granularity itself already
    # reducing tail latency -- see the module docstring's "Parallelism"
    # section.)
    jobs.sort(key=lambda j: j.mem_estimate, reverse=True)
    return jobs, failed_datasets


def _worker_init() -> None:
    # Re-assert the thread-limiting env vars inside each freshly spawned
    # worker process (belt-and-braces alongside the module-level
    # os.environ.setdefault calls above, which already apply here too
    # since "spawn" re-imports this module fresh in every worker before
    # this initializer runs -- see main()'s comment on why "spawn" is
    # used instead of the platform-default fork on Linux).
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
        "--memory-budget-gb", type=float, default=None,
        help="Override detected available RAM (GB) used for scheduling.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the detected resources and the planned job schedule, then exit without running anything.",
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cpu_count = detect_cpu_count()
    n_cpu_workers = args.max_workers or max(1, cpu_count - RESERVED_CORES)
    memory_budget = (
        int(args.memory_budget_gb * 1024**3) if args.memory_budget_gb is not None
        else int(detect_available_memory_bytes() * MEMORY_BUDGET_FRACTION)
    )
    print(
        f"Detected {cpu_count} CPU core(s) -> {n_cpu_workers} worker(s) "
        f"(reserving {RESERVED_CORES} for the main process).\n"
        f"Memory budget for scheduling: {memory_budget / 1024**3:.1f} GB."
    )

    datasets = dev_subset_datasets()
    jobs, failed_datasets = build_jobs(datasets)
    if failed_datasets:
        pd.DataFrame(failed_datasets, columns=["Dataset", "Error"]).to_csv(
            RESULTS_DIR / "load_errors.csv", index=False
        )
        print(
            f"\nWARNING: {len(failed_datasets)}/{len(datasets)} dataset(s) could not be loaded at all "
            f"and were skipped entirely (see {RESULTS_DIR / 'load_errors.csv'}): "
            f"{', '.join(d for d, _ in failed_datasets)}"
        )
    n_loadable_datasets = len(datasets) - len(failed_datasets)
    n_total_possible_jobs = n_loadable_datasets * NUM_RESAMPLES * len(FAMILIES)
    n_already_done = n_total_possible_jobs - len(jobs)
    print(
        f"\n{len(jobs)} job(s) to run ({n_already_done} already done, resuming) "
        f"across {n_loadable_datasets} loadable dataset(s) x {NUM_RESAMPLES} resample(s) "
        f"x {len(FAMILIES)} families."
    )
    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    # concurrent.futures.ProcessPoolExecutor's default start method on
    # Linux is "fork", which -- since the parent has already imported
    # numpy/scipy by the time workers are forked -- can leave a forked
    # worker's BLAS thread pool initialized from whatever state it was in
    # at fork time, bypassing the env vars this module sets at import
    # time. "spawn" instead starts each worker as a fresh interpreter
    # that re-imports this module (and so re-applies those env vars, plus
    # runs _worker_init) before touching numpy at all -- the only start
    # method that reliably makes the 1-thread-per-worker limiting above
    # take effect regardless of platform.
    import concurrent.futures as cf

    mp_context = multiprocessing.get_context("spawn")
    memory_used = 0
    completed, errored = 0, 0

    with cf.ProcessPoolExecutor(
        max_workers=n_cpu_workers, mp_context=mp_context, initializer=_worker_init
    ) as executor:
        in_flight = {}  # Future -> Job
        remaining = list(jobs)
        start_time = time.perf_counter()

        while remaining or in_flight:
            # Admit as many jobs as fit under the current CPU-slot and
            # memory-budget constraints. An empty in_flight set always
            # admits the next job regardless of its own estimate exceeding
            # the whole budget -- otherwise a single unit larger than the
            # detected budget would deadlock the run instead of just
            # running alone (with a warning).
            while remaining and len(in_flight) < n_cpu_workers:
                job = remaining[0]
                fits_budget = memory_used + job.mem_estimate <= memory_budget
                if not fits_budget and in_flight:
                    break  # wait for something to finish and free memory
                if not fits_budget:
                    print(
                        f"WARNING: {job.dataset} resample{job.resample} {job.family}'s estimated "
                        f"{job.mem_estimate / 1024**3:.1f} GB exceeds the {memory_budget / 1024**3:.1f} GB "
                        "budget on its own; running it alone rather than blocking forever."
                    )
                remaining.pop(0)
                future = executor.submit(_process_family_job, job.dataset, job.resample, job.family)
                in_flight[future] = job
                memory_used += job.mem_estimate

            if not in_flight:
                continue  # shouldn't happen, but avoids a hang if it somehow does

            done, _ = cf.wait(in_flight.keys(), return_when=cf.FIRST_COMPLETED)
            for future in done:
                job = in_flight.pop(future)
                memory_used -= job.mem_estimate
                result = future.result()
                if result["status"] == "done":
                    completed += 1
                    print(
                        f"[{completed + errored}/{len(jobs)}] OK   {job.dataset} resample{job.resample} "
                        f"{job.family} ({result['elapsed']:.1f}s)"
                    )
                else:
                    errored += 1
                    print(
                        f"[{completed + errored}/{len(jobs)}] FAIL {job.dataset} resample{job.resample} "
                        f"{job.family}: {result.get('error')}"
                    )

        elapsed = time.perf_counter() - start_time
        print(f"\nDone in {elapsed / 60:.1f} min: {completed} succeeded, {errored} failed.")

    finalize()


if __name__ == "__main__":
    main()
