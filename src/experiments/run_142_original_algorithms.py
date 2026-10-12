"""Benchmark the 4 *original* literature algorithms (RDST, QUANT, KG-MTP,
WEASEL 2.0 -- not this project's own 12-learner stacking portfolio) across
the full 142-dataset x 30-resample UCR archive, measuring both per-step
runtime and predictive performance.

Meant to run unattended, for hours, on the user's own server (NOT the
linked development Mac) -- see "Parallelism" below for exactly how it's
built to do that safely.

Algorithm -> (family extractor, classifier) mapping
-----------------------------------------------------
Each of the 4 algorithms here is exactly the single, paper-matched
extractor + classifier pair already identified, verified, and used
elsewhere on this branch -- not a new implementation. This script only
assembles them and adds timing/orchestration:

    RDST      : tsfm_ensemble.families.ShapeletFamily (aeon's
                RandomDilatedShapeletTransform, max_shapelets=10_000) +
                tsfm_ensemble.base_learners.make_learner("shapelet", "ridge")
                (StandardScaler + RidgeClassifierCV(alphas=logspace(-4,4,20)))
                -- matches aeon's own RDSTClassifier default estimator
                exactly (see base_learners.py's module docstring).
    QUANT     : tsfm_ensemble.families.IntervalFamily (this project's own
                torch-free QuantTransformer, depth=6, div=4) +
                make_learner("interval", "extra_trees")
                (ExtraTreesClassifier(n_estimators=200, max_features=0.1,
                criterion="entropy")) -- matches aeon's own QUANTClassifier
                default estimator exactly.
    KG-MTP    : tsfm_ensemble.families.ConvolutionFamily (this project's
                own KGMTP transform -- no aeon equivalent exists, this
                *is* the reference implementation here) +
                make_learner("convolution", "ridge")
                (StandardScaler + RidgeClassifierCV(alphas=logspace(-3,3,10)))
                -- matches KG-MTP's own reference script (run_KGMTP.py).
    WEASEL2.0 : tsfm_ensemble.families.DictionaryFamily (aeon's
                WEASELTransformerV2, feature_selection="chi2_top_k") +
                make_learner("dictionary", "ridge")
                (RidgeClassifierCV(alphas=logspace(-1,5,10)), no scaler)
                -- matches aeon's own WEASEL_V2 default estimator exactly.

This is deliberately the single "native" (family, learner) pairing per
algorithm -- not all 3 of base_learners.py's LEARNER_NAMES on every
family, which is a different, already-answered question (see the dev-
subset study earlier on this branch). The point of *this* run is a clean,
literature-faithful reproduction of each of the 4 algorithms on the full
archive, comparable to their own published numbers in comparisons/.

Every extra_trees learner's own ``n_jobs=-1`` (base_learners.py's
default) is overridden to 1 here -- see "Parallelism" below for why.

Dictionary-family output is densified (``.toarray()``) before fitting/
scoring, timed as part of the transform step -- ``RidgeClassifierCV`` is
measured 38-62x slower on WEASEL's nominally-sparse-but-actually-dense
(>=18% nonzero on every dev-subset dataset) CSR output than on a plain
dense array (see save_dev_subset_transforms.py's own module docstring,
which made the same choice for the same measured reason).

Metrics
-------
The 5 metrics come from the user's own ``moment-quant`` project
(``src/classification/utils.py``, ``predict_with_proba`` and
``compute_classification_metrics``), reproduced verbatim below rather
than imported (this repo and that one are separate projects; the two
functions are small and self-contained) -- Accuracy, BalancedAccuracy,
LogLoss, ROCAUC, F1. ``predict_with_proba`` already handles
``RidgeClassifierCV`` having no ``predict_proba`` (every one of the 4
algorithms here uses Ridge except QUANT/extra_trees) by falling back to
a one-hot pseudo-probability from ``.predict()`` -- exactly the behavior
wanted here, so no separate decision_function/column-realignment
handling is needed on top.

Per-step timing
----------------
Each unit records 4 wall-clock numbers (seconds, ``time.perf_counter``):
``TrainTransformSeconds`` (extractor.fit_transform on the training
split), ``TrainClassifySeconds`` (classifier.fit), ``TestTransformSeconds``
(extractor.transform on the test split), ``TestClassifySeconds``
(``predict_with_proba`` on the test split -- classification only, the
test-transform time above is charged separately). Dictionary-family
densification time is folded into the corresponding transform step (see
above), not split out separately.

A 5th number, ``UnitTotalSeconds``, is also recorded: the total
wall-clock for the whole unit (dataset/resample load+split, label
encoding, the 4 timed steps above, metric computation, manifest I/O),
measured directly rather than left to be reconstructed from the 4 step
timings after the fact -- this script is meant to run once, so anything
worth having later needs to be captured now. Unlike the sibling
``cv_ablation_142.py`` script, one manifest row already *is* one whole
unit here, so ``UnitTotalSeconds`` can be summed directly across rows
with no double-counting caveat.

Dataset loading -- load once, share via copy-on-write fork
-------------------------------------------------------------
All 142 datasets' full train+test arrays are loaded exactly once, in the
main process, into a *module-level global* (``_DATASETS``) before any
worker is started -- not passed as a function argument through
``ProcessPoolExecutor.submit``, which would re-pickle (and thus re-copy)
the whole dict once per *job* (17,040 of them at 30 resamples x 4
algorithms), not once per worker. Instead, ``multiprocessing.get_context(
"fork")`` is used explicitly: a forked worker process starts as a
copy-on-write clone of the parent's already-populated memory, so every
worker sees ``_DATASETS`` fully populated for free, without any IPC/
pickling cost, and without duplicating the data in RAM unless/until a
worker actually mutates its own copy (it doesn't -- workers only read
``_DATASETS`` and call ``common.load_resample``, which returns new
arrays without touching the originals).

Per-(dataset, resample) splitting itself (``common.load_resample``,
against ``PythonResampleIndices/``) is cheap -- reading small per-
resample index text files and re-slicing already-in-memory arrays -- so
it happens once per *unit*, inside the worker, directly against the
shared ``_DATASETS`` entry; only the one-time full-dataset load is
hoisted to the parent.

Parallelism and thread pinning under fork
--------------------------------------------
This is the one place this script deliberately diverges from
``save_dev_subset_transforms.py``'s own convention (env vars set before
any numpy import, plus "spawn" so every worker re-applies them from a
genuinely fresh interpreter): "spawn" is incompatible with the load-once
design above (a spawned worker does *not* inherit the parent's memory,
so ``_DATASETS`` would need to be pickled and IPC'd to every worker, or
reloaded independently per worker -- exactly what this script is trying
to avoid). "fork" is required instead, but a forked child is a memory
copy of the *already-running* parent -- if the parent's BLAS/numba
runtime already spun up its own internal thread pool (which can happen
lazily, on first heavy call, not necessarily at import time) before the
fork, env vars set at the top of this file would not reliably reach into
that already-initialized native thread pool.

The fix used here is ``threadpoolctl.threadpool_limits(1)`` (not as a
context manager -- called once, its effect persists for the life of the
process) plus ``numba.set_num_threads(1)``, both called from
``_worker_init`` -- ``ProcessPoolExecutor``'s ``initializer`` callback,
which runs once per worker process at startup regardless of start
method (fork included). Unlike env vars, both of these are *runtime* APIs
that reconfigure the actual loaded BLAS library / numba threading layer
directly, so they work correctly no matter what state that library was
in at fork time. The top-of-file env var block below is kept anyway,
belt-and-braces (harmless, and still the correct mechanism for the
*parent* process's own light numpy use while loading datasets) but is
not relied upon alone.

One more fork hazard, hit and fixed during development (not hypothetical
-- this crashed a real smoke test): ``kg_mtp.py``'s ``_transform`` is a
``@njit(..., parallel=True, cache=True)`` function declared with an
*explicit* signature string, which makes numba compile it *eagerly*, at
*import* time -- i.e. in the parent process, the moment this script's own
``from src.tsfm_ensemble.families import FAMILY_REGISTRY`` (which pulls
in ``kg_mtp.py`` transitively) executes, well before ``main()`` loads
any dataset or forks any worker. Eager-compiling a ``parallel=True``
function selects and initializes numba's threading layer right then --
which resolved, on the platform this was tested on, to GNU OpenMP -- and
GNU OpenMP's own runtime deliberately aborts any process that later
calls ``fork()`` ("Terminating: fork() called from a process already
using GNU OpenMP, this is unsafe."), taking down every worker fork this
script does. Fixed by forcing ``NUMBA_THREADING_LAYER=workqueue`` (numba's
plain-pthreads layer, no GNU OpenMP dependency, no such fork
restriction) at the very top of this file, before that import chain runs
-- see the comment there. With every worker pinned to 1 thread regardless
(above), the choice of threading layer has no performance consequence
here, only a correctness one.

Each worker is pinned to 1 BLAS/numba thread because parallelism here is
entirely at the process level (``--max-workers`` worker processes, each
handling one (dataset, resample, algorithm) unit at a time) -- letting
each worker's own libraries also multithread would oversubscribe
whatever core budget ``--max-workers`` was set to, and would also
contaminate the runtime measurements this script exists to collect (a
unit's measured seconds should reflect 1 core's worth of work, not
however many cores happened to be free/contended at that moment).

Physical vs. logical cores
----------------------------
``--max-workers`` should be the number of *physical* cores, not
hyperthreaded logical ones, for CPU-bound BLAS/numba workloads like
these (oversubscribing hyperthreads for this kind of work does not
scale linearly and mainly adds scheduling noise to the timing numbers
this script is collecting). ``detect_physical_cpu_count()`` below makes
a best effort (``psutil`` if installed, else ``/proc/cpuinfo`` on Linux)
and is used as the default only as a fallback if ``--max-workers`` isn't
passed explicitly -- pass it explicitly (e.g. ``--max-workers 40``) to
not depend on that detection being right on an unfamiliar server.

Resumability
-------------
Jobs are scheduled at (dataset, resample, algorithm) granularity --
142 x 30 x 4 = 17,040 possible units. Each writes to its own
``units/<dataset>/resample<k>/<algorithm>/`` directory: a ``DONE``
marker plus a one-row ``manifest.csv`` fragment on success, or
``ERROR.txt`` (with the full traceback) on failure -- one bad unit
never aborts the run, and a killed/restarted run skips every unit whose
``DONE`` marker is already on disk (``build_jobs``). ``finalize()``
concatenates every completed unit's fragment into
``RESULTS_DIR/manifest.csv`` -- called automatically at the end of a run
and periodically during a long one (``--finalize-every``), and safe to
call standalone (``--finalize-only``) at any time, including while the
run is still in progress elsewhere.

Usage
-----
    python -m src.experiments.run_142_original_algorithms --dry-run
    python -m src.experiments.run_142_original_algorithms --max-workers 40
    python -m src.experiments.run_142_original_algorithms --max-workers 40 --resamples 30
    python -m src.experiments.run_142_original_algorithms --datasets Computers,OliveOil --resamples 3
    python -m src.experiments.run_142_original_algorithms --finalize-only
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Belt-and-braces: still set before any numpy import, same convention as
# save_dev_subset_transforms.py. This is NOT the mechanism this script
# actually relies on for correctness under fork -- see the module
# docstring's "Parallelism and thread pinning under fork" section --
# threadpoolctl/numba runtime calls in _worker_init are. Kept because it's
# free, still correct for the main process's own light numpy use, and a
# harmless second line of defense for anything reached before
# _worker_init runs.
# --------------------------------------------------------------------------- #
import os  # noqa: E402

for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

# NOT belt-and-braces -- this one is load-bearing, and must be set before
# ``src.tsfm_ensemble.kg_mtp`` (imported transitively below, via
# ``families``) is ever imported. ``kg_mtp.py``'s ``_transform`` is a
# ``@njit(..., parallel=True, cache=True)`` function declared with an
# *explicit* signature string -- which makes numba compile it eagerly, at
# *import* time, instead of lazily on first call. Confirmed empirically
# (this crashed a real smoke test before this line was added): eager
# compilation of a ``parallel=True`` function selects and initializes
# numba's default threading layer immediately, in the parent process,
# before ``main()`` has even loaded a single dataset -- and on this
# platform that default resolves to the GNU OpenMP ("omp") layer. GNU
# OpenMP's own runtime aborts any process that calls ``fork()`` after it
# has been initialized ("Terminating: fork() called from a process
# already using GNU OpenMP, this is unsafe."), which is exactly what
# every worker fork in this script would otherwise do. "workqueue" is
# numba's plain-pthreads threading layer, with no GNU OpenMP dependency
# and no such fork restriction -- and since every worker is pinned to 1
# thread anyway (``_worker_init``'s ``numba.set_num_threads(1)``), which
# threading layer is selected has no performance consequence here, only
# a correctness one.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse  # noqa: E402
import inspect  # noqa: E402
import multiprocessing  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

# Benign, high-volume noise confirmed during smoke testing: every
# extra_trees (QUANT) unit's forced n_jobs=1 (see _run_unit) makes
# sklearn/joblib fall back to a 'SequentialBackend' that doesn't support
# the 'timeout' kwarg ExtraTreesClassifier's own internals pass to
# Parallel() regardless -- purely informational, not a sign of anything
# actually failing to run sequentially. Left unfiltered this would print
# twice per QUANT unit (fit + predict), i.e. thousands of times across a
# 17,040-unit run -- suppressed by exact message match only, same
# targeted-suppression convention as src/experiments/common.py.
warnings.filterwarnings(
    "ignore",
    message=r"The backend class 'SequentialBackend' does not support timeout\.",
    category=UserWarning,
)

# Benign, sklearn-internal noise: fires whenever sklearn's own internal
# ensemble code (ExtraTreesClassifier here -- QUANT's learner) calls
# joblib's plain Parallel/delayed under the hood instead of sklearn's
# own config-propagating wrapper -- purely informational (only affects
# whether sklearn's global config, e.g. array-API dispatch, gets
# propagated into joblib workers, which this project doesn't use), not
# a sign of anything computed incorrectly. Same targeted-suppression
# convention as the SequentialBackend warning just above -- confirmed
# to fire routinely under this script's threadpoolctl/n_jobs=1 pinning
# (see _worker_init/module docstring).
warnings.filterwarnings(
    "ignore",
    message=(
        r"`sklearn\.utils\.parallel\.delayed` should be used with "
        r"`sklearn\.utils\.parallel\.Parallel` to make it possible to propagate "
        r"the scikit-learn configuration of the current thread to the joblib workers\."
    ),
    category=UserWarning,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse as sp  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from src.experiments.common import (  # noqa: E402
    RESAMPLE_INDICES_DIR,
    load_classification_with_fallback,
    load_resample,
)
from src.tsfm_ensemble.base_learners import RANDOM_STATE, make_learner  # noqa: E402
from src.tsfm_ensemble.families import FAMILY_REGISTRY  # noqa: E402


def list_142_datasets(resample_indices_dir: Path = RESAMPLE_INDICES_DIR) -> List[str]:
    """The 142-dataset list, taken from ``PythonResampleIndices/``'s own
    subdirectory names rather than aeon's ``tsc_datasets`` registry.

    ``check_dataset_loading.py``'s own ``list_142_datasets`` (``sorted(
    univariate_equal_length + redux)``) turns out to be a real, live bug
    against the aeon version actually installed here (1.3.0): that
    version's ``tsc_datasets`` module has no ``redux`` attribute at all
    (renamed to ``univariate_bake_off_2024`` at some point -- confirmed
    ``univariate_equal_length`` (112) + ``univariate_bake_off_2024`` (30)
    = 142, matching the count), and even with that rename applied, 7 of
    those 30 "bake off redux" names don't match this project's own
    dataset-folder naming for the regression-derived classification
    datasets (e.g. aeon's ``AsphaltPavementType``/``Covid3Month``/
    ``FloodModeling1`` vs. this project's own ``AsphaltPavementTypeUni``/
    ``Covid3Month_disc``/``FloodModeling1_disc``, which is what
    ``PythonResampleIndices/`` and the local ``datasets/`` cache actually
    use -- verified directly against both directories' listings).
    ``PythonResampleIndices/`` is the actual ground truth this project's
    own ``load_resample`` depends on regardless of aeon's registry, so
    it's used directly here instead of importing the broken function.
    """
    return sorted(p.name for p in resample_indices_dir.iterdir() if p.is_dir())


RESULTS_DIR = Path(
    os.environ.get("RUN_142_ORIGINAL_ALGORITHMS_RESULTS_DIR", str(Path("results") / "run_142_original_algorithms"))
)
UNITS_DIR = RESULTS_DIR / "units"

NUM_RESAMPLES = 30

# Scheduling-only randomness (see build_jobs's own docstring) -- kept
# distinct from RANDOM_STATE (the scientific seed used for extractor/
# learner init) so it's unambiguous this has zero effect on any result,
# only on which unit runs when. Same constant/value as
# cv_ablation_142.py's own JOB_SHUFFLE_SEED, for consistency between the
# two sibling scripts -- not load-bearing that they match, just tidy.
JOB_SHUFFLE_SEED = 20260915

# algorithm name -> (family key into FAMILY_REGISTRY / make_learner,
# learner_name into make_learner, extractor kwargs beyond random_state).
# See the module docstring's "Algorithm -> (family extractor, classifier)
# mapping" section for exactly why each of these is the "native" pairing.
ALGORITHMS: Dict[str, Dict] = {
    "RDST": dict(family="shapelet", learner="ridge", extractor_kwargs=dict(max_shapelets=10_000, n_jobs=1)),
    "QUANT": dict(family="interval", learner="extra_trees", extractor_kwargs=dict(depth=6, div=4)),
    "KG-MTP": dict(
        family="convolution",
        learner="ridge",
        extractor_kwargs=dict(num_features=50_000, max_dilations_per_kernel=32, n_features_per_kernel=5),
    ),
    "WEASEL2.0": dict(family="dictionary", learner="ridge", extractor_kwargs=dict(n_jobs=1)),
}


# --------------------------------------------------------------------------- #
# Metric utilities -- verbatim from the user's own moment-quant project
# (src/classification/utils.py), reproduced here rather than imported
# since that's a separate project. See the module docstring's "Metrics"
# section.
# --------------------------------------------------------------------------- #
def predict_with_proba(classifier, X, n_classes):
    """Predict labels and class probabilities, with a one-hot fallback
    for classifiers (e.g. ``RidgeClassifierCV``) that have no
    ``predict_proba``."""
    if hasattr(classifier, "predict_proba"):
        y_proba = classifier.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = classifier.predict(X)
        y_proba = np.zeros((len(y_pred), n_classes))
        y_proba[np.arange(len(y_pred)), y_pred] = 1.0
    return y_pred, y_proba


def compute_classification_metrics(y_true, y_pred, y_proba, n_classes):
    """Accuracy, BalancedAccuracy, LogLoss, ROCAUC (weighted OvR), and
    F1 (binary on the minority class, else weighted)."""
    labels = np.arange(n_classes)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_proba, labels=labels)
    if n_classes == 2:
        auroc = roc_auc_score(y_true, y_proba[:, 1], labels=labels, average="weighted", multi_class="ovr")
    else:
        auroc = roc_auc_score(y_true, y_proba, labels=labels, average="weighted", multi_class="ovr")
    unique, counts = np.unique(y_true, return_counts=True)
    sort_order = np.argsort(unique)
    unique, counts = unique[sort_order], counts[sort_order]
    minority_class = unique[np.flatnonzero(counts == counts.min())[0]]
    if n_classes == 2:
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=minority_class, zero_division=0.0)
    else:
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0.0)
    return dict(Accuracy=accuracy, BalancedAccuracy=balanced_accuracy, LogLoss=logloss, ROCAUC=auroc, F1=f1)


# --------------------------------------------------------------------------- #
# Physical-core detection -- see module docstring's "Physical vs. logical
# cores" section. Best-effort only; --max-workers should still be passed
# explicitly on an unfamiliar server.
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
# worker via copy-on-write. See the module docstring's "Dataset loading"
# section.
# --------------------------------------------------------------------------- #
_DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _load_all_datasets(datasets: List[str]) -> List[Tuple[str, str]]:
    """Populate the module-level ``_DATASETS`` global in place. Returns a
    list of ``(dataset, error_repr)`` for any dataset that failed to load
    at all -- those datasets contribute zero jobs rather than aborting
    the whole run (same convention as ``save_dev_subset_transforms.py``'s
    ``build_jobs``/``failed_datasets``).
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
    algorithm: str


def _unit_dir(dataset: str, resample: int, algorithm: str) -> Path:
    return UNITS_DIR / dataset / f"resample{resample}" / algorithm


def _make_extractor(family: str, extractor_kwargs: Dict, random_state: int):
    """Build one family's extractor, passing ``random_state`` only if its
    constructor actually accepts it (``IntervalFamily``'s QUANT wrapper
    doesn't -- it's deterministic given input shape). Same pattern as
    ``save_dev_subset_transforms.py``'s own ``_make`` helper.
    """
    cls = FAMILY_REGISTRY[family]
    kwargs = dict(extractor_kwargs)
    accepted = inspect.signature(cls.__init__).parameters
    if "random_state" in accepted:
        kwargs["random_state"] = random_state
    return cls(**kwargs)


def _maybe_densify(Xt):
    """Densify WEASEL/dictionary's sparse CSR output -- see the module
    docstring's "Algorithm -> (family extractor, classifier) mapping"
    section for why (RidgeClassifierCV measured 38-62x slower on sparse
    input at this scale)."""
    if sp.issparse(Xt):
        return np.asarray(Xt.todense())
    return np.asarray(Xt)


def _worker_init() -> None:
    """Runs once per worker process at startup (any start method,
    including "fork" -- see the module docstring's "Parallelism and
    thread pinning under fork" section for why this, not env vars alone,
    is what actually pins thread counts here)."""
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


def _run_unit(dataset: str, resample: int, algorithm: str) -> Dict:
    """Run one (dataset, resample, algorithm) unit against the shared,
    already-loaded ``_DATASETS`` global, timing each of the 4 steps and
    computing the 5 metrics on the test split. Any exception is caught
    and recorded to ``ERROR.txt`` rather than propagated -- one bad unit
    must not take down the whole run (same convention as
    ``save_dev_subset_transforms.py``'s ``_process_family_job``).
    """
    job_dir = _unit_dir(dataset, resample, algorithm)
    job_dir.mkdir(parents=True, exist_ok=True)
    spec = ALGORITHMS[algorithm]
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

        extractor = _make_extractor(spec["family"], spec["extractor_kwargs"], RANDOM_STATE)
        learner = make_learner(spec["family"], spec["learner"], random_state=RANDOM_STATE)
        if spec["learner"] == "extra_trees":
            # base_learners.make_learner's extra_trees defaults to
            # n_jobs=-1; parallelism here is at the process level (one
            # unit per worker) -- see module docstring's "Parallelism"
            # section for why oversubscribing per-worker threads would
            # both contend for cores and contaminate the timing numbers.
            learner.set_params(n_jobs=1)

        t0 = time.perf_counter()
        Xt_train = _maybe_densify(extractor.fit_transform(X_train, y_train_enc))
        train_transform_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        learner.fit(Xt_train, y_train_enc)
        train_classify_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        Xt_test = _maybe_densify(extractor.transform(X_test))
        test_transform_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred, y_proba = predict_with_proba(learner, Xt_test, n_classes)
        test_classify_s = time.perf_counter() - t0

        metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba, n_classes)

        row = {
            "Dataset": dataset, "Resample": resample, "Algorithm": algorithm,
            "NumTrainInstances": X_train.shape[0], "NumTestInstances": X_test.shape[0],
            "NumClasses": n_classes, "NumFeaturesRaw": Xt_train.shape[1],
            "TrainTransformSeconds": train_transform_s, "TrainClassifySeconds": train_classify_s,
            "TestTransformSeconds": test_transform_s, "TestClassifySeconds": test_classify_s,
            # Total wall-clock for this whole unit -- should be close to
            # the sum of the 4 step timings above (there's little else in
            # this function), but measured directly rather than left to
            # be inferred, same reasoning as cv_ablation_142.py's
            # UnitTotalSeconds: this script is meant to run once, so any
            # runtime worth having later (e.g. untimed overhead the step
            # timings miss -- label encoding, metric computation, npz/csv
            # I/O) needs to be captured now, not reconstructed after the
            # fact. Unlike cv_ablation_142.py, one manifest row already
            # *is* one whole unit here, so there's no duplication/
            # double-counting caveat to document.
            "UnitTotalSeconds": time.perf_counter() - start,
            **metrics,
        }
        pd.DataFrame([row]).to_csv(job_dir / "manifest.csv", index=False)

        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        return {"dataset": dataset, "resample": resample, "algorithm": algorithm, "status": "done",
                "elapsed": time.perf_counter() - start}
    except Exception:  # noqa: BLE001 - one bad unit must not abort the whole run
        (job_dir / "ERROR.txt").write_text(traceback.format_exc())
        return {"dataset": dataset, "resample": resample, "algorithm": algorithm, "status": "error",
                "elapsed": time.perf_counter() - start}


# --------------------------------------------------------------------------- #
# Job building / resumability / finalize
# --------------------------------------------------------------------------- #
def build_jobs(datasets: List[str], resamples: int, algorithms: List[str]) -> List[Job]:
    """Same (dataset, resample, algorithm) unit set as the original
    nested-loop version, but dataset-interleaved rather than
    dataset-major -- same fix, same reasoning, as
    ``cv_ablation_142.py``'s own ``build_jobs`` (see that function's
    docstring for the full story): dataset-major order lets a burst of
    ``n_workers`` jobs for the SAME (large) dataset all get dispatched
    at once whenever the scheduler crosses into a new dataset's block,
    multiplying that one dataset's peak per-unit memory (RDST's
    ``max_shapelets=10_000`` distances, KG-MTP's ``num_features=50_000``
    convolution features -- both scale with instance count) instead of
    spreading load across many differently-sized datasets.

    Lower-stakes here than in ``cv_ablation_142.py`` -- no external
    bootstrap-bagging arm (B=15/25 replicate fits) multiplying peak
    memory further on top of extraction, and this script's own full run
    already completed successfully without hitting this. Applied
    anyway, for the same underlying risk and for consistency between
    the two sibling scripts -- future reruns/resumes (e.g. adding more
    resamples later) get the same protection cv_ablation_142.py does.

    Fix: group pending jobs by dataset, shuffle both the dataset order
    and each dataset's own job order (fixed seed -- ``random``, not
    ``numpy``, since this is a plain list of ``Job`` dataclasses, not
    numeric data), then interleave round-robin across datasets one job
    at a time -- guarantees no two of the next ``n_workers`` dispatched
    jobs share a dataset, as long as more than ``n_workers`` datasets
    still have jobs left (true for nearly this whole run).
    """
    by_dataset: Dict[str, List[Job]] = {}
    for dataset in datasets:
        for resample in range(resamples):
            for algorithm in algorithms:
                if not (_unit_dir(dataset, resample, algorithm) / "DONE").is_file():
                    by_dataset.setdefault(dataset, []).append(Job(dataset, resample, algorithm))

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
    for manifest_path in UNITS_DIR.glob("*/*/*/manifest.csv"):
        if (manifest_path.parent / "DONE").is_file():
            rows.append(pd.read_csv(manifest_path))
    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame(
            columns=[
                "Dataset", "Resample", "Algorithm", "NumTrainInstances", "NumTestInstances",
                "NumClasses", "NumFeaturesRaw", "TrainTransformSeconds", "TrainClassifySeconds",
                "TestTransformSeconds", "TestClassifySeconds", "UnitTotalSeconds", "Accuracy",
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
        help="Number of worker processes (should be the server's PHYSICAL core count -- "
             "see module docstring's 'Physical vs. logical cores' section). "
             "Defaults to detect_physical_cpu_count(), falling back to 40 with a warning if detection fails.",
    )
    parser.add_argument("--resamples", type=int, default=NUM_RESAMPLES, help=f"Number of resamples (default {NUM_RESAMPLES}).")
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset subset to run (default: all 142). Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--algorithms", type=str, default=None,
        help=f"Comma-separated algorithm subset (default: all of {list(ALGORITHMS)}).",
    )
    parser.add_argument(
        "--finalize-every", type=int, default=200,
        help="Re-run finalize() every N completed units, so results/manifest.csv stays current during a long run "
             "(default 200; 0 disables periodic finalization, still finalizes once at the end).",
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
    algorithms = args.algorithms.split(",") if args.algorithms else list(ALGORITHMS)
    for algorithm in algorithms:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algorithm!r}; expected one of {list(ALGORITHMS)}.")

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
    jobs = build_jobs(loadable_datasets, args.resamples, algorithms)
    n_total_possible = len(loadable_datasets) * args.resamples * len(algorithms)
    n_already_done = n_total_possible - len(jobs)
    print(f"\n{len(jobs)} job(s) to run ({n_already_done} already done, resuming) across "
          f"{len(loadable_datasets)} loadable dataset(s) x {args.resamples} resample(s) x "
          f"{len(algorithms)} algorithm(s) = {n_total_possible} possible unit(s).")

    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    import concurrent.futures as cf

    # "fork", not "spawn" -- required for the load-once/copy-on-write
    # design (see module docstring's "Dataset loading" and "Parallelism
    # and thread pinning under fork" sections). Only available on
    # Linux/macOS (not Windows), which is fine -- this is meant to run
    # on the user's Linux server.
    mp_context = multiprocessing.get_context("fork")

    completed, errored = 0, 0
    start_time = time.perf_counter()

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context, initializer=_worker_init) as executor:
        in_flight = {}
        remaining = list(jobs)

        while remaining or in_flight:
            while remaining and len(in_flight) < n_workers:
                job = remaining.pop(0)
                future = executor.submit(_run_unit, job.dataset, job.resample, job.algorithm)
                in_flight[future] = job

            done_futures, _ = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            for future in done_futures:
                job = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker-process-level crash must not abort the run
                    print(f"WORKER CRASH on {job.dataset} resample{job.resample} {job.algorithm}: {exc!r}")
                    errored += 1
                    continue
                if result["status"] == "done":
                    completed += 1
                else:
                    errored += 1
                n_finished = completed + errored
                if n_finished % 10 == 0 or not remaining and not in_flight:
                    elapsed = time.perf_counter() - start_time
                    rate = n_finished / elapsed if elapsed > 0 else 0.0
                    eta = (len(jobs) - n_finished) / rate if rate > 0 else float("nan")
                    print(
                        f"[{n_finished}/{len(jobs)}] {completed} done, {errored} errored "
                        f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min remaining) -- "
                        f"last: {job.dataset} resample{job.resample} {job.algorithm} ({result['status']}, "
                        f"{result['elapsed']:.1f}s)"
                    )
                if args.finalize_every and n_finished % args.finalize_every == 0:
                    finalize()

    print(f"\nRun complete: {completed} done, {errored} errored, "
          f"{time.perf_counter() - start_time:.1f}s total.")
    df = finalize()
    print(f"Finalized {len(df)} completed unit(s) -> {RESULTS_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
