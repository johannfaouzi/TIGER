"""Ablation: does Arsenal/HC2-style bagging-OOB scoring give more honest
out-of-fold (OOF) signal than the current adaptive-fold-count CV
(``base_learners._make_cv``) for the stacking layer's 12-learner base
portfolio (4 families x 3 learners)? Run across the full 142-dataset x
30-resample archive, not just the 20-dataset dev subset -- a targeted
re-check found the dev subset only ever exercised this question on 2
non-independent (both "Pig") datasets, and never exercised the 3-fold or
singleton-class-fallback regimes at all (see ``results/
class_count_scan_142.csv``, produced while scoping this script, for the
full per-dataset fold-count breakdown across all 142 datasets).

Scored arms per (dataset, resample, family, learner)
-----------------------------------------------------------
    "cv"    : exactly ``base_learners.run_family_base_learners`` --
              ``_make_cv``'s adaptive-fold-count StratifiedKFold (or
              unstratified KFold fallback for a singleton class) via
              ``cross_val_predict`` for OOF train scores, plus a
              fresh fit on the full training set scored on the test
              set. Native per-learner scores (``decision_function`` for
              ridge, ``predict_proba`` for extra_trees/naive_bayes),
              exactly as saved by ``build_stacking_features.py``.
    "bagB15"/"bagB25" (ridge and naive_bayes only -- see next section
              for why extra_trees doesn't get this arm) : Arsenal-style
              bootstrap bagging (see the Arsenal source this project
              already read, aeon's ``_arsenal.py``): B bootstrap
              replicates of the *same* training set size, each a fresh
              classifier fit on that replicate; a training row's OOF/OOB
              score is the normalized vote count over only the
              replicates that left it out-of-bag (uniform 1/n_classes
              fallback if a row is OOB in zero replicates -- rare but
              possible at small B/n); a test row's score is the
              normalized vote count over *every* replicate (a test row
              is never in any bootstrap sample regardless, so there's no
              OOB/in-bag distinction there).
    "nativeoob" (extra_trees only) : see next section.

Extra Trees is already an ensemble -- no external bagging wrapper for it
----------------------------------------------------------------------------
Caught during review (not hypothetical): wrapping our own external
bootstrap-OOB loop around ``make_learner(family, "extra_trees", ...)``
means each of the B=15/25 "replicates" is a *whole independent 200-tree
forest* fit on an outer-bootstrapped sample -- nesting a coarse outer
ensemble around an already-fine inner one. (Precisely: sklearn's
``ExtraTreesClassifier`` defaults to ``bootstrap=False`` -- confirmed on
our exact ``make_learner`` config -- so it isn't literally row-resampling
bagging internally, its variance reduction comes from random split
thresholds instead; but it's still a 200-estimator ensemble, so the
"ensemble wrapped around an ensemble" problem holds regardless of that
technicality.) Three concrete problems this caused: (1) cost -- fitting
15+25=40 *additional* full 200-tree forests per unit for this one
learner, dwarfing the 6 forests the "cv" arm fits; (2) it isn't the
analogous fix -- Arsenal's own external bootstrap-OOB exists because
Ridge has no internal ensembling to produce an honest held-out estimate
from; Extra Trees already has a standard, native way to get exactly
that (``bootstrap=True, oob_score=True``, one fit, each tree scored only
on rows it didn't see); (3) extra_trees is interval/QUANT's literature-
paired learner, so this landed squarely on the comparison that matters
most for the paper-reproduction angle.

Fixed by giving extra_trees its own arm, "nativeoob", instead of the
external bagB15/bagB25 wrapper: the same per-family ExtraTreesClassifier
config ``make_learner`` builds, with ``bootstrap=True, oob_score=True``
added, fit once. ``oob_decision_function_`` (NaN-filled for a row that's
never OOB across all 200 trees -- replaced with uniform 1/n_classes,
same fallback convention as the bagging arms, and counted the same way)
is this arm's OOF/OOB train score; the *same* fitted forest's
``predict_proba`` on the test split is its test score -- one fit
produces both, unlike the external-wrapper arms which need B separate
fits to get a test-side estimate at all.

Single extraction, not per-replicate -- a known, deliberate simplification
--------------------------------------------------------------------------
Checked directly against aeon's ``Arsenal`` source
(``_fit_arsenal``/``_train_probas_for_estimator``) rather than assumed:
Arsenal's B (default 25) base estimators each get their *own*,
independently-seeded ROCKET transform (`_clone_estimator(base_rocket,
rng.randint(...))`), fit on the *full* training set (not bootstrapped) --
and only *afterward*, separately, does its OOB train-estimate procedure
bootstrap-resample rows to fit a fresh Ridge on top of that estimator's
own already-computed transformed features. So real Arsenal decorrelates
its ensemble two ways at once: a different random representation per
estimator, *and* a different row sample per estimator's classifier fit.

``bagging_oob_scores`` here only does the second: every bootstrap
replicate reuses the *one* extraction this unit already computed for the
"cv" arm (``Xt_train``/``Xt_test``, built once in ``_run_unit``) --
identical representation across all B replicates, decorrelated only by
which rows their classifier is fit on. This is a deliberate choice (see
conversation), not an oversight: re-extracting a fresh transform per
replicate would be Arsenal-faithful, but for the three genuinely
stochastic transforms here (RDST/shapelet, KG-MTP/convolution,
WEASEL 2.0/dictionary -- each ``random_state``-controlled: shapelet
sampling, kernel bias-threshold example draws, and SFA configuration
sampling respectively), extraction is already this pipeline's dominant
cost, and doing it 15-25x more per unit was judged infeasible at full
142x30 scale even on a 40-core box. QUANT/interval is the one exception
where this simplification costs nothing at all: ``QuantTransformer``/
aeon's ``QUANTTransformer`` take no ``random_state`` whatsoever -- their
dyadic intervals are a fixed function of series length, so a "fresh
seed per replicate" wouldn't change QUANT's features even if we paid
for it.

Practical upshot: this ablation's bagging arms test a narrower question
than true Arsenal -- "does row-resampling-based OOB classifier scoring,
holding the representation fixed, give a more honest OOF signal than
CV" -- not "does Arsenal's full representation-diversity-plus-row-
diversity bagging help." Worth remembering when interpreting a result
here as "bagging doesn't help": it specifically means *this* form of
bagging (classifier-only) didn't help, not that representation-diversity
bagging (the more expensive, more Arsenal-faithful version) wouldn't.

Weighting (bagB15/bagB25, ridge and naive_bayes only): unweighted
----------------------------------------------------------------------
Arsenal's own OOB voting weights each replicate by its fitted
``RidgeClassifierCV.best_score_`` -- a Ridge-specific artifact (the CV
score at its internally-selected alpha). ``GaussianNB``/``MultinomialNB``
expose no equivalent scalar from a plain ``.fit()`` call, so there's
nothing to weight by for one of the two learners that still get this
arm without inventing a proxy. Decided (see conversation) to use plain
unweighted vote-count averaging for both, including ridge -- consistent
between the two, and the same scheme the earlier dev-subset bagging/OOB
check already used (so this run's results stay comparable to that one),
at the cost of not reproducing Arsenal's own weighting exactly for
ridge.

Score representation caveat -- read before comparing arms on LogLoss/ROCAUC
--------------------------------------------------------------------------
The "cv" arm's scores are each learner's *native* output
(``decision_function``/``predict_proba``). The bagB15/bagB25 arms'
scores (ridge, naive_bayes) are, structurally, a *hard-vote fraction*
(``count of replicates voting for class c`` / ``count of replicates
scored``) -- Arsenal-style bagging only ever calls ``.predict()`` per
replicate, not ``.decision_function()``/``.predict_proba()``, so there's
no native per-replicate score to average even before the weighting
question above. The nativeoob arm (extra_trees) is the one exception:
``oob_decision_function_``/``predict_proba`` are genuine (if
tree-averaged, not bootstrap-hard-vote) probabilities, the same
representation as the "cv" arm's extra_trees scores. So across the
portfolio, "cv" vs "nativeoob" scores are apples-to-apples for any
metric, but "cv"/"nativeoob" vs bagB15/bagB25 are only apples-to-apples
for Accuracy (argmax) -- a vote fraction is not a calibrated
probability, so LogLoss/ROCAUC computed directly from it would penalize
bagging for a representational difference, not necessarily worse
ranking. This run only computes/compares Accuracy-based OOF-honesty
(|OOF acc - true test acc|, exactly the earlier dev-subset check's
metric) for that reason -- the full score matrices are saved anyway
(per your request, for later uniform-averaging/CAWPE/meta-learner
experiments), just with this caveat attached.

Output layout, under RESULTS_DIR (results/cv_ablation_142/)
----------------------------------------------------------------
    fragments/<dataset>/resample<k>/<family>.npz
        One file per (dataset, resample, family) job -- the scheduling
        granularity (see "Parallelism" below, same reasoning as
        save_dev_subset_transforms.py: per-family extraction cost varies
        too much across datasets to schedule any coarser on a
        high-core-count box). Deliberately NOT merged into one
        combined per-(dataset, resample) file the way
        build_stacking_features.py's scores.npz is -- see
        ``build_stacking_features.py``'s own docstring for why saving
        "individually... so later ablation experiments can pick any
        subset... without needing to know column offsets" is already
        this project's convention; per-family *files* (rather than
        per-family *keys* in one file) is the same idea at a coarser
        grain, and skips a separate merge step/failure mode. For each
        learner in base_learners.LEARNER_NAMES:
            "{learner}__cv__oof_train", "{learner}__cv__test"          -- all 3 learners
            "{learner}__bagB15__oof_train", "{learner}__bagB15__test"  -- ridge, naive_bayes only
            "{learner}__bagB25__oof_train", "{learner}__bagB25__test"  -- ridge, naive_bayes only
            "{learner}__nativeoob__oof_train", "{learner}__nativeoob__test"  -- extra_trees only
        (any arm/learner that errored has none of its keys -- check
        manifest.csv's own Error columns). Plus, shared across every
        learner in this family: "y_train_enc", "y_test_enc" (absent if
        the test split has a label never seen in training -- see
        _run_unit), "classes_".
    units/<dataset>/resample<k>/<family>/{DONE,ERROR.txt,manifest.csv}
        Per-job bookkeeping, same resumable convention as every other
        script on this branch.
    manifest.csv
        Every completed job's manifest fragment, concatenated by
        finalize(). Columns: Dataset, Resample, Family, Learner,
        NTrain, NTest, NClasses, "Cv{OOFAcc,TestAcc,Seconds,Error}"
        always; ridge/naive_bayes rows additionally get
        "Bag{15,25}{OOFAcc,TestAcc,NeverOOB,Seconds,Error}"; extra_trees
        rows additionally get
        "NativeOOB{OOFAcc,TestAcc,NeverOOB,Seconds,Error}" instead --
        the Bag15/Bag25 columns are simply absent/NaN for extra_trees
        rows, and NativeOOB is absent/NaN for the other two.

        Each arm's own total ``*Seconds`` column is additionally
        decomposed into two sub-components, so the total can be broken
        down further than just "which arm":
        "CvOOFSeconds"/"CvFinalFitSeconds" (the OOF cross_val_predict
        call vs. the final full-train-fit-plus-test-score call --
        CvSeconds == their sum exactly); "Bag{15,25}FitSeconds"/
        "Bag{15,25}PredictSeconds" (summed across all B bootstrap
        replicates -- fitting each replicate's classifier vs. scoring
        it on both its OOB rows and the test rows; won't sum to exactly
        Bag{15,25}Seconds, since bootstrap-resampling/OOB-index
        bookkeeping between the timed calls isn't counted in either --
        see bagging_oob_scores's own docstring); "NativeOOBFitSeconds"/
        "NativeOOBTestSeconds" (the one ``.fit()`` call, which is also
        where sklearn computes ``oob_decision_function_`` internally, so
        NativeOOBFitSeconds already includes this arm's OOF/OOB scoring
        cost and isn't a pure "fit" number the way the other two arms'
        Fit columns are -- vs. the separate test-set ``predict_proba``
        call). These sub-columns are per-learner-row, same as the totals
        they decompose -- no extra double-counting caveat beyond the one
        the totals already have.

        Also "ExtractionTrainSeconds", "ExtractionTestSeconds" (the
        family's one extraction pass, shared by the "cv" arm and every
        bagging/nativeoob arm alike -- see "Single extraction, not
        per-replicate" above) and "UnitTotalSeconds" (this whole
        (dataset, resample, family) job's wall-clock, i.e. what
        _run_unit's own "elapsed" return value is, persisted here rather
        than only printed during the run). This script is meant to run
        once; every runtime worth reporting later (e.g. total compute
        cost of this ablation, or per-family/per-arm timing breakdowns)
        has to be captured now. IMPORTANT: these 3 columns are
        family-level, not learner-level -- identical across a family's 3
        learner rows (same duplication convention as NTrain/NTest/
        NClasses above). Sum "UnitTotalSeconds" or the Extraction*
        columns over *distinct (Dataset, Resample, Family) combinations*
        for a true total, not over every manifest row, or extraction
        time gets triple-counted. The per-arm Seconds columns (Cv/
        Bag15/Bag25/NativeOOB) are genuinely per-learner and don't have
        this caveat.

Everything else (load-once + fork + copy-on-write dataset sharing,
threadpoolctl/numba thread pinning under fork, NUMBA_THREADING_LAYER=
workqueue to avoid the GNU-OpenMP fork-abort, physical-core detection,
resumability) is identical in mechanism to
``run_142_original_algorithms.py`` -- see that module's docstring for
the full reasoning; not re-derived here, just reused directly (small
self-contained helpers duplicated rather than cross-imported, matching
this project's existing convention of experiment scripts not depending
on each other's import-time side effects -- see common.py's own
docstring on why DEV_SUBSET_TRANSFORMS_RESULTS_DIR is duplicated rather
than imported).

Usage
-----
    python -m src.experiments.cv_ablation_142 --dry-run
    python -m src.experiments.cv_ablation_142 --max-workers 40
    python -m src.experiments.cv_ablation_142 --datasets OliveOil,PigCVP --resamples 2
    python -m src.experiments.cv_ablation_142 --finalize-only
"""
from __future__ import annotations

import os  # noqa: E402

for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")
# Load-bearing, not belt-and-braces -- see run_142_original_algorithms.py's
# module docstring ("Parallelism and thread pinning under fork") for the
# full story: kg_mtp.py's eager-compiled parallel=True numba function
# initializes GNU OpenMP at import time, which aborts on a later fork()
# unless numba is forced onto its plain-pthreads "workqueue" layer first.
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
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=r"The backend class 'SequentialBackend' does not support timeout\.",
    category=UserWarning,
)

# Expected, not a bug -- exactly the small-dataset CV degradation this
# whole script exists to study (see the module docstring's opening
# paragraph and results/class_count_scan_142.csv): _make_cv's adaptive
# fold-count fallback (down to 2/3/4-fold, or an unstratified KFold as
# a last resort) can leave a fold's *training* split missing one or
# more classes that the full dataset has, for the ~15/142 datasets
# where that fallback kicks in. sklearn's cross_val_predict warns about
# this every time it happens (once per affected fold, per learner, per
# resample -- i.e. potentially thousands of times across the full run)
# rather than raising, since a missing-but-not-ALL-classes fold is
# still scoreable (unlike the genuinely-singleton-class case, which
# raises ValueError and is already handled separately just below in
# run_family_base_learners). This ablation's whole point is to measure
# how honest the OOF signal is *despite* this -- the warning is telling
# us something we already know and are deliberately measuring, not
# something to silently work around by re-engineering the CV here.
warnings.filterwarnings(
    "ignore",
    message=r"Number of classes in training fold \(\d+\) does not match total number of classes \(\d+\)\.",
    category=RuntimeWarning,
)

# Benign, sklearn-internal noise: fires whenever sklearn's own internal
# CV/ensemble code (cross_val_predict, RidgeClassifierCV,
# ExtraTreesClassifier) calls joblib's plain Parallel/delayed under the
# hood instead of sklearn's own config-propagating wrapper -- purely
# informational (only affects whether sklearn's global config, e.g.
# array-API dispatch, gets propagated into joblib workers, which this
# project doesn't use), not a sign of anything computed incorrectly.
# Confirmed during smoke testing to fire routinely under this script's
# threadpoolctl/n_jobs=1 pinning (see _worker_init/module docstring) --
# left unfiltered this would print per-fold/per-replicate, i.e.
# thousands of times across the full 142-dataset run.
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
    EXTRA_TREES_CRITERION,
    EXTRA_TREES_MAX_FEATURES,
    EXTRA_TREES_N_ESTIMATORS,
    FAMILIES,
    LEARNER_NAMES,
    RANDOM_STATE,
    make_learner,
    run_family_base_learners,
)
from src.tsfm_ensemble.families import FAMILY_REGISTRY  # noqa: E402

RESULTS_DIR = Path(os.environ.get("CV_ABLATION_142_RESULTS_DIR", str(Path("results") / "cv_ablation_142")))
FRAGMENTS_DIR = RESULTS_DIR / "fragments"
UNITS_DIR = RESULTS_DIR / "units"

NUM_RESAMPLES = 30
B_VALUES = (15, 25)
# Scheduling-only randomness (see build_jobs's own docstring) -- kept
# distinct from RANDOM_STATE (the scientific seed used for CV folds,
# bootstrap replicates, learner init, etc.) so it's unambiguous this
# has zero effect on any result, only on which unit runs when.
JOB_SHUFFLE_SEED = 20260915


def list_142_datasets(resample_indices_dir: Path = RESAMPLE_INDICES_DIR) -> List[str]:
    """See run_142_original_algorithms.py's own version of this function
    for why it reads PythonResampleIndices/ directly instead of using
    aeon's tsc_datasets registry (version-sensitive, and 7 of its names
    don't match this project's own dataset-folder naming). Duplicated
    here rather than imported -- see this module's docstring on why
    sibling experiment scripts aren't cross-imported.
    """
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
# worker via copy-on-write. See run_142_original_algorithms.py's module
# docstring ("Dataset loading") for the full reasoning.
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
# Bagging/OOB scoring -- generalized from the earlier dev-subset check
# (~/scratch/bagging_oob_check.py) to also return full score matrices
# (OOF/OOB train + test), not just a summary accuracy. See the module
# docstring's "Weighting" and "Score representation caveat" sections.
# --------------------------------------------------------------------------- #
def bagging_oob_scores(X_train, y_train_enc, X_test, n_classes, family, learner_name, B, rng):
    """Return (oof_scores, test_scores, n_never_oob, fit_seconds, predict_seconds).

    oof_scores : (n_train, n_classes) -- normalized OOB vote fraction
        per training row, over only the B replicates that left it out.
        Uniform 1/n_classes for a row OOB in zero replicates.
    test_scores : (n_test, n_classes) -- normalized vote fraction over
        every successfully-fitted replicate (uniform 1/n_classes if
        every single replicate failed to fit).
    n_never_oob : count of training rows OOB in zero replicates.
    fit_seconds : total wall time summed across all B replicates'
        ``.fit()`` calls (a failed replicate's partial fit time, if any,
        is not counted -- the ``try`` wraps the fit call itself).
    predict_seconds : total wall time summed across all B replicates'
        prediction work (both the OOB-row ``.predict()`` and the
        test-row ``.predict()`` for that replicate, combined -- splitting
        those two apart wasn't asked for and would double the timer
        calls in this already B=15/25-iteration loop for no expected
        difference in per-call cost). ``fit_seconds + predict_seconds``
        will be a little less than this function's own wall-clock (the
        bootstrap resampling / OOB-index bookkeeping outside the timed
        blocks isn't counted in either) -- same "don't expect an exact
        match" caveat as ``run_142_original_algorithms.py``'s
        ``UnitTotalSeconds`` vs. its 4 step timings.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    oof_votes = np.zeros((n_train, n_classes))
    oof_coverage = np.zeros(n_train, dtype=int)
    test_votes = np.zeros((n_test, n_classes))
    n_fitted = 0
    fit_seconds = 0.0
    predict_seconds = 0.0

    # learner_name is ridge or naive_bayes only -- extra_trees gets its
    # own arm (extra_trees_native_oob_scores below), never this one. See
    # the module docstring's "Extra Trees is already an ensemble" section.
    for _ in range(B):
        sample = rng.choice(n_train, size=n_train, replace=True)
        oob = np.setdiff1d(np.arange(n_train), sample, assume_unique=False)
        try:
            clf = make_learner(family, learner_name, random_state=RANDOM_STATE)
            t0 = time.perf_counter()
            clf.fit(X_train[sample], y_train_enc[sample])
            fit_seconds += time.perf_counter() - t0
        except Exception:  # noqa: BLE001 - one bad bootstrap replicate must not abort the whole unit
            continue
        n_fitted += 1

        t0 = time.perf_counter()
        if len(oob) > 0:
            preds_oob = clf.predict(X_train[oob])
            for row, pred in zip(oob, preds_oob):
                oof_votes[row, int(pred)] += 1
                oof_coverage[row] += 1

        preds_test = clf.predict(X_test)
        for row, pred in enumerate(preds_test):
            test_votes[row, int(pred)] += 1
        predict_seconds += time.perf_counter() - t0

    never_oob = oof_coverage == 0
    oof_scores = np.divide(
        oof_votes, oof_coverage[:, None], out=np.full_like(oof_votes, 1.0 / n_classes), where=~never_oob[:, None]
    )
    if n_fitted == 0:
        test_scores = np.full((n_test, n_classes), 1.0 / n_classes)
    else:
        test_scores = test_votes / n_fitted
    return oof_scores, test_scores, int(never_oob.sum()), fit_seconds, predict_seconds


def extra_trees_native_oob_scores(X_train, y_train_enc, X_test, family):
    """extra_trees's "bagging" arm: sklearn's own native OOB machinery,
    one fit, instead of an external bootstrap wrapper around a whole
    200-tree forest many times over. See the module docstring's "Extra
    Trees is already an ensemble" section for why.

    Builds the exact same per-family ExtraTreesClassifier config
    ``make_learner`` produces (n_estimators/max_features/criterion),
    only with ``bootstrap=True, oob_score=True`` added.

    Returns (oof_scores, test_scores, n_never_oob, fit_seconds,
    test_predict_seconds) -- same first-3 shapes/semantics as
    bagging_oob_scores's return value, for a uniform downstream
    manifest/fragment schema, but genuine (tree-averaged) probabilities
    rather than hard-vote fractions -- see the module docstring's "Score
    representation caveat". ``fit_seconds`` covers the one ``.fit()``
    call, which is also where sklearn computes
    ``oob_decision_function_`` internally (no separate "OOB predict"
    call exists to time here, unlike the external bagging arms' OOB
    scoring) -- so unlike ``bagging_oob_scores``, ``fit_seconds`` already
    includes this arm's OOF/OOB scoring cost, not just model fitting.
    ``test_predict_seconds`` is the separate ``predict_proba(X_test)``
    call only.
    """
    clf = ExtraTreesClassifier(
        n_estimators=EXTRA_TREES_N_ESTIMATORS,
        max_features=EXTRA_TREES_MAX_FEATURES[family],
        criterion=EXTRA_TREES_CRITERION,
        bootstrap=True, oob_score=True,
        random_state=RANDOM_STATE, n_jobs=1,  # 1 thread -- see _worker_init/module docstring
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train_enc)
    fit_seconds = time.perf_counter() - t0

    oof_scores = np.array(clf.oob_decision_function_, dtype=np.float64, copy=True)
    never_oob = np.isnan(oof_scores).any(axis=1)
    n_classes = oof_scores.shape[1]
    if never_oob.any():
        oof_scores[never_oob] = 1.0 / n_classes

    t0 = time.perf_counter()
    test_scores = clf.predict_proba(X_test)
    test_predict_seconds = time.perf_counter() - t0
    return oof_scores, test_scores, int(never_oob.sum()), fit_seconds, test_predict_seconds


# --------------------------------------------------------------------------- #
# Per-unit work
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    dataset: str
    resample: int
    family: str


def _unit_dir(dataset: str, resample: int, family: str) -> Path:
    return UNITS_DIR / dataset / f"resample{resample}" / family


def _fragment_path(dataset: str, resample: int, family: str) -> Path:
    return FRAGMENTS_DIR / dataset / f"resample{resample}" / f"{family}.npz"


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
    fragment_path = _fragment_path(dataset, resample, family)
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
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
            # A label in the test split never seen in training -- rare
            # but possible for some resamples of some datasets. Test-side
            # accuracy can't be computed for this unit; OOF-side honesty
            # still can, so don't abort the whole unit over it.
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

        cv_results = run_family_base_learners(family, Xt_train, y_train, Xt_test, random_state=RANDOM_STATE)

        npz_payload: Dict = {"y_train_enc": y_train_enc, "classes_": le.classes_}
        if y_test_enc is not None:
            npz_payload["y_test_enc"] = y_test_enc

        for learner_name in LEARNER_NAMES:
            row = {
                "Dataset": dataset, "Resample": resample, "Family": family, "Learner": learner_name,
                "NTrain": Xt_train.shape[0], "NTest": Xt_test.shape[0], "NClasses": n_classes,
            }

            # -- "cv" arm --
            cv_entry = cv_results[learner_name]
            if "error" in cv_entry:
                row["CvError"] = cv_entry["error"]
            else:
                npz_payload[f"{learner_name}__cv__oof_train"] = cv_entry["oof_train"]
                npz_payload[f"{learner_name}__cv__test"] = cv_entry["test"]
                row["CvOOFAcc"] = accuracy_score(y_train_enc, cv_entry["oof_train"].argmax(axis=1))
                if y_test_enc is not None:
                    row["CvTestAcc"] = accuracy_score(y_test_enc, cv_entry["test"].argmax(axis=1))
                row["CvSeconds"] = cv_entry["fit_seconds"]
                # Decomposed into the OOF cross_val_predict call vs. the
                # final full-train fit+test-score call -- CvSeconds ==
                # CvOOFSeconds + CvFinalFitSeconds exactly (see
                # run_family_base_learners's own docstring). Neither
                # sub-call exposes fit/predict separately any further
                # (cross_val_predict doesn't return per-fold timing).
                row["CvOOFSeconds"] = cv_entry["oof_seconds"]
                row["CvFinalFitSeconds"] = cv_entry["final_fit_seconds"]

            # -- "bagging" arm(s): external bootstrap wrapper for ridge/
            # naive_bayes, native OOB for extra_trees -- see the module
            # docstring's "Extra Trees is already an ensemble" section.
            if learner_name == "extra_trees":
                try:
                    t0 = time.perf_counter()
                    oof_scores, test_scores, n_never_oob, fit_s, test_predict_s = extra_trees_native_oob_scores(
                        Xt_train, y_train_enc, Xt_test, family
                    )
                except Exception as exc:  # noqa: BLE001
                    row["NativeOOBError"] = repr(exc)
                else:
                    npz_payload[f"{learner_name}__nativeoob__oof_train"] = oof_scores
                    npz_payload[f"{learner_name}__nativeoob__test"] = test_scores
                    row["NativeOOBOOFAcc"] = accuracy_score(y_train_enc, oof_scores.argmax(axis=1))
                    if y_test_enc is not None:
                        row["NativeOOBTestAcc"] = accuracy_score(y_test_enc, test_scores.argmax(axis=1))
                    row["NativeOOBNeverOOB"] = n_never_oob
                    row["NativeOOBSeconds"] = time.perf_counter() - t0
                    # NativeOOBFitSeconds includes the OOB/OOF scoring
                    # cost (computed inside .fit() itself, see
                    # extra_trees_native_oob_scores's own docstring) --
                    # NativeOOBTestSeconds is the separate test-set
                    # predict_proba call only.
                    row["NativeOOBFitSeconds"] = fit_s
                    row["NativeOOBTestSeconds"] = test_predict_s
            else:
                for B in B_VALUES:
                    arm = f"Bag{B}"
                    try:
                        t0 = time.perf_counter()
                        rng = np.random.default_rng(RANDOM_STATE)  # same seed every (learner, B) -- see module docstring
                        oof_scores, test_scores, n_never_oob, fit_s, predict_s = bagging_oob_scores(
                            Xt_train, y_train_enc, Xt_test, n_classes, family, learner_name, B, rng
                        )
                    except Exception as exc:  # noqa: BLE001
                        row[f"{arm}Error"] = repr(exc)
                        continue
                    npz_payload[f"{learner_name}__bagB{B}__oof_train"] = oof_scores
                    npz_payload[f"{learner_name}__bagB{B}__test"] = test_scores
                    row[f"{arm}OOFAcc"] = accuracy_score(y_train_enc, oof_scores.argmax(axis=1))
                    if y_test_enc is not None:
                        row[f"{arm}TestAcc"] = accuracy_score(y_test_enc, test_scores.argmax(axis=1))
                    row[f"{arm}NeverOOB"] = n_never_oob
                    row[f"{arm}Seconds"] = time.perf_counter() - t0
                    # Summed across all B replicates -- see
                    # bagging_oob_scores's own docstring for why these
                    # two won't sum to exactly {arm}Seconds (bootstrap
                    # resampling/bookkeeping outside the timed blocks
                    # isn't counted in either).
                    row[f"{arm}FitSeconds"] = fit_s
                    row[f"{arm}PredictSeconds"] = predict_s

            manifest_rows.append(row)

        # Family-level timings (extraction) and this whole unit's total
        # wall-clock are the same for all 3 learner rows -- duplicated
        # onto each rather than a separate file, matching how NTrain/
        # NTest/NClasses (also family-level, not learner-level) are
        # already duplicated across the 3 rows above. See the module
        # docstring's "manifest.csv" section: read from any ONE row for
        # these columns, don't sum across a family's 3 learner rows or
        # you'll triple-count. This script is not meant to be re-run --
        # every timing worth having later must be captured now.
        unit_total_s = time.perf_counter() - start
        for row in manifest_rows:
            row["ExtractionTrainSeconds"] = extraction_train_s
            row["ExtractionTestSeconds"] = extraction_test_s
            row["UnitTotalSeconds"] = unit_total_s

        np.savez_compressed(fragment_path, **npz_payload)
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
# Job building / resumability / finalize
# --------------------------------------------------------------------------- #
def build_jobs(datasets: List[str], resamples: int, families: List[str]) -> List[Job]:
    """Same (dataset, resample, family) unit set the original nested-loop
    version built, but returned in a dataset-interleaved order instead of
    dataset-major.

    Real crash, not hypothetical: the original version appended jobs in
    dataset-major order (all ``resamples x families`` for one dataset
    contiguously, then the next dataset). ``main()`` feeds this list to
    the ``ProcessPoolExecutor`` front-to-back, so the moment the
    scheduler works through one dataset's block and reaches the next,
    up to ``n_workers`` of that NEW dataset's jobs get dispatched to
    idle workers all at once -- multiplying that one dataset's peak
    per-unit memory (extraction output matrices, mainly -- e.g.
    KG-MTP's ``num_features=50_000`` convolution features, or RDST's
    ``max_shapelets=10_000`` shapelet distances, both scale with
    n_instances) by however many workers happened to be free right
    then, instead of spreading load across many differently-sized
    datasets. This is exactly what crashed on ``Crop`` (24,000
    instances total, one of the largest datasets in the archive by row
    count): several workers ended up extracting/fitting Crop
    simultaneously instead of Crop's ~120 jobs being spread out across
    the run.

    Fix: group pending jobs by dataset, shuffle both the dataset order
    and each dataset's own job order (fixed seed -- ``random``, not
    ``numpy``, since this is a plain list of ``Job`` dataclasses, not
    numeric data; and a distinct seed from ``RANDOM_STATE``, since this
    is scheduling order, not a scientific parameter -- shuffling here
    has zero effect on any result, only on which unit runs when), then
    interleave round-robin across datasets, one job at a time. As long
    as more than ``n_workers`` datasets still have jobs left (true for
    nearly this whole run -- 142 datasets vs. a much smaller worker
    count), round-robin interleaving *guarantees* no two of the next
    ``n_workers`` dispatched jobs share a dataset -- strictly stronger
    than a plain full shuffle of the flat job list, which would only
    make same-dataset bursts *less likely*, not impossible.
    """
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
    FRAGMENTS_DIR.mkdir(parents=True, exist_ok=True)
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
