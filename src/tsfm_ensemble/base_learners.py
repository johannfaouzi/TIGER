"""Per-family base learners for the stacking architecture.

This is the *base* layer of the new design (see the branch's git history
for the design discussion this replaces the old per-family-TabICLv2 +
reduction + CAWPE-fusion pipeline -- see ``reduction.py``/``ensemble.py``/
``pipeline.py``'s removal earlier on this branch): a small, cheap,
deliberately diverse portfolio of classifiers fit directly on each
family's *raw* (pre-reduction) extractor output -- no dimensionality
reduction step at all, unlike the old design. Each learner's out-of-fold
(OOF) scores across the training set, plus its scores on the held-out
test set, are meant to be concatenated across every (family, learner)
combination into one wide feature matrix for a downstream meta-model
(not implemented in this module -- see ``run_family_base_learners``'s
docstring for exactly what this module does and does not do).

Portfolio
---------
Three classifier types per family (12 base learners total: FAMILIES x
LEARNER_NAMES), chosen for genuinely different inductive biases (so
their errors are not perfectly correlated -- the actual point of
stacking), for being fast/cheap to fit many times over (once per CV
fold, per family, per (dataset, resample) -- see
``run_family_base_learners``), and, for the two non-Ridge choices,
deliberately not narrowed to "the literature's single best pick per
family": per-family literature signal (see QUANT's own Ridge-vs-
ExtraTrees comparison, cited below) shows the *runner-up* algorithm
still wins outright on a real minority of datasets, so the practical
question worth answering with this project's own dev-subset runs is
whether that complementarity generalizes across every family, not just
interval -- easy to prune per family later once that's measured (see
this module's docstring's parting note).

    ridge        : sklearn.linear_model.RidgeClassifierCV, with a
                   per-family ``alphas`` grid and (for 3 of 4 families) a
                   preceding ``StandardScaler`` -- both checked directly
                   against each family's own original-paper/aeon-
                   implementation source (not assumed), and matched
                   exactly where a literature precedent exists:
                     convolution (KG-MTP): ``alphas=np.logspace(-3, 3, 10)``,
                       preceded by ``StandardScaler()`` -- KG-MTP's own
                       reference script (``run_KGMTP.py``) builds
                       ``make_pipeline(StandardScaler(), RidgeClassifierCV(
                       alphas=np.logspace(-3, 3, 10), normalize=True))``;
                       ``normalize`` was removed from scikit-learn in 1.2,
                       but the pipeline's own explicit ``StandardScaler``
                       already does that job, so nothing is lost by
                       relying on it alone.
                     shapelet (RDST): ``alphas=np.logspace(-4, 4, 20)``,
                       preceded by ``StandardScaler(with_mean=True)`` --
                       aeon's own ``_rdst.py`` builds exactly
                       ``make_pipeline(StandardScaler(with_mean=True),
                       RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)))``
                       as ``RDSTClassifier``'s default estimator.
                     dictionary (WEASEL 2.0): ``alphas=np.logspace(-1, 5, 10)``,
                       explicitly NOT scaled -- aeon's own
                       ``_weasel_v2.py`` fits
                       ``RidgeClassifierCV(alphas=np.logspace(-1, 5, 10))``
                       directly on the raw bag-of-words counts, no
                       scaler. Also structurally required here regardless
                       of what the paper does: these features are
                       sparse, and a mean-centering ``StandardScaler``
                       would both densify a ~20-70k-wide matrix and
                       introduce negative values incompatible with this
                       family's ``MultinomialNB`` choice below.
                     interval (QUANT): no literature precedent either
                       way -- QUANT's own paper/aeon implementation uses
                       ExtraTrees, not Ridge, as its classifier (see
                       below), so there is nothing to match here. Scaled
                       anyway (``StandardScaler()``, grid reused from
                       convolution's), by extrapolation from the other
                       two dense continuous-valued families both scaling
                       before Ridge, and because QUANT's quantile
                       features are computed over intervals of many
                       different lengths and so can differ in scale
                       across columns in a way Ridge's uniform L2 penalty
                       would otherwise weight unevenly.
                   No ``predict_proba`` on ``RidgeClassifierCV`` itself --
                   scored via ``decision_function`` (delegated through
                   the ``Pipeline`` for the three scaled families; see
                   ``_as_2d_scores``'s handling of the binary-
                   classification 1-D special case).
    extra_trees  : sklearn.ensemble.ExtraTreesClassifier, configured as
                   ``n_estimators=200, criterion="entropy"`` everywhere
                   (shared across families -- cheap regardless of
                   feature count) plus a per-family ``max_features``:
                   ``0.1`` for interval, matching QUANT's own aeon
                   implementation (``_quant.py``)'s default estimator
                   exactly (confirmed: QUANT's own transform output
                   feeds this classifier directly, no scaling); sklearn's
                   own ``"sqrt"`` default for the other three. ``0.1`` is
                   a *fraction of that family's own raw feature count*
                   considered per split, so applying QUANT's literal
                   value outside interval means ~5,000-7,000 candidate
                   features per split on convolution's ~58k or
                   dictionary's ~69k raw columns -- measured, this made
                   convolution's extra_trees 1.6s -> 24.7s and
                   dictionary's 2.2s -> 33.3s on one real dataset, which
                   works directly against the "fast, cheap" goal this
                   whole portfolio is built around (see this branch's git
                   history) for two families QUANT's own tuning was never
                   meant to cover in the first place. QUANT's own paper
                   additionally found ExtraTrees clearly better than
                   Ridge on average for interval features specifically,
                   but *not* universally -- Ridge still wins outright on
                   a real minority of datasets, which is exactly the "not
                   fully correlated -> stacking might help" case this
                   whole portfolio is built to test; checking whether
                   that same complementarity shows up in the other three
                   families too is why this (and ridge) are run
                   everywhere, not just on their own "native" family.
    naive_bayes  : sklearn.naive_bayes.GaussianNB or MultinomialNB,
                   picked per family (``_NAIVE_BAYES_CLASS`` below) by
                   its raw features' actual sign/type -- checked
                   directly against real cached dev-subset .npz files
                   (Computers/resample0), not assumed:
                     convolution: GaussianNB -- real-valued, includes
                       negative values (observed range [-34.1, 862]).
                     interval:    GaussianNB -- real-valued, includes
                       negative values (observed range [-67.7, 454]).
                     shapelet:    GaussianNB -- non-negative (observed
                       range [0, 710]) but continuous distance/position/
                       count features, not genuine count data --
                       MultinomialNB's multinomial-count assumption
                       doesn't fit these the way it fits an actual
                       word-count matrix, so GaussianNB (which only
                       needs real-valued input, sign notwithstanding) is
                       the better-matched choice here despite the
                       non-negativity.
                     dictionary:  MultinomialNB -- genuine non-negative
                       integer bag-of-SFA-words counts, confirmed via
                       ``DictionaryFamily``'s own docstring and the raw
                       cache's dtype (``uint32``, observed range
                       [1, 707]) -- the textbook use case for this
                       class. ``MultinomialNB`` itself accepts sparse
                       input natively, but see ``run_family_base_learners``
                       for why this family's input is densified before
                       any of the three learners here ever see it.
                   Naive Bayes's feature-independence assumption is
                   almost certainly wrong for all four families (these
                   features are highly correlated by construction --
                   e.g. convolution's PPV/max/mean triplets computed per
                   kernel), so it is expected to be individually weak;
                   it is included anyway because a weak-but-differently-
                   wrong learner can still be useful for stacking
                   diversity, and it costs almost nothing to fit. This
                   is the one member of the portfolio most likely to get
                   pruned per family once the meta-model shows whether
                   it actually earns its place -- see this module's
                   parting note.

Output convention
------------------
Every learner's score matrix -- OOF on train (``run_family_base_learners``'s
``"oof_train"``), and scored on the held-out test set from a model fit on
the *full* training set (``"test"``) -- is shaped (n_samples, n_classes)
after ``_as_2d_scores``, regardless of learner or family, with columns
always in ``classes_`` (the fitted ``LabelEncoder``'s label order).

This is deliberately NOT put on a common scale across learners:
``RidgeClassifierCV``'s ``decision_function`` is an unbounded signed
margin, while ``ExtraTreesClassifier``/``GaussianNB``/``MultinomialNB``'s
``predict_proba`` is a bounded [0, 1] probability -- left heterogeneous
here rather than calibrated, matching this project's earlier conclusion
(see the design discussion this branch's git history records) that
TabICLv2's own built-in feature scaling/normalization makes explicit
calibration unnecessary for a TabICLv2 meta-model; a non-TabICLv2
meta-learner that does care about input scale can standardize these
columns itself downstream -- this module's job is only to produce
honest (no-leakage), unscaled scores, not to pre-decide what the
meta-model needs.

No dimensionality reduction
----------------------------
Every learner here fits directly on a family's full raw feature width
(convolution ~50-60k, shapelet ~30k, dictionary ~20-70k depending on
configuration, interval ~200-13k depending on series length) -- unlike
the old design's TabICLv2 branches, none of these three classifier
types has TabICLv2's own pretraining-range constraint (2-100 columns),
so there is no group_size/wide_selection machinery here at all; see
``reduction.py``'s removal from this branch.

What this module does not do
------------------------------
Load anything from disk (see ``src.experiments.common.load_raw_family_features``
for reading ``save_dev_subset_transforms.py``'s cached raw/ features),
loop over datasets/resamples/families, assemble the final wide
(n_samples, 12 x n_classes) stacking matrix across families, evaluate
any metric, or fit a meta-model on top. Those are a separate
orchestration script's job, layered on top of ``run_family_base_learners``
below -- kept out of this module so the base-learner logic itself stays
easy to unit-test in isolation (see ``tests/test_base_learners.py``).
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# sklearn's type_of_target (called internally by cross_val_predict/CV
# splitters/estimator.fit on our label arrays) warns whenever a target has
# more than 20 samples and its unique-class count exceeds 50% of that --
# a heuristic aimed at catching a *mislabeled regression target* passed to
# a classifier by mistake. It's a false positive for this project's actual
# data: UCR/UEA time series classification datasets routinely have dozens
# of genuine classes against a modest per-resample sample count (e.g. this
# project's own dev subset includes datasets with 52 classes at only 2
# training instances each), which is completely expected multiclass
# structure, not a regression problem in disguise. Silenced by exact
# message match (not a blanket UserWarning filter) so this doesn't also
# swallow this project's own deliberate warnings (e.g. _make_cv's
# "cannot stratify" or run_family_base_learners's per-learner OOF-failure
# warning), which use different message text and stay fully visible.
warnings.filterwarnings(
    "ignore",
    message=r"The number of unique classes is greater than 50% of the number of samples\.",
    category=UserWarning,
)

RANDOM_STATE = 0

FAMILIES = ("convolution", "interval", "shapelet", "dictionary")
LEARNER_NAMES = ("ridge", "extra_trees", "naive_bayes")

# Per-family RidgeClassifierCV alphas grid -- see the module docstring's
# "Portfolio" section for the exact source verified per family (three of
# four match a literature precedent; "interval" does not have one and is
# a documented extrapolation).
RIDGE_ALPHAS = {
    "convolution": np.logspace(-3, 3, 10),   # KG-MTP (run_KGMTP.py)
    "interval": np.logspace(-3, 3, 10),      # no native precedent -- reuses convolution's grid, see module docstring
    "shapelet": np.logspace(-4, 4, 20),      # RDST (aeon _rdst.py)
    "dictionary": np.logspace(-1, 5, 10),    # WEASEL 2.0 (aeon _weasel_v2.py)
}

# Families whose "ridge" learner is preceded by StandardScaler -- see
# the module docstring's "Portfolio" section for the exact reasoning per
# family (two literature-matched, one structurally required not to
# scale, one extrapolated).
RIDGE_SCALE_FAMILIES = frozenset({"convolution", "interval", "shapelet"})

# QUANT's own aeon implementation's default ExtraTreesClassifier
# configuration (see the module docstring's "Portfolio" section),
# applied to every family's "extra_trees" learner, not just interval --
# n_estimators and criterion are cheap regardless of feature count, so
# those stay shared across all four families. max_features=0.1 is not:
# it's a *fraction of this family's own raw feature count* considered
# per split, so applying QUANT's literal 0.1 to convolution's ~58k or
# dictionary's ~69k raw columns means ~5,000-7,000 candidate features
# per split (measured: convolution's extra_trees went 1.6s -> 24.7s,
# dictionary's 2.2s -> 33.3s on one real dataset once max_features=0.1
# was applied everywhere -- see this branch's git history). Only
# interval actually has literature backing for 0.1 (it's QUANT's own
# family); the other three fall back to sklearn's own "sqrt" default,
# which is what keeps per-split cost roughly log-scale in feature count
# instead of linear.
EXTRA_TREES_N_ESTIMATORS = 200
EXTRA_TREES_MAX_FEATURES = {
    "convolution": "sqrt",
    "interval": 0.1,      # QUANT (aeon _quant.py) -- the one family this is actually sourced from
    "shapelet": "sqrt",
    "dictionary": "sqrt",
}
EXTRA_TREES_CRITERION = "entropy"

# Family -> Naive Bayes class matched to that family's raw feature type;
# see the module docstring's "Portfolio" section for the empirical basis
# (checked against real cached dev-subset .npz files, not assumed).
_NAIVE_BAYES_CLASS = {
    "convolution": GaussianNB,
    "interval": GaussianNB,
    "shapelet": GaussianNB,
    "dictionary": MultinomialNB,
}

# Which fitted-estimator method produces this learner's per-class scores.
# RidgeClassifierCV has no predict_proba (it's a regression-on-one-hot-
# targets classifier under the hood); the other two are genuine
# probabilistic classifiers.
LEARNER_METHOD = {
    "ridge": "decision_function",
    "extra_trees": "predict_proba",
    "naive_bayes": "predict_proba",
}


def make_learner(family: str, learner_name: str, random_state: int = RANDOM_STATE):
    """Construct one (family, learner_name) base learner, unfit.

    ``family`` affects every learner type -- see the module docstring's
    "Portfolio" section for exactly what varies and why: ``"ridge"``
    picks a per-family ``alphas`` grid (``RIDGE_ALPHAS``) and whether a
    preceding ``StandardScaler`` is included (``RIDGE_SCALE_FAMILIES``,
    returning a ``Pipeline`` when it is, a bare ``RidgeClassifierCV``
    when it isn't); ``"extra_trees"`` shares ``n_estimators``/
    ``criterion`` across families but picks a per-family
    ``max_features`` (``EXTRA_TREES_MAX_FEATURES`` -- QUANT's own 0.1
    for interval, sklearn's cheaper "sqrt" default for the other three);
    ``"naive_bayes"`` picks ``GaussianNB`` vs ``MultinomialNB`` per
    ``_NAIVE_BAYES_CLASS``.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}.")
    if learner_name == "ridge":
        ridge = RidgeClassifierCV(alphas=RIDGE_ALPHAS[family])
        if family in RIDGE_SCALE_FAMILIES:
            return make_pipeline(StandardScaler(), ridge)
        return ridge
    if learner_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=EXTRA_TREES_N_ESTIMATORS,
            max_features=EXTRA_TREES_MAX_FEATURES[family],
            criterion=EXTRA_TREES_CRITERION,
            random_state=random_state, n_jobs=-1,
        )
    if learner_name == "naive_bayes":
        return _NAIVE_BAYES_CLASS[family]()
    raise ValueError(f"Unknown learner_name {learner_name!r}; expected one of {LEARNER_NAMES}.")


def _as_2d_scores(raw: np.ndarray, method: str) -> np.ndarray:
    """Normalize one scoring call's output to a 2-D (n_samples,
    n_classes_seen) array, regardless of learner or dataset.

    ``decision_function`` returns a 1-D (n_samples,) array in the binary-
    classification case only (a single signed margin toward the second
    of the two classes in ``classes_`` order) -- every other case
    (multiclass decision_function, or predict_proba regardless of class
    count) is already 2-D. Expanding the binary case to
    ``[-margin, margin]`` gives every learner a uniform 2-D shape with
    columns in ``classes_`` order, so downstream code (assembling the
    wide stacking matrix across learners/families) never needs to
    special-case binary datasets.
    """
    if method == "decision_function" and raw.ndim == 1:
        return np.column_stack([-raw, raw])
    return raw


def _make_cv(y_enc: np.ndarray, max_folds: int = 5, random_state: int = RANDOM_STATE):
    """A ``StratifiedKFold`` with as many folds as the rarest class
    safely allows (capped at ``max_folds``), falling back to a plain
    (unstratified) ``KFold`` -- with a loud warning -- if some class has
    fewer than 2 training examples.

    ``StratifiedKFold`` requires ``n_splits <= min(class counts)`` (a
    singleton class can appear in at most one fold's test set, so
    ``n_splits >= 2`` would otherwise raise); the smallest dev-subset
    datasets here have as few as 20 training instances total (e.g.
    MoteStrain, resample0), so this is a real, not hypothetical, edge
    case worth handling gracefully rather than letting
    ``cross_val_predict`` crash the whole job.
    """
    counts = np.bincount(y_enc)
    min_class_count = int(counts.min())
    if min_class_count >= 2:
        n_folds = max(2, min(max_folds, min_class_count))
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    n_folds = max(2, min(max_folds, len(y_enc)))
    warnings.warn(
        f"At least one class has only {min_class_count} training example(s) -- "
        f"cannot stratify; falling back to unstratified {n_folds}-fold CV for OOF "
        "scoring, which may leave that class out of some training folds entirely."
    )
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_state)


def _maybe_densify(X_train, X_test):
    """Convert ``X_train``/``X_test`` to dense ``np.ndarray`` when
    ``X_train`` is sparse (currently only ``DictionaryFamily``/WEASEL
    2.0's bag-of-words output) -- a no-op otherwise.

    Mirrors ``save_dev_subset_transforms.py``'s ``_to_dense_output``,
    which made this same call, unconditionally, for its own cache-writing
    path based on direct measurement (see ``run_family_base_learners``'s
    docstring for the numbers). Duplicated here, rather than relied on
    solely via that upstream cache, so this function's own documented
    input contract ("either sparse or dense in, dense-fit out") holds
    for every caller -- including one that hands this function raw,
    never-cached ``DictionaryFamily`` output directly, which would
    otherwise silently take the much slower sparse-fit path.

    The density decision is made from ``X_train`` alone and applied to
    both splits (same reasoning as ``_to_dense_output``: train and test
    come from the same extractor fit on the same dataset, so treating
    them differently would be a pointless on/off inconsistency, not a
    real trade-off).
    """
    if not sp.issparse(X_train):
        return X_train, X_test
    return np.asarray(X_train.todense()), np.asarray(X_test.todense())


def _fit_and_score_final(
    family: str,
    learner_name: str,
    X_train,
    y_train_enc: np.ndarray,
    X_test,
    random_state: int = RANDOM_STATE,
):
    """Fit a fresh instance of ``learner_name`` on the *full* training
    set, then score it on ``X_test`` -- this is "step 2" of
    ``run_family_base_learners`` (the model that is actually deployed),
    pulled out into its own function so its fit call and its scoring
    call can be timed separately (``fit_seconds``, ``predict_seconds``)
    as well as together. This is a pure refactor, not a behavior change:
    ``run_family_base_learners`` calls this directly for its own
    ``final_fit_seconds`` (== ``fit_seconds + predict_seconds`` exactly,
    same combined duration it always measured as one block), and
    ``measure_base_learner_final_split_142.py`` calls it directly too,
    to get this step's timing split at 142-dataset scale without also
    re-running the (already-cached, and considerably more expensive)
    OOF cross-validation step above it in ``run_family_base_learners``.

    Parameters
    ----------
    family, learner_name : as in ``make_learner``.
    X_train, X_test : already-extracted, already-densified features
        (see ``run_family_base_learners``'s own docstring -- this
        function does no densification of its own; callers must have
        already done it, exactly as ``run_family_base_learners`` does
        before its own call site).
    y_train_enc : label-encoded training labels (``LabelEncoder``
        already applied by the caller).

    Returns
    -------
    ``(test_scores, fit_seconds, predict_seconds)`` -- ``test_scores``
    is ``(n_test, n_classes)``, via ``_as_2d_scores`` exactly as
    ``run_family_base_learners`` already returns it.
    """
    method = LEARNER_METHOD[learner_name]

    fit_start = time.perf_counter()
    estimator = make_learner(family, learner_name, random_state=random_state)
    estimator.fit(X_train, y_train_enc)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    test_scores = _as_2d_scores(getattr(estimator, method)(X_test), method)
    predict_seconds = time.perf_counter() - predict_start

    return test_scores, fit_seconds, predict_seconds


def run_family_base_learners(
    family: str,
    X_train,
    y_train: np.ndarray,
    X_test,
    max_cv_folds: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """Fit every learner in ``LEARNER_NAMES`` for one family, on one
    (dataset, resample)'s already-loaded raw features (e.g. from
    ``src.experiments.common.load_raw_family_features``), and return
    each learner's out-of-fold train scores plus its test-set scores.

    For each learner:

    1. Out-of-fold (OOF) scores on ``X_train`` via
       ``sklearn.model_selection.cross_val_predict`` under a
       label-stratified CV split (``_make_cv``) -- an *honest* score for
       every training row (each row is scored by a model that never saw
       it during that fold's fit), which is what a meta-model needs to
       train on without leaking its own base learners' in-sample
       overfitting into its own training signal. This is deliberately
       *not* combined with calibration (``CalibratedClassifierCV`` or
       similar) -- see the module docstring's "Output convention"
       section for why that's a separate, not-yet-needed concern here.
    2. A fresh instance of the same learner, fit once on the *full*
       ``X_train``, then scored on ``X_test`` -- this is the model that
       would actually be deployed; its test-set scores are what feeds
       the meta-model at evaluation time.

    Both scoring calls go through ``_as_2d_scores`` so every entry has a
    uniform (n_samples, n_classes) shape regardless of learner type or
    how many classes this particular dataset has.

    Parameters
    ----------
    family : one of ``FAMILIES`` -- only affects which Naive Bayes
        variant ``make_learner`` builds (see its own docstring).
    X_train, X_test : this family's raw (pre-reduction) extractor output
        for the training/test split -- dense ``np.ndarray`` for
        convolution/interval/shapelet; dictionary may arrive as sparse
        ``scipy.sparse.csr_matrix`` (``DictionaryFamily``'s own output
        format) or already-dense (e.g. from a caller reading a
        pre-densified cache), and either way this function densifies it
        itself, once, before fitting any learner -- see the module-level
        ``_maybe_densify`` for why: measured directly on this project's
        own dev-subset dictionary features, ``RidgeClassifierCV`` was
        38-62x slower on sparse input than dense at the densities WEASEL
        2.0 actually produces (every dev-subset dataset >=18% dense), and
        ``ExtraTreesClassifier`` ~2.5x slower, with no measured case where
        sparse storage won on speed (matches
        ``save_dev_subset_transforms.py``'s own ``_to_dense_output``,
        which made the same call for its cache-writing path; this
        function makes it here too so the same is true regardless of
        caller).
    y_train : this split's raw (un-encoded) training labels -- encoded
        internally via a fresh ``LabelEncoder`` fit on ``y_train`` alone
        (never on test labels, which this function doesn't even see).

    Returns
    -------
    ``{learner_name: {"oof_train": (n_train, n_classes) array,
    "test": (n_test, n_classes) array, "classes_": label encoder classes
    in this dataset's own label order, "fit_seconds": total wall time for
    both fits (OOF cross_val_predict + the final full-train fit),
    "oof_seconds": wall time for just the ``cross_val_predict`` call
    (step 1 -- covers that call's own internal per-fold fit+predict,
    which ``cross_val_predict`` doesn't expose split out any further),
    "final_fit_seconds": wall time for step 2 (the fresh full-train fit
    plus its scoring call on ``X_test``, combined -- ``fit_seconds`` is
    exactly ``oof_seconds + final_fit_seconds``, kept alongside the two
    for backward compatibility with existing callers, e.g.
    ``build_stacking_features.py``, that only read ``"fit_seconds"``)}}``
    for every entry in ``LEARNER_NAMES`` this call could actually score.

    A learner whose OOF scoring fails is recorded instead as
    ``{"error": repr(exc)}`` rather than raising and losing every other
    learner's (already-computed) results. In practice this only happens
    for ``"ridge"`` (``decision_function``) on a pathological dataset
    where some class has fewer training examples than the CV can ever
    keep in every fold's *training* set -- e.g. a genuinely singleton
    class (exactly 1 training example): that example must sit in some
    fold's test set, so the class is absent from every fold's training
    set no matter how ``_make_cv`` splits.

    This module used to assume (and this docstring used to claim)
    that ``cross_val_predict(..., method="decision_function")`` always
    *raises* when a fold's training set is missing a class, the same way
    it does in the binary (2-class) case ("Only {n} class/es in training
    fold... This is not supported for decision_function with imbalanced
    folds", straight from sklearn's own
    ``sklearn.model_selection._validation`` module) -- which the
    ``except ValueError`` below would catch and turn into this same
    ``{"error": ...}`` outcome. That is only true for 2 classes. For 3+
    classes (every dataset that actually hit this in the real 142-scale
    run: DiatomSizeReduction/4 classes, FiftyWords/50, Phoneme/39,
    confirmed by direct investigation and a full scan of every one of
    this project's 142 datasets x 30 resamples -- these three are the
    *only* ones where any resample has a genuinely singleton class),
    sklearn does NOT raise: it silently fills the missing class's
    ``decision_function`` column with ``np.finfo(dtype).min`` (~-1.8e308,
    the most negative representable float64, chosen so that class can
    never win an argmax) for every OOF row scored by the fold(s) missing
    it, and returns normally -- see that same module's fold-combining
    helper. ``predict_proba`` (extra_trees, naive_bayes) is unaffected:
    sklearn 0-fills that case instead, which is already correct/benign
    for a probability column. A ``decision_function`` column silently
    filled with a legitimate-looking (if extreme) finite float is *not*
    caught by the ``except ValueError`` below, so without the explicit
    sentinel check just after the ``cross_val_predict`` call, this would
    silently poison every downstream consumer of this learner's OOF
    scores instead of erroring loudly here -- confirmed in practice: it
    doesn't crash anything nearby, it crashes ``StandardScaler`` inside
    ``fit_predict_logistic_regression``, three layers downstream, when
    squaring ~1.8e308 for a variance computation overflows to ``inf``.
    The sentinel check below restores the originally-intended behaviour
    (a clean, recorded ``{"error": ...}`` for this learner on this job)
    for the 3+-class case too, matching this project's existing "one bad
    (sub-)problem must not take down the whole job" convention (see
    ``save_dev_subset_transforms.py``'s per-job error handling).
    """
    X_train, X_test = _maybe_densify(X_train, X_test)

    label_encoder = LabelEncoder().fit(y_train)
    y_train_enc = label_encoder.transform(y_train)
    cv = _make_cv(y_train_enc, max_folds=max_cv_folds, random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}
    for learner_name in LEARNER_NAMES:
        method = LEARNER_METHOD[learner_name]
        start = time.perf_counter()

        try:
            oof_train = cross_val_predict(
                make_learner(family, learner_name, random_state=random_state),
                X_train, y_train_enc, cv=cv, method=method,
            )
        except ValueError as exc:
            warnings.warn(
                f"{family}/{learner_name}: OOF scoring failed ({exc!r}) -- skipping this "
                "learner for this job rather than failing every other learner too."
            )
            results[learner_name] = {"error": repr(exc)}
            continue
        oof_seconds = time.perf_counter() - start
        oof_train = _as_2d_scores(oof_train, method)

        if method == "decision_function":
            # See this function's own docstring's correction above: for
            # 3+ classes, sklearn's cross_val_predict silently fills a
            # missing-from-some-fold class's column with
            # np.finfo(dtype).min instead of raising. Detect that exact,
            # deterministic sentinel (never a value a real ridge
            # decision_function would produce by chance) and treat it as
            # the OOF-scoring failure it actually is, rather than letting
            # it silently poison every downstream consumer.
            sentinel = np.finfo(oof_train.dtype).min
            if np.any(oof_train == sentinel):
                exc = ValueError(
                    f"{family}/{learner_name}: cross_val_predict(method='decision_function') "
                    f"silently filled >=1 column with sklearn's missing-class sentinel "
                    f"({sentinel!r}) instead of raising -- at least one CV fold's training "
                    "set was missing a class entirely (see _make_cv's stratification-"
                    "fallback warning). Treating this as an OOF-scoring failure for this "
                    "learner rather than returning contaminated scores."
                )
                warnings.warn(
                    f"{family}/{learner_name}: OOF scoring failed ({exc!r}) -- skipping this "
                    "learner for this job rather than failing every other learner too."
                )
                results[learner_name] = {"error": repr(exc)}
                continue

        test_scores, final_fit_only_seconds, final_predict_only_seconds = _fit_and_score_final(
            family, learner_name, X_train, y_train_enc, X_test, random_state=random_state
        )
        final_fit_seconds = final_fit_only_seconds + final_predict_only_seconds

        results[learner_name] = {
            "oof_train": oof_train,
            "test": test_scores,
            "classes_": label_encoder.classes_,
            "fit_seconds": oof_seconds + final_fit_seconds,
            "oof_seconds": oof_seconds,
            "final_fit_seconds": final_fit_seconds,
            "final_fit_only_seconds": final_fit_only_seconds,
            "final_predict_only_seconds": final_predict_only_seconds,
        }
    return results
