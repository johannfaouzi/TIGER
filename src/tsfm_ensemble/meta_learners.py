"""The meta-model layer: combines the 12 (family, learner) base learners'
already-saved out-of-fold (OOF) train scores and test scores
(``build_stacking_features.py``'s ``scores.npz``, itself built on
``base_learners.run_family_base_learners``) into final per-(dataset,
resample) predictions, via the six methods this branch's design
discussion settled on:

    hard_majority_vote  : plurality vote over each base learner's own
                           argmax prediction -- the non-learned baseline
                           (see its own docstring for the tie-break rule).
    logistic_regression : a scaled multinomial logistic regression fit
                           directly on the wide OOF meta-feature matrix.
    lightgbm             : a gradient-boosted-tree meta-classifier, same
                           input.
    cawpe                : HIVE-COTE 2.0's own weighted-probability
                           combiner (weight = OOF accuracy**4 per base
                           learner), with the four ridge learners'
                           uncalibrated ``decision_function`` margins
                           replaced by ``CalibratedClassifierCV``-
                           calibrated probabilities so they can
                           participate on the same scale as the other
                           eight (genuinely probabilistic) learners.
    cawpe_hard_vote      : a hard-voting counterpart to ``cawpe`` --
                           same per-learner weight (OOF accuracy**4)
                           as CAWPE, but applied to each base
                           learner's hard argmax vote instead of its
                           probability vector, so (unlike ``cawpe``)
                           it never needs calibrated ridge
                           probabilities -- see its own docstring.
                           Added to answer Johann's question about a
                           CV-weighted hard-voting scheme; evaluated
                           on the 20 development datasets only, not
                           (yet) on the 142-dataset benchmark.
    tabiclv2              : ``tsfm_ensemble.backbone.GroupedICLClassifier``
                           (this branch's existing TabICLv2 wrapper),
                           fit directly on the wide meta-feature matrix,
                           with ``ecoc`` selecting native mixed-radix
                           many-class handling vs. an additional ECOC
                           wrapper -- see
                           ``compare_tabiclv2_multiclass_handling.py``
                           for the harness that measures which to use.
                           Uses ``backbone="tabicl"`` (strict -- see
                           ``fit_predict_tabiclv2``'s own docstring): no
                           silent fallback to a placeholder classifier,
                           ever. If the real TabICLv2 backbone can't be
                           imported or fit, this method raises instead
                           of quietly substituting a different model.
    meta_oracle            : diagnostic-only upper bound over the SIX
                           methods just above (hard_majority_vote,
                           logistic_regression, lightgbm, cawpe,
                           cawpe_hard_vote, tabiclv2), NOT over
                           oracle_cv/oracle below -- selects whichever
                           of those six had the best
                           TEST accuracy for this (dataset, resample) and
                           reports that accuracy. Same "looks at test
                           labels to pick a winner, never deployable"
                           caveat as ``oracle`` below, at the meta-method
                           level instead of the single-base-learner
                           level -- the point is comparing the two
                           oracles' headroom against each other (does
                           stacking-with-hindsight beat
                           picking-one-base-learner-with-hindsight?), not
                           using either in practice. Implemented in the
                           orchestration scripts' own ``_run_methods``
                           (not here): it only needs the five methods'
                           already-computed outputs from the same job, no
                           separate fit/predict of its own.
    oracle_cv             : leakage-free, deployable "pick the single
                           best base learner" method -- selects whichever
                           (family, learner) block has the best OOF
                           (training-set, cross-validated) accuracy (see
                           ``best_block_by_oof_accuracy``) and returns
                           that block's own test-set scores. Never looks
                           at the test labels to make its choice.
    oracle                 : diagnostic-only upper bound, NOT a real
                           meta-learner -- selects whichever (family,
                           learner) block has the best TEST accuracy (see
                           ``best_block_by_test_accuracy``) and returns
                           that block's own test-set scores. This is the
                           one function in this module that looks at
                           ``y_test_enc`` for anything other than final
                           scoring -- see this docstring's "Meta-level
                           leakage discipline" section. Useful only to
                           quantify how much headroom the other methods
                           are leaving on the table; never deployable,
                           since there is no way to know a dataset's best
                           base learner ahead of time without already
                           knowing the answer.

This module holds the algorithmic building blocks only (mirroring
``base_learners.py``'s own split): no disk I/O, no looping over
datasets/resamples, no resilience/retry logic -- every function here
raises on failure rather than catching anything, so a caller (e.g.
``src/experiments/build_meta_model_predictions.py``) can decide per
(dataset, resample, method) whether a failure should skip just that one
method or abort the whole job, the same "one bad (sub-)problem must not
take down the whole job" convention used throughout this project (see
``run_family_base_learners``'s and ``build_stacking_features.py``'s own
docstrings).

Meta-level leakage discipline
------------------------------
Every method below is trained on the OOF ``meta.X_train``/``y_train_enc``
and evaluated on ``meta.X_test``/``y_test_enc`` -- the classic Wolpert
"stacking generalizer" setup, which is leak-free as long as (a) the base
layer's OOF scores are themselves honest (verified true for
``run_family_base_learners``'s ``cross_val_predict``-based scoring -- see
that module's own docstring) and (b) the meta-layer never touches the
test set for anything other than the final prediction call. This module
follows (b) throughout:

    * ``fit_predict_logistic_regression`` tunes its regularization
      strength via ``LogisticRegressionCV``'s own internal k-fold CV,
      built entirely from ``meta.X_train``/``y_train_enc`` (the same
      adaptive ``_make_cv`` the base layer already uses) -- the test set
      never enters that selection.
    * ``fit_predict_lightgbm`` early-stops against a validation split
      carved out of the OOF training rows only (never the test set),
      falling back to a small fixed-size model (no early stopping) when
      the training set is too small/imbalanced to safely carve one out
      without dropping a class entirely.
    * ``cawpe_combine``'s per-learner weight is
      ``accuracy_score(y_train_enc, argmax(oof_block))**4`` -- purely a
      function of the already-honest OOF scores and the training labels,
      no test-set access at all (see ``learner_weights``).
    * ``calibrate_ridge_family`` fits ``CalibratedClassifierCV`` once on
      the full training set (its own internal CV for calibration is
      itself entirely within that training set) and only ever calls
      ``predict_proba`` on the test set at the very end -- exactly the
      same "fit on full train, score test once" shape as every other
      "test" array already saved in ``scores.npz``
      (``run_family_base_learners``'s step 2).
    * ``fit_predict_tabiclv2`` is a plain fit-on-train/predict-on-test
      call, like every other method here, and (as of the ``backbone="tabicl"``
      default) never silently substitutes a placeholder model on failure.
    * ``oracle_by_oof_accuracy``/``best_block_by_oof_accuracy`` selects its
      single winning base learner using only the already-honest OOF scores
      and ``y_train_enc`` -- no test-set access, same discipline as
      ``learner_weights``.
    * ``oracle_best_single_learner``/``best_block_by_test_accuracy`` is the
      one deliberate, marked exception to this whole section: it selects
      its winning base learner using ``y_test_enc`` directly, which is
      exactly why it is documented everywhere as diagnostic-only and never
      deployable -- see its own docstring.

Test-set peeking caution (inherited from ``build_stacking_features.py``'s
own docstring, "On not using the test scores to make development
decisions") still applies to any *human* decision that looks at test-set
accuracy while choosing between methods or hyperparameters -- e.g.
deciding which of TabICLv2's native-vs-ECOC many-class handling to use
going forward, which
``src/experiments/compare_tabiclv2_multiclass_handling.py`` measures
directly on the dev subset's test sets. That is a deliberate, acknowledged
exception (the same kind of architecture-level decision
``build_stacking_features.py``'s docstring already flags), not a leak in
the sense the rest of this docstring is about -- no *individual* method's
own fit/predict here ever uses test labels or test features for anything
but its own final scoring call.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from src.tsfm_ensemble.base_learners import (
    FAMILIES,
    LEARNER_METHOD,
    LEARNER_NAMES,
    RANDOM_STATE,
    _make_cv,
    make_learner,
)

META_METHODS = (
    "hard_majority_vote",
    "logistic_regression",
    "lightgbm",
    "cawpe",
    "cawpe_hard_vote",
    "tabiclv2",
    "meta_oracle",
    "oracle_cv",
    "oracle",
)


@dataclass
class ColumnBlock:
    """One (family, learner)'s columns within an assembled meta-feature
    matrix -- ``[start, stop)`` into ``MetaFeatures.X_train``/``X_test``'s
    second axis. ``method`` is ``base_learners.LEARNER_METHOD[learner]``
    ("decision_function" for ridge, "predict_proba" for the other two),
    kept here so downstream code (e.g. CAWPE's calibration step) can tell
    which blocks are genuine probabilities without re-deriving it.
    """

    family: str
    learner: str
    method: str
    start: int
    stop: int


@dataclass
class MetaFeatures:
    """The assembled wide (n_samples, sum of each present block's width)
    meta-feature matrix for one (dataset, resample) job, plus everything
    needed to interpret its columns -- see ``assemble_meta_features``.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train_enc: np.ndarray
    y_test_enc: np.ndarray
    classes_: np.ndarray
    blocks: List[ColumnBlock]
    missing: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.classes_)


def assemble_meta_features(
    scores, families: Sequence[str] = FAMILIES, learners: Sequence[str] = LEARNER_NAMES,
) -> MetaFeatures:
    """Concatenate every available (family, learner) block's OOF-train
    and test score arrays (``build_stacking_features.py``'s
    ``"{family}__{learner}__oof_train"``/``"...__test"`` keys) into one
    wide matrix each, in a fixed, deterministic column order
    (``families`` x ``learners``, both in their default order unless
    overridden).

    ``scores`` is anything dict-like with those keys plus
    ``"classes_"``/``"y_train_enc"``/``"y_test_enc"`` -- in practice
    either a loaded ``np.load(...)`` ``NpzFile`` or a plain ``dict`` (as
    used by this module's own tests).

    A (family, learner) pair with no OOF/test key (it errored during
    ``run_family_base_learners`` -- see that function's docstring for the
    one realistic way that happens, a genuinely singleton training class
    breaking ridge's OOF scoring) is silently skipped and recorded in the
    returned ``MetaFeatures.missing`` list, rather than raising --
    matching ``base_learners``'s own "one bad learner must not take down
    every other learner" convention. Every meta-method downstream simply
    sees fewer columns/voters for that one job; raises only if *every*
    block is missing (nothing at all to build a meta-model from).
    """
    classes_ = np.asarray(scores["classes_"])
    y_train_enc = np.asarray(scores["y_train_enc"])
    y_test_enc = np.asarray(scores["y_test_enc"])

    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    blocks: List[ColumnBlock] = []
    missing: List[Tuple[str, str]] = []
    col = 0
    for family in families:
        for learner in learners:
            oof_key = f"{family}__{learner}__oof_train"
            test_key = f"{family}__{learner}__test"
            if oof_key not in scores or test_key not in scores:
                missing.append((family, learner))
                continue
            oof = np.asarray(scores[oof_key])
            test = np.asarray(scores[test_key])
            width = oof.shape[1]
            train_blocks.append(oof)
            test_blocks.append(test)
            blocks.append(ColumnBlock(family, learner, LEARNER_METHOD[learner], col, col + width))
            col += width

    if not train_blocks:
        raise ValueError(
            "assemble_meta_features: no (family, learner) blocks available -- "
            "every one of them is missing/errored for this job."
        )

    X_train = np.concatenate(train_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)
    return MetaFeatures(X_train, X_test, y_train_enc, y_test_enc, classes_, blocks, missing)


def _class_group_key(meta: "MetaFeatures") -> np.ndarray:
    """Per-column class index for ``meta.X_train``/``X_test``'s columns --
    ``backbone.GroupedICLClassifier``'s ``class_group_key`` grouping
    strategy, so ``fit_predict_tabiclv2`` can guarantee no TabICLv2 group
    ends up blind to a class (see that class's module docstring's
    "Grouping" section for why plain random grouping doesn't give that
    guarantee, and how often it actually matters at this project's own
    class counts).

    Every block in ``meta.blocks`` is one (family, learner)'s
    ``[start, stop)`` columns, always ``stop - start == meta.n_classes``
    wide (one predicted-probability-or-decision_function column per
    class, in the same ``0..n_classes-1`` order as ``meta.classes_`` --
    true for every learner regardless of ``method``: sklearn's own
    multiclass convention keeps both ``predict_proba`` and
    ``decision_function`` columns aligned to the fitted estimator's own
    ``classes_``, and every OOF/test score array here was produced from
    the same shared ``y_train_enc`` label encoding, see
    ``run_family_base_learners``). So a column's offset *within* its own
    block is exactly its class index -- this needs no access to the
    actual score values, just ``meta.blocks``' own recorded widths.
    """
    n_features = meta.X_train.shape[1]
    key = np.empty(n_features, dtype=int)
    for block in meta.blocks:
        width = block.stop - block.start
        key[block.start:block.stop] = np.arange(width)
    return key


def _realign_columns(proba: np.ndarray, classes, n_classes: int) -> np.ndarray:
    """Defensively realign a ``predict_proba``-style output's columns to
    the shared ``0..n_classes-1`` encoded-label order, in case the
    fitted estimator's own ``classes_`` skips one (e.g. a class absent
    from an internal CV fold's training subset). A no-op copy when the
    columns already match -- which is the normal case throughout this
    project, since every meta-classifier here is fit on the full
    ``y_train_enc`` (which by construction already covers every class
    0..n_classes-1: it's the very array ``LabelEncoder`` produced from
    ``y_train`` in ``run_family_base_learners``).
    """
    classes = np.asarray(classes).astype(int)
    if proba.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return proba
    full = np.zeros((proba.shape[0], n_classes))
    for j, c in enumerate(classes):
        full[:, c] = proba[:, j]
    return full


def accuracy_from_proba(proba: np.ndarray, y_true_enc: np.ndarray) -> float:
    """Accuracy of a (n_samples, n_classes) probability/score matrix's
    argmax predictions against encoded true labels -- a small shared
    convenience so every method's own accuracy is computed the same way.
    """
    return float(accuracy_score(y_true_enc, proba.argmax(axis=1)))


# --------------------------------------------------------------------------- #
# Hard majority voting
# --------------------------------------------------------------------------- #
def _block_argmax_predictions(X: np.ndarray, blocks: Sequence[ColumnBlock]) -> np.ndarray:
    """Each learner's own argmax prediction -- scale-invariant per learner
    (this is exactly what ``RidgeClassifierCV.predict()`` itself does
    with ``decision_function``, and what any ``predict_proba``-based
    ``.predict()`` does), so this is meaningful even though the blocks
    themselves are on incomparable scales (unbounded signed margins vs.
    bounded [0, 1] probabilities) -- see ``base_learners.py``'s own
    "Output convention" docstring section for why that heterogeneity is
    deliberate. Returns (n_samples, n_blocks) encoded class indices.
    """
    preds = np.empty((X.shape[0], len(blocks)), dtype=int)
    for j, block in enumerate(blocks):
        preds[:, j] = X[:, block.start:block.stop].argmax(axis=1)
    return preds


def hard_majority_vote(meta: MetaFeatures) -> np.ndarray:
    """Plurality vote over every available base learner's own argmax
    prediction (see ``_block_argmax_predictions``) -- the non-learned
    baseline, used instead of uniform probability averaging because
    ridge's ``decision_function`` margins are not on a comparable scale
    to the other two learners' genuine probabilities (Johann's own
    observation; see this branch's design discussion).

    Tie-break rule (ties are the norm here, not an edge case: with up to
    12 voters, results like 6-6 or 4-4-4 happen routinely) -- Johann's
    own rule, verbatim: "predict the majority class if the classes are
    imbalanced, otherwise predict the 'first' class (to have something
    reproducible)". Implemented as: among the classes tied for the
    plurality vote,

        * if the TRAINING set's class distribution is imbalanced (not
          every class has exactly the same number of training examples),
          pick the tied class with the most training examples -- i.e.
          the overall majority class, restricted to genuine contenders
          for this sample's vote (predicting a class no learner even
          voted for would defeat the point of a tie-break). A further
          tie among those training counts (e.g. two tied classes that
          also happen to have equal training-set representation) is
          broken by the smaller encoded class index, for determinism.
        * if the training distribution is perfectly balanced (every
          class has the same count), pick the smallest encoded class
          index directly among the tied candidates -- the "first" class,
          reproducible by construction.

    Returns the (n_test,) array of encoded-label predictions.
    """
    if not meta.blocks:
        raise ValueError("hard_majority_vote: MetaFeatures has no blocks to vote over.")

    preds = _block_argmax_predictions(meta.X_test, meta.blocks)
    n_classes = meta.n_classes
    class_counts = np.bincount(meta.y_train_enc, minlength=n_classes)
    imbalanced = not np.all(class_counts == class_counts[0])

    n_samples = preds.shape[0]
    result = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        vote_counts = np.bincount(preds[i], minlength=n_classes)
        best = vote_counts.max()
        tied = np.flatnonzero(vote_counts == best)
        if len(tied) == 1:
            result[i] = tied[0]
        elif imbalanced:
            tied_training_counts = class_counts[tied]
            # Sort by (-training_count, class_index) ascending -> largest
            # training count first, smallest index breaks further ties.
            order = np.lexsort((tied, -tied_training_counts))
            result[i] = tied[order[0]]
        else:
            result[i] = tied.min()
    return result


# --------------------------------------------------------------------------- #
# CAWPE
# --------------------------------------------------------------------------- #
def _block_oof_accuracy(meta: MetaFeatures) -> Dict[Tuple[str, str], float]:
    """Raw OOF (training-set, already-honest) argmax accuracy per
    (family, learner) block -- the shared ingredient behind
    ``learner_weights`` (raises this to the 4th power) and
    ``best_block_by_oof_accuracy`` (picks the single largest one, no
    exponent -- selecting a winner, not building a weighted blend).
    Purely a function of ``meta.X_train``/``meta.y_train_enc``: no
    test-set access.
    """
    accs: Dict[Tuple[str, str], float] = {}
    for block in meta.blocks:
        preds = meta.X_train[:, block.start:block.stop].argmax(axis=1)
        accs[(block.family, block.learner)] = accuracy_score(meta.y_train_enc, preds)
    return accs


def learner_weights(meta: MetaFeatures) -> Dict[Tuple[str, str], float]:
    """CAWPE's per-learner weight: ``accuracy_score(y_train, argmax(oof
    block))**4`` -- HIVE-COTE 2.0's own formula (exponent confirmed = 4
    by reading aeon's ``_hivecote_v2.py`` directly), computed from each
    block's already-honest OOF scores and the actual training labels --
    no test-set access, no calibration needed (accuracy from argmax is
    identical whether or not the block's scores are later calibrated
    into genuine probabilities -- calibration only matters for
    *combining* probabilities, not for this weight; see
    ``calibrate_ridge_family``'s own docstring).
    """
    return {key: acc ** 4 for key, acc in _block_oof_accuracy(meta).items()}


def _cawpe_hard_vote_tally(meta: MetaFeatures, weights: Dict[Tuple[str, str], float]) -> np.ndarray:
    """The test-time half of ``cawpe_hard_vote``: given already-computed
    per-learner ``weights`` (``learner_weights(meta)`` -- train-side only,
    needs only OOF scores and training labels, no test data), compute
    each test row's weighted-tally vote and resolve ties. Pulled out of
    ``cawpe_hard_vote`` so the (near-instant either way, but still
    conceptually separate) "compute weights from training data" and
    "vote on test data" steps can be timed separately --
    ``measure_meta_classification_split_142.py`` calls ``learner_weights``
    and this function directly, timed as two blocks, instead of calling
    ``cawpe_hard_vote`` (which times them as one). Pure extraction, not a
    behavior change: ``cawpe_hard_vote`` still returns exactly what it
    always did.
    """
    preds = _block_argmax_predictions(meta.X_test, meta.blocks)
    block_weights = np.array([weights[(b.family, b.learner)] for b in meta.blocks])

    n_classes = meta.n_classes
    class_counts = np.bincount(meta.y_train_enc, minlength=n_classes)
    imbalanced = not np.all(class_counts == class_counts[0])

    n_samples = preds.shape[0]
    result = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        tally = np.zeros(n_classes)
        np.add.at(tally, preds[i], block_weights)
        best = tally.max()
        tied = np.flatnonzero(np.isclose(tally, best))
        if len(tied) == 1:
            result[i] = tied[0]
        elif imbalanced:
            tied_training_counts = class_counts[tied]
            order = np.lexsort((tied, -tied_training_counts))
            result[i] = tied[order[0]]
        else:
            result[i] = tied.min()
    return result


def cawpe_hard_vote(meta: MetaFeatures) -> np.ndarray:
    """Weighted hard-voting counterpart to ``cawpe_combine``: instead of
    blending each block's own *probabilities* by CAWPE's per-learner
    weight, each block casts a single hard (argmax) vote and that vote is
    weighted by the same CAWPE weight (``learner_weights`` -- OOF
    accuracy**4). Concretely: for each test sample, every block's argmax
    prediction contributes ``weight`` to that class's running tally
    (``_block_argmax_predictions``' (n_test, n_blocks) output, summed
    per class via ``np.add.at``); the predicted class is whichever tally
    is largest.

    This answers Johann's question about "a hard-voting version of
    CAWPE" -- distinct from both ``hard_majority_vote`` (every block's
    vote counts equally) and ``cawpe_combine`` (soft/probability-
    weighted, needs calibrated ridge probabilities). Like
    ``hard_majority_vote`` and unlike ``cawpe_combine``, this never needs
    ``calibrate_ridge_family``'s calibrated probabilities -- only each
    block's own argmax, which ridge's raw ``decision_function`` already
    gives for free -- so every block always participates (no
    calibration-failure carve-out the way ``cawpe_combine`` has).

    Tie-break: identical rule to ``hard_majority_vote`` (Johann's own
    rule -- majority-class-among-the-tied if the training distribution
    is imbalanced, else smallest encoded class index), applied to
    whichever classes are tied for the largest *weighted* tally instead
    of the largest raw vote count. Ties are compared with ``np.isclose``
    since the tally is now a sum of floating-point weights, not integer
    votes.

    Dev-subset only, for now: added to ``META_METHODS`` and
    ``build_meta_model_predictions.py``'s (20-dataset) candidate pool per
    Johann's request; deliberately NOT wired into
    ``build_meta_model_predictions_142.py``, which hardcodes its own
    method list rather than iterating ``META_METHODS`` -- consistent
    with keeping any new, not-yet-validated candidate off the
    142-dataset evaluation set until it has been assessed on the 20
    development datasets alone.

    Returns the (n_test,) array of encoded-label predictions -- no
    probabilistic output, same convention as ``hard_majority_vote``.
    """
    if not meta.blocks:
        raise ValueError("cawpe_hard_vote: MetaFeatures has no blocks to vote over.")

    weights = learner_weights(meta)
    return _cawpe_hard_vote_tally(meta, weights)


def calibrate_ridge_family(
    family: str,
    X_train,
    y_train_enc: np.ndarray,
    X_test,
    n_classes: int,
    cv=None,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Sigmoid(Platt)-calibrate this family's ridge learner
    (``base_learners.make_learner(family, "ridge", ...)``) and return its
    calibrated ``predict_proba`` on ``X_test``, fit once on the *full*
    training set -- the same "fit on full train, score test" shape as
    every other "test" array in ``scores.npz`` (see
    ``run_family_base_learners``'s step 2). ``cv`` defaults to this
    family's own adaptive CV (``base_learners._make_cv`` on
    ``y_train_enc``) if not supplied -- the same split scheme the base
    layer already uses for OOF scoring, reused here for
    ``CalibratedClassifierCV``'s internal calibration folds.

    Only test-set probabilities are produced (not an OOF train matrix):
    CAWPE's weight for this block comes from the cheaper, already-saved
    uncalibrated OOF ridge scores instead (see ``learner_weights``), so a
    second, *nested*-CV OOF calibration pass (``cross_val_predict``
    wrapped around a ``CalibratedClassifierCV`` that already does its own
    internal CV) would cost real compute for a number this function's
    caller never needs.
    """
    if cv is None:
        cv = _make_cv(y_train_enc, max_folds=5, random_state=random_state)
    base = make_learner(family, "ridge", random_state=random_state)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    calibrated.fit(X_train, y_train_enc)
    test_proba = calibrated.predict_proba(X_test)
    return _realign_columns(test_proba, calibrated.classes_, n_classes)


def cawpe_combine(
    meta: MetaFeatures, test_proba_by_block: Dict[Tuple[str, str], np.ndarray],
) -> np.ndarray:
    """HIVE-COTE 2.0's weighted-and-normalized probability combination:
    ``sum_c weight_c * proba_c``, rows re-normalized to sum to 1 (aeon's
    own ``_hivecote_v2.py`` formula).

    ``test_proba_by_block`` must supply one genuine (n_test, n_classes)
    probability matrix per block this call should include, keyed by
    ``(family, learner)`` -- build it with ``build_cawpe_test_probabilities``
    (non-ridge blocks sliced straight out of ``meta.X_test``, ridge blocks
    substituted with ``calibrate_ridge_family``'s output). A block present
    in ``meta.blocks`` but absent from this mapping (e.g. ridge
    calibration failed/was skipped for that family) is simply left out of
    the combination -- CAWPE proceeds on whatever blocks *are* available,
    matching this project's "one bad component must not take down the
    whole job" convention; raises only if the mapping is empty (nothing
    at all to combine).
    """
    weights = learner_weights(meta)
    n_test = meta.X_test.shape[0]
    acc = np.zeros((n_test, meta.n_classes))
    used = 0
    for block in meta.blocks:
        key = (block.family, block.learner)
        if key not in test_proba_by_block:
            continue
        acc += test_proba_by_block[key] * weights[key]
        used += 1
    if used == 0:
        raise ValueError("cawpe_combine: test_proba_by_block has no usable blocks to combine.")
    row_sums = acc.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return acc / row_sums


def build_cawpe_test_probabilities(
    meta: MetaFeatures, ridge_calibrated_test_proba: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], np.ndarray]:
    """Assemble the ``{(family, learner): (n_test, n_classes) proba}``
    mapping ``cawpe_combine`` needs: non-ridge blocks are genuine
    ``predict_proba`` output already, sliced straight out of
    ``meta.X_test``; ridge blocks are substituted with
    ``ridge_calibrated_test_proba[family]`` (keyed by family alone, since
    there is exactly one ridge learner per family) -- see
    ``calibrate_ridge_family``. A ridge block whose family is missing
    from ``ridge_calibrated_test_proba`` (calibration failed/was skipped
    for that family) is simply omitted from the result -- see
    ``cawpe_combine``'s docstring for what happens then.
    """
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for block in meta.blocks:
        if block.learner == "ridge":
            if block.family not in ridge_calibrated_test_proba:
                continue
            out[(block.family, block.learner)] = ridge_calibrated_test_proba[block.family]
        else:
            out[(block.family, block.learner)] = meta.X_test[:, block.start:block.stop]
    return out



def calibrate_ridge_family_from_scores(
    oof_scores: np.ndarray, y_train_enc: np.ndarray, test_scores: np.ndarray, n_classes: int,
) -> np.ndarray:
    """Sigmoid(Platt)-calibrate one family's ridge block directly from its
    already-computed OOF/test ``decision_function`` scores (e.g.
    ``cv_ablation_142.py``'s fragment ``.npz`` arrays, or ``scores.npz``'s
    own ``"{family}__ridge__oof_train"``/``"__test"`` blocks) -- no raw
    feature matrix needed, unlike ``calibrate_ridge_family``.

    Why this is equivalent, not a shortcut
    ------------------------------------------
    ``calibrate_ridge_family`` fits ``CalibratedClassifierCV(base, cv=cv)``
    on raw features: with a non-prefit ``cv``, that internally *refits* a
    fresh ridge classifier per fold to get the held-out scores it
    calibrates against -- i.e. it silently redoes its own from-scratch
    OOF cross-validation, on top of (and independent from) whatever OOF
    generation already happened for the "cv"/"cawpe"'s own scores.npz
    block. The scores this function calibrates against instead ARE
    already-honest OOF decision_function values (each training row
    scored by a model that never saw it) -- calibration only needs
    (score, true label) pairs, never the underlying feature matrix that
    produced those scores (Platt's method is literally a 1-D logistic
    regression of ``true label`` on ``score``, fit independently per
    class in the one-vs-rest style ``CalibratedClassifierCV`` already
    uses for a multiclass ``decision_function`` estimator -- see
    ``sklearn.calibration._SigmoidCalibration``, the exact private class
    ``CalibratedClassifierCV(method="sigmoid")`` uses internally, called
    here directly). Reusing already-computed OOF scores instead of
    refitting is not just cheaper (no raw feature cache, no repeated
    ridge fits) -- it also means CAWPE's ridge calibration is now
    consistent with whichever arm ("cv" vs bagging/native-OOB) an
    ablation found more honest for ridge in that family, instead of
    always regenerating its own fixed, unablated 5-fold OOF pass.

    A per-class fit that raises (e.g. a pathological column) falls back
    to that class's uninformative OOF prevalence for every test row --
    the same "one bad component doesn't take down the whole thing"
    convention as this module's other combinators -- rather than
    crashing the whole CAWPE combination.
    """
    from sklearn.calibration import _SigmoidCalibration

    n_test = test_scores.shape[0]
    calibrated_test = np.empty((n_test, n_classes))
    for c in range(n_classes):
        target = (y_train_enc == c).astype(int)
        try:
            sigmoid = _SigmoidCalibration()
            sigmoid.fit(oof_scores[:, c], target)
            calibrated_test[:, c] = sigmoid.predict(test_scores[:, c])
        except Exception:  # noqa: BLE001 - one bad class calibration must not sink the whole block
            calibrated_test[:, c] = target.mean() if len(target) else 1.0 / n_classes
    row_sums = calibrated_test.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return calibrated_test / row_sums


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #
def _sanitize_finite(
    X_train: np.ndarray, X_test: np.ndarray, *, context: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace any non-finite (NaN/+-inf) entries in ``X_train``/``X_test``
    with values derived ONLY from ``X_train`` (never leaking test-set
    information into the substitution), for a meta-learner (e.g.
    ``LogisticRegressionCV``) that cannot accept non-finite input at all.

    Real, observed root cause (not hypothetical): ridge's
    ``decision_function`` margins are deliberately left unbounded (see
    ``base_learners.py``'s "Output convention" docstring section), and on
    this project's most extreme small-``NTrain``/many-class 142-scale
    datasets (DiatomSizeReduction, NTrain=16/4 classes; FiftyWords,
    NTrain=450/50 classes; Phoneme, NTrain=214/39 classes -- confirmed
    100% reproducible across every one of their 30 resamples in the real
    142-scale run) ``RidgeClassifierCV``'s own generalized-CV alpha search
    becomes numerically unstable enough to occasionally emit an actual
    NaN/inf margin. ``StandardScaler`` silently propagates that into an
    all-NaN column, and ``LogisticRegressionCV`` then raises
    ``ValueError: Input X contains NaN`` -- this function exists so that
    failure becomes a graceful, loud substitution instead of a crashed
    job.

    Per affected column: NaN -> that column's median across its finite
    ``X_train`` entries (0.0 if the whole column is non-finite there);
    ``+inf``/``-inf`` -> that column's max/min finite ``X_train`` value
    (a "clip to the observed range" substitution, consistent with these
    being unbounded-by-design margins rather than a bug to paper over).
    Emits one ``UserWarning`` per call when anything was substituted,
    naming the affected column count -- never silent, matching this
    project's "loud, auditable fallback" convention (e.g.
    ``build_meta_model_predictions_142.py``'s ``_auto_select_tabiclv2_ecoc``;
    ``backbone.py``'s own TabICLv2 wrapper no longer has a silent fallback
    at all -- see that module's docstring).
    A no-op (returns the inputs unchanged, no copy) when nothing needs
    fixing -- the normal case for every family/learner outside the three
    datasets above.
    """
    train_bad = ~np.isfinite(X_train)
    test_bad = ~np.isfinite(X_test)
    if not train_bad.any() and not test_bad.any():
        return X_train, X_test

    X_train = X_train.copy()
    X_test = X_test.copy()
    n_cols_affected = 0
    for j in range(X_train.shape[1]):
        col_train_bad = train_bad[:, j]
        col_test_bad = test_bad[:, j]
        if not col_train_bad.any() and not col_test_bad.any():
            continue
        n_cols_affected += 1
        finite_vals = X_train[~col_train_bad, j]
        if finite_vals.size == 0:
            median = lo = hi = 0.0
        else:
            median = float(np.median(finite_vals))
            lo, hi = float(finite_vals.min()), float(finite_vals.max())
        for col in (X_train[:, j], X_test[:, j]):
            col[np.isnan(col)] = median
            col[np.isposinf(col)] = hi
            col[np.isneginf(col)] = lo

    warnings.warn(
        f"meta_learners{f' ({context})' if context else ''}: replaced non-finite "
        f"(NaN/inf) entries in {n_cols_affected} meta-feature column(s) before "
        "fitting -- see _sanitize_finite's docstring for why this happens "
        "(unbounded ridge decision_function margins on small/many-class "
        "datasets) and how the substitution is chosen. Imputation statistics "
        "were computed from the training split only.",
        stacklevel=3,
    )
    return X_train, X_test


def fit_predict_logistic_regression(
    meta: MetaFeatures, random_state: int = RANDOM_STATE, max_iter: int = 5000, n_cs: int = 10,
    fallback_C: float = 1.0,
) -> np.ndarray:
    """A scaled multinomial logistic regression meta-classifier:
    ``StandardScaler`` (the wide meta-feature matrix mixes unbounded
    ridge margins with bounded [0, 1] probabilities -- logistic
    regression's L2 penalty needs a comparable scale across columns the
    way ridge itself does per-family, see ``base_learners.py``'s
    "Portfolio" section) feeding ``LogisticRegressionCV``, whose
    regularization strength is tuned via its own internal k-fold CV --
    see this module's docstring's "Meta-level leakage discipline" section
    for why that CV never touches the test set.

    Falls back to a fixed-``C`` (default 1.0, sklearn's own default),
    non-CV-tuned ``LogisticRegression`` when ``meta.y_train_enc`` has a
    genuinely singleton class (< 2 training examples) -- the exact same
    condition ``_make_cv`` itself already warns about and falls back to
    unstratified KFold for (see that function's docstring). A singleton
    class can, structurally, never sit in *every* fold's training set no
    matter how CV is split; ``LogisticRegressionCV``'s own internal
    per-fold coefficient-path aggregation requires exactly that, and
    raises a numpy "inhomogeneous shape" ``ValueError`` when one fold's
    fit sees fewer classes than another's -- confirmed directly (a
    minimal synthetic reproduction with plain sklearn hits the identical
    error) on the real 142-scale run's DiatomSizeReduction, FiftyWords,
    and Phoneme, the same three datasets whose genuinely-singleton class
    is what originally surfaced this whole investigation. This is a
    *different* failure from, and not fixed by, the ridge-OOF sentinel
    fix in ``base_learners.run_family_base_learners``: that fix stops
    ridge's OOF scores from being silently contaminated by sklearn's
    missing-class placeholder value; this fallback instead avoids a crash
    that's internal to ``LogisticRegressionCV`` itself, entirely
    independent of what the input features look like. Mirrors
    ``fit_predict_lightgbm``'s own ``can_hold_out``-driven fallback to a
    fixed, non-CV-tuned configuration for the same class of small/
    pathological dataset -- loud (one ``UserWarning``), never silent.

    Also sanitizes any remaining non-finite (NaN/inf) meta-feature
    entries via ``_sanitize_finite`` first, as a defense-in-depth safety
    net for any *other* source of extreme/non-finite scores -- see that
    function's docstring. A no-op in the normal case (including, now,
    the three datasets above: with ridge's OOF scores no longer
    contaminated at the source, this rarely has anything left to do).
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test = _sanitize_finite(meta.X_train, meta.X_test, context="logistic_regression")

    counts = np.bincount(meta.y_train_enc)
    if counts.min() < 2:
        warnings.warn(
            "fit_predict_logistic_regression: at least one class has < 2 training "
            f"examples (per-class counts={counts.tolist()}) -- LogisticRegressionCV's "
            "internal k-fold CV cannot keep every class in every fold's training set no "
            "matter how it's split, and crashes (a numpy 'inhomogeneous shape' error) "
            f"when it can't. Falling back to a fixed C={fallback_C} LogisticRegression "
            "(no internal CV/regularization tuning) for this job instead -- see this "
            "function's own docstring.",
            stacklevel=2,
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=fallback_C, max_iter=max_iter, random_state=random_state),
        )
    else:
        inner_cv = _make_cv(meta.y_train_enc, max_folds=5, random_state=random_state)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=n_cs, cv=inner_cv, max_iter=max_iter, n_jobs=1, random_state=random_state,
            ),
        )
    model.fit(X_train, meta.y_train_enc)
    proba = model.predict_proba(X_test)
    return _realign_columns(proba, model.classes_, meta.n_classes)


# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
def _lgbm_min_child_samples(n_rows: int) -> int:
    """The ``min_child_samples`` to use for a LightGBM fit on ``n_rows``
    training rows.

    LightGBM's own default (20) silently prevents ANY split from ever
    happening once ``n_rows < 2 * 20 = 40`` (a split's two children must
    each retain >=20 rows) -- the model then degenerates to a single
    root leaf predicting the constant training-set class prior, with no
    error or warning. See ``fit_predict_lightgbm``'s docstring for the
    real, confirmed failure this caused (Coffee: accuracy 0.536,
    constant-class prediction) and how widespread it is across this
    project's datasets (28 of 142).

    Scaled down for small ``n_rows`` (floored at 1, capped at LightGBM's
    own default of 20 so larger datasets are unaffected and keep the
    exact previously-tested behaviour).
    """
    return max(1, min(20, n_rows // 5))


def fit_predict_lightgbm(
    meta: MetaFeatures,
    random_state: int = RANDOM_STATE,
    valid_fraction: float = 0.2,
    n_estimators: int = 500,
    fallback_n_estimators: int = 100,
    early_stopping_rounds: int = 30,
) -> np.ndarray:
    """A LightGBM meta-classifier fit directly on the wide OOF
    meta-feature matrix, early-stopped against an internal validation
    split carved out of the OOF training rows only -- never the test
    set (see this module's docstring's "Meta-level leakage discipline").

    Falls back to a small, fixed-size (no early stopping) model, fit on
    every OOF training row, when the training set is too small or too
    imbalanced to carve out a validation split without dropping some
    class from the *training* half entirely (LightGBM's multiclass
    objective needs every class present in its training data) -- this is
    a real, not hypothetical, edge case among this project's dev-subset
    datasets (e.g. MoteStrain, n_train=20).

    LightGBM's own default ``min_child_samples`` (20) requires >=40 rows
    to ever make a single split (both of a split's children must retain
    >=20 rows each). Below that, every tree in the ensemble stays a
    single root leaf that just predicts the constant training-set class
    prior -- completely silently, no error or warning from LightGBM --
    and the whole model degenerates to a constant prediction regardless
    of the input features. This is a real, not hypothetical, failure:
    28 of this project's 142 datasets have too few rows for the default
    to ever let a single split happen (confirmed on Coffee, n_train=28:
    default settings predicted a single constant class for all 28 test
    rows, accuracy 0.536, while every other meta-method scored >=1.0;
    scaling ``min_child_samples`` down for small fits fixed it, 1.0).
    ``min_child_samples`` is therefore scaled to the actual number of
    rows fed to each fit call (see ``_lgbm_min_child_samples``) rather
    than left at LightGBM's fixed default.
    """
    from lightgbm import LGBMClassifier, early_stopping
    from sklearn.model_selection import train_test_split

    X, y = meta.X_train, meta.y_train_enc
    n_classes = meta.n_classes
    counts = np.bincount(y, minlength=n_classes)

    params = dict(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        random_state=random_state, n_jobs=1, verbosity=-1,
    )

    can_hold_out = len(y) >= 20 and counts.min() >= 2
    idx_tr = idx_va = None
    if can_hold_out:
        idx = np.arange(len(y))
        stratify = y if counts.min() >= 2 else None
        try:
            idx_tr, idx_va = train_test_split(
                idx, test_size=valid_fraction, stratify=stratify, random_state=random_state,
            )
        except ValueError:
            # sklearn's stratified split additionally requires the
            # (train, test) row counts to each be >= n_classes -- not
            # covered by the len(y)/counts.min() guard above. Real,
            # observed failure mode on this project's many-class,
            # small-N 142-scale datasets (52-class Pig* series, 6-class
            # Symbols, at n_train ~90-150): fall back to the no-hold-out
            # path below instead of crashing the job.
            can_hold_out = False
        else:
            if len(np.unique(y[idx_tr])) < n_classes:
                can_hold_out = False

    if can_hold_out:
        fit_params = {**params, "min_child_samples": _lgbm_min_child_samples(len(idx_tr))}
        model = LGBMClassifier(**fit_params)
        model.fit(
            X[idx_tr], y[idx_tr],
            eval_X=X[idx_va], eval_y=y[idx_va],
            callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
    else:
        fit_params = {
            **params,
            "n_estimators": fallback_n_estimators,
            "min_child_samples": _lgbm_min_child_samples(len(y)),
        }
        model = LGBMClassifier(**fit_params)
        model.fit(X, y)

    proba = model.predict_proba(meta.X_test)
    return _realign_columns(proba, model.classes_, n_classes)


# --------------------------------------------------------------------------- #
# TabICLv2
# --------------------------------------------------------------------------- #
def fit_predict_tabiclv2(
    meta: MetaFeatures,
    ecoc: bool = False,
    group_size: int = 100,
    stratify_groups_by_class: bool = True,
    ecoc_backend: str = "many_class",
    ecoc_alphabet_size: int = 10,
    ecoc_redundancy: int = 4,
    ecoc_code_size: float = 1.5,
    backbone: str = "tabicl",
    backbone_kwargs: Optional[dict] = None,
    random_state: int = RANDOM_STATE,
):
    """TabICLv2 (``tsfm_ensemble.backbone.GroupedICLClassifier``), fit on
    the wide OOF meta-feature matrix and scored on the test meta-feature
    matrix. ``ecoc`` selects native mixed-radix many-class handling
    (``False`` -- ``tabicl.TabICLClassifier``'s own default internal
    behaviour) vs. also wrapping each group in an ECOC layer (``True``,
    ``ecoc_backend`` selects which implementation) -- see ``backbone.py``'s
    module docstring for exactly what each does, and
    ``compare_tabiclv2_multiclass_handling.py`` for the harness that
    measures which one to use going forward on this project's own
    >10-class dev-subset datasets, per the TabICLv2 paper's own finding
    that native+ECOC beats native alone on >10-class TALENT benchmarks
    (paper text: "slightly better but 3x slower" than native alone) -- not
    assumed to transfer here without checking.

    ``stratify_groups_by_class`` (default ``True``): when ``group_size``
    is small enough that ``GroupedICLClassifier`` splits the meta-feature
    matrix into more than one group, this decides *how* columns are
    assigned to groups. ``True`` passes ``_class_group_key(meta)`` as
    ``class_group_key`` -- a key-stratified round-robin that guarantees
    every group gets at least some of every class's 12 (one per learner)
    direct-evidence columns, whenever ``n_groups <= 12`` (true for every
    dev-subset job at the default ``group_size=100``). ``False`` falls
    back to the original random-shuffle assignment, which gives no such
    guarantee -- e.g. at 60 classes (720 columns, 8 groups of 90 under
    the default ``group_size``), plain random assignment has a ~90%
    chance that *some* group ends up with none of a given class's 12
    columns at all (measured directly against the real grouping code, not
    just estimated). See ``backbone.py``'s module docstring's "Grouping"
    section for the full derivation, and
    ``compare_tabiclv2_multiclass_handling.py``'s ``--stratify-groups-by-
    class``/``--no-stratify-groups-by-class`` flags for comparing the two
    empirically.

    ``ecoc_backend="many_class"`` (the default) is
    ``tabpfn_extensions.many_class.ManyClassClassifier`` -- the TabICLv2
    paper's own text says it "also supports TabPFNv2's ECOC wrapper", and
    this is that wrapper (a real, maintained module from PriorLabs, the
    TabPFN authors; confirmed by reading its actual source that it wraps
    any sklearn-compatible estimator via nothing but standard
    ``fit``/``predict_proba`` calls, and genuinely implements
    ``predict_proba`` rather than falling back to a one-hot vote). Its own
    codebook-size formula scales as roughly
    ``max(log_alphabet(n_classes), n_classes / alphabet_size) * redundancy``
    (``ecoc_redundancy``, default 4, matching its own
    ``n_estimators_redundancy`` default), capped near
    ``4 * log_alphabet(n_classes)`` -- e.g. 52 classes at the default
    ``ecoc_alphabet_size=10`` needs 8 estimators per group, not the ~78 a
    naive ``int(n_classes * code_size)`` formula would give, which lines
    up with the paper's own reported ~3x far better than that naive
    formula does. ``ecoc_alphabet_size`` defaults to 10, TabICLv2's own
    known pretrained ``max_classes`` (``TabICLClassifier`` has no
    ``MAX_NUMBER_OF_CLASSES`` attribute for ``ManyClassClassifier`` to
    auto-detect the way it does for actual TabPFN estimators, so this must
    be, and is, passed explicitly). Requires the ``tabpfn-extensions``
    package (see ``requirements.txt``) -- raises a clear ``ImportError``
    with install instructions if missing, not a silent fallback.

    ``ecoc_backend="sklearn"`` is the original implementation: sklearn's
    generic, off-the-shelf ``OutputCodeClassifier(code_size=ecoc_code_size)``
    (sklearn's own default 1.5) -- no real ``predict_proba``, and each
    group holds ``int(n_classes * ecoc_code_size)`` separate backbone
    instances simultaneously (sklearn's own ``OutputCodeClassifier``
    docstring: "Estimators used for predictions" -- all of them, not just
    one), each an independent, unshared copy of TabICLv2's weights
    (confirmed by reading the real installed ``tabicl`` package's
    ``TabICLClassifier._load_model()``, which always allocates and loads
    fresh rather than sharing/caching across instances) -- e.g. the same
    52-class case needs ``int(52*1.5)=78`` instances per group, a ~78x
    memory multiplier over native mode's single instance per group. Kept
    available for comparison against ``"many_class"`` and because it needs
    no extra dependency, but not believed to match what the TabICLv2 paper
    itself measured.

    Either backend's per-instance memory footprint is separately governed
    by ``backbone_kwargs``'s ``batch_size``/``offload_mode``/
    ``disk_offload_dir`` (passed straight through to ``TabICLClassifier``):
    the per-instance "column-wise embedding" tensor
    (``batch_size x n_rows x n_columns x embed_dim``) that ``tabicl``'s
    own ``offload_mode`` docstring calls "the main memory bottleneck". See
    ``src/experiments/profile_tabiclv2_memory.py`` for a script that
    measures real peak RAM for these levers on your own data/machine.

    Returns the fitted model's ``predict_proba(meta.X_test)``, column-
    realigned to ``0..n_classes-1`` order.
    """
    from src.tsfm_ensemble.backbone import GroupedICLClassifier

    model = GroupedICLClassifier(
        group_size=group_size, backbone=backbone, backbone_kwargs=backbone_kwargs,
        ecoc=ecoc, ecoc_backend=ecoc_backend, ecoc_alphabet_size=ecoc_alphabet_size,
        ecoc_redundancy=ecoc_redundancy, ecoc_code_size=ecoc_code_size, random_state=random_state,
    )
    class_group_key = _class_group_key(meta) if stratify_groups_by_class else None
    model.fit(meta.X_train, meta.y_train_enc, class_group_key=class_group_key)
    proba = model.predict_proba(meta.X_test)
    return _realign_columns(proba, model.classes_, meta.n_classes)


# --------------------------------------------------------------------------- #
# Oracle single-learner selection
# --------------------------------------------------------------------------- #
def best_block_by_oof_accuracy(meta: MetaFeatures) -> ColumnBlock:
    """Pick whichever (family, learner) block has the best OOF/training
    accuracy (``_block_oof_accuracy``) -- ties broken by ``meta.blocks``'
    own deterministic (family, learner) iteration order, i.e. the first
    block encountered wins a tie. See ``oracle_by_oof_accuracy``, which
    returns this block's own test-set scores as its output. Exposed
    separately so a caller (e.g. an orchestration script's manifest) can
    record *which* block won without redoing the computation.

    Never looks at ``meta.y_test_enc`` -- unlike
    ``best_block_by_test_accuracy``, this is a real, deployable
    selection rule.
    """
    if not meta.blocks:
        raise ValueError("best_block_by_oof_accuracy: MetaFeatures has no blocks.")
    oof_acc = _block_oof_accuracy(meta)
    return max(meta.blocks, key=lambda b: oof_acc[(b.family, b.learner)])


def best_block_by_test_accuracy(meta: MetaFeatures) -> ColumnBlock:
    """Pick whichever (family, learner) block has the best TEST accuracy
    -- ties broken by ``meta.blocks``' own deterministic iteration order.
    See ``oracle_best_single_learner``'s docstring for why this is
    diagnostic-only and never a real meta-learner: it uses
    ``meta.y_test_enc`` directly, the one deliberate, marked exception to
    this module's "no test-label access outside a method's own final
    scoring call" discipline (see this module's own docstring's
    "Meta-level leakage discipline" section).
    """
    if not meta.blocks:
        raise ValueError("best_block_by_test_accuracy: MetaFeatures has no blocks.")
    best_block, best_acc = None, -1.0
    for block in meta.blocks:
        preds = meta.X_test[:, block.start:block.stop].argmax(axis=1)
        acc = accuracy_score(meta.y_test_enc, preds)
        if acc > best_acc:
            best_acc, best_block = acc, block
    return best_block


def oracle_by_oof_accuracy(meta: MetaFeatures) -> np.ndarray:
    """The leakage-free, deployable "pick a single best base learner"
    meta-method: selects the (family, learner) block with the best OOF
    (training-set, cross-validated) accuracy via
    ``best_block_by_oof_accuracy``, and returns that block's own
    (n_test, n_classes) test-set scores unchanged (ridge blocks keep
    their raw unbounded ``decision_function`` margins -- fine here, since
    only the argmax is ever scored against, same as every other block
    consumer in this module).

    This answers Johann's question "could we estimate the oracle single
    learner on the training set, using cross-validation?" -- yes: the OOF
    scores this picks from are already exactly a cross-validated
    (adaptive k-fold, see ``base_learners._make_cv``) estimate of each
    base learner's held-out accuracy, computed with zero test-set access.
    It is not guaranteed to match ``oracle_best_single_learner``'s
    test-accuracy-best pick (OOF accuracy is an estimate, not the answer
    key), but it is a real, deployable method a caller could actually run
    in production, unlike that one.
    """
    block = best_block_by_oof_accuracy(meta)
    return meta.X_test[:, block.start:block.stop].copy()


def oracle_best_single_learner(meta: MetaFeatures) -> np.ndarray:
    """Diagnostic-only upper bound, NOT a real meta-learner: selects the
    (family, learner) block with the best TEST accuracy via
    ``best_block_by_test_accuracy``, and returns that block's own
    (n_test, n_classes) test-set scores unchanged.

    This deliberately violates the rest of this module's "never touch
    the test set except for a method's own final scoring call" rule --
    picking the winner *is* here a function of the test labels, which is
    exactly why this can never be deployed: there is no way to know,
    ahead of time and without already knowing the answer, which single
    base learner will turn out to score best on a given dataset's test
    set. Its only purpose is as a quantified upper bound the other,
    genuinely deployable methods (including ``oracle_by_oof_accuracy``)
    can be measured against -- see this module's own docstring's
    "Meta-level leakage discipline" section for where this exception is
    called out.
    """
    block = best_block_by_test_accuracy(meta)
    return meta.X_test[:, block.start:block.stop].copy()
