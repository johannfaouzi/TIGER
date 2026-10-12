"""
Per-family time series feature extractors.

Each class below wraps one or more ``aeon`` transformers and exposes a
common, minimal interface:

    extractor.fit(X, y)          # X: (n_cases, n_channels, n_timepoints)
    extractor.transform(X)       # -> ndarray or scipy.sparse matrix,
                                  #    shape (n_cases, n_raw_features)
    extractor.fit_transform(X, y)

The families follow the eight-way taxonomy from "Bake off redux"
(Middlehurst, Schafer & Bagnall, 2024) -- specifically the four
representations that HIVE-COTE 2.0 combines: convolution-based,
interval-based, shapelet-based, and dictionary-based. See the accompanying
report (Sec. 2-3, 5.3) for why these four are the ones worth ensembling
here, as opposed to e.g. redundant hand-crafted statistical feature sets.

Every extractor deliberately returns the *raw* representation, before any
family-specific dimension reduction -- reduction is a separate, swappable
step (see ``reduction.py``) precisely because the right reduction strategy
differs by family (report Sec. 5.5).
"""

import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.base import BaseEstimator, TransformerMixin

from src.tsfm_ensemble.kg_mtp import KGMTP
from src.tsfm_ensemble.quant import QuantTransformer


class BaseFamily(BaseEstimator, TransformerMixin):
    """Common interface for all family extractors.

    Attributes
    ----------
    name : str
        Short family name, used as a key elsewhere (reduction registry,
        branch names in the ensemble, etc.).
    sparse_output : bool
        Whether ``transform`` returns a ``scipy.sparse`` matrix -- true
        only for the dictionary family (WEASEL 2.0's bag-of-words counts
        are mostly zero); the other three return a dense ``ndarray``.

    Inference-time pruning
    -----------------------
    ``prune(keep_mask)`` is the hook the pipeline calls, once, right after
    a family-matched reducer (``reduction.py``) has decided which of this
    extractor's *raw* output columns are actually used downstream. A
    family that implements it reconfigures its already-fitted internal
    transformer(s) so that every *subsequent* ``transform()`` call (i.e.
    every call at inference time, and on held-out folds) computes only
    what's needed to reproduce those columns -- not the full raw feature
    set followed by a column mask, which is what this codebase did before
    and is precisely what was flagged as wasteful: "if a transformed
    feature is not used, then this transformed feature should not be
    created in the first place during inference."

    ``keep_mask`` is a boolean array the same length as this extractor's
    (pre-pruning) ``transform()`` output width. Returns ``True`` if
    pruning was applied (in which case ``transform()`` now returns exactly
    ``keep_mask.sum()`` columns, in the same relative order as
    ``keep_mask``'s ``True`` entries -- i.e. equivalent to
    ``full_output[:, keep_mask]``), or ``False`` if this extractor/variant
    doesn't support it, in which case the caller must keep doing the
    select-after-compute masking itself. The base implementation always
    returns ``False``; this is a real, honest fallback, not an error, since
    not every representation's internals are amenable to it (see each
    subclass's docstring for the specific reason where it applies).

    Of the current families, only ``ShapeletFamily`` (RDST) implements
    this: its shapelets each independently produce their own block of
    output columns (distance/argmin/count), so pruning can recompute only
    the surviving shapelets' blocks (the dominant cost), then apply one
    small leftover column mask (``_post_prune_column_mask_``) to trim any
    not-actually-wanted columns that came along for the ride within a kept
    block. ``ConvolutionFamily`` (KGMTP), ``IntervalFamily`` (QUANT), and
    ``DictionaryFamily`` (WEASEL 2.0) all fall back to the default,
    compute-then-mask behaviour -- see each class's docstring for why.

    Constant-feature removal
    -------------------------
    Separately from (and unconditionally, unlike) pruning, every family
    also drops raw output columns that are literally constant on the
    training data, via the shared ``_nonconstant_mask`` helper below.
    Every one of these representations can produce dead columns for
    reasons specific to its own algorithm: a convolutional kernel whose
    PPV never crosses its bias threshold on a given dataset, a QUANT
    interval too short/degenerate to vary, a shapelet that matches (or
    fails to match) identically across every training case, a dictionary
    word that never occurs. These carry no signal, cost memory and compute
    for nothing, and can even break a univariate selection test downstream
    (e.g. an undefined chi-squared statistic for a zero-variance column).
    Each family's ``fit``/``fit_transform`` computes ``_nonconstant_mask_``
    once from the training data and both ``fit_transform`` and
    ``transform`` apply it before returning -- so, unlike ``prune()``, this
    always happens, for every family, not only when a variant/config
    supports it.
    """

    name: str = "base"
    sparse_output: bool = False

    def fit(self, X, y=None):  # noqa: D102 - documented on subclasses
        raise NotImplementedError

    def transform(self, X):  # noqa: D102
        raise NotImplementedError

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.transform(X)

    def prune(self, keep_mask: np.ndarray) -> bool:
        """Reconfigure this (already-fitted) extractor to stop computing
        discarded features. See the class docstring. Default: unsupported.
        """
        return False

    @staticmethod
    def _apply_post_prune_mask(Xt, mask: Optional[np.ndarray]):
        """Shared helper for the block-granularity families: trim a
        just-computed, block-pruned matrix down to the exact originally
        requested columns."""
        if mask is None:
            return Xt
        return Xt[:, mask]

    @staticmethod
    def _nonconstant_mask(Xt) -> np.ndarray:
        """Boolean mask over ``Xt``'s columns with nonzero variance --
        shared by every family for constant-feature removal (see the
        class docstring). Works for both dense ``ndarray`` and
        ``scipy.sparse`` input. Falls back to keeping every column (with a
        warning) in the degenerate case where all of them are constant,
        rather than handing the rest of the pipeline a zero-width
        representation.

        Deliberately does *not* cast a dense ``Xt`` to ``float64`` before
        calling ``.var(axis=0)``: that reduction already promotes
        internally to a float64 accumulator on its own, so casting the
        *input* array first (an earlier version of this method did) just
        allocates a second full-width dense copy for nothing -- at
        WEASEL's up-to-30,000-raw-column scale (``DictionaryFamily``, one
        of the two widest of the four families' raw outputs, alongside
        convolution's ~50-60k) that copy alone doubled the memory this
        one variance check needed, on top of whatever ``Xt`` itself
        already cost.
        """
        if sp.issparse(Xt):
            Xt = Xt.toarray()
        mask = np.asarray(Xt).var(axis=0) > 0
        if not mask.any():
            warnings.warn(
                "All raw features are constant on the training data; "
                "keeping every column rather than returning an empty "
                "representation.",
                stacklevel=2,
            )
            mask = np.ones(mask.shape[0], dtype=bool)
        return mask

    @staticmethod
    def _expand_mask(mask: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Expand `mask` (defined over the ``True`` positions of
        `base_mask`) back to `base_mask`'s own length, with ``False``
        everywhere `base_mask` is ``False``.

        Used by a family whose ``prune()`` does its own block-level
        bookkeeping (currently only ``ShapeletFamily``) to translate a
        `keep_mask` expressed in this extractor's *public*
        (constant-feature-filtered) output width back into indices over
        its full, pre-filter raw output -- a column that constant-feature
        removal already dropped was never something the reducer could
        have selected, so it must map to ``False`` here.
        """
        expanded = np.zeros(base_mask.shape[0], dtype=bool)
        expanded[base_mask] = mask
        return expanded


def _as_float64_3d(X: np.ndarray) -> np.ndarray:
    """Some aeon transformers (notably the numba-jitted shapelet ones)
    require float64 input and will raise a fairly opaque numba TypingError
    otherwise. Cast defensively.
    """
    X = np.asarray(X)
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    return X


# --------------------------------------------------------------------------- #
# Convolution-based: KG-MTP (kernel-grouping features under multiple input
# transformations and pooling operators)
# --------------------------------------------------------------------------- #
class ConvolutionFamily(BaseFamily):
    """Random convolutional-kernel features, via our own ``KGMTP`` transform.

    Feature structure (see report Sec. 5.5): thousands of near-independent
    features from *random, unlearned* kernels, of which typically only a
    small dataset-specific subset is discriminative. This is the family
    RocketPFN (arXiv:2606.21786) builds on.

    ``KGMTP`` (see ``kg_mtp.py``) already handles multivariate input
    natively -- it treats X as 3D (n_cases, n_channels, n_timepoints) and
    concatenates each channel's own PPV/Hydra features internally -- so
    this wrapper passes X straight through without any per-channel
    looping of its own.

    No inference-time pruning is implemented for this family: unlike the
    ROCKET variant this replaces, ``KGMTP``'s fitted state (per-channel,
    per-branch kernel arrays plus a fitted Hydra sparse scaler) isn't a
    simple flat, independently-sliceable-by-output-column structure, so
    pruning would mean reimplementing meaningful parts of ``KGMTP`` itself
    rather than slicing existing fitted state. The pipeline falls back to
    computing the full raw feature set and masking afterward, same as any
    other family that doesn't support pruning (see ``BaseFamily.prune``).
    Constant-feature removal (``BaseFamily._nonconstant_mask``) still
    applies unconditionally, independent of pruning support.
    """

    name = "convolution"
    sparse_output = False

    def __init__(
        self,
        num_features: int = 50_000,
        max_dilations_per_kernel: int = 32,
        n_features_per_kernel: int = 5,
        random_state: Optional[int] = None,
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.random_state = random_state

    def _build(self):

        return KGMTP(
            num_features=self.num_features,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_features_per_kernel=self.n_features_per_kernel,
            random_state=self.random_state,
        )

    def fit(self, X, y=None):
        self._transformer_ = self._build()
        Xt = np.asarray(self._transformer_.fit_transform(X, y))
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        self._post_prune_column_mask_ = None
        return self

    def fit_transform(self, X, y=None, **kwargs):
        # ``KGMTP.fit_transform()`` is materially cheaper than the base
        # class's default ``fit(); transform()`` composition: fitting its
        # Hydra scaler needs the full-training-set convolution pass that
        # ``fit_transform`` already computed, so calling ``transform()``
        # again afterward would redundantly recompute it. Call it
        # directly instead of relying on the inherited default.
        self._transformer_ = self._build()
        Xt = np.asarray(self._transformer_.fit_transform(X, y))
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        self._post_prune_column_mask_ = None
        return Xt[:, self._nonconstant_mask_]

    def transform(self, X):
        Xt = np.asarray(self._transformer_.transform(X))
        Xt = Xt[:, self._nonconstant_mask_]
        return self._apply_post_prune_mask(Xt, self._post_prune_column_mask_)


# --------------------------------------------------------------------------- #
# Interval-based: QUANT (quantiles over dyadic/random intervals)
# --------------------------------------------------------------------------- #
class IntervalFamily(BaseFamily):
    """Interval summary-statistic features, via our own ``QuantTransformer``.

    We use QUANT (Dempster, Schmidt & Webb, arXiv:2308.00928) as the
    representative interval-based method: quantiles computed over many
    (dyadic) intervals of the series and several of its representations
    (first difference, etc). Feature structure (report Sec. 5.5):
    continuous, correlated-by-construction statistics. Its many
    correlated quantile features are matched to ``reduction.PLSDASelector``
    (PLS discriminant projection), rather than the tree ensembles
    (Extra Trees) the family's own designers pair it with -- see that
    reducer's docstring for why.

    ``QuantTransformer`` (see ``quant.py``) is our own torch-free
    reimplementation and already handles multivariate input natively (3D
    ``(n_cases, n_channels, n_timepoints)``, one shared quantile layout
    reused across channels), so this wrapper passes X straight through.
    Its own ``fit()`` only derives interval boundaries from the input
    *shape* (not the data itself), so it's cheap -- but computing the
    constant-feature mask (``BaseFamily._nonconstant_mask``) needs one
    real ``transform()`` pass regardless, which is why ``fit_transform``
    is overridden below (to do that pass once, not twice).

    No inference-time pruning is implemented for this family (the previous
    aeon-``QUANTTransformer``-based implementation reached into torch
    internals to skip whole unused representations/intervals; that logic
    was specific to aeon's torch implementation and isn't reproduced
    here). The pipeline falls back to computing the full raw feature set
    and masking afterward, same as any other family that doesn't support
    pruning (see ``BaseFamily.prune``). Constant-feature removal still
    applies unconditionally, independent of pruning support.
    """

    name = "interval"
    sparse_output = False

    def __init__(self, depth: int = 6, div: int = 4):
        self.depth = depth
        self.div = div

    def _build(self):
        return QuantTransformer(depth=self.depth, div=self.div)

    def fit(self, X, y=None):
        self._transformer_ = self._build()
        self._transformer_.fit(X, y)
        Xt = np.asarray(self._transformer_.transform(X))
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        self._post_prune_column_mask_ = None
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self._transformer_ = self._build()
        self._transformer_.fit(X, y)
        Xt = np.asarray(self._transformer_.transform(X))
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        self._post_prune_column_mask_ = None
        return Xt[:, self._nonconstant_mask_]

    def transform(self, X):
        Xt = np.asarray(self._transformer_.transform(X))
        Xt = Xt[:, self._nonconstant_mask_]
        return self._apply_post_prune_mask(Xt, self._post_prune_column_mask_)


def _rdst_transform_ignoring_known_warning(transformer, X) -> np.ndarray:
    """``RandomDilatedShapeletTransform.transform`` (aeon) emits a plain
    ``UserWarning`` ("Some invalid values (inf or nan) where converted
    from to 0 during the shapelet transformation.") whenever any of its
    computed distance/argmin/count features come out inf or nan -- e.g.
    a degenerate shapelet-to-series distance against a near-constant
    segment. aeon itself already corrects this before returning
    (``np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`` in its own
    ``_dilated_shapelet_transform.py``), so the warning is purely
    informational about a fix-up aeon has already applied, not a sign
    this family's own code did something wrong. It's frequent enough
    across a benchmark this wide (a routine occurrence on some
    dataset/shapelet combination, not a per-dataset anomaly) that
    surfacing it on every occurrence is just noise in real experiment
    runs -- suppressed here, at the one place every ``ShapeletFamily``
    call into RDST's own ``transform`` goes through, rather than left to
    accumulate in every calling script's own output.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Some invalid values \(inf or nan\).*",
            category=UserWarning,
        )
        return np.asarray(transformer.transform(X))


# --------------------------------------------------------------------------- #
# Shapelet-based: Random Dilated Shapelet Transform (RDST)
# --------------------------------------------------------------------------- #
class ShapeletFamily(BaseFamily):
    """Shapelet-distance features.

    We use RDST (Guillaume et al., arXiv:2109.13514) as the representative
    shapelet-based method: it samples many random, dilated shapelets
    cheaply and, for each series, returns (min distance, argmin position,
    count of matches) per shapelet. ``max_shapelets`` is deliberately left
    at a high default: the RDST paper itself reports better results from
    more shapelets, and an early, single-dataset investigation into
    lowering it was not strong enough evidence to override that (see
    ``reduction.py``'s module docstring).

    Feature structure (report Sec. 5.5): numerous, highly correlated
    derived features (each shapelet contributes 3: distance/argmin/count,
    and many sampled shapelets end up capturing overlapping patterns).
    Empirical investigation on real UCR data (ACSF1) found this defeats
    per-feature selection approaches -- FDR filtering barely compressed
    it, and even a "proper" top-k-by-score shapelet ranking needed nearly
    every shapelet to match full-feature accuracy -- the same signature as
    QUANT's raw features, which is why ``reduction.PLSDASelector`` (a
    fixed-budget PLS-DA projection, not a per-feature/per-shapelet
    selector) is the matched reducer here; see that class's docstring for
    the full comparison.

    Note: RDST's numba-jitted core requires float64 input; ``transform``
    casts defensively (a float32 input otherwise fails with an opaque
    ``numba.core.errors.TypingError``, confirmed while building this).

    Constant-feature removal (``BaseFamily._nonconstant_mask``) sits in
    front of this family's own ``prune()``: ``prune()`` receives a
    ``keep_mask`` sized to the already-constant-filtered *public* output
    width, and expands it back to one slot per raw (pre-filter) shapelet
    column (``BaseFamily._expand_mask``) before doing its own shapelet-
    block bookkeeping, since a column constant-feature removal already
    dropped was never something the reducer could have selected in the
    first place.
    """

    name = "shapelet"
    sparse_output = False

    def __init__(
        self,
        max_shapelets: int = 10_000,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ):
        self.max_shapelets = max_shapelets
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = _as_float64_3d(X)
        self._transformer_ = RandomDilatedShapeletTransform(
            max_shapelets=self.max_shapelets,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._transformer_.fit(X, y)
        # Remember how many derived features (3: dist/argmin/count) RDST
        # produces per shapelet -- used by this family's own prune() below
        # for its shapelet-level block bookkeeping (independent of whether
        # the currently-registered reducer ever calls prune(); see
        # PLSDASelector's docstring for why it currently doesn't).
        n_shapelets = len(self._transformer_.shapelets_[0])
        self.n_features_per_shapelet_ = 3
        self.n_shapelets_ = n_shapelets
        Xt = _rdst_transform_ignoring_known_warning(self._transformer_, X)
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        self._post_prune_column_mask_ = None
        return self

    def fit_transform(self, X, y=None, **kwargs):
        # As with ``IntervalFamily``: fitting RDST doesn't itself produce
        # the transformed features, so building the constant-feature mask
        # needs one real transform() pass regardless -- do it here and
        # reuse it, instead of the base class's default
        # ``fit(); transform()`` composition (which would call
        # transform() a second time).
        X = _as_float64_3d(X)
        self._transformer_ = RandomDilatedShapeletTransform(
            max_shapelets=self.max_shapelets,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._transformer_.fit(X, y)
        n_shapelets = len(self._transformer_.shapelets_[0])
        self.n_features_per_shapelet_ = 3
        self.n_shapelets_ = n_shapelets
        Xt = _rdst_transform_ignoring_known_warning(self._transformer_, X)
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        self._post_prune_column_mask_ = None
        return Xt[:, self._nonconstant_mask_]

    def transform(self, X):
        X = _as_float64_3d(X)
        Xt = _rdst_transform_ignoring_known_warning(self._transformer_, X)
        if not self._pruned_:
            # Not pruned yet: `self._transformer_` still computes the full
            # pre-filter raw width, so the constant-feature mask (sized to
            # that same width) still applies here. Once pruned, the
            # transformer itself only computes the surviving shapelets'
            # columns, and `_post_prune_column_mask_` below already maps
            # straight from that smaller width to the requested one.
            Xt = Xt[:, self._nonconstant_mask_]
        return self._apply_post_prune_mask(Xt, self._post_prune_column_mask_)

    def prune(self, keep_mask: np.ndarray) -> bool:
        """Drop unneeded shapelets before the next ``transform()`` call.

        RDST's ``self.shapelets_`` is a 9-tuple of parallel arrays, one
        entry per shapelet (values, startpoints, lengths, dilations,
        thresholds, normalises, means, stds, classes -- see aeon's
        ``dilated_shapelet_transform``), and each shapelet independently
        produces exactly its own 3 consecutive output columns
        (distance / argmin / count). That makes this the cleanest of the
        four families to prune: slice all 9 arrays down to the shapelets
        whose triple has at least one surviving column, and hand the
        result straight back to aeon's own (unmodified)
        ``dilated_shapelet_transform``.
        """
        keep_mask = np.asarray(keep_mask, dtype=bool)
        # `keep_mask` is over our *public* (constant-filtered) output --
        # expand it back to the full pre-filter width (see class
        # docstring) before the shapelet-block logic below, which assumes
        # one slot per raw shapelet-feature column.
        keep_mask = self._expand_mask(keep_mask, self._nonconstant_mask_)
        n_per_shapelet = self.n_features_per_shapelet_
        n_shapelets = self.n_shapelets_
        if keep_mask.sum() == 0 or keep_mask.shape[0] != n_per_shapelet * n_shapelets:
            return False

        mask2d = keep_mask.reshape(n_shapelets, n_per_shapelet)
        keep_shapelet = mask2d.any(axis=1)
        idx = np.flatnonzero(keep_shapelet)

        (
            values,
            startpoints,
            lengths,
            dilations,
            thresholds,
            normalises,
            means,
            stds,
            classes,
        ) = self._transformer_.shapelets_
        self._transformer_.shapelets_ = (
            values[idx],
            startpoints[idx],
            lengths[idx],
            dilations[idx],
            thresholds[idx],
            normalises[idx],
            means[idx],
            stds[idx],
            classes[idx],
        )
        self.n_shapelets_ = int(keep_shapelet.sum())
        self._post_prune_column_mask_ = mask2d[keep_shapelet].reshape(-1)
        self._pruned_ = True
        return True


# --------------------------------------------------------------------------- #
# Dictionary-based: WEASEL 2.0
# --------------------------------------------------------------------------- #
class DictionaryFamily(BaseFamily):
    """Bag-of-SFA-words count features, via aeon's ``WEASELTransformerV2``.

    ``WEASEL_V2`` bundles this same representation together with its own
    feature selection and a final ``RidgeClassifierCV`` into one
    classifier, with no public ``transform``. Rather than go through that
    classifier and reach into its internals, this wraps
    ``WEASELTransformerV2`` directly -- the plain transformer WEASEL 2.0
    is built on (see aeon's
    ``aeon/classification/dictionary_based/_weasel_v2.py``) -- so we get
    the bag-of-words matrix on its own, to feed this project's own
    per-family base learners (``tsfm_ensemble.base_learners``) directly,
    instead of WEASEL 2.0's own bundled classifier.

    ``WEASELTransformerV2`` has no separate ``fit()``: its
    ``fit_transform()`` already does the (expensive) per-configuration SFA
    fitting *and* transform together in one parallel pass, so this
    family's own ``fit_transform`` is overridden to call it directly
    rather than the base class's default ``fit(); transform()``
    composition, which would otherwise redundantly recompute it (the same
    concern as ``ConvolutionFamily``/``KGMTP``).

    Feature selection: ``feature_selection="chi2_top_k"`` with
    ``max_feature_count`` left at ``WEASELTransformerV2``'s own default
    (30,000) -- matching aeon's own reference configuration exactly (see
    ``_weasel_v2.py``), the same "match the original literature/
    implementation's own hyperparameters" convention already followed for
    every other family (see ``tsfm_ensemble.base_learners``'s module
    docstring). An earlier version of this class instead used
    ``feature_selection="none"``, deferring word selection to a
    downstream ``reduction.ChiSquareThenSVD`` step -- but that reduction
    stage was removed from this branch entirely (this design fits base
    learners directly on each family's raw extractor output, no
    dimensionality-reduction step at all; see
    ``tsfm_ensemble.base_learners``'s "No dimensionality reduction"
    section), which left ``feature_selection="none"`` stale: with nothing
    downstream to prune the vocabulary, the raw per-configuration SFA
    word count grew unbounded (measured on the actual 20-dataset dev
    subset: 10,302-72,285 raw columns, some datasets over 2x aeon's own
    30,000 cap and, for wide/long-series datasets, dense enough
    (50-90% nonzero, not sparse in any performance sense) that
    ``RidgeClassifierCV``'s default generalized-CV path -- which builds
    an n_samples x n_samples Gram matrix from this "sparse" input --
    scaled roughly quadratically in sample count and took several minutes
    on a single (dataset, resample) job for the dev subset's largest
    dataset). Restoring aeon's own chi2-top-k selection fixes both
    problems at once (bounds the vocabulary to WEASEL 2.0's own tested
    scale, and lets a discriminative pre-filter -- not an unfiltered
    random word set -- reach every downstream learner) without
    reintroducing a separate reduction stage.

    Two things this wrapper does on top of the (already chi2-filtered)
    transformer output, both fit once on the training data and then
    reapplied identically at transform time (no peeking at test data, no
    refitting):

    1. Constant-feature removal (``BaseFamily._nonconstant_mask``, shared
       by every family). Even after chi2 selection, a handful of selected
       words can still be constant across the training set actually seen
       here (chi2_top_k's own internal p-value ranking doesn't guarantee
       nonzero variance) -- purely dead columns that carry no signal,
       cost memory, and can upset univariate feature-selection tests
       downstream (e.g. a zero-variance column always has an undefined/
       degenerate chi-squared statistic).
    2. Sparse output. ``WEASELTransformerV2`` itself always materializes a
       dense array internally, but the result is still a mostly-zero
       bag-of-words count matrix -- so we convert it to a
       ``scipy.sparse.csr_matrix`` right away, rather than carrying the
       dense array any further. This is a real memory saving, not just a
       cosmetic one, at WEASEL's now-bounded (<=30,000) raw-column scale.
       Deliberately *sparsify first, mask constant columns second*
       (``_to_sparse``'s own docstring has the full reasoning): applying
       the constant-feature mask to the still-dense array first, the way
       an earlier version of this class did, meant a second full-width
       dense copy on top of the first, plus a third from an unnecessary
       ``float64`` upcast in the same expression -- roughly 3-4x one raw
       dense array's worth of memory transiently alive at once, for what
       should be a single sparsify-and-shrink step.

    Every other hyperparameter is left at ``WEASELTransformerV2``'s own
    default. Note this transformer is univariate-only (it squeezes out
    the channel axis internally), matching this project's current scope
    (the 142 UCR univariate datasets).
    """

    name = "dictionary"
    sparse_output = True

    def __init__(self, random_state: Optional[int] = None, n_jobs: int = 1):
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _build(self):
        return WEASELTransformerV2(
            feature_selection="chi2_top_k",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y=None):
        self._transformer_ = self._build()
        Xt = self._transformer_.fit_transform(X, y)
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self._transformer_ = self._build()
        Xt = self._transformer_.fit_transform(X, y)
        self._nonconstant_mask_ = self._nonconstant_mask(Xt)
        self._pruned_ = False
        return self._to_sparse(Xt)[:, self._nonconstant_mask_]

    def transform(self, X):
        Xt = self._transformer_.transform(X)
        return self._to_sparse(Xt)[:, self._nonconstant_mask_]

    @staticmethod
    def _to_sparse(Xt):
        """Convert to (or recompress as) a CSR matrix (see the class
        docstring's point 2) -- and, deliberately, *before* the caller's
        column mask is applied, not after.

        An earlier version masked the still-dense ``Xt`` first
        (``Xt[:, mask]``, a second full-width dense copy) and only then
        sparsified, upcasting to ``float64`` in the same expression (a
        third full-width copy, at 2x the size of the other two since
        WEASEL's own raw output is ``uint32``) -- at WEASEL's ~20,000-raw-
        column scale that meant roughly 3-4x one raw dense array's worth
        of memory transiently alive at once for what should be a single
        sparsify-and-shrink step. Sparsifying first means the mask is
        applied to an already-sparse (nnz-proportional, not full-width)
        matrix instead, and preserving ``Xt``'s own dtype (rather than
        upcasting to ``float64`` here) avoids doubling it again for no
        reason -- the downstream base learners (``tsfm_ensemble.
        base_learners``) already accept sparse input directly and promote
        internally, only where they actually need to, on the much
        smaller (chi2-selected, <=30,000-column) data.
        """
        if sp.issparse(Xt):
            return Xt.tocsr()
        return sp.csr_matrix(np.asarray(Xt))


FAMILY_REGISTRY = {
    "convolution": ConvolutionFamily,
    "interval": IntervalFamily,
    "shapelet": ShapeletFamily,
    "dictionary": DictionaryFamily,
}


def make_family(name: str, **kwargs) -> BaseFamily:
    """Convenience constructor: ``make_family("convolution", num_features=20_000)``."""
    if name not in FAMILY_REGISTRY:
        raise ValueError(
            f"Unknown family {name!r}. Available: {sorted(FAMILY_REGISTRY)}"
        )
    return FAMILY_REGISTRY[name](**kwargs)
