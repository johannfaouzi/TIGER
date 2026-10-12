"""Per-family dimension reduction / feature selection.

Each selector below is matched to one family's feature structure, following
the report's Sec. 5.5 ("per-family dimension-reduction playbook") and
Sec. 5.6 ("how many features to keep"). ``DetachRocketSelector`` and
``ChiSquareThenSVD`` use an adaptive, data-driven stopping rule (a
validation-curve optimum, or a multiple-testing-corrected p-value threshold
feeding a variance-target SVD) that lets the *number* of retained features
fall out of how much signal that particular dataset/branch actually has.

``PLSDASelector`` -- used for *both* the interval and shapelet families --
combines correlated raw features together via a supervised PLS-DA
projection, rather than asking each one to individually prove itself the
way a per-feature test does. Both families' raw features are numerous and
highly correlated by construction (QUANT's overlapping dyadic-interval
quantiles; RDST's near-duplicate dilated shapelets), and empirical
investigation on real UCR data (ACSF1) found that per-feature selection
methods -- Boruta-style shadow importance for QUANT, FDR filtering and
top-k-by-score ranking for RDST -- all struggle at this scale: signal is
spread thin across many individually-weak, mutually-correlated features
rather than concentrated in a subset that a per-feature test can cleanly
separate from noise.

Earlier versions of this module *fixed* the number of PLS components at a
single generous budget (200, capped by the usual ``n_samples - 1`` safety
net -- see the class docstring) and let TabICLv2's own in-context feature
weighting decide which of those components mattered downstream. Follow-up
investigation (again on ACSF1, plus a spread of other real UCR datasets --
ItalyPowerDemand, GunPoint, ArrowHead, OSULeaf, the only other ones bundled
with the ``aeon`` install in the sandbox this was built in, which has no
network access to timeseriesclassification.com) found that assumption only
partly holds: pushing PLS-DA all the way to its capped budget clearly hurt
QUANT/interval on 2 of the 5 datasets (ACSF1: -16pt test accuracy, OSULeaf:
-12pt, vs. RDST/shapelet's -1 to -3pt on the same two), was roughly a wash
on the other 3 (GunPoint, ItalyPowerDemand, ArrowHead -- where the two
families degraded similarly little, or GunPoint's near-ceiling accuracy
just had no room left to show it), but *always* made held-out log-loss
worse, monotonically, for both families, on every single dataset tested --
even the ones where hard accuracy didn't budge. In other words: "let
TabICLv2 sort it out downstream" wasn't a safe assumption at the fixed
generous budget, but neither is any single fixed smaller number picked
from one or two datasets' curves -- the right cutoff clearly depends on
the dataset (and possibly the family), not just the family.

``PLSDASelector`` therefore now picks its own component count per
branch/dataset by cross-validating mean held-out log-loss of a fast proxy
classifier (a small ``ExtraTreesClassifier``) over a grid of candidate
counts, all the way up to the same capped budget as before -- see the
class docstring's ``select_n_components`` for the mechanics, including how
this is done for roughly the cost of a single extra PLS fit per CV fold
(not one per candidate) by exploiting NIPALS's own sequentially-deflated,
nested component structure. This is deliberately the same style already
used for ``BranchFusion``'s CAWPE alpha search (cheap CV over an
already-fit model's outputs, minimizing log-loss) rather than yet another
bespoke mechanism.

Note this is a reduction-stage decision only: ``max_shapelets`` at
*extraction* time is deliberately left high (see ``families.ShapeletFamily``)
-- the RDST paper (Guillaume et al., arXiv:2109.13514) itself reports
better results from more shapelets, and one dataset's noisy small-sample
CV curve is not enough evidence to override that in the other direction.

    convolution -> DetachRocketSelector   (ridge-coefficient pruning)
    interval    -> PLSDASelector          (PLS discriminant projection)
    shapelet    -> PLSDASelector          (PLS discriminant projection)
    dictionary  -> ChiSquareThenSVD       (chi2 filter + truncated SVD)
"""

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFdr, chi2, f_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder


class PrunableReducerMixin:
    """Shared contract that lets ``pipeline.PrunedFamilyBranch`` hand this
    reducer's selection decision back to the family extractor, so the
    extractor can stop computing discarded raw features at inference time
    (the family's own ``prune()`` -- see ``families.py``), instead of this
    reducer masking them out of an already-fully-computed matrix on every
    call.

    ``raw_support_`` : boolean array over the *raw extractor output*
        columns this reducer was fit on, set by ``fit()``. For the three
        plain selectors below it's just ``support_``; for
        ``ChiSquareThenSVD`` it's the chi2-filter stage's mask (the stage
        before its SVD, which mixes surviving raw columns together and so
        can't be "pruned away" at the extractor level the same way).
    ``mark_input_preselected()`` : called by the pipeline once the
        extractor confirms it *did* prune down to exactly
        ``raw_support_``'s columns. After this, ``transform()`` must
        assume its input already *is* that reduced set (in the same
        column order) and skip re-selecting -- doing the selection again
        would silently select the wrong columns a second time.

    A separate, independent convention (not part of this mixin's own
    interface, since it's optional and every reducer below happens to set
    it, rather than mixin-enforced): ``feature_importance_``, one score
    per *output* column, aligned column-for-column with what
    ``transform()`` returns (higher = more important; the columns
    themselves are never reordered by it -- only the values carry the
    ranking). ``pipeline.PrunedFamilyBranch`` passes this through to
    ``backbone.GroupedICLClassifier``, which uses it (via
    ``utils.split_into_groups``) to assign TabICLv2 groups by stratified
    round-robin instead of pure random shuffling, whenever a family ends
    up with more surviving features than ``group_size``. See each
    reducer's own docstring for where its particular ``feature_importance_``
    comes from, and ``utils.split_into_groups``'s docstring for the
    empirical case for doing this at all.
    """

    def mark_input_preselected(self) -> None:
        self._input_preselected_ = True

    def _apply_max_features_cap(self) -> None:
        """Final ceiling on top of whatever this reducer's own selection
        logic already chose -- shared by every selection-based reducer
        that exposes ``max_features``, ``support_``, ``feature_importance_``,
        and ``n_features_kept_`` in the conventions documented above
        (originally ``DetachRocketSelector``'s own method; moved here once
        ``FDRUnivariateSelector`` and ``ChiSquareSelector`` needed the exact
        same behaviour -- see ``DetachRocketSelector.max_features``'s
        docstring for the full rationale: without a ceiling, a weak or
        adaptive selection criterion can leave far more columns than
        TabICLv2's grouped backbone can afford to fit in reasonable time).
        A no-op unless the current support exceeds the cap. Keeps only the
        ``max_features`` highest-importance kept columns, re-deriving
        ``support_``/``feature_importance_`` in the same ascending-
        original-column-index order ``transform()`` and every other
        reducer in this module already use, rather than the importance-
        descending order ``argsort`` itself returns.
        """
        if self.max_features is None or self.n_features_kept_ <= self.max_features:
            return
        kept_positions = np.flatnonzero(self.support_)
        top = kept_positions[np.argsort(self.feature_importance_)[::-1][: self.max_features]]
        new_support = np.zeros_like(self.support_)
        new_support[top] = True
        self.feature_importance_ = self.feature_importance_[np.isin(kept_positions, top)]
        self.support_ = new_support
        self.n_features_kept_ = int(self.support_.sum())

    @staticmethod
    def _nonconstant_mask(X) -> np.ndarray:
        """Boolean mask over ``X``'s columns that are not literally
        constant (peak-to-peak range > 0 -- for sparse input, max minus
        min per column, which correctly counts an all-zero/never-occurring
        word column as constant too). Sparse-safe without densifying the
        whole matrix: a sparse per-column min/max reduction is itself
        cheap, only the small (1, n_features) result gets densified.

        A fit-time safety net specific to this mixin's own selectors, not
        a replacement for ``families.BaseFamily._nonconstant_mask`` (which
        already drops columns with zero *variance* across an entire
        family's raw output, once, before any reducer ever sees them --
        see that method's docstring). That upstream check can still leave
        a column here that trips a statistical test's own internal
        zero-division guard: scikit-learn's ``f_classif``/``f_oneway``
        warns "Features [...] are constant" and divides by an exactly-zero
        within-group mean square for such a column -- a warning about a
        column whose F-statistic is undefined (no information), not a
        fluke worth silencing after the fact by filtering warnings.
        Peak-to-peak range (vs. the upstream check's variance) also
        catches a column whose values are identical enough to have zero
        true range, but which floating-point summation could otherwise
        report as a tiny nonzero variance -- so this is deliberately not
        skipped just because the upstream check already ran.
        """
        if sp.issparse(X):
            X = X.tocsc()
            col_min = np.asarray(X.min(axis=0).toarray()).ravel()
            col_max = np.asarray(X.max(axis=0).toarray()).ravel()
            return (col_max - col_min) > 0
        X = np.asarray(X)
        return np.ptp(X, axis=0) > 0


# --------------------------------------------------------------------------- #
# Convolution family: Detach-ROCKET-style sequential coefficient pruning
# --------------------------------------------------------------------------- #
class DetachRocketSelector(PrunableReducerMixin, BaseEstimator, TransformerMixin):
    """Ridge-coefficient-magnitude feature pruning for convolution features.

    Mirrors Detach-ROCKET (Uribarri et al., arXiv:2309.14518): fit a ridge
    classifier, repeatedly drop the lowest-|coefficient| features, and
    (in "auto" mode) stop at the point that maximizes a validation
    objective ``accuracy(q) + c * q`` where ``q`` is the fraction of
    features kept -- i.e. the model picks its own stopping point per
    dataset rather than using a target count. "fixed" mode (drop straight
    to ``fixed_fraction``) is offered only as the fallback the original
    paper itself recommends "when a validation split is not feasible"
    (report Sec. 5.6) -- prefer "auto" whenever you can spare the split.

    Parameters
    ----------
    mode : {"auto", "fixed"}
    fixed_fraction : float
        Used only in "fixed" mode.
    val_fraction : float
        Fraction of the training fold held out to evaluate the stopping
        objective, in "auto" mode.
    step_fraction : float
        Fraction of the *currently active* features dropped at each
        iteration (5% in the original paper).
    c : float
        Compression-preference weight in ``accuracy(q) + c * q``. Larger
        values favour smaller feature sets; the paper's own experiments use
        small values (we default to 0.01) so accuracy dominates unless two
        candidate sizes are close.
    min_features : int
        Never prune below this many features, regardless of the objective.
    max_features : int, optional
        Final ceiling applied *after* whichever selection ``mode`` chose
        (``None``, the default, applies no ceiling at all). Neither
        ``mode="auto"``'s own stopping objective nor ``fixed_fraction``
        is changed by this -- it's a safety valve on top, not a different
        selection strategy: if the chosen support already has more than
        ``max_features`` columns, only the ``max_features`` highest-
        |ridge-coefficient| of them survive (the same ``feature_importance_``
        ranking ``mode="fixed"`` itself uses), otherwise nothing changes.
        Exists because ``mode="auto"``'s stopping objective
        (``accuracy(q) + c * q``, see above) rewards compression only
        weakly at the default ``c=0.01``, so on a dataset where pruning
        doesn't clearly hurt validation accuracy it can stop having barely
        pruned at all -- observed empirically up to ~18,600 of a raw
        ~50-60k-wide feature matrix on some real UCR datasets (see the dev-
        subset feature-count study), which at TabICLv2's roughly-linear-
        in-feature-count cost (~1.3s/feature observed there) can turn one
        (dataset, resample, family) branch fit into hours. A generous cap
        (e.g. 2000, which bound roughly a quarter of dev-subset jobs to
        well under an hour each while leaving the other three-quarters'
        own auto-selected count untouched) trades away only the tail of
        that distribution, not the family's genuine benefit from using
        more than the coarse feature-count grid's own maximum (150) --
        see that study's report for why convolution specifically needed
        this and the other three families didn't.
    alpha : float or "auto"
        Ridge regularization strength; "auto" tries a small grid.

    Attributes
    ----------
    feature_importance_ : ndarray
        Ridge-|coefficient| importance for each *kept* feature, aligned
        column-for-column with ``transform()``'s own output order (not
        sorted by importance itself -- the values carry the ranking, the
        columns stay in their original relative order). Consumed by
        ``backbone.GroupedICLClassifier`` (via ``pipeline.PrunedFamilyBranch``)
        to assign TabICLv2 groups by stratified round-robin instead of
        pure random shuffling, when this family ends up with more surviving
        features than ``group_size`` -- see ``utils.split_into_groups``'s
        docstring for why that's an improvement over random assignment,
        and the empirical numbers behind it.
    used_fixed_fallback_ : bool
        True if ``mode="auto"`` couldn't stratify a validation split (some
        class too small) and silently fell back to the fixed-fraction
        heuristic for this fit -- see ``_fit_auto``. Always False when
        ``mode="fixed"`` is requested directly. Doesn't affect behaviour;
        it's here so this (expected, by-design, but accuracy-relevant)
        fallback stays inspectable after the fact without needing a
        warning on every occurrence -- small-class datasets are routine
        enough across the UCR benchmark that one would otherwise fire on
        a sizeable share of real experiment runs.
    """

    name = "convolution"

    def __init__(
        self,
        mode: str = "auto",
        fixed_fraction: float = 0.10,
        val_fraction: float = 0.2,
        step_fraction: float = 0.05,
        c: float = 0.01,
        min_features: int = 32,
        max_features: Optional[int] = None,
        alpha: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.mode = mode
        self.fixed_fraction = fixed_fraction
        self.val_fraction = val_fraction
        self.step_fraction = step_fraction
        self.c = c
        self.min_features = min_features
        self.max_features = max_features
        self.alpha = alpha
        self.random_state = random_state

    @staticmethod
    def _coef_importance(clf: RidgeClassifier) -> np.ndarray:
        coef = clf.coef_
        # Binary: coef_ has shape (n_features,); multiclass: (n_classes, n_features).
        # Detach-ROCKET's own approach: max-pool across the class dimension.
        return np.abs(coef) if coef.ndim == 1 else np.abs(coef).max(axis=0)

    def _fit_fixed(self, X, y):
        n_features = X.shape[1]
        clf = RidgeClassifier(alpha=self.alpha).fit(X, y)
        importance = self._coef_importance(clf)
        n_keep = max(self.min_features, int(round(self.fixed_fraction * n_features)))
        keep_idx = np.argsort(importance)[::-1][:n_keep]
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[keep_idx] = True
        self.n_features_kept_ = int(self.support_.sum())
        # `importance` is indexed by original (full-``X``) feature
        # position, same as `self.support_` -- masking with `support_`
        # itself (rather than `keep_idx`, which is sorted by descending
        # importance, not original position) keeps this aligned with
        # `transform()`'s own output column order (`X[:, support_]`,
        # always ascending original index). See `feature_importance_`'s
        # own note on the class docstring for what this is used for.
        self.feature_importance_ = importance[self.support_]

    def _fit_auto(self, X, y):
        n_features = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        try:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X,
                y,
                test_size=self.val_fraction,
                stratify=y,
                random_state=self.random_state,
            )
        except ValueError:
            # Some class has too few members to stratify a held-out split;
            # fall back to the fixed-fraction heuristic, exactly as the
            # original paper recommends when a validation split isn't
            # feasible. This is expected, by-design behaviour (not a sign
            # of a problem with the dataset or the fit) -- small-class
            # datasets are routine across the 142-dataset UCR benchmark,
            # so this fallback fires often enough in normal experiment
            # runs that warning on every occurrence is just noise. Record
            # it on the fitted estimator instead of warning, so it's still
            # inspectable after the fact (e.g. when auditing why a branch
            # ended up using the fixed-fraction heuristic on some dataset)
            # without cluttering experiment logs.
            self.used_fixed_fallback_ = True
            self._fit_fixed(X, y)
            return

        active = np.ones(n_features, dtype=bool)
        best_score = -np.inf
        best_support = active.copy()
        best_importance = None

        while active.sum() > self.min_features:
            idx_active = np.flatnonzero(active)
            clf = RidgeClassifier(alpha=self.alpha).fit(X_fit[:, idx_active], y_fit)
            val_acc = clf.score(X_val[:, idx_active], y_val)
            importance = self._coef_importance(clf)
            # q = proportion of features *pruned so far* (not kept): the
            # objective rewards compression, so between two candidates with
            # similar accuracy the more heavily pruned one wins. Getting
            # this the other way around (rewarding a larger kept fraction)
            # would make "prune nothing" trivially optimal whenever
            # accuracy is roughly flat across pruning levels -- exactly the
            # bug this comment is here to prevent regressing back into.
            q_pruned = 1.0 - (active.sum() / n_features)
            objective = val_acc + self.c * q_pruned
            if objective > best_score:
                best_score = objective
                best_support = active.copy()
                # `importance` here is `clf.coef_`-derived, indexed by
                # `idx_active` (ascending, since `np.flatnonzero` always
                # returns ascending order) -- exactly the order
                # `transform()` will later return these same columns in
                # (`X[:, support_]`), since `idx_active == np.flatnonzero
                # (best_support)` at this exact point. No reordering
                # needed, unlike `_fit_fixed` above.
                best_importance = importance.copy()

            n_drop = max(1, int(round(self.step_fraction * idx_active.size)))
            if idx_active.size - n_drop < self.min_features:
                break
            drop_local = np.argsort(importance)[:n_drop]
            active[idx_active[drop_local]] = False

        if best_importance is None:
            # The loop body never ran (n_features <= min_features from the
            # very start, so nothing was ever pruned) -- feature_importance_
            # should still reflect *some* ranking for downstream grouping
            # to use, so fit one ridge model on the (unpruned) active set
            # rather than leaving it unset.
            idx_active = np.flatnonzero(best_support)
            clf = RidgeClassifier(alpha=self.alpha).fit(X_fit[:, idx_active], y_fit)
            best_importance = self._coef_importance(clf)

        self.support_ = best_support
        self.n_features_kept_ = int(self.support_.sum())
        self.feature_importance_ = best_importance

    def fit(self, X, y):
        X = np.asarray(X)
        # Always defined after fit(), regardless of mode: True only when
        # mode="auto" itself fell back to the fixed-fraction heuristic
        # (see _fit_auto) -- an explicit mode="fixed" request is not a
        # "fallback" and leaves this False.
        self.used_fixed_fallback_ = False
        if self.mode == "fixed":
            self._fit_fixed(X, y)
        elif self.mode == "auto":
            self._fit_auto(X, y)
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")
        self._apply_max_features_cap()
        self.raw_support_ = self.support_
        self._input_preselected_ = False
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self._input_preselected_:
            # The extractor already pruned itself down to exactly
            # self.support_'s columns (families.ConvolutionFamily.prune,
            # "rocket" variant only) -- selecting again would pick the
            # wrong columns out of an already-reduced matrix.
            return X
        return X[:, self.support_]


# --------------------------------------------------------------------------- #
# Interval AND shapelet families: PLS-DA projection
# --------------------------------------------------------------------------- #
class PLSDASelector(PrunableReducerMixin, BaseEstimator, TransformerMixin):
    """Partial Least Squares Discriminant Analysis projection.

    Used for *both* the interval family (QUANT) and the shapelet family
    (RDST) -- see the module docstring for why the same reducer matches
    both: their raw features are numerous, continuous, and highly
    correlated by construction (QUANT's overlapping dyadic-interval
    quantiles; RDST's near-duplicate dilated shapelets), which is exactly
    the "many correlated features, p >> n" regime PLS-DA (a chemometrics
    standard) is built for -- fit a PLS regression of the features against
    one-hot-encoded class labels and project onto the resulting latent
    components, each a supervised linear combination of the original,
    correlated features.

    For the interval family, this replaced a Boruta-style shadow-feature
    selector (``BorutaSelector`` -- see git history), matched to QUANT's
    designers' own pairing with tree ensembles, which on real UCR data
    (ACSF1) was found to systematically under-confirm real signal at
    QUANT's feature counts: as the shadow pool grows into the thousands,
    the max of many i.i.d. shadow importances grows with it (extreme-value
    statistics), raising the bar any single real feature must individually
    clear.

    For the shapelet family, this replaced ``FDRUnivariateSelector`` (see
    below -- kept in this module, just no longer any family's default):
    on the same ACSF1 data, FDR selection barely compressed RDST's raw
    features (retaining 40-60% of them regardless of settings), a "proper"
    top-k-by-score shapelet ranking needed to keep nearly *all* shapelets
    before matching full-feature accuracy, and ridge-coefficient pruning
    (``DetachRocketSelector``, convolution's matched reducer) badly
    underperformed even after correcting for a genuine scale mismatch
    across RDST's three derived feature types (distance/argmin/count).
    All three symptoms point the same way as QUANT's Boruta failure:
    signal spread thin across many individually-weak, mutually-correlated
    features rather than concentrated in a subset a per-feature test can
    cleanly separate from noise. PLS-DA sidesteps this by combining
    correlated features together instead of asking each one to
    individually prove itself.

    Because PLS-DA's output columns are dense linear mixtures of every raw
    input feature, ``raw_support_`` is always ``None`` here (see
    ``PrunableReducerMixin``) -- there is no discardable raw-column subset
    to hand back to either family's extractor. For the shapelet family in
    particular this gives up ``ShapeletFamily.prune()``'s inference-time
    speedup (previously the only family/reducer pairing in this codebase
    where that actually engaged) -- every raw shapelet still needs to be
    computed at predict time, same as the other three families that never
    supported pruning in the first place.

    Unlike the other reducers in this module, ``n_components`` here is
    still a fixed, generous *budget* (see the module docstring for why the
    number itself isn't dataset-adaptive by hand) -- but it's now a ceiling
    on the search, not the count actually used. By default
    (``select_n_components=True``) the number of components actually
    handed downstream is chosen per branch/dataset by cross-validating
    mean held-out log-loss of a small proxy classifier over a grid of
    candidate counts up to that ceiling, and picking the minimizer -- see
    the module docstring for why this replaced always using the full
    capped budget. Note this whole reduction-stage decision is independent
    of each family's own *extraction*-time feature budget (e.g.
    ``max_shapelets``) -- see the module docstring for why that stays high
    for the shapelet family specifically.

    Parameters
    ----------
    n_components : int, optional
        Ceiling on PLS components -- both the search grid's upper end (see
        ``select_n_components``) and, when that search is disabled or
        infeasible, the count used directly. By default (``None``) this
        ceiling is simply ``min(n_features, n_samples - 1)`` -- i.e. no
        fixed budget at all, just whatever the data itself allows -- which
        is one short of the hard limit scikit-learn's ``PLSRegression``
        actually enforces (``min(n_samples, n_features)``): the component
        at that exact boundary is a numerically-degenerate artifact of
        exhausting every available degree of freedom (mean-centering plus
        the label-residual construction each use one up) -- confirmed
        empirically on real data (ACSF1): its variance was ~1e-27, i.e.
        constant. Handing a downstream consumer a supposedly-informative
        feature that's actually all-but-constant is a known trigger for
        numerical instability elsewhere in the pipeline -- concretely,
        this is what led to a ``RuntimeWarning: overflow encountered in
        power`` inside TabICLv2's own internal Yeo-Johnson preprocessing
        (its per-feature ``PowerTransformer``'s maximum-likelihood lambda
        search is prone to exactly this failure mode on near-constant
        columns; the same class of issue is independently documented in
        the TabPFN ecosystem this backbone is architecturally related to,
        e.g. PriorLabs/TabPFN#183). Trimming the boundary by one component
        removes the degenerate column at the source, mirroring
        ``ChiSquareThenSVD``'s own ``n_avail - 1`` convention below for
        the same underlying reason. If explicitly set to an int, that
        value is instead treated as a hard budget and capped down (with a
        warning) whenever ``min(n_features, n_samples - 1)`` is smaller --
        useful for tests/experiments that want a fixed, small ceiling
        regardless of dataset size; the ``None`` default never warns,
        since there's no fixed request to fall short of.
    select_n_components : bool, default True
        If True, pick the actual component count by ``cv``-fold
        cross-validated log-loss over a grid of candidates up to the
        capped ``n_components`` budget (see ``n_components_grid``), the
        same style as ``BranchFusion``'s CAWPE alpha search. This costs
        roughly one extra ``PLSRegression`` fit per fold, not one per
        (fold, candidate) pair: NIPALS extracts components sequentially
        via deflation, so a model fit with ``K`` components' first ``k <
        K`` columns are numerically identical (~1e-14 in a direct check)
        to a model fit fresh with ``n_components=k`` -- fitting once per
        fold at the full candidate ceiling and slicing columns reproduces
        scoring every candidate directly, for the price of one fit. If
        False, or if cross-validation isn't feasible (see
        ``min_class_count_for_cv``), the full capped ``n_components``
        budget is used directly -- the old, unconditional behaviour.
    n_components_grid : sequence of float, optional
        Candidate component counts to search, as fractions of the capped
        ``n_components`` budget (rounded, deduplicated, and always
        including the budget itself). Defaults to
        ``(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)`` -- dense at the low end,
        since the datasets where this search matters most (see the module
        docstring) had their best log-loss well under a third of the
        capped budget.
    cv : int, default 5
        Number of stratified folds for the component-count search.
    min_class_count_for_cv : int, default 5
        If the rarest class has fewer members than this (and fewer than
        ``cv``), the search isn't feasible; falls back to the full capped
        budget directly, same as ``select_n_components=False``.
    proxy_classifier_n_estimators : int, default 50
        Tree count for the small ``ExtraTreesClassifier`` used only to
        *score* candidate component counts during the search -- never the
        classifier actually used downstream (that's each branch's own
        backbone, typically ``GroupedICLClassifier``/TabICLv2). Kept small
        because this fits ``cv * len(n_components_grid)`` times per branch
        fit, on data that's already been projected down to at most
        ``n_components`` columns.
    max_iter : int, default 1000
        Maximum NIPALS iterations per ``PLSRegression`` fit (both the
        final fit and every per-fold fit during the component-count
        search). Raised from scikit-learn's own default of 500, since
        pushing ``n_components`` up toward its full ``min(n_features,
        n_samples - 1)`` ceiling (see ``n_components`` above) means some
        component, somewhere in the sequence, can need many iterations to
        converge -- empirically this is *not* reliably the last few
        (lowest-signal) components specifically, just wherever two
        singular values of the progressively-deflated residual happen to
        land close together, which is data-dependent. Even at this raised
        budget a component can still fail to converge within ``max_iter``
        on some real datasets; that specific warning (``Convergence
        Warning: Maximum number of iterations reached``) is deliberately
        suppressed during both fits (see the inline comment at the final
        fit) rather than chased with an ever-larger budget -- see that
        comment for why.
    random_state : int, optional
        Controls the CV fold split (and the proxy classifier's own
        randomness) during the component-count search.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        select_n_components: bool = True,
        n_components_grid: Optional[Sequence[float]] = None,
        cv: int = 5,
        min_class_count_for_cv: int = 5,
        proxy_classifier_n_estimators: int = 50,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.select_n_components = select_n_components
        self.n_components_grid = n_components_grid
        self.cv = cv
        self.min_class_count_for_cv = min_class_count_for_cv
        self.proxy_classifier_n_estimators = proxy_classifier_n_estimators
        self.max_iter = max_iter
        self.random_state = random_state

    _DEFAULT_N_COMPONENTS_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

    def _can_cross_validate(self, y: np.ndarray) -> bool:
        _, counts = np.unique(y, return_counts=True)
        return counts.min() >= max(2, min(self.cv, self.min_class_count_for_cv))

    def _candidate_grid(self, upper_bound: int) -> list:
        fractions = self.n_components_grid or self._DEFAULT_N_COMPONENTS_GRID
        candidates = {max(1, int(round(f * upper_bound))) for f in fractions}
        candidates = {c for c in candidates if 1 <= c <= upper_bound}
        candidates.add(upper_bound)
        return sorted(candidates)

    def _select_n_components(
        self, X: np.ndarray, y: np.ndarray, upper_bound: int
    ) -> Tuple[int, Dict[int, float]]:
        """Pick the candidate component count (<= ``upper_bound``) with the
        best mean cross-validated log-loss, fitting only one PLS model per
        fold (at ``upper_bound`` components) and slicing its already-fit
        scores for every candidate -- see ``select_n_components``'s
        docstring for why that slicing is equivalent to a from-scratch fit
        at each candidate count.
        """
        candidates = self._candidate_grid(upper_bound)
        if len(candidates) <= 1:
            return candidates[0], {}

        classes = np.unique(y)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        fold_scores: Dict[int, list] = {c: [] for c in candidates}
        for tr_idx, val_idx in skf.split(X, y):
            # A CV fold has fewer rows than the full training set this
            # selector was capped against, so `upper_bound` itself can
            # exceed *this fold's* actual PLSRegression rank limit
            # (min(n_fold_samples, n_features)) -- cap per fold, with the
            # same -1 margin as the outer safety net (see the class
            # docstring) to avoid a degenerate boundary component here too.
            fold_cap = max(1, min(upper_bound, len(tr_idx) - 1))
            enc = OneHotEncoder(sparse_output=False)
            y_oh_tr = enc.fit_transform(y[tr_idx].reshape(-1, 1))
            pls = PLSRegression(n_components=fold_cap, max_iter=self.max_iter)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="y residual is constant.*", category=UserWarning
                )
                # See the final fit's own comment below for why a NIPALS
                # ConvergenceWarning here is expected and benign, not
                # something raising max_iter further should chase away.
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                pls.fit(X[tr_idx], y_oh_tr)
            scores_tr_full = pls.transform(X[tr_idx])
            scores_val_full = pls.transform(X[val_idx])
            for c in candidates:
                c_fold = min(c, fold_cap)
                clf = ExtraTreesClassifier(
                    n_estimators=self.proxy_classifier_n_estimators,
                    random_state=self.random_state,
                )
                clf.fit(scores_tr_full[:, :c_fold], y[tr_idx])
                proba = clf.predict_proba(scores_val_full[:, :c_fold])
                full_proba = np.zeros((len(val_idx), len(classes)))
                for j, cls in enumerate(clf.classes_):
                    col = np.searchsorted(classes, cls)
                    full_proba[:, col] = proba[:, j]
                fold_scores[c].append(
                    float(log_loss(y[val_idx], full_proba, labels=classes))
                )
        mean_scores = {c: float(np.mean(v)) for c, v in fold_scores.items()}
        best = min(mean_scores, key=mean_scores.get)
        return best, mean_scores

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        # See the docstring above: -1 relative to sklearn's own hard limit
        # to avoid the numerically-degenerate boundary component.
        upper_bound = max(1, min(n_samples, n_features) - 1)
        if self.n_components is None:
            # Default: no fixed budget, just whatever the data allows --
            # never falls short of a "request", so nothing to warn about.
            n_components = upper_bound
        else:
            n_components = max(1, min(self.n_components, upper_bound))
            if n_components < self.n_components:
                warnings.warn(
                    f"PLSDASelector: requested n_components={self.n_components} "
                    f"exceeds min(n_samples={n_samples}, n_features={n_features}) - 1; "
                    f"capping to {n_components}.",
                    stacklevel=2,
                )

        if self.select_n_components and n_components > 1 and self._can_cross_validate(y):
            selected, grid_scores = self._select_n_components(X, y, n_components)
        else:
            selected, grid_scores = n_components, {}
        self.n_components_selected_ = selected
        self.n_components_grid_scores_ = grid_scores

        self._encoder_ = OneHotEncoder(sparse_output=False)
        y_onehot = self._encoder_.fit_transform(y.reshape(-1, 1))

        self._pls_ = PLSRegression(n_components=n_components, max_iter=self.max_iter)
        with warnings.catch_warnings():
            # Harmless internals warning ("y residual is constant at
            # iteration K") that sklearn itself emits when n_components
            # approaches n_samples with few classes -- expected here since
            # we deliberately push n_components as high as the safety net
            # allows, not a sign of a problem with this fit.
            warnings.filterwarnings(
                "ignore", message="y residual is constant.*", category=UserWarning
            )
            # NIPALS's per-component power iteration can still fail to
            # converge within max_iter on some components even at
            # max_iter=1000 -- inherent to deliberately pushing
            # n_components up to min(n_features, n_samples - 1) (see the
            # class docstring): components get extracted from
            # progressively deflated, lower-signal residuals, and
            # whenever two of the *remaining* singular values happen to
            # be close together (data-dependent -- empirically this isn't
            # reliably the last few components specifically, just
            # wherever a near-tie in the residual spectrum happens to
            # land), the power method can genuinely take very many
            # iterations, or not settle to `tol` at all. This is
            # deflation-sequential and one-directional (component k's fit
            # only reads the deflation left by components < k), so a
            # slow/non-converged component never changes the values
            # already extracted for earlier components -- it can only
            # blur its own direction and, going forward, the deflation
            # basis for components after it. Practically this mostly
            # lands on components that `select_n_components`'s adaptive
            # search (biased toward the low end -- see its own docstring)
            # ends up discarding anyway, so silently uncapping max_iter
            # further chases a warning without a clear accuracy payoff,
            # while materially slowing every fit that hits it (each
            # unconverged component then burns its full max_iter budget).
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._pls_.fit(X, y_onehot)

        # PLS-DA's output columns are dense linear mixtures of every raw
        # input feature, so (like ChiSquareThenSVD's post-SVD stage) there
        # is no discardable raw-column subset to hand back to the
        # extractor -- nothing here is prunable at the extractor level.
        self.raw_support_ = None
        self.n_features_kept_ = selected
        self._input_preselected_ = False
        # NIPALS extracts PLS components sequentially, each one maximizing
        # covariance with y conditional on the previous components'
        # deflation -- so component 0 is (by construction, not by any
        # extra computation here) the most informative, component 1 the
        # next, and so on, in exactly the order `transform()` returns
        # them. A simple descending rank is enough to capture that for
        # downstream grouping (see `utils.split_into_groups`) -- no need
        # for an explicit magnitude when only the *order* matters.
        self.feature_importance_ = np.arange(selected, 0, -1, dtype=np.float64)
        return self

    def transform(self, X):
        X = np.asarray(X)
        # The final fit is always done at the full capped `n_components`
        # (see fit()); slicing to `n_components_selected_` here reuses
        # that one fit rather than refitting at the selected count -- see
        # `select_n_components`'s docstring for why that's equivalent.
        return self._pls_.transform(X)[:, : self.n_components_selected_]


# --------------------------------------------------------------------------- #
# General-purpose: F-statistic ranking with FDR control
# (not currently any family's default -- see PLSDASelector's docstring for
# why this no longer matches the shapelet family; kept here as a tested,
# working, algorithm-agnostic building block, e.g. for reducer_kwargs
# overrides or future families whose feature structure suits it.)
# --------------------------------------------------------------------------- #
class FDRUnivariateSelector(PrunableReducerMixin, BaseEstimator, TransformerMixin):
    """Univariate filter with Benjamini-Hochberg FDR control.

    Originally matched to shapelet-distance features (report Sec. 5.5),
    reusing post-hoc the exact selection principle the classic
    (non-random) Shapelet Transform already applies at candidate-generation
    time (information-gain ranking of candidate shapelets; F-statistic /
    Kruskal-Wallis-type measures give equivalent rankings at lower cost,
    per Hills et al. 2014). No longer the shapelet family's default --
    see ``PLSDASelector``'s docstring for the empirical investigation that
    replaced it (FDR barely compressed RDST's raw features on real UCR
    data) -- but left in place as a working, tested, family-agnostic
    reducer. We use scikit-learn's ``SelectFdr``, which keeps every
    feature whose association p-value clears a Benjamini-Hochberg-corrected
    significance threshold -- an adaptive count, not a fixed top-k (report
    Sec. 5.6): a dataset with many genuinely discriminative features keeps
    more, one with few keeps fewer.

    ``group_size`` lets the caller dedupe correlated derived features that
    belong to the same underlying object before ranking (e.g. RDST's 3
    features -- distance / argmin / count -- per shapelet): within each
    group of that size, only the single best-scoring feature competes for
    selection, so the result isn't dominated by 2-3 redundant views of a
    handful of shapelets (report Sec. 5.5's "one wrinkle"). Note this
    dedup only works when the input's columns are still laid out in exact,
    unbroken blocks of ``group_size`` -- upstream constant-feature removal
    (``families.BaseFamily._nonconstant_mask``) can (and in practice does)
    break that alignment by dropping individual columns out of a block, at
    which point the ``n_features % group_size == 0`` guard below silently
    disables the dedup rather than deduping on a misaligned reshape.

    max_features : int, optional
        Final ceiling applied *after* the FDR threshold has already
        decided which columns clear the significance bar (``None``, the
        default, applies no ceiling) -- same safety-valve role as
        ``DetachRocketSelector.max_features`` (see that docstring), via
        the shared ``PrunableReducerMixin._apply_max_features_cap``.
        Added specifically to let this selector stand in for a
        "much-less-aggressive-than-PLS, but still bounded" reduction
        strategy on interval/shapelet: since ``SelectFdr`` keeps every
        column that clears its Benjamini-Hochberg bar rather than a fixed
        top-k, and this class was previously found to "barely compress"
        RDST/QUANT's raw features (see the class docstring above), that
        adaptive count alone can still run into the tens of thousands of
        columns -- a generous cap here (paired with a correspondingly
        larger ``backbone.GroupedICLClassifier.group_size`` downstream,
        so the *number* of TabICLv2 groups stays bounded too, not just
        the feature count feeding each one) is what actually makes "keep
        almost everything" affordable to run at all.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        group_size: int = 1,
        min_features: int = 16,
        max_features: "int | None" = None,
    ):
        self.alpha = alpha
        self.group_size = group_size
        self.min_features = min_features
        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X)
        n_features = X.shape[1]
        # Drop literally-constant columns before calling f_classif at all
        # -- scikit-learn's own f_oneway divides by each feature's
        # within-group mean square, which is exactly zero for a constant
        # column, and warns "Features [...] are constant". That warning
        # flags a column whose F-statistic is undefined (no information
        # to rank), not a fluke worth silencing after the fact -- see
        # PrunableReducerMixin._nonconstant_mask.
        nonconstant = self._nonconstant_mask(X)
        f_scores = np.zeros(n_features)
        p_values = np.ones(n_features)
        if nonconstant.any():
            f_scores[nonconstant], p_values[nonconstant] = f_classif(X[:, nonconstant], y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0)

        if self.group_size > 1 and n_features % self.group_size == 0:
            # Keep only the best-scoring feature within each group of
            # correlated derived features as a *candidate*; the rest of the
            # group is never eligible, which avoids the top-k clustering
            # around a few redundant shapelets' features. A constant
            # column is scored -inf here so it never wins its group over a
            # nonconstant alternative; if every column in a group is
            # constant, that group ends up with no eligible column at all
            # (via `eligible &= nonconstant` below) rather than picking
            # one of them essentially at random.
            group_scores = np.where(nonconstant, f_scores, -np.inf).reshape(-1, self.group_size)
            best_in_group = group_scores.argmax(axis=1)
            eligible = np.zeros(n_features, dtype=bool)
            eligible[np.arange(len(best_in_group)) * self.group_size + best_in_group] = True
        else:
            eligible = np.ones(n_features, dtype=bool)
        eligible &= nonconstant

        selector = SelectFdr(score_func=lambda X_, y_: (f_scores, p_values), alpha=self.alpha)
        selector.fit(X, y)
        support = selector.get_support() & eligible

        if support.sum() < self.min_features:
            order = np.argsort(np.where(eligible, f_scores, -np.inf))[::-1]
            support = np.zeros(n_features, dtype=bool)
            support[order[: self.min_features]] = True

        self.support_ = support
        self.n_features_kept_ = int(self.support_.sum())
        self._input_preselected_ = False
        # `f_scores` is indexed by original feature position, same as
        # `self.support_` -- masking with `support_` keeps this aligned
        # with `transform()`'s own output column order (`X[:, support_]`,
        # always ascending original index), same convention as the other
        # reducers' `feature_importance_` (see `utils.split_into_groups`).
        self.feature_importance_ = f_scores[self.support_]
        self._apply_max_features_cap()
        # Set after the cap (same ordering as DetachRocketSelector.fit)
        # so raw_support_ reflects the final, post-cap selection.
        self.raw_support_ = self.support_
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self._input_preselected_:
            return X
        return X[:, self.support_]


# --------------------------------------------------------------------------- #
# Dictionary family: chi-squared filter only (no SVD projection) -- the
# pure-selection sibling of ChiSquareThenSVD below.
# --------------------------------------------------------------------------- #
class ChiSquareSelector(PrunableReducerMixin, BaseEstimator, TransformerMixin):
    """Chi-squared FDR selection for sparse bag-of-words count features,
    with no second-stage SVD projection.

    ``ChiSquareThenSVD`` (below) runs the same chi2 FDR filter as stage 1,
    then always projects the survivors onto a small fixed/adaptive number
    of SVD components -- so its final output columns are linear mixtures
    of many raw words, not literal word counts. This class stops after
    stage 1: it keeps the original, chi2-surviving raw count columns
    themselves (capped by ``max_features``, see below), the same
    "genuine selection, not extraction" character as
    ``DetachRocketSelector``/``FDRUnivariateSelector`` -- added so
    dictionary has a selection-only option directly comparable to the
    other three families' own (see ``max_features``'s docstring on
    ``FDRUnivariateSelector`` for the fuller motivation: pairing a
    generous cap here with a correspondingly larger
    ``backbone.GroupedICLClassifier.group_size`` downstream, rather than
    the family's own reduced-then-still-narrow-grouped default).

    Note the module docstring's own history is relevant context before
    reaching for this: dictionary's raw bag-of-words width is typically
    already close to WEASEL's own ~20,000-column native scale (see
    ``families.DictionaryFamily``), so a generous ``max_features`` cap
    here is likely to bind only lightly -- this is close to "keep almost
    every word that clears the FDR significance bar," not a meaningful
    compression step on its own; whatever accuracy/log-loss effect shows
    up is mostly about *not projecting* (selection vs. extraction) and
    about the *group_size* TabICLv2 sees them through, not about how
    aggressively words were filtered.

    Parameters
    ----------
    chi2_alpha : float
        Benjamini-Hochberg FDR significance threshold for the chi2 test,
        same semantics and default as ``ChiSquareThenSVD.chi2_alpha``.
    max_features : int, optional
        Final ceiling on the number of chi2-surviving columns kept, by
        descending chi2 score (``None``, the default, applies no
        ceiling) -- see ``PrunableReducerMixin._apply_max_features_cap``.
    min_features : int
        Never select fewer than this many columns, regardless of how few
        clear the FDR bar -- mirrors ``FDRUnivariateSelector``'s own
        floor.

    Attributes
    ----------
    feature_importance_ : ndarray
        Chi2 statistic for each *kept* feature, aligned column-for-column
        with ``transform()``'s own output order -- same convention as
        every other reducer in this module (see ``PrunableReducerMixin``).
    """

    name = "dictionary"

    def __init__(
        self,
        chi2_alpha: float = 0.05,
        max_features: "int | None" = None,
        min_features: int = 16,
    ):
        self.chi2_alpha = chi2_alpha
        self.max_features = max_features
        self.min_features = min_features

    def fit(self, X, y):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
        X.data = np.clip(X.data, 0, None)  # chi2 requires non-negative input
        n_features = X.shape[1]

        # Same guard as FDRUnivariateSelector.fit -- an all-zero/never-
        # occurring word column is constant by this family's own raw
        # scale, and chi2 already degrades to a silent `nan` for one
        # (caught by the nan_to_num below); excluding it up front is more
        # direct than relying on that nan collapsing to a score of 0, and
        # keeps this selector's behaviour symmetric with
        # FDRUnivariateSelector's for the other two families.
        nonconstant = self._nonconstant_mask(X)
        chi2_scores = np.zeros(n_features)
        p_values = np.ones(n_features)
        if nonconstant.any():
            chi2_scores[nonconstant], p_values[nonconstant] = chi2(X[:, nonconstant], y)
        chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0)

        selector = SelectFdr(score_func=lambda X_, y_: (chi2_scores, p_values), alpha=self.chi2_alpha)
        selector.fit(X, y)
        support = selector.get_support() & nonconstant

        if support.sum() < self.min_features:
            order = np.argsort(np.where(nonconstant, chi2_scores, -np.inf))[::-1]
            support = np.zeros(n_features, dtype=bool)
            support[order[: self.min_features]] = True

        self.support_ = support
        self.n_features_kept_ = int(self.support_.sum())
        self._input_preselected_ = False
        # `chi2_scores` is indexed by original feature position, same
        # convention as every other reducer's `feature_importance_`.
        self.feature_importance_ = chi2_scores[self.support_]
        self._apply_max_features_cap()
        self.raw_support_ = self.support_
        return self

    def transform(self, X):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
        X.data = np.clip(X.data, 0, None)
        if self._input_preselected_:
            return X
        return X[:, self.support_]


# --------------------------------------------------------------------------- #
# Dictionary family: chi-squared filter + truncated SVD
# --------------------------------------------------------------------------- #
class ChiSquareThenSVD(PrunableReducerMixin, BaseEstimator, TransformerMixin):
    """Two-stage reduction for sparse bag-of-words count features.

    Matched to dictionary-based features (report Sec. 5.5 / 5.6): stage 1
    reuses WEASEL's own native reduction step (a chi-squared association
    test, well-suited to non-negative count data), via scikit-learn's
    ``SelectFdr(chi2, ...)`` so the cutoff is an FDR-controlled p-value
    threshold rather than a fixed top-k. That alone typically still leaves
    far more features than TabICLv2's comfort zone, so stage 2 applies
    truncated SVD (PCA's sparse-matrix-friendly cousin) -- the standard
    second stage in text classification for the same kind of bag-of-words
    object (this is literally Latent Semantic Analysis), and the family
    where the user's own prior clustering result (WEASEL features + PCA)
    gives an independent reason to trust variance-based compression.

    ``n_components`` may be an int (fixed SVD target) or "auto", in which
    case a modest upper-bound SVD is fit first and then truncated to the
    number of components needed to reach ``explained_variance_target``
    cumulative explained variance -- an adaptive count in the same spirit
    as the other selectors, not a fixed number chosen in advance.
    """

    name = "dictionary"

    def __init__(
        self,
        chi2_alpha: float = 0.05,
        n_components: "int | str" = "auto",
        explained_variance_target: float = 0.95,
        max_components: int = 200,
        min_components: int = 16,
        random_state: Optional[int] = None,
    ):
        self.chi2_alpha = chi2_alpha
        self.n_components = n_components
        self.explained_variance_target = explained_variance_target
        self.max_components = max_components
        self.min_components = min_components
        self.random_state = random_state

    def fit(self, X, y):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
        X.data = np.clip(X.data, 0, None)  # chi2 requires non-negative input
        n_raw_features = X.shape[1]

        self._chi2_selector_ = SelectFdr(chi2, alpha=self.chi2_alpha)
        X_filtered = self._chi2_selector_.fit_transform(X, y)
        if X_filtered.shape[1] == 0:
            # Degenerate case: nothing cleared the FDR bar (e.g. a tiny
            # pilot dataset). Fall back to keeping everything and let SVD
            # do the compression.
            warnings.warn(
                "ChiSquareThenSVD: chi2 FDR filter removed all features; "
                "skipping the filter stage for this branch.",
                stacklevel=2,
            )
            X_filtered = X
            self._chi2_selector_ = None

        # Unlike the shapelet family's selector, our *final* output
        # columns (post-SVD) are linear mixtures of every chi2-surviving
        # raw column, so they can't be traced back to one raw feature each
        # -- there's nothing to hand the extractor as a "keep exactly
        # these final columns" mask. Even the chi2 filter stage itself
        # isn't prunable at the extractor here: ``families.DictionaryFamily``
        # (WEASELTransformerV2) doesn't implement ``prune()`` (see its
        # docstring), so ``raw_support_`` below only feeds a prune
        # *attempt* that honestly reports unsupported -- the reducer's own
        # ``transform()`` still masks the fully-computed raw columns down,
        # same as it always did.
        self.raw_support_ = (
            self._chi2_selector_.get_support()
            if self._chi2_selector_ is not None
            else np.ones(n_raw_features, dtype=bool)
        )
        self._input_preselected_ = False

        n_avail = X_filtered.shape[1]
        cap = min(self.max_components, n_avail - 1) if n_avail > 1 else 1
        cap = max(cap, 1)

        if self.n_components == "auto":
            probe_k = max(cap, self.min_components)
            probe_k = min(probe_k, n_avail - 1) if n_avail > 1 else 1
            probe_k = max(probe_k, 1)
            svd_probe = TruncatedSVD(
                n_components=probe_k, random_state=self.random_state
            )
            svd_probe.fit(X_filtered)
            cum_var = np.cumsum(svd_probe.explained_variance_ratio_)
            n_components = int(np.searchsorted(cum_var, self.explained_variance_target) + 1)
            n_components = int(np.clip(n_components, self.min_components, cap))
        else:
            n_components = int(np.clip(self.n_components, 1, cap))

        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.svd_.fit(X_filtered)
        self.n_features_kept_ = n_components
        # Deliberately *not* a synthetic descending rank (unlike
        # PLSDASelector's component ordering above, which genuinely does
        # match column order by construction): `TruncatedSVD.components_`
        # are sorted by descending *singular value*, but singular value
        # order and `explained_variance_ratio_` order are NOT the same
        # thing for uncentered sparse data like this (checked empirically
        # while building this) -- when the surviving chi2-filtered word
        # counts share a strong common "DC" offset (most rows having
        # broadly similar counts in many columns), the leading singular
        # vector often aligns with that offset direction: large magnitude
        # (so ranked first by singular value), but low actual cross-sample
        # *variance* (so a low `explained_variance_ratio_`, i.e. not
        # actually where the class-discriminative signal lives). Using the
        # real `explained_variance_ratio_` here, rather than assuming
        # transform()'s column position already reflects true importance,
        # is exactly why downstream grouping (`utils.split_into_groups`)
        # is told to *rank* by this array's values rather than trust
        # column order.
        self.feature_importance_ = self.svd_.explained_variance_ratio_
        return self

    def transform(self, X):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
        X.data = np.clip(X.data, 0, None)
        if not self._input_preselected_ and self._chi2_selector_ is not None:
            X = self._chi2_selector_.transform(X)
        return self.svd_.transform(X)


REDUCTION_REGISTRY = {
    "convolution": DetachRocketSelector,
    "interval": PLSDASelector,
    "shapelet": PLSDASelector,
    "dictionary": ChiSquareThenSVD,
}


def make_reducer(family_name: str, **kwargs):
    """Convenience constructor mapping a family name to its matched reducer."""
    if family_name not in REDUCTION_REGISTRY:
        raise ValueError(
            f"No reducer registered for family {family_name!r}. "
            f"Available: {sorted(REDUCTION_REGISTRY)}"
        )
    return REDUCTION_REGISTRY[family_name](**kwargs)
