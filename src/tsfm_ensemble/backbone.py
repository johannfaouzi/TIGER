"""
Grouped in-context-learning backbone (the TabICLv2 wrapper).

This module implements the two mechanisms from the report that address the
mismatch between a family's (possibly still large, post-reduction) feature
count and TabICLv2's validated operating range, and TabICLv2's many-class
handling:

1. **Grouping** (report Sec. 5.4/5.6): split a branch's feature matrix into
   column groups no larger than ``group_size`` (TabICLv2's comfort zone --
   pretrained on 2-100 columns), run a fresh backbone instance per group,
   and average predicted probabilities across groups. The number of groups
   is a *consequence* of how many features the upstream reducer decided to
   keep, not a fixed target -- this is precisely the "let the ensemble size
   absorb whatever k turns out to be" design from Sec. 5.6.

   Which columns land in which group matters for many-class meta-models
   specifically: ``meta_learners.py``'s wide meta-feature matrix has one
   column per (learner, class) pair (12 learners x n_classes columns), so
   a given class's *direct* evidence -- the 12 columns where some learner
   actually estimated P(that class) -- is a small minority of the total
   column count once n_classes is large. Plain random group assignment
   (``fit()``'s default, no ``class_group_key``) gives *no* guarantee
   every group gets some of those 12 columns: for this project's own
   worst case (60 classes, 720 columns, ``group_size=100`` -> 8 groups of
   90), the exact hypergeometric probability that a *specific* group gets
   *none* of a given class's 12 columns is ~20% per group, and simulating
   the actual random split (20,000 trials of the real
   ``utils.split_into_groups``) puts the chance that *some* group is
   entirely blind to a given class at ~90% (averaging ~1.6 of the 8
   groups per class) -- not a rare edge case, essentially the norm at
   this class count. A "blind" group still fits and still returns an
   n_classes-column ``predict_proba`` (nothing crashes), but its estimate
   for that class is necessarily built on indirect signal only, and
   uniform averaging then gives it the same weight as groups that did
   have direct evidence. Passing ``class_group_key`` (see ``fit()`` and
   ``meta_learners._class_group_key``) switches to the key-stratified
   round-robin strategy instead, which *guarantees* every group gets at
   least ``floor(12 / n_groups)`` columns per class whenever
   ``n_groups <= 12`` (true for every dev-subset job at the default
   ``group_size=100``, since a class's 12 learner-columns only run out
   past 12 groups) -- see ``utils.split_into_groups``'s own docstring for
   the mechanism. ``fit_predict_tabiclv2`` uses this by default
   (``stratify_groups_by_class=True``); ``GroupedICLClassifier`` itself
   stays opt-in (``class_group_key=None`` unless a caller passes one) so
   this doesn't silently change behaviour for other callers of the class.

2. **ECOC on top of native many-class handling** (report Sec. 5.2): the
   TabICLv2 authors' own Figure 7 result is that native mixed-radix
   many-class handling *and* an ECOC wrapper *combined* beat either alone
   (paper text: the ECOC wrapper is "slightly better but 3x slower" than
   native handling alone). Real ``tabicl.TabICLClassifier`` already does
   native mixed-radix handling internally when ``support_many_classes=True``
   (its default). Setting ``ecoc=True`` here additionally wraps each
   group's backbone in an ECOC layer for an extra, independent coding-level
   of redundancy -- ``ecoc_backend`` selects which implementation:

   * ``"many_class"`` (default): ``tabpfn_extensions.many_class.
     ManyClassClassifier`` -- a real, maintained module from PriorLabs (the
     TabPFN authors; TabICLv2 draws on the same lineage). The TabICLv2
     paper's own text says it "also supports TabPFNv2's ECOC wrapper" --
     this is that wrapper, not a generic stand-in: it wraps *any*
     sklearn-compatible estimator (confirmed by reading its actual source
     -- ``fit``/``predict_proba`` only ever call the wrapped estimator's
     own sklearn-standard ``fit``/``predict_proba``, nothing TabPFN-
     specific), and its own codebook-size formula
     (``max(log-cover, covering-bound) * redundancy``, default
     ``n_estimators_redundancy=4``, capped near ``4 * log-cover`` -- see
     its own ``_get_n_estimators``) is a genuinely different, much cheaper
     scaling than a naive ``int(n_classes * code_size)`` -- e.g. 52 classes
     at ``ecoc_alphabet_size=10`` (TabICLv2's own pretrained
     ``max_classes``) needs 8 estimators, not 78, which lines up with the
     paper's own reported ~3x (native mode's own mixed-radix decomposition
     needs 2 digit-passes for the same 52 classes; 8/2 = 4x, close to the
     paper's 3x given this is a different, if closely related, package
     implementing the same idea rather than the paper's own private code).
     It also implements a genuine ``predict_proba`` (weighted/log-
     likelihood decoding across codebook rows) -- unlike the ``"sklearn"``
     backend below, so groups using it don't fall back to the coarser
     one-hot-vote approximation ``_group_proba`` uses when ``has_proba``
     is False. Requires the ``tabpfn-extensions`` package (see
     ``requirements.txt``); raises a clear ``ImportError`` with install
     instructions if it's missing, rather than silently falling back.
     ``TabICLClassifier`` has no ``MAX_NUMBER_OF_CLASSES`` attribute (that's
     a TabPFN-specific convention ``ManyClassClassifier`` auto-detects), so
     ``ecoc_alphabet_size`` must be, and is, passed explicitly (default 10,
     TabICLv2's own known pretrained ``max_classes`` -- see
     ``tabicl.TabICLClassifier``'s own ``support_many_classes`` docstring
     and its ``model_.max_classes`` check).
   * ``"sklearn"``: the original implementation -- scikit-learn's generic,
     off-the-shelf ``OutputCodeClassifier`` (random code matrix, Hamming-
     distance decoding at *prediction* time only -- no ``predict_proba``).
     Kept available for comparison against ``"many_class"`` (e.g. via
     ``compare_tabiclv2_multiclass_handling.py``) and because it needs no
     extra dependency, but it is **not** believed to match what the
     TabICLv2 paper itself measured -- see ``"many_class"`` above.

   IMPORTANT TIMING CAVEAT for ``"many_class"``: unlike native mode (and
   unlike the ``"sklearn"`` ECOC backend), ``ManyClassClassifier.fit()``
   only builds the codebook -- every one of its per-row estimator
   fit+predict cycles happens lazily inside ``predict_proba()`` (see its
   own source). So for ECOC-via-``many_class`` groups, this class's own
   ``fit_timings_`` will look artificially cheap and ``predict_timings_``
   will carry essentially all of that group's real compute -- expected,
   not a bug, but worth knowing when reading those timings.

Because TabICLv2 requires downloading a model checkpoint from the network
(and a GPU for anything beyond toy-sized inputs), this module falls back to
a plain scikit-learn classifier when ``tabicl`` is not installed or not
reachable, with a loud warning -- so the rest of the pipeline (grouping,
ECOC wrapping, cross-branch fusion) can be built, tested, and unit-tested
without requiring TabICLv2 itself to be runnable in a given environment.
See ``README.md`` for exactly what was and wasn't actually exercised
against the real model while building this.

A note on RAM/VRAM over a long run: every group's fitted model is kept
(``predict_proba`` has to query all of them), and every branch keeps every
group, so a fitted ensemble's memory footprint scales with (branches x
groups per branch) -- that part is inherent to grouping/ensembling, not a
bug (see ``GroupedICLClassifier``'s own docstring). Separately from that,
real ``tabicl.TabICLClassifier`` loads a checkpoint fresh per group
(``.fit()`` is called once per group, see ``GroupedICLClassifier.fit``
below), and PyTorch's own MPS backend has a confirmed, currently-open bug
where memory grows specifically from repeated model/state-dict loading
(cuda and cpu are unaffected -- see pytorch/pytorch#154329), which is
exactly that pattern over a many-group, many-branch, many-dataset run.
``_release_device_memory()`` below is a best-effort mitigation (a
``gc.collect()`` plus, if available, ``torch.cuda.empty_cache()`` /
``torch.mps.empty_cache()``) called after every group fit -- it cannot
un-retain the models grouping requires keeping, but it does let PyTorch's
allocator release whatever *isn't* still referenced instead of silently
holding onto it for the rest of the process. If VRAM is still the binding
constraint, ``backbone_kwargs={"device": "cpu"}`` (passed straight through
to ``TabICLClassifier``) sidesteps the MPS-specific leak entirely, at the
cost of CPU-speed inference.
"""
from __future__ import annotations

import gc
import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder

from src.tsfm_ensemble.timing import timed
from src.tsfm_ensemble.utils import split_into_groups

# Two known-benign numpy RuntimeWarnings raised from *inside* real
# tabicl.TabICLClassifier's own preprocessing, not from this project's own
# code -- confirmed by pulling the actual installed tabicl==2.2.0 wheel and
# reading its source (tabicl/_sklearn/preprocessing.py): TabICLClassifier's
# ``norm_methods`` defaults to ``None``, which resolves to ``["none",
# "power"]`` -- i.e. its default 8-member ensemble (``n_estimators=8``)
# always includes at least one member that runs
# ``PreprocessingPipeline(normalization_method="power")``, which fits a
# plain ``sklearn.preprocessing.PowerTransformer(method="yeo-johnson")`` on
# whatever this project hands it. This project's own meta-feature matrices
# (assemble_meta_features's wide concatenation of every base learner's
# scores) mix RidgeClassifierCV's raw, unbounded ``decision_function``
# margins with bounded [0, 1] probability columns -- a much wider dynamic
# range per column than typical tabular data, which is exactly what makes
# Yeo-Johnson's own lambda search (``scipy.optimize.brent``, inside
# ``PowerTransformer.fit``) and the fitted transform itself (``(x+1)**lmbda``,
# inside ``PowerTransformer.transform``) prone to float overflow on the
# largest-magnitude ridge columns.
#
# Both are numpy's ordinary "ignore, computed inf/nan for one intermediate
# value" RuntimeWarnings, not exceptions -- they don't stop the fit, and
# TabICLClassifier's own ensemble already blends this "power" member with a
# "none" member (plus feature/class-shuffle variants), with its own
# OutlierRemover(z>4) clipping stage downstream of the normalizer either
# way. Scoped by *both* exact message text and the specific module that
# raises it (scipy's own Brent optimizer internals; sklearn's own Yeo-
# Johnson transform application) -- verified this scoping is precise, not
# blanket: a synthetic PowerTransformer stress test (an extreme-valued
# column, mirroring a wide-margin ridge column) produced several *other*
# overflow/invalid-value RuntimeWarnings from neighbouring numpy/sklearn
# call sites (e.g. "overflow encountered in square" from
# sklearn.utils.extmath, "overflow encountered in scalar power" from a
# different line of sklearn.preprocessing._data) that these two filters
# correctly leave alone -- and the same message text raised from an
# unrelated module is also left alone. If a sibling overflow/invalid-value
# warning from this same TabICLv2-preprocessing path turns up later (a real
# possibility -- see that stress test), it needs its own narrowly-scoped
# filter added here the same way, not a broadened version of these two.
for _message, _module in (
    (r"overflow encountered in scalar multiply", r"scipy\.optimize\._optimize"),
    (r"overflow encountered in power", r"sklearn\.preprocessing\._data"),
):
    warnings.filterwarnings("ignore", message=_message, category=RuntimeWarning, module=_module)

# NOTE: this module used to latch a process-wide "TabICLv2 unavailable,
# stop retrying" flag here and have backbone="auto" silently substitute a
# HistGradientBoostingClassifier placeholder on any import/fit failure --
# removed. Per Johann's explicit instruction, TabICLv2 (backbone="auto" or
# "tabicl", identical strict behaviour) never falls back to a placeholder:
# any failure to import or fit the real model raises immediately (see
# ``make_backbone`` and ``GroupedICLClassifier.fit`` below). The
# placeholder classifier itself (``_fallback_backbone``) still exists and
# is still reachable, but only via an *explicit* ``backbone="fallback"``
# request (used by this project's own tests to exercise the surrounding
# grouping/ECOC/fusion machinery without needing the real, heavy model) --
# never chosen automatically.


def _import_many_class_classifier():
    """Import ``tabpfn_extensions.many_class.ManyClassClassifier`` -- see
    the module docstring's "many_class" ECOC-backend section for what it
    is and why it's the default. A missing ``tabpfn-extensions`` package
    is a loud, explicit ``ImportError`` here, not a silent fallback to a
    different backend or to the one-hot-vote approximation -- consistent
    with this project's "no un-auditable silent fallback" convention
    (e.g. ``build_meta_model_predictions.py``'s own ``_auto_select_tabiclv2_ecoc``).
    """
    try:
        from tabpfn_extensions.many_class import ManyClassClassifier
    except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised below
        raise ImportError(
            "ecoc_backend='many_class' (the default) requires the "
            "`tabpfn-extensions` package (see requirements.txt), which "
            f"could not be imported: {exc!r}. Install it with `pip install "
            "tabpfn-extensions`, or pass ecoc_backend='sklearn' to use the "
            "original scikit-learn-based ECOC implementation instead (see "
            "the module docstring for how the two differ)."
        ) from exc
    return ManyClassClassifier


def _fallback_backbone(random_state: Optional[int] = None):
    """A plain, dependency-light stand-in classifier used when TabICLv2 is
    unavailable, so the surrounding grouping/ECOC/fusion machinery remains
    testable. Not a substitute for TabICLv2's actual in-context-learning
    behaviour -- see the module docstring.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=random_state)


def make_backbone(
    backbone: str = "auto",
    backbone_kwargs: Optional[dict] = None,
    random_state: Optional[int] = None,
):
    """Construct one (unfitted) backbone classifier instance.

    Parameters
    ----------
    backbone : {"auto", "tabicl", "fallback"} or an sklearn estimator
        "auto" and "tabicl" are strict synonyms (no behavioural
        difference -- kept as two names for readability at call sites):
        both *construct* a ``TabICLClassifier``, and both raise a loud
        ``ImportError`` if the package can't be imported. Note a
        successful import does **not** guarantee TabICLv2 will actually
        run: ``tabicl`` loads its model checkpoint lazily, on the first
        call to ``.fit()`` (typically downloading it from Hugging Face
        Hub on first use), so an import-time success can still fail
        later at fit-time if that download isn't reachable (as it wasn't
        in the sandbox this package was developed in -- see README.md).
        That fit-time failure is handled by ``GroupedICLClassifier.fit``,
        not here -- this function only handles *import*-time failure.
        Neither name ever falls back to a placeholder classifier, at
        import time or fit time -- see this module's module-level note
        (just above ``_import_many_class_classifier``) for why that
        behaviour was removed. "fallback" is the one way to get the
        placeholder: it must be requested explicitly (this project's own
        tests do, to exercise the grouping/ECOC/fusion machinery without
        the real, heavy model) -- never chosen automatically. Anything
        else is assumed to already be an (unfitted) scikit-learn-
        compatible classifier instance or class, which is cloned/
        instantiated per group.
    """
    backbone_kwargs = dict(backbone_kwargs or {})

    if isinstance(backbone, str):
        if backbone in ("auto", "tabicl"):
            try:
                from tabicl import TabICLClassifier

                kwargs = {"random_state": random_state, **backbone_kwargs}
                return TabICLClassifier(**kwargs)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised below
                raise ImportError(
                    f"backbone={backbone!r} requires the `tabicl` package, "
                    f"which could not be imported (original error: {exc!r}). "
                    "Install it with `pip install tabicl`. There is no "
                    "automatic fallback -- pass backbone='fallback' "
                    "explicitly if you specifically want the dependency-"
                    "light placeholder classifier instead (e.g. for testing "
                    "this module's grouping/ECOC/fusion machinery)."
                ) from exc
        if backbone == "fallback":
            return _fallback_backbone(random_state=random_state)
        raise ValueError(f"Unknown backbone spec: {backbone!r}")

    # Assume an sklearn estimator instance or class was passed directly.
    if isinstance(backbone, type):
        return backbone(**backbone_kwargs)
    return clone(backbone)


def _release_device_memory() -> None:
    """Best-effort release of accelerator memory back to the OS/driver,
    beyond plain Python garbage collection -- see the module docstring's
    "RAM/VRAM over a long run" note for why this exists and what it can
    and can't fix. ``gc.collect()`` runs first regardless of whether torch
    is even installed (it's cheap, and it's a prerequisite for the cache
    calls below to free anything: PyTorch's allocators can only release
    tensors that no longer have a live Python reference). Safe to call
    unconditionally -- on a machine with no GPU, with ``tabicl``/``torch``
    never installed (e.g. the dependency-light ``fallback`` backbone), or
    mid-``ImportError`` for any other reason, this is a silent no-op.
    """
    gc.collect()
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch may simply not be installed
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - never let cleanup itself fail a fit/predict
        pass
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - same as above
        pass


class GroupedICLClassifier(BaseEstimator, ClassifierMixin):
    """Split a (possibly wide) feature matrix into TabICLv2-sized column
    groups, classify each group independently, and average across groups.
    See predict_proba_per_group to inspect each group's own
    probabilities before that averaging is applied.

    Parameters
    ----------
    group_size : int
        Maximum columns per group. TabICLv2 was pretrained on 2-100
        columns (report Sec. 4); we default conservatively within that
        range rather than TabPFN v2.5's much larger 2,000-feature cap that
        RocketPFN used, per the report's revised Sec. 5.4 recommendation.
    backbone, backbone_kwargs : see ``make_backbone``.
    ecoc : bool
        If True, wrap each group's backbone in an ECOC layer -- see
        ``ecoc_backend`` and the module docstring's "ECOC on top of native
        many-class handling" section. Only meaningful for genuinely
        multiclass problems; silently ignored for binary ones (ECOC over
        2 classes is not a meaningful construction).
    ecoc_backend : {"many_class", "sklearn"}
        Which ECOC implementation ``ecoc=True`` uses -- see the module
        docstring for what each one actually is and why ``"many_class"``
        (``tabpfn_extensions.many_class.ManyClassClassifier``) is the
        default. Requires the ``tabpfn-extensions`` package when
        ``"many_class"`` is selected; raises ``ImportError`` (with install
        instructions) if it's missing, not a silent fallback.
    ecoc_alphabet_size : int
        ``"many_class"`` backend only: the base estimator's own max
        classes-per-fit limit (``ManyClassClassifier``'s own
        ``alphabet_size``). Default 10 matches TabICLv2's own pretrained
        ``max_classes`` (see ``tabicl.TabICLClassifier``'s
        ``support_many_classes`` docstring) -- override if a different
        checkpoint/backbone has a different limit. Ignored (but still
        accepted) for ``ecoc_backend="sklearn"`` or non-TabICL backbones.
    ecoc_redundancy : int
        ``"many_class"`` backend only: forwarded to ``ManyClassClassifier``'s
        own ``n_estimators_redundancy`` (its own default is 4) -- higher
        values add more codebook redundancy (and estimator fits) per
        group; see the module docstring's own worked example.
    ecoc_code_size : float
        ``"sklearn"`` backend only: passed through to
        ``OutputCodeClassifier`` (fraction of classes_-many bits, default
        matches sklearn's own default of 1.5x).
    n_jobs : int
        Reserved for future parallelisation across groups; groups are
        currently fit sequentially (kept simple and debuggable -- the
        expensive part in a real run is each TabICLv2 forward pass, which
        should be parallelised at a higher level, e.g. across datasets or
        via TabICLv2's own batching, not by this class).

    Attributes
    ----------
    grouping_strategy_ : {"class_stratified", "stratified", "random"}
        Set by ``fit()``: "class_stratified" if it was called with a
        ``class_group_key`` array (key-stratified round-robin, see the
        module docstring's "Grouping" section), "stratified" if instead
        called with a ``feature_importance`` array (importance-stratified
        round-robin), "random" otherwise (the original, still-default-for-
        this-class behaviour -- ``fit_predict_tabiclv2`` opts into
        "class_stratified" itself; this class doesn't do so on its own).
        Purely informational -- doesn't affect prediction, just useful for
        confirming which path a given branch/call actually took.
    """

    def __init__(
        self,
        group_size: int = 100,
        backbone: str = "auto",
        backbone_kwargs: Optional[dict] = None,
        ecoc: bool = False,
        ecoc_backend: str = "many_class",
        ecoc_alphabet_size: int = 10,
        ecoc_redundancy: int = 4,
        ecoc_code_size: float = 1.5,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ):
        self.group_size = group_size
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.ecoc = ecoc
        self.ecoc_backend = ecoc_backend
        self.ecoc_alphabet_size = ecoc_alphabet_size
        self.ecoc_redundancy = ecoc_redundancy
        self.ecoc_code_size = ecoc_code_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X,
        y,
        feature_importance: Optional[np.ndarray] = None,
        class_group_key: Optional[np.ndarray] = None,
    ):
        """See the class docstring for ``group_size``, etc.

        ``feature_importance`` : one score per input column (higher =
        more important), optional. When given, TabICLv2 groups are
        assigned by stratified round-robin instead of a random shuffle --
        see ``utils.split_into_groups``'s docstring for the mechanism and
        the empirical case for it. ``pipeline.PrunedFamilyBranch`` passes
        this through automatically from whichever reducer produced ``X``,
        when that reducer exposes a ``feature_importance_`` (see
        ``reduction.PrunableReducerMixin``'s docstring); ``None`` (the
        default, e.g. when called directly, or the reducer has no such
        ranking) falls back to the previous random-shuffle behaviour.

        ``class_group_key`` : one *categorical* label per input column
        (e.g. "which class does this column directly evidence"),
        optional, mutually exclusive with ``feature_importance``. When
        given, groups are assigned by key-stratified round-robin instead
        of a random shuffle -- see ``utils.split_into_groups``'s
        docstring for the mechanism and the guarantee it gives (no group
        ends up with zero columns for a given key, as long as that key
        has at least ``n_groups`` columns). ``meta_learners.py``'s
        ``fit_predict_tabiclv2`` passes this through automatically (see
        ``_class_group_key``) so that, e.g., a many-class meta-model's
        groups are never blind to one of the classes -- see this class's
        module docstring's "Grouping" section for why plain random
        assignment doesn't give that guarantee and how likely it is to
        matter in practice.
        """
        if feature_importance is not None and class_group_key is not None:
            raise ValueError(
                "feature_importance and class_group_key are mutually "
                "exclusive -- pass at most one."
            )
        # ``per_group`` records each group's own fit time, in the same
        # order as ``self.groups_``;
        # ``total`` is the whole loop, so ``total - sum(per_group)`` is
        # roughly the (usually negligible) group-splitting/label-encoding
        # overhead around it.
        self.fit_timings_: dict = {"per_group": []}
        with timed(self.fit_timings_, "total"):
            X = np.asarray(X)
            self._label_encoder_ = LabelEncoder().fit(y)
            y_enc = self._label_encoder_.transform(y)
            self.classes_ = self._label_encoder_.classes_
            n_classes = len(self.classes_)

            n_features = X.shape[1]
            self.groups_ = split_into_groups(
                n_features,
                self.group_size,
                random_state=self.random_state,
                importance=feature_importance,
                group_key=class_group_key,
            )
            self.n_groups_ = len(self.groups_)
            if class_group_key is not None:
                self.grouping_strategy_ = "class_stratified"
            elif feature_importance is not None:
                self.grouping_strategy_ = "stratified"
            else:
                self.grouping_strategy_ = "random"

            use_ecoc = self.ecoc and n_classes > 2

            def _wrap(base):
                if use_ecoc:
                    if self.ecoc_backend == "many_class":
                        ManyClassClassifier = _import_many_class_classifier()
                        return (
                            ManyClassClassifier(
                                estimator=base,
                                alphabet_size=self.ecoc_alphabet_size,
                                n_estimators_redundancy=self.ecoc_redundancy,
                                random_state=self.random_state,
                            ),
                            # Real predict_proba (weighted/log-likelihood decoding
                            # across codebook rows) -- see the module docstring's
                            # "many_class" section, unlike the sklearn backend below.
                            True,
                        )
                    if self.ecoc_backend == "sklearn":
                        return (
                            OutputCodeClassifier(
                                base,
                                code_size=self.ecoc_code_size,
                                random_state=self.random_state,
                                n_jobs=self.n_jobs,
                            ),
                            False,
                        )
                    raise ValueError(
                        f"Unknown ecoc_backend: {self.ecoc_backend!r} (expected "
                        "'many_class' or 'sklearn')."
                    )
                return base, hasattr(base, "predict_proba")

            self._fitted_groups_ = []
            for gi, cols in enumerate(self.groups_):
                base = make_backbone(
                    self.backbone, self.backbone_kwargs, random_state=self.random_state,
                )
                model, has_proba = _wrap(base)
                group_timings: dict = {}
                with timed(group_timings, "fit"):
                    try:
                        model.fit(X[:, cols], y_enc)
                    except Exception as exc:  # noqa: BLE001 - re-raised with context below, never a fallback
                        # tabicl loads its checkpoint lazily on first `.fit()` call
                        # (typically a Hugging Face Hub download), so a failure
                        # that `make_backbone` couldn't have caught at construction
                        # time (e.g. no network route to the checkpoint) surfaces
                        # here instead. No fallback, for any backbone value: per
                        # Johann's explicit instruction, TabICLv2 (backbone="auto"
                        # or "tabicl") never silently substitutes a placeholder
                        # classifier on a fit-time failure -- this re-raises,
                        # adding context, rather than retrying with
                        # `_fallback_backbone`. A user-supplied custom estimator's
                        # errors are likewise always raised, never papered over.
                        raise RuntimeError(
                            f"TabICLv2 (backbone={self.backbone!r}) failed to fit "
                            f"group {gi} of {self.n_groups_}: {exc!r}. No "
                            "fallback -- see make_backbone's and this method's "
                            "own docstrings. A common cause is no network route "
                            "to TabICLv2's Hugging Face Hub checkpoint on first "
                            "`.fit()`; fix connectivity/installation and retry, "
                            "or pass backbone='fallback' explicitly if you "
                            "specifically want the placeholder classifier."
                        ) from exc
                self.fit_timings_["per_group"].append(group_timings["fit"])
                self._fitted_groups_.append(
                    {"cols": cols, "model": model, "has_proba": has_proba}
                )
                # See the module docstring's RAM/VRAM note: real TabICLv2
                # loads a checkpoint fresh on every one of these per-group
                # fits, which is exactly the repeated-model-loading pattern
                # behind a confirmed, open PyTorch MPS memory-growth bug.
                # This can't un-retain the group we just kept (grouping
                # needs it for predict_proba), but it releases anything
                # from this group's own fit that isn't referenced anymore.
                _release_device_memory()
        return self

    def _group_proba(self, group: dict, X: np.ndarray, n_classes: int) -> np.ndarray:
        model = group["model"]
        X_g = X[:, group["cols"]]
        if group["has_proba"]:
            proba = np.asarray(model.predict_proba(X_g))
            # predict_proba's column order follows model.classes_, which
            # (since we fit on label-encoded y) should already be
            # 0..n_classes-1, but don't assume -- realign defensively.
            full = np.zeros((X_g.shape[0], n_classes))
            for j, c in enumerate(np.asarray(model.classes_).astype(int)):
                full[:, c] = proba[:, j]
            return full
        # No predict_proba available (currently only ecoc_backend="sklearn"
        # groups -- "many_class" has a real predict_proba, see the module
        # docstring): approximate a probability with a one-hot vote. This
        # is a coarser signal than a real probability.
        preds = np.asarray(model.predict(X_g)).astype(int)
        full = np.zeros((X_g.shape[0], n_classes))
        full[np.arange(X_g.shape[0]), preds] = 1.0
        return full

    def predict_proba(self, X):
        self.predict_timings_: dict = {"per_group": []}
        with timed(self.predict_timings_, "total"):
            X = np.asarray(X)
            n_classes = len(self.classes_)
            acc = np.zeros((X.shape[0], n_classes))
            for group in self._fitted_groups_:
                group_timings: dict = {}
                with timed(group_timings, "predict"):
                    acc += self._group_proba(group, X, n_classes)
                self.predict_timings_["per_group"].append(group_timings["predict"])
            acc /= len(self._fitted_groups_)
            row_sums = acc.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            result = acc / row_sums
        # One cleanup pass after the whole predict_proba call, not per
        # group: unlike fit(), predict doesn't reload a checkpoint per
        # group (TabICLv2's own KV cache, built once at fit time, is
        # reused across predict calls -- see the module docstring), so the
        # per-group repeated-loading pattern that motivates
        # _release_device_memory() in fit() doesn't apply here as
        # strongly; this just catches stray forward-pass buffers.
        _release_device_memory()
        return result

    def predict_proba_per_group(self, X) -> np.ndarray:
        """Each group's own predicted probabilities, before averaging.

        Uniform averaging across groups is a real modeling choice, not a
        law of nature -- it implicitly assumes every group is equally
        discriminative, which need not hold when groups are formed from a
        heavy-tailed feature-importance ranking (see
        ``utils.split_into_groups``'s docstring): a group built mostly
        from the lower-ranked end of that ranking can end up with little
        discriminative signal, especially for a family/reducer that kept
        a wide, weakly-filtered feature set (e.g. ``wide_selection``).
        This method exposes the per-group building block ``predict_proba``
        itself averages, so that averaging strategy can be revisited
        offline (e.g. weighting groups by held-out group-level accuracy,
        or by their features' mean ``feature_importance_``) without
        needing to refit anything -- it does not itself change how
        ``predict_proba``/``predict`` behave; uniform averaging remains
        this class's actual prediction rule unless/until that's revisited.

        Returns
        -------
        ndarray of shape (n_groups_, n_samples, n_classes_)
            ``result[i]`` is group ``i``'s (``self.groups_[i]`` /
            ``self._fitted_groups_[i]``) own class-probability estimate,
            already realigned to the shared 0..n_classes_-1 column order
            (see ``_group_proba``) and already row-normalized (each
            group's own ``predict_proba`` rows already sum to 1, or are a
            one-hot vote when ``has_proba`` is False -- see
            ``_group_proba``). ``result.mean(axis=0)`` reproduces what
            ``predict_proba(X)`` computes before its own defensive
            re-normalization (a no-op in the normal case where every
            group's rows already sum to 1).
        """
        X = np.asarray(X)
        n_classes = len(self.classes_)
        return np.stack(
            [self._group_proba(group, X, n_classes) for group in self._fitted_groups_],
            axis=0,
        )

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self._label_encoder_.inverse_transform(idx)
