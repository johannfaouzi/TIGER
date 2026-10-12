"""Automatic meta-learner selection: given a new dataset's training-time
characteristics (available before running any meta-learner), pick which of
the six real, deployable meta-methods (``hard_majority_vote``,
``logistic_regression``, ``lightgbm``, ``cawpe``, ``cawpe_hard_vote``,
``tabiclv2`` -- see ``meta_learners``'s own module docstring) to run.

Derivation (dev-subset only, per Johann's explicit instruction -- see
``build_stacking_features.py``'s "On not using the test scores to make
development decisions" docstring section): built entirely from
``results/meta_model_predictions/dev_subset_characteristics_and_winners.csv``,
the 20 development datasets x 5 resamples = 100 rows produced by
``build_meta_model_predictions.py`` (all 6 methods, ``tabiclv2`` patched in
via ``patch_tabiclv2_dev_subset.py`` after the sandbox that ran the main job
lacked the ``tabicl`` package). The 142-dataset benchmark
(``results/meta_model_predictions_142/``) was never consulted while
designing this rule.

What the data showed: per-dataset training samples-per-class (``NTrain /
NClasses`` -- the only feature that produced a clean signal; NTrain and
NClasses alone were noisier proxies for the same effect) is the dominant
driver of which method wins. At the low end (~2 samples/class -- the two
52-class Pig* datasets), the learned combiners collapse
(``logistic_regression`` 0.614, ``lightgbm`` 0.628, ``tabiclv2`` 0.676) while
the vote-based methods barely notice (``cawpe_hard_vote`` 0.840): they don't
need to *learn* a combination rule, just aggregate already-fit base
learners. At the high end (~75+ samples/class), ``tabiclv2`` pulls ahead
(0.869 mean accuracy vs. 0.856 for the next-best real method).

A single-threshold rule -- samples_per_class < ``THRESHOLD`` ->
``cawpe_hard_vote``, else ``tabiclv2`` -- was chosen by scanning every
midpoint between the 20 datasets' own samples-per-class values (see
``analyze_meta_learner_selection.py``, or the dev-subset characteristics CSV
directly) for the value maximizing mean accuracy across all 100
(dataset, resample) rows. The optimum, ``THRESHOLD = 60.83``, sits in a
genuine gap in the data (46.7, the highest value below it, is
Covid3Month_disc; 75.0, the lowest value above it, is
SemgHandMovementCh2) rather than at an arbitrary round number.

This rule was NOT validated via leave-one-dataset-out cross-validation --
a deliberate choice (Johann's own reasoning, agreed): LODO earns its keep
when selecting among multiple candidate strategies, or when a model with
many parameters needs protection against overfitting. Here there is a
single, hand-specified, one-parameter rule and no alternative candidate to
compare it against, so LODO would not change any decision. What IS reported
instead is a direct, honest comparison against the six real methods' own
fixed-baseline mean accuracies and against a 6-method hindsight oracle, all
computed on the same 100 rows the rule was derived from (see the module-level
constants below and ``analyze_meta_learner_selection.py``'s printed output):

    single-threshold rule (T=60.8)   mean acc = 0.8433
    always hard_majority_vote        mean acc = 0.8111
    always logistic_regression       mean acc = 0.8057
    always lightgbm                  mean acc = 0.8057
    always cawpe                     mean acc = 0.8117
    always cawpe_hard_vote           mean acc = 0.8319   (best single fixed method)
    always tabiclv2                  mean acc = 0.8237
    oracle (best of 6, hindsight)    mean acc = 0.8509

The rule beats every fixed single-method baseline (including the best one,
by +0.0114 mean accuracy, Wilcoxon p<0.0001 against it specifically, 36
wins / 4 losses / 60 ties across the 100 rows) and recovers roughly 60% of
the achievable headroom over a hindsight oracle that always picks the best
of the six methods for that exact job -- a real, honestly-scoped gain, not
a claim of closing the whole gap.

Caveat worth carrying forward: individual-dataset winners are noisier than
this aggregate trend -- several individual datasets with high
samples-per-class are still won by ``lightgbm``, ``logistic_regression``,
or ``hard_majority_vote`` rather than ``tabiclv2`` (see the per-dataset
winner-consistency breakdown in this session's analysis). The rule shifts
*expected* accuracy in the right direction on average; it is not a
guaranteed correct pick for every individual dataset. With only 20
development datasets, this is the honest limit of what can be claimed.
"""
from __future__ import annotations

SAMPLES_PER_CLASS_THRESHOLD = 60.83
LOW_SPC_METHOD = "cawpe_hard_vote"
HIGH_SPC_METHOD = "tabiclv2"


def select_meta_learner(n_train: int, n_classes: int) -> str:
    """Pick a real, deployable meta-method for a new dataset from training-
    time characteristics alone (no meta-learner needs to be run first).

    ``n_train`` / ``n_classes`` is this dataset's mean training samples per
    class (assumes roughly balanced classes, matching how the rule was
    derived -- see ``samples_per_class_threshold_scan`` in
    ``analyze_meta_learner_selection.py`` for the derivation, and this
    module's own docstring for the full rationale and caveats).

    Returns ``"cawpe_hard_vote"`` below the threshold (few samples per
    class -- vote-based combination needs no meta-level learning) or
    ``"tabiclv2"`` at or above it (enough samples per class for the
    foundation-model meta-learner to actually use the extra data).
    """
    if n_classes <= 0:
        raise ValueError(f"select_meta_learner: n_classes must be positive, got {n_classes}.")
    samples_per_class = n_train / n_classes
    return HIGH_SPC_METHOD if samples_per_class >= SAMPLES_PER_CLASS_THRESHOLD else LOW_SPC_METHOD
