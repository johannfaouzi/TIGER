"""Small shared helpers."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def split_into_groups(
    n_features: int,
    group_size: int,
    random_state: Optional[int] = None,
    importance: Optional[np.ndarray] = None,
    group_key: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Split ``n_features`` column indices into groups of at most
    ``group_size``, for TabICLv2's grouped-backbone trick (report Sec.
    5.4): rather than truncating a branch's features to fit one backbone
    call, run several backbone calls, each on a different subset of
    columns, and average the results (see ``backbone.py``).

    Three assignment strategies -- ``importance`` and ``group_key`` are
    mutually exclusive (pass at most one; both come from a different
    upstream notion of structure among the columns):

    - Neither given (the default -- used whenever the upstream reducer
      doesn't expose a ranking and the caller has no categorical column
      structure to preserve, see below): a *random* assignment -- shuffle
      all column indices, then take contiguous chunks of the shuffled
      order. Matters when the upstream features are not exchangeable --
      e.g. a reducer that returns features sorted by importance would
      otherwise put all the best features in the first group and leave
      the rest starved, if grouped positionally rather than after
      shuffling.
    - ``importance`` given (one score per column, ``0..n_features-1``,
      higher = more important -- e.g. ``reduction.PLSDASelector`` and
      ``reduction.ChiSquareThenSVD``'s components are already ordered by
      construction, and ``reduction.DetachRocketSelector`` /
      ``reduction.FDRUnivariateSelector`` expose their own selection
      statistic as ``feature_importance_``, see each class's docstring):
      a *stratified round-robin* assignment -- sort columns by descending
      importance, then deal them out one at a time across the groups
      (rank 0 to group 0, rank 1 to group 1, ..., rank ``n_groups`` back
      to group 0, and so on), same idea as dealing a sorted deck of cards
      evenly. This *guarantees* every group gets a representative spread
      across the full importance range, rather than only making that
      likely the way a random shuffle does -- checked empirically on real
      UCR data (ACSF1, convolution family): random assignment's
      best-group/worst-group mean-importance spread averaged ~10% of the
      overall mean importance across 200 seeds; stratified round-robin
      brought that down to ~3%, deterministically. Note this is a
      *different* index pattern from the random case -- contiguous chunks
      of an importance-*sorted* order would reproduce exactly the
      clustering problem this whole mechanism exists to avoid, so the
      stratified path deliberately uses a strided ``order[i::n_groups]``
      slice per group instead of ``np.array_split``.
    - ``group_key`` given (one *categorical* label per column,
      ``0..n_features-1`` -- e.g. ``meta_learners``'s per-column class
      index, see ``_class_group_key``): a *key-stratified round-robin*
      assignment -- for each distinct key value, shuffle that key's own
      member columns and deal them round-robin across the groups, using a
      random per-key starting offset (so that when a key's column count
      isn't a multiple of ``n_groups``, which groups get its "extra"
      column is randomized independently per key rather than always
      being the same low-index groups). This *guarantees* every group
      gets at least ``floor(count_for_that_key / n_groups)`` of that
      key's columns -- in particular, a group can only end up with
      *zero* columns for some key if that key has fewer member columns
      than there are groups (``count_for_that_key < n_groups``); pure
      random assignment gives no such guarantee at all (e.g. checked
      directly for this project's own tabiclv2 meta-feature layout: 60
      classes x 12 learners = 720 columns split into 8 groups of 90 under
      pure random assignment has a ~90% chance that *some* group ends up
      with literally none of a given class's 12 columns -- see
      ``backbone.py``'s ECOC/grouping module docstring for the full
      calculation). This is the mechanism ``meta_learners.py``'s
      ``fit_predict_tabiclv2`` uses (via ``GroupedICLClassifier``'s
      ``class_group_key``) so that no group is ever blind to a class that
      has at least ``n_groups`` learners' worth of direct evidence for it
      -- true for every dev-subset dataset at the default ``group_size``
      (at most 8 groups, always 12 learners).

    Each returned group's column indices are sorted ascending (a cosmetic,
    stable-ordering choice -- unrelated to the importance/key that decided
    group membership, which never affects each group's own internal
    column order).
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if importance is not None and group_key is not None:
        raise ValueError(
            "importance and group_key are mutually exclusive grouping "
            "strategies -- pass at most one."
        )
    n_groups = int(np.ceil(n_features / group_size))
    rng = np.random.RandomState(random_state)

    if group_key is not None:
        group_key = np.asarray(group_key)
        if group_key.shape != (n_features,):
            raise ValueError(
                f"group_key must have shape ({n_features},) to match "
                f"n_features, got {group_key.shape}."
            )
        groups: List[List[int]] = [[] for _ in range(n_groups)]
        # Sorted so iteration order (and hence how many rng draws happen
        # before each key is dealt) is deterministic given random_state,
        # independent of e.g. dict/set iteration order.
        for key in np.sort(np.unique(group_key)):
            cols = np.flatnonzero(group_key == key)
            cols = rng.permutation(cols)
            # A fresh random starting group per key: when this key's
            # column count isn't a multiple of n_groups, this is what
            # spreads which groups get the "extra" column across
            # different keys instead of always favouring group 0.
            offset = int(rng.randint(n_groups))
            for i, col in enumerate(cols):
                groups[(offset + i) % n_groups].append(int(col))
        return [np.sort(np.asarray(g, dtype=int)) for g in groups]

    if importance is not None:
        importance = np.asarray(importance)
        if importance.shape != (n_features,):
            raise ValueError(
                f"importance must have shape ({n_features},) to match "
                f"n_features, got {importance.shape}."
            )
        # Descending by importance; ties broken by a random shuffle
        # (rather than left in, e.g., original-column-index order) so
        # that features tied on this particular statistic still don't
        # end up positionally clustered by some other correlated
        # property (e.g. many chi2-tied dictionary words that happen to
        # come from the same SFA word-length configuration).
        tie_break = rng.permutation(n_features)
        order = np.lexsort((tie_break, -importance))
        return [np.sort(order[i::n_groups]) for i in range(n_groups)]

    order = rng.permutation(n_features)
    return [np.sort(chunk) for chunk in np.array_split(order, n_groups)]
