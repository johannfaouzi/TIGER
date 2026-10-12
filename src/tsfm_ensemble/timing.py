"""Tiny timing helper shared across the package.

Every class that does discrete, separately-interesting steps (feature
extraction, pruning, dimensionality reduction, TabICLv2 fit/predict, branch
fusion, ...) records how long each step took in a plain ``dict[str, float]``
attribute -- ``fit_timings_`` after ``fit``/``fit_transform``,
``predict_timings_`` after ``predict``/``predict_proba`` -- rather than
requiring a bespoke profiling framework wrapped around the outside. The
"clean" outer class (e.g. ``pipeline.MultiFamilyTSCEnsemble``) still just
does ``clf.fit(X, y)`` / ``clf.predict(X)``; the timings are extra
attributes to inspect afterward, at whatever level of granularity you want,
by following the same object composition the classes already have:

    clf.fit(X_train, y_train)
    clf.fit_timings_                                          # top-level fusion steps
    clf.fusion_.fitted_branches_["convolution"].fit_timings_  # one branch's own steps
    clf.fusion_.fitted_branches_["convolution"].backbone_.fit_timings_  # per TabICLv2 group

Each ``*_timings_`` dict holds the *latest* call's per-step wall-clock time
in seconds, in-place-overwritten on every call -- not a cumulative total
across repeated calls. That is the simplest thing that answers "where did
the time go" for one ``fit``/``predict`` call, which is what the report's
ablation ladder (Sec. 6/7) actually needs; if you need timings across many
calls (e.g. one row per UCR dataset in a benchmark loop), read the dict
right after each call and append it to your own list/DataFrame -- don't
rely on these objects to accumulate history for you.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timed(timings: Dict[str, float], key: str):
    """Record wall-clock seconds spent in the ``with`` block into ``timings[key]``.

    ``timings`` is mutated in place -- usually a class's own ``fit_timings_``
    / ``predict_timings_`` dict -- so it's fine for a timed block to itself
    call into another object that populates *its own*, separate
    ``fit_timings_``/``predict_timings_`` dict: the two nest naturally
    without this helper needing to know about that structure at all. A
    ``return`` or a raised exception inside the ``with`` block still records
    the elapsed time before propagating (the timer is recorded in a
    ``finally``), so a failed step's time up to the failure is not lost.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - start
