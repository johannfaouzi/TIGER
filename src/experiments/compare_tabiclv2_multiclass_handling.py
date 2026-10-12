"""Harness: measure TabICLv2's native mixed-radix many-class handling
against its ECOC-wrapped variant (``tsfm_ensemble.backbone.GroupedICLClassifier``'s
``ecoc=False`` vs ``ecoc=True`` -- see that module's own docstring) on the
dev subset's own >10-class datasets, before picking one as the fixed
default for ``build_meta_model_predictions.py``'s ``--tabiclv2-ecoc`` flag.

Why this needs measuring here, not just assuming the paper's result
transfers
---------------------------------------------------------------------------
The TabICLv2 paper (arXiv 2602.11139, Sec. 6 / Fig. 7 / App. A.3.4) reports
that native+ECOC "substantially outperforms" native alone on TALENT-
benchmark datasets with >10 classes (paper text: the ECOC wrapper is
"slightly better but 3x slower" than native handling alone) -- but that
result is on generic tabular TALENT datasets. ``GroupedICLClassifier(ecoc=True)``
here defaults to ``ecoc_backend="many_class"``
(``tabpfn_extensions.many_class.ManyClassClassifier``, believed to be the
actual "TabPFNv2 ECOC wrapper" the paper's own text says it supports --
see ``backbone.py``'s module docstring for the full source-level
verification); ``ecoc_backend="sklearn"`` (the original implementation,
scikit-learn's generic ``OutputCodeClassifier``) is kept available for
comparison. This project's own meta-feature matrices (up to 12 base
learners x n_classes columns, itself an ensemble's score outputs rather
than raw tabular features, additionally split into TabICLv2-sized groups
and averaged) are different enough from TALENT's own setting that Johann
asked to measure both directly rather than assume the paper's finding
carries over -- exactly what this script does.

Test-set peeking, acknowledged
---------------------------------
This is the one deliberate exception to this branch's "OOF-only for
development decisions" discipline (see ``build_stacking_features.py``'s
own docstring, and ``meta_learners.py``'s "Meta-level leakage discipline"
section): choosing which of TabICLv2's many-class handling modes to use
going forward is an architecture decision, and this script measures it on
the dev subset's real held-out test sets, the same way
``build_stacking_features.py``'s docstring already flags as an acceptable,
narrow exception for this *kind* of choice -- not a per-learner or
per-hyperparameter tuning loop. The same caution that docstring raises
about MoteStrain/CinCECGTorso/FacesUCR (whose "test set" is essentially
the same ~90%+ of pooled samples across all 5 resamples, so their
test-set signal is far less diluted by repeated peeking) applies here too
if any of them happen to have >10 classes.

Scope
-----
Restricted to dev-subset datasets with more than 10 classes -- the paper's
own qualifying regime for its native-vs-ECOC comparison (Sec. 6); below
that, ``GroupedICLClassifier`` doesn't even use ECOC (see its own
``use_ecoc = self.ecoc and n_classes > 2`` guard, and Fig. 7's own
TALENT-dataset selection). Which dev-subset datasets qualify is
determined empirically per job from that job's own ``scores.npz``
(``classes_``'s length), not hardcoded -- printed once at start-up.

Output layout, under RESULTS_DIR (results/compare_tabiclv2_multiclass_handling/)
------------------------------------------------------------------------------------
    units/<dataset>/resample<k>/manifest.csv
        The layout for a run at the exact canonical default config
        (``group_size=100``, ``ecoc_backend="many_class"``,
        ``ecoc_alphabet_size=10``, ``ecoc_redundancy=4``,
        ``ecoc_code_size=1.5``, ``stratify_groups_by_class=True`` -- see
        ``_DEFAULT_RUN_CONFIG``). Two rows (Ecoc=False, Ecoc=True):
        Dataset, Resample, Ecoc, EcocBackend/EcocAlphabetSize/
        EcocRedundancy/EcocCodeSize (None for the Ecoc=False row),
        GroupSize, StratifyGroupsByClass, NClasses, NTrain, NTest,
        FitPredictSeconds, Accuracy, Error.
    units/<tag>/<dataset>/resample<k>/manifest.csv
        Same, for any run whose config differs from the canonical
        default in at least one of those six knobs -- ``<tag>`` is
        ``_run_config_tag(...)``'s deterministic encoding of exactly
        which knobs differ (e.g. ``group_size=1000``). This is what lets
        two differently-configured runs (e.g. comparing
        ``--group-size 100`` against ``--group-size 1000``) coexist and
        both stay independently resumable instead of one silently
        clobbering or being skipped-as-already-done because of the
        other -- see ``_run_config_tag``'s own docstring for the full
        reasoning, including why this matters even if you've only ever
        varied ``--ecoc-backend`` so far.
    manifest.csv
        All completed jobs' manifest fragments (from every config tag,
        default or not), concatenated by ``finalize()`` (safe to re-run
        standalone) -- this is the file that persists past this process
        exiting: every per-job Accuracy/FitPredictSeconds/Error, one row
        per (dataset, resample, Ecoc, GroupSize, StratifyGroupsByClass,
        EcocBackend), suitable for a paper's own table/appendix (group by
        those same columns for the mean-accuracy/mean-timing summary
        ``finalize()`` prints to stdout, or keep it per-dataset for a
        finer breakdown). This is also the exact file
        ``build_meta_model_predictions.py``'s ``--tabiclv2-ecoc auto``
        (its default) reads to decide which mode to use -- see that
        module's own docstring and ``_auto_select_tabiclv2_ecoc`` (which
        filters to one ``GroupSize`` before comparing Ecoc/EcocBackend,
        since this manifest can now hold more than one).

Parallelism and memory
------------------------
Unlike ``build_stacking_features.py`` (cheap sklearn fits), every job here
fits real TabICLv2 *twice* (native + ECOC), and ``GroupedICLClassifier``
loads a model checkpoint per feature-group of each fit (see ``backbone.py``'s
own "RAM/VRAM over a long run" docstring section) -- reusing that other
script's ``cpu_count - 1`` worker-count default here means running that
many heavy TabICLv2 fits simultaneously, which is what produced a
``concurrent.futures.process.BrokenProcessPool`` ("A process in the process
pool was terminated abruptly") in practice on an 80-core/503GB machine:
that error means a worker was killed by the OS mid-fit (almost certainly an
OOM kill -- it isn't a catchable Python exception, which one *failing*
TabICLv2 fit already would be, and already is handled per-row above), not
a bug in this script's own logic. ``DEFAULT_MAX_TABICLV2_WORKERS`` caps the
default worker count well below ``cpu_count - 1`` accordingly -- an
**unmeasured, conservative starting guess** (this project hasn't profiled
real TabICLv2's actual per-fit memory footprint on this environment: the
one place that could actually run real ``tabicl`` while this cap was being
picked had no network route to its Hugging Face Hub checkpoint), safe to
raise via ``--max-workers`` if you watch memory (``free -h`` / ``htop``
while a run is going) and see headroom, or lower further if you still see
kills.

What *is* known (source-verified from the real installed ``tabicl==2.2.0``,
``tabpfn-extensions`` and ``scikit-learn`` packages, not measured) about
what drives one job's peak RAM here:

    * ECOC mode holds several separate backbone instances *simultaneously*
      per feature-group, vs. native mode's one -- but exactly how many
      depends heavily on ``--ecoc-backend``:

      - ``many_class`` (default): ``ManyClassClassifier``'s own codebook
        formula (``max(log_alphabet(n_classes), n_classes/alphabet_size) *
        redundancy``, ``--ecoc-redundancy`` default 4, capped near
        ``4*log_alphabet(n_classes)``) -- e.g. a 52-class dataset at the
        default ``--ecoc-alphabet-size 10`` needs only 8 instances per
        group, not 78. Much less memory pressure than the sklearn backend
        below, and the recommended backend for both accuracy (see the
        docstring above) and memory reasons.
      - ``sklearn``: ``int(n_classes * ecoc_code_size)`` (default
        ``--ecoc-code-size 1.5``) instances per group -- sklearn's own
        ``OutputCodeClassifier`` keeps every one of them for prediction
        (see its ``estimators_`` attribute docstring). For a 52-class
        dataset that's ``int(52*1.5)=78`` instances per group -- ECOC jobs
        on many-class datasets under this backend are the ones most
        likely to trigger a ``BrokenProcessPool`` even at a worker count
        that's fine for native jobs.

      Either way, each retained instance is an independent, unshared copy
      of TabICLv2's weights (``TabICLClassifier._load_model()`` always
      allocates+loads fresh, confirmed from source).
    * Within each backbone instance, ``tabicl``'s own ``offload_mode``
      docstring names a ``(batch_size, n_rows, n_columns, embed_dim)``
      "column-wise embedding" tensor as "the main memory bottleneck" --
      scaling with ``batch_size`` (default 8, ``--batch-size`` here),
      ``n_rows`` (train+test row count) and ``n_columns`` (per-group
      width, capped at ``--group-size``, default 100). ``--offload-mode
      disk`` (with ``--disk-offload-dir``) is a real, paper-documented
      (arXiv 2602.11139, Appendix H, "Offloading technique") mechanism
      that memory-maps this specific tensor to disk instead of holding it
      in RAM -- a higher-leverage lever than lowering worker count when
      *this* tensor, not the number of retained models, is the binding
      constraint (large ``n_rows`` even in native mode).

Neither of these gives an absolute MB number -- that depends on this
machine's actual ``tabicl``/torch build and this project's actual input
shapes. ``src/experiments/profile_tabiclv2_memory.py`` runs real TabICLv2
fits (both modes, real or synthetic shapes) on *this* machine and reports
measured peak RSS per job, plus a suggested ``--max-workers`` for a given
total-RAM budget -- run it once here (where the checkpoint download is
reachable, unlike the sandboxed environment this project was partly built
in) before raising ``--max-workers``/lowering ``--ecoc-code-size``/
choosing ``--offload-mode`` by feel.

Usage
-----
    python -m src.experiments.compare_tabiclv2_multiclass_handling
    python -m src.experiments.compare_tabiclv2_multiclass_handling --max-workers 4
    python -m src.experiments.compare_tabiclv2_multiclass_handling --min-classes 11
    python -m src.experiments.compare_tabiclv2_multiclass_handling --device cpu
    python -m src.experiments.compare_tabiclv2_multiclass_handling --device cpu \\
        --offload-mode disk --disk-offload-dir /tmp/tabicl_offload
    python -m src.experiments.compare_tabiclv2_multiclass_handling --device cpu \\
        --ecoc-backend sklearn --ecoc-code-size 1.0

    # Compare --group-size 100 (the default) against a much larger value
    # (e.g. testing a single group covering every meta-feature column) --
    # each is its own run, safe to run one after the other (or in
    # parallel, in separate invocations): they land in independently
    # resumable output subtrees (see this module's "Output layout"
    # section and _run_config_tag) and both end up as separate rows in
    # the same aggregated manifest.csv, directly comparable by MeanAccuracy.
    python -m src.experiments.compare_tabiclv2_multiclass_handling --device cpu
    python -m src.experiments.compare_tabiclv2_multiclass_handling --device cpu --group-size 1000

    # Compare class-stratified (default) against plain random group
    # assignment -- see backbone.py's "Grouping" docstring section.
    python -m src.experiments.compare_tabiclv2_multiclass_handling --device cpu \\
        --no-stratify-groups-by-class
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from src.experiments.common import dev_subset_datasets, detect_cpu_count  # noqa: E402
from src.tsfm_ensemble.base_learners import RANDOM_STATE  # noqa: E402
from src.tsfm_ensemble.meta_learners import assemble_meta_features, fit_predict_tabiclv2  # noqa: E402

STACKING_FEATURES_RESULTS_DIR = Path(
    os.environ.get("STACKING_FEATURES_RESULTS_DIR", str(Path("results") / "stacking_features"))
)
RESULTS_DIR = Path(
    os.environ.get(
        "COMPARE_TABICLV2_RESULTS_DIR", str(Path("results") / "compare_tabiclv2_multiclass_handling")
    )
)

NUM_RESAMPLES = 5
RESERVED_CORES = 1
DEFAULT_MIN_CLASSES = 11  # ">10 classes", matching the paper's own regime

# See this module's docstring's "Parallelism and memory" section -- a
# conservative cap on default worker count specifically because every job
# here fits real TabICLv2 twice (unlike build_stacking_features.py's cheap
# sklearn fits), which is heavy enough that cpu_count - 1 parallel fits
# produced a BrokenProcessPool (OS-killed worker, almost certainly OOM) in
# practice. Override with --max-workers if you've checked memory headroom.
DEFAULT_MAX_TABICLV2_WORKERS = 8


def _scores_path(dataset: str, resample: int) -> Path:
    return STACKING_FEATURES_RESULTS_DIR / dataset / f"resample{resample}" / "scores.npz"


def _n_classes(dataset: str, resample: int) -> Optional[int]:
    path = _scores_path(dataset, resample)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as scores:
        return int(len(scores["classes_"]))


def qualifying_jobs(datasets: List[str], min_classes: int) -> List["Job"]:
    """Every (dataset, resample) with >= ``min_classes`` classes and an
    already-built ``scores.npz`` to read -- printed once so it's obvious
    which datasets this run actually covers.
    """
    jobs = []
    for dataset in datasets:
        for resample in range(NUM_RESAMPLES):
            n_classes = _n_classes(dataset, resample)
            if n_classes is not None and n_classes >= min_classes:
                jobs.append(Job(dataset=dataset, resample=resample))
    return jobs


# The exact argparse defaults for every knob that actually changes what's
# being measured (as opposed to how it's computed -- see
# ``_run_config_tag``'s own docstring for the device/batch_size/
# offload_mode distinction). A run matching this exactly keeps the
# original, pre-existing ``units/<dataset>/resample<k>/`` output layout
# (no extra path segment) -- see ``_run_config_tag``.
_DEFAULT_RUN_CONFIG = {
    "group_size": 100,
    "ecoc_backend": "many_class",
    "ecoc_alphabet_size": 10,
    "ecoc_redundancy": 4,
    "ecoc_code_size": 1.5,
    "stratify_groups_by_class": True,
}


def _run_config_tag(**config) -> str:
    """A short, filesystem-safe tag identifying a non-default run
    configuration for use as an extra path segment in ``_paths`` -- empty
    string for the exact canonical default (``_DEFAULT_RUN_CONFIG``), so a
    plain default-args run keeps this script's original output layout
    unchanged (in particular: every job already completed on the server
    under that layout stays valid, and this change doesn't silently
    re-run or orphan it).

    Why this exists: before this, ``_paths``/``_is_done`` keyed a job's
    output directory (and its ``DONE`` marker) by ``(dataset, resample)``
    alone -- completely blind to ``--group-size``/``--ecoc-backend``/etc.
    Re-running this script with a different value of any of those for a
    (dataset, resample) job already completed under a *different* value
    would either get silently skipped (the ``DONE`` marker already
    exists, regardless of which config produced it) or, if it did run,
    silently overwrite that job's ``manifest.csv`` in place -- either way
    losing the ability to compare two configs against each other, which
    is exactly what this script is for (see its own module docstring).
    If this script was run more than once with different
    ``--group-size``/``--ecoc-backend``/etc. values for overlapping
    (dataset, resample) jobs *before* this fix, check
    ``results/compare_tabiclv2_multiclass_handling/units/*/resample*/manifest.csv``
    for which config's numbers actually ended up there -- they may be
    from whichever run happened to go second, not a genuine per-config
    result.

    Only ``group_size``, ``ecoc_backend``, ``ecoc_alphabet_size``,
    ``ecoc_redundancy``, ``ecoc_code_size`` and ``stratify_groups_by_class``
    participate. ``--device``/``--batch-size``/``--offload-mode``/
    ``--disk-offload-dir`` deliberately don't -- those are execution/memory
    levers (see this module's "Parallelism and memory" docstring section),
    not expected to change the comparison being measured, and treating
    them as config-distinguishing would fragment results (and force
    re-runs) over what's meant to be an invisible implementation detail.
    """
    if config == _DEFAULT_RUN_CONFIG:
        return ""
    parts = [f"{k}={config[k]}" for k in sorted(config)]
    tag = "_".join(parts).replace(" ", "")
    if len(tag) > 100:
        import hashlib

        tag = "cfg-" + hashlib.sha1(tag.encode()).hexdigest()[:16]
    return tag


def _paths(dataset: str, resample: int, run_config_tag: str = ""):
    base = RESULTS_DIR / "units"
    if run_config_tag:
        base = base / run_config_tag
    return base / dataset / f"resample{resample}"


def _is_done(dataset: str, resample: int, run_config_tag: str = "") -> bool:
    job_dir = _paths(dataset, resample, run_config_tag)
    return (job_dir / "DONE").is_file() and (job_dir / "manifest.csv").is_file()


def _process_job(
    dataset: str,
    resample: int,
    random_state: int,
    backbone_kwargs: Optional[dict],
    group_size: int = 100,
    stratify_groups_by_class: bool = True,
    ecoc_backend: str = "many_class",
    ecoc_alphabet_size: int = 10,
    ecoc_redundancy: int = 4,
    ecoc_code_size: float = 1.5,
    run_config_tag: str = "",
) -> dict:
    job_dir = _paths(dataset, resample, run_config_tag)
    job_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        with np.load(_scores_path(dataset, resample), allow_pickle=False) as scores:
            scores_dict = dict(scores.items())
        meta = assemble_meta_features(scores_dict)

        rows = []
        for ecoc in (False, True):
            row_start = time.perf_counter()
            try:
                fit_start = time.perf_counter()
                proba = fit_predict_tabiclv2(
                    meta, ecoc=ecoc, group_size=group_size,
                    stratify_groups_by_class=stratify_groups_by_class, ecoc_backend=ecoc_backend,
                    ecoc_alphabet_size=ecoc_alphabet_size, ecoc_redundancy=ecoc_redundancy,
                    ecoc_code_size=ecoc_code_size, random_state=random_state,
                    backbone_kwargs=backbone_kwargs,
                )
                elapsed = time.perf_counter() - fit_start
                acc = accuracy_score(meta.y_test_enc, proba.argmax(axis=1))
                rows.append({
                    "Dataset": dataset, "Resample": resample, "Ecoc": ecoc,
                    "EcocBackend": ecoc_backend if ecoc else None,
                    "GroupSize": group_size, "StratifyGroupsByClass": stratify_groups_by_class,
                    "EcocAlphabetSize": ecoc_alphabet_size if ecoc else None,
                    "EcocRedundancy": ecoc_redundancy if ecoc else None,
                    "EcocCodeSize": ecoc_code_size if ecoc else None,
                    "NClasses": meta.n_classes, "NTrain": meta.X_train.shape[0],
                    "NTest": meta.X_test.shape[0], "FitPredictSeconds": elapsed,
                    "Accuracy": acc, "Error": None,
                })
            except Exception as exc:  # noqa: BLE001 - one mode failing must not sink the other
                rows.append({
                    "Dataset": dataset, "Resample": resample, "Ecoc": ecoc,
                    "EcocBackend": ecoc_backend if ecoc else None,
                    "GroupSize": group_size, "StratifyGroupsByClass": stratify_groups_by_class,
                    "EcocAlphabetSize": ecoc_alphabet_size if ecoc else None,
                    "EcocRedundancy": ecoc_redundancy if ecoc else None,
                    "EcocCodeSize": ecoc_code_size if ecoc else None,
                    "NClasses": meta.n_classes, "NTrain": meta.X_train.shape[0],
                    "NTest": meta.X_test.shape[0], "FitPredictSeconds": time.perf_counter() - row_start,
                    "Accuracy": None, "Error": repr(exc),
                })

        pd.DataFrame(rows).to_csv(job_dir / "manifest.csv", index=False)
        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        return {
            "dataset": dataset, "resample": resample, "status": "done",
            "elapsed": time.perf_counter() - start,
        }
    except Exception as exc:  # noqa: BLE001
        (job_dir / "ERROR.txt").write_text(f"{exc!r}\n\n{traceback.format_exc()}")
        return {
            "dataset": dataset, "resample": resample, "status": "error",
            "elapsed": time.perf_counter() - start, "error": repr(exc),
        }


def finalize() -> None:
    # Recursive: covers both the untagged default-config layout
    # (units/<dataset>/resample<k>/manifest.csv) and any tagged
    # non-default-config layout (units/<tag>/<dataset>/resample<k>/
    # manifest.csv) -- see _run_config_tag.
    fragments = sorted(RESULTS_DIR.glob("units/**/manifest.csv"))
    if not fragments:
        print("No completed jobs found; nothing to aggregate.")
        return
    combined = pd.concat([pd.read_csv(p) for p in fragments], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    print(f"Aggregated {len(fragments)} completed job(s) -> {RESULTS_DIR / 'manifest.csv'}")

    # dropna=False: EcocBackend/EcocAlphabetSize/EcocRedundancy/EcocCodeSize
    # are None for every Ecoc=False row by design (see _process_job) -- the
    # default groupby would otherwise silently drop that whole native-mode
    # group from the summary. GroupSize/StratifyGroupsByClass are always
    # set (both Ecoc=True and Ecoc=False rows of the same job share them),
    # included here specifically so results from different --group-size/
    # --stratify-groups-by-class runs (now safely coexisting in the same
    # manifest.csv, see _run_config_tag) are never silently averaged
    # together -- a manifest.csv.fillna("sklearn")-style backward-compat
    # read for old rows without these columns belongs in whatever script
    # consumes this file (e.g. build_meta_model_predictions.py's
    # _auto_select_tabiclv2_ecoc), not here.
    summary = combined.groupby(
        ["GroupSize", "StratifyGroupsByClass", "Ecoc", "EcocBackend"], dropna=False
    ).agg(
        MeanAccuracy=("Accuracy", "mean"),
        MeanFitPredictSeconds=("FitPredictSeconds", "mean"),
        NErrors=("Error", lambda s: s.notna().sum()),
        NJobs=("Accuracy", "size"),
    )
    print(
        "\nNative (Ecoc=False) vs ECOC (Ecoc=True, by EcocBackend), by GroupSize/"
        "StratifyGroupsByClass, averaged across qualifying (dataset, resample) jobs:"
    )
    print(summary.to_string())


@dataclass
class Job:
    dataset: str
    resample: int


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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help=(
            f"Override the default worker count (min(cpu_count - {RESERVED_CORES}, "
            f"{DEFAULT_MAX_TABICLV2_WORKERS}) -- see this module's docstring's 'Parallelism and "
            "memory' section for why the default is capped well below cpu_count for this script)."
        ),
    )
    parser.add_argument(
        "--min-classes", type=int, default=DEFAULT_MIN_CLASSES,
        help=f"Only compare on (dataset, resample) jobs with at least this many classes (default {DEFAULT_MIN_CLASSES}).",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu"), default="auto",
        help=(
            "'cpu' forces TabICLv2 onto CPU (backbone_kwargs={'device': 'cpu'}, passed straight "
            "through to TabICLClassifier -- see backbone.py's make_backbone) instead of leaving "
            "device selection to tabicl/torch's own auto-detection. 'auto' (default) doesn't pass "
            "backbone_kwargs at all."
        ),
    )
    parser.add_argument(
        "--group-size", type=int, default=100,
        help=(
            "Max columns per TabICLv2 feature-group (GroupedICLClassifier's own default is 100 -- "
            "see backbone.py). Fewer, wider groups mean fewer retained backbone instances but a "
            "larger per-instance embedding tensor; more, narrower groups is the opposite tradeoff -- "
            "see this module's docstring's 'Parallelism and memory' section."
        ),
    )
    parser.add_argument(
        "--no-stratify-groups-by-class", dest="stratify_groups_by_class", action="store_false", default=True,
        help=(
            "Disable class-stratified TabICLv2 group assignment (see meta_learners.fit_predict_tabiclv2's "
            "and backbone.py's 'Grouping' docstring section) -- reverts to the original random group "
            "assignment, which gives no guarantee every group keeps some direct evidence (one column per "
            "learner) for every class. Default: stratify (on)."
        ),
    )
    parser.add_argument(
        "--ecoc-backend", choices=("many_class", "sklearn"), default="many_class",
        help=(
            "Which ECOC implementation the Ecoc=True fit uses. 'many_class' (default) is "
            "tabpfn_extensions.many_class.ManyClassClassifier -- believed to be the actual "
            "'TabPFNv2 ECOC wrapper' the TabICLv2 paper's own text says it supports (requires the "
            "tabpfn-extensions package). 'sklearn' is the original scikit-learn OutputCodeClassifier "
            "implementation, kept for comparison -- see this module's and backbone.py's docstrings."
        ),
    )
    parser.add_argument(
        "--ecoc-alphabet-size", type=int, default=10,
        help=(
            "'many_class' backend only: the base estimator's own max classes-per-fit limit "
            "(ManyClassClassifier's own alphabet_size). Default 10 matches TabICLv2's own pretrained "
            "max_classes."
        ),
    )
    parser.add_argument(
        "--ecoc-redundancy", type=int, default=4,
        help="'many_class' backend only: forwarded to ManyClassClassifier's own n_estimators_redundancy (its own default is 4).",
    )
    parser.add_argument(
        "--ecoc-code-size", type=float, default=1.5,
        help=(
            "'sklearn' backend only: forwarded to OutputCodeClassifier(code_size=...) for the "
            "Ecoc=True fit (sklearn's own default is 1.5). Each group then holds "
            "int(n_classes * ecoc_code_size) backbone instances simultaneously; see this module's "
            "docstring's 'Parallelism and memory' section."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=(
            "Forwarded to TabICLClassifier(batch_size=...) (tabicl's own default is 8). Scales the "
            "per-instance 'column-wise embedding' tensor tabicl's own offload_mode docstring calls "
            "the main memory bottleneck. Default (unset) leaves tabicl's own default in place."
        ),
    )
    parser.add_argument(
        "--offload-mode", choices=("auto", "gpu", "cpu", "disk"), default=None,
        help=(
            "Forwarded to TabICLClassifier(offload_mode=...) -- controls where that embedding "
            "tensor is stored (tabicl's own default is 'auto'). 'disk' memory-maps it to "
            "--disk-offload-dir instead of holding it in RAM (paper-documented, arXiv 2602.11139 "
            "Appendix H) -- the highest-leverage lever when that tensor, not the number of retained "
            "ECOC instances, is the binding constraint. Default (unset) leaves tabicl's own default."
        ),
    )
    parser.add_argument(
        "--disk-offload-dir", type=str, default=None,
        help="Forwarded to TabICLClassifier(disk_offload_dir=...). Required when --offload-mode disk is set.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the qualifying job list, then exit without running anything.",
    )
    args = parser.parse_args(argv)
    if args.offload_mode == "disk" and not args.disk_offload_dir:
        parser.error("--offload-mode disk requires --disk-offload-dir.")
    backbone_kwargs = {}
    if args.device == "cpu":
        backbone_kwargs["device"] = "cpu"
    if args.batch_size is not None:
        backbone_kwargs["batch_size"] = args.batch_size
    if args.offload_mode is not None:
        backbone_kwargs["offload_mode"] = args.offload_mode
    if args.disk_offload_dir is not None:
        backbone_kwargs["disk_offload_dir"] = args.disk_offload_dir
    backbone_kwargs = backbone_kwargs or None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_config_tag = _run_config_tag(
        group_size=args.group_size, ecoc_backend=args.ecoc_backend,
        ecoc_alphabet_size=args.ecoc_alphabet_size, ecoc_redundancy=args.ecoc_redundancy,
        ecoc_code_size=args.ecoc_code_size, stratify_groups_by_class=args.stratify_groups_by_class,
    )

    datasets = dev_subset_datasets()
    all_jobs = qualifying_jobs(datasets, args.min_classes)
    jobs = [job for job in all_jobs if not _is_done(job.dataset, job.resample, run_config_tag)]
    print(
        f"{len(all_jobs)} (dataset, resample) job(s) with >= {args.min_classes} classes "
        f"across {len(datasets)} dev-subset dataset(s); {len(jobs)} remaining to run."
    )
    qualifying_datasets = sorted({job.dataset for job in all_jobs})
    print(f"Qualifying datasets: {qualifying_datasets}")
    print(
        f"Run config: group_size={args.group_size}, ecoc_backend={args.ecoc_backend}, "
        f"ecoc_alphabet_size={args.ecoc_alphabet_size}, ecoc_redundancy={args.ecoc_redundancy}, "
        f"ecoc_code_size={args.ecoc_code_size}, stratify_groups_by_class={args.stratify_groups_by_class} "
        f"-> output tag {run_config_tag!r} ({'default layout, units/<dataset>/resample<k>/' if not run_config_tag else 'units/' + run_config_tag + '/<dataset>/resample<k>/'})"
    )
    if not all_jobs:
        print(f"No dev-subset dataset has >= {args.min_classes} classes -- nothing to compare.")
        return
    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        for job in jobs:
            print(f"  {job.dataset} resample{job.resample}")
        print("--dry-run: exiting without running anything.")
        return

    cpu_count = detect_cpu_count()
    n_workers = args.max_workers or min(max(1, cpu_count - RESERVED_CORES), DEFAULT_MAX_TABICLV2_WORKERS)
    print(
        f"Using {n_workers} worker(s) (device={args.device}, group_size={args.group_size}, "
        f"stratify_groups_by_class={args.stratify_groups_by_class}, ecoc_backend={args.ecoc_backend}, "
        f"ecoc_alphabet_size={args.ecoc_alphabet_size}, ecoc_redundancy={args.ecoc_redundancy}, "
        f"ecoc_code_size={args.ecoc_code_size}, batch_size={args.batch_size}, "
        f"offload_mode={args.offload_mode}, disk_offload_dir={args.disk_offload_dir}) -- see this "
        "module's docstring's 'Parallelism and memory' section if you hit a BrokenProcessPool at this "
        "or a higher count."
    )

    import concurrent.futures as cf
    import multiprocessing

    mp_context = multiprocessing.get_context("spawn")
    completed, errored = 0, 0

    with cf.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp_context, initializer=_worker_init
    ) as executor:
        start_time = time.perf_counter()
        futures = {
            executor.submit(
                _process_job, job.dataset, job.resample, RANDOM_STATE, backbone_kwargs,
                args.group_size, args.stratify_groups_by_class, args.ecoc_backend,
                args.ecoc_alphabet_size, args.ecoc_redundancy, args.ecoc_code_size, run_config_tag,
            ): job
            for job in jobs
        }
        for future in cf.as_completed(futures):
            job = futures[future]
            result = future.result()
            if result["status"] == "done":
                completed += 1
                print(f"[{completed + errored}/{len(jobs)}] OK   {job.dataset} resample{job.resample} ({result['elapsed']:.1f}s)")
            else:
                errored += 1
                print(f"[{completed + errored}/{len(jobs)}] FAIL {job.dataset} resample{job.resample}: {result.get('error')}")
        elapsed = time.perf_counter() - start_time
        print(f"\nDone in {elapsed / 60:.1f} min: {completed} succeeded, {errored} failed.")

    finalize()


if __name__ == "__main__":
    main()
