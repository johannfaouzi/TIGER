"""Build and cache the meta-model layer's final predictions for every
(dataset, resample) in the development subset, on top of
``build_stacking_features.py``'s already-saved ``scores.npz`` (the 12
(family, learner) base learners' OOF-train and test scores).

Runs all nine meta-methods this branch's design settled on (see
``src.tsfm_ensemble.meta_learners``'s module docstring for the full
rationale of each): ``hard_majority_vote``, ``logistic_regression``,
``lightgbm``, ``cawpe``, ``cawpe_hard_vote``, ``tabiclv2``,
``meta_oracle``, ``oracle_cv``, ``oracle``. Each method is attempted
independently per (dataset, resample)
-- one method failing (e.g. TabICLv2's checkpoint unreachable -- it has no
fallback, see ``fit_predict_tabiclv2``'s docstring -- or CAWPE's ridge
calibration failing on a pathological small dataset) is recorded as that
method's own error and does not take down the other seven, matching this
project's existing "one bad (sub-)problem must not take down the whole
job" convention (see
``base_learners.run_family_base_learners``'s and
``build_stacking_features.py``'s own docstrings).

TabICLv2's native-vs-ECOC many-class handling
------------------------------------------------
``--tabiclv2-ecoc`` selects which of TabICLv2's two many-class handling
modes this run uses. Default is ``"auto"``: read
``compare_tabiclv2_multiclass_handling.py``'s own aggregated
``manifest.csv`` (its ``finalize()`` output -- see that module's
docstring), compare native vs. ECOC's mean accuracy across whatever
qualifying (dataset, resample) jobs it has completed, and use whichever
scored higher (see ``_auto_select_tabiclv2_ecoc``). This is what makes
``compare_tabiclv2_multiclass_handling.py && build_meta_model_predictions.py``
work as a single unattended pipeline -- no manual step in between to read
the comparison and pass a flag. If that manifest doesn't exist yet (the
harness hasn't been run) or has no valid rows for either mode, this falls
back to native (``ecoc=False``, ``tabicl.TabICLClassifier``'s own default
internal mixed-radix handling) and says so loudly in the printed summary
-- a silent, unflagged fallback here would be exactly the kind of
un-auditable default this project has tried to avoid throughout (e.g. see
``base_learners.py``'s own approach to the same tension). Pass
``--tabiclv2-ecoc native`` or ``--tabiclv2-ecoc ecoc`` to override the
auto-selection and force one mode explicitly.

CAWPE and raw family features
--------------------------------
Unlike the other four methods (which only need ``scores.npz``), CAWPE
also needs each family's cached *raw* extractor output
(``save_dev_subset_transforms.py``'s ``raw/`` cache, loaded via
``src.experiments.common.load_raw_family_features``) to fit
``CalibratedClassifierCV`` on top of that family's ridge learner (see
``meta_learners.calibrate_ridge_family``) -- ``scores.npz`` only has the
*uncalibrated* ``decision_function`` margins, not a fitted model to
recalibrate. A family whose raw cache is missing or whose calibration
fails is simply left out of CAWPE's combination for that job (see
``meta_learners.cawpe_combine``'s docstring) rather than failing CAWPE
entirely.

Output layout, under RESULTS_DIR (results/meta_model_predictions/)
----------------------------------------------------------------------
    <dataset>/resample<k>/predictions.npz
        For every method that succeeded:
            "{method}__test_pred_enc"   (n_test,) encoded label predictions
            "{method}__test_proba"      (n_test, n_classes) -- every method
                                         except hard_majority_vote and
                                         cawpe_hard_vote, which have no
                                         probabilistic output.
        For every method that failed: "{method}__error" (string).
        Shared: "classes_", "y_test_enc", "y_train_enc" (copied straight
        from the input scores.npz, for convenience).
    units/<dataset>/resample<k>/
        Per-job bookkeeping: a ``DONE`` marker (resumable reruns) and a
        ``manifest.csv`` fragment (one row per method: Accuracy,
        FitSeconds, Error, EcocUsed / EcocBackend [tabiclv2 only, None
        when EcocUsed is False], NClasses, NTrain, NTest). ``ERROR.txt``
        instead of ``DONE`` if the whole job raised
        (e.g. the input scores.npz itself is missing/corrupt -- distinct
        from one *method* failing inside the job, which is recorded per
        row and does not fail the job).
    manifest.csv
        All completed jobs' manifest fragments, concatenated by
        ``finalize()`` (safe to re-run standalone).

Parallelism and memory
------------------------
Every job here includes a real TabICLv2 fit (one of the five methods) --
heavy enough (model checkpoint load, grouped-ensemble inference) that this
script's default worker count is capped well below ``cpu_count - 1`` (see
``compare_tabiclv2_multiclass_handling.py``'s docstring's own
"Parallelism and memory" section: that script hit a
``concurrent.futures.process.BrokenProcessPool`` -- an OS-killed worker,
almost certainly OOM -- at ``cpu_count - 1`` parallel TabICLv2 fits on an
80-core/503GB machine, and ``DEFAULT_MAX_TABICLV2_WORKERS`` is an
unmeasured, conservative placeholder, not a profiled number). That same
section also derives (source-verified, not measured) what drives one
job's peak RAM here -- ECOC mode's retained backbone instances per group
(``--ecoc-backend``/``--ecoc-alphabet-size``/``--ecoc-redundancy`` for the
default ``many_class`` backend, ``--ecoc-code-size`` for the ``sklearn``
backend, below), and the per-instance embedding tensor ``tabicl``'s own
``offload_mode`` docstring calls "the main memory bottleneck"
(``--batch-size``, ``--offload-mode``, ``--disk-offload-dir`` below) --
and points at ``src/experiments/profile_tabiclv2_memory.py`` for
measuring real peak RSS for these levers on your own data/machine before
raising ``--max-workers`` by feel. Only the ``tabiclv2`` method uses any
of ``--group-size``/``--ecoc-*``/``--batch-size``/``--offload-mode``/
``--disk-offload-dir`` -- the other four methods are plain, cheap sklearn/
LightGBM fits unaffected by them.

Usage
-----
    python -m src.experiments.build_meta_model_predictions
    python -m src.experiments.build_meta_model_predictions --max-workers 8
    python -m src.experiments.build_meta_model_predictions --tabiclv2-ecoc ecoc
    python -m src.experiments.build_meta_model_predictions --device cpu
    python -m src.experiments.build_meta_model_predictions --dry-run
    python -m src.experiments.build_meta_model_predictions --device cpu \\
        --offload-mode disk --disk-offload-dir /tmp/tabicl_offload
    python -m src.experiments.build_meta_model_predictions --device cpu \\
        --ecoc-backend sklearn --ecoc-code-size 1.0

    # Run the native-vs-ECOC comparison, then build predictions using
    # whichever mode it measured as better -- no manual step in between:
    python -m src.experiments.compare_tabiclv2_multiclass_handling --device cpu && \\
        python -m src.experiments.build_meta_model_predictions --device cpu
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Same reasoning as build_stacking_features.py's/save_dev_subset_transforms.py's
# identical block -- must happen before numpy (or anything that
# transitively imports it) is imported. See those modules' own top-of-file
# comments.
# --------------------------------------------------------------------------- #
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

from src.experiments.common import dev_subset_datasets, detect_cpu_count, load_raw_family_features  # noqa: E402
from src.experiments.compare_tabiclv2_multiclass_handling import (  # noqa: E402
    DEFAULT_MAX_TABICLV2_WORKERS,
    RESULTS_DIR as COMPARE_TABICLV2_RESULTS_DIR,
)
from src.tsfm_ensemble.base_learners import FAMILIES, RANDOM_STATE  # noqa: E402
from src.tsfm_ensemble.meta_learners import (  # noqa: E402
    accuracy_from_proba,
    assemble_meta_features,
    best_block_by_oof_accuracy,
    best_block_by_test_accuracy,
    build_cawpe_test_probabilities,
    calibrate_ridge_family,
    cawpe_combine,
    cawpe_hard_vote,
    fit_predict_lightgbm,
    fit_predict_logistic_regression,
    fit_predict_tabiclv2,
    hard_majority_vote,
    oracle_best_single_learner,
    oracle_by_oof_accuracy,
)

STACKING_FEATURES_RESULTS_DIR = Path(
    os.environ.get("STACKING_FEATURES_RESULTS_DIR", str(Path("results") / "stacking_features"))
)
RESULTS_DIR = Path(
    os.environ.get("META_MODEL_PREDICTIONS_RESULTS_DIR", str(Path("results") / "meta_model_predictions"))
)

NUM_RESAMPLES = 5
RESERVED_CORES = 1


def _paths(dataset: str, resample: int):
    job_dir = RESULTS_DIR / "units" / dataset / f"resample{resample}"
    scores_path = STACKING_FEATURES_RESULTS_DIR / dataset / f"resample{resample}" / "scores.npz"
    predictions_path = RESULTS_DIR / dataset / f"resample{resample}" / "predictions.npz"
    return job_dir, scores_path, predictions_path


def _is_done(dataset: str, resample: int) -> bool:
    job_dir, _, predictions_path = _paths(dataset, resample)
    return (job_dir / "DONE").is_file() and predictions_path.is_file()


def _auto_select_tabiclv2_ecoc(
    results_dir: Path = COMPARE_TABICLV2_RESULTS_DIR,
    ecoc_backend: str = "many_class",
    group_size: int = 100,
    stratify_groups_by_class: bool = True,
):
    """Read ``compare_tabiclv2_multiclass_handling.py``'s own aggregated
    ``manifest.csv`` and decide native vs. ECOC by mean accuracy across
    whatever qualifying (dataset, resample) jobs it has completed --
    see this module's docstring's "TabICLv2's native-vs-ECOC many-class
    handling" section.

    ``ecoc_backend`` restricts the ECOC side of the comparison to rows
    whose own ``EcocBackend`` column matches this run's actual backend
    (this module's own ``--ecoc-backend``, threaded through here) --
    that manifest can accumulate rows from more than one
    ``--ecoc-backend`` across separate ``compare_tabiclv2_multiclass_handling.py``
    invocations (``many_class`` and ``sklearn`` are not comparable: very
    different accuracy/cost profiles -- see that module's own docstring),
    so blending them into one mean would silently conflate two different
    things. Native (``Ecoc=False``) rows have no backend and are always
    included.

    ``group_size``/``stratify_groups_by_class`` similarly restrict *both*
    sides of the comparison to rows matching this run's actual values --
    that manifest can now also accumulate rows from more than one
    ``--group-size``/``--stratify-groups-by-class`` (see
    ``compare_tabiclv2_multiclass_handling.py``'s own ``_run_config_tag``,
    which is what makes that safe to accumulate in the first place rather
    than one config's rows clobbering another's). An older manifest
    written before these columns existed has them missing for every row;
    those rows are this project's own historical default before either
    knob existed (``group_size=100``, plain random group assignment), so
    they're treated as ``GroupSize=100``/``StratifyGroupsByClass=False``
    rather than dropped, matching the ``EcocBackend`` backward-compat
    handling below.

    Returns ``(ecoc, detail)``: ``ecoc`` is ``True``/``False`` when a
    decision could be made, or ``None`` when it couldn't (manifest
    missing, or neither mode has any valid -- ``Error`` empty,
    ``Accuracy`` not NaN -- row matching this ``ecoc_backend``/
    ``group_size``/``stratify_groups_by_class``); ``detail`` is a
    human-readable string explaining the decision (or why none could be
    made), always safe to print, meant to keep this auto-selection
    auditable rather than a silent default.
    """
    manifest_path = results_dir / "manifest.csv"
    if not manifest_path.is_file():
        return None, (
            f"{manifest_path} not found -- run compare_tabiclv2_multiclass_handling.py "
            "first for a measured choice; falling back to native."
        )

    manifest = pd.read_csv(manifest_path)
    valid = manifest[manifest["Error"].isna() & manifest["Accuracy"].notna()]
    # Native rows (Ecoc=False) have no EcocBackend (see _process_job); an
    # older manifest written before EcocBackend existed also has it
    # missing for every row -- treat that as "sklearn" (this project's
    # only backend before many_class was added), not as "no backend",
    # so those older comparisons still count rather than being silently
    # dropped by the filter below.
    ecoc_backend_col = valid["EcocBackend"] if "EcocBackend" in valid.columns else pd.Series(index=valid.index, dtype=object)
    group_size_col = valid["GroupSize"] if "GroupSize" in valid.columns else pd.Series(index=valid.index, dtype=float)
    stratify_col = (
        valid["StratifyGroupsByClass"] if "StratifyGroupsByClass" in valid.columns
        else pd.Series(False, index=valid.index)  # a bool-dtype (not object) all-False fallback avoids a pandas fillna downcast warning
    )
    is_native = valid["Ecoc"] == False  # noqa: E712 - pandas boolean column, `is` doesn't work here
    is_matching_ecoc_backend = ecoc_backend_col.fillna("sklearn") == ecoc_backend
    is_matching_group_size = group_size_col.fillna(100) == group_size
    is_matching_stratify = stratify_col.fillna(False) == stratify_groups_by_class
    valid = valid[(is_native | is_matching_ecoc_backend) & is_matching_group_size & is_matching_stratify]
    means = valid.groupby("Ecoc")["Accuracy"].agg(["mean", "count"])

    has_native = False in means.index and means.loc[False, "count"] > 0
    has_ecoc = True in means.index and means.loc[True, "count"] > 0
    if not has_native and not has_ecoc:
        return None, (
            f"{manifest_path} has no valid (Error-free) rows for native or for ecoc_backend="
            f"{ecoc_backend!r} at group_size={group_size}, stratify_groups_by_class="
            f"{stratify_groups_by_class} -- falling back to native."
        )

    native_mean = means.loc[False, "mean"] if has_native else float("-inf")
    ecoc_mean = means.loc[True, "mean"] if has_ecoc else float("-inf")
    ecoc = ecoc_mean > native_mean
    return ecoc, (
        f"auto-selected from {manifest_path} (ecoc_backend={ecoc_backend!r}, group_size={group_size}, "
        f"stratify_groups_by_class={stratify_groups_by_class}): "
        f"native mean accuracy = {native_mean:.4f} ({means.loc[False, 'count'] if has_native else 0} job(s)), "
        f"ecoc mean accuracy = {ecoc_mean:.4f} ({means.loc[True, 'count'] if has_ecoc else 0} job(s)) "
        f"-> using {'ecoc' if ecoc else 'native'}."
    )


def _cawpe_test_probabilities(dataset: str, resample: int, meta, random_state: int):
    """Best-effort per-family CalibratedClassifierCV-calibrated ridge test
    probabilities for every family with a ridge block present in ``meta``
    -- see this module's docstring's "CAWPE and raw family features"
    section. A family whose raw cache can't be loaded, or whose
    calibration raises, is simply omitted (returns fewer entries than
    requested) rather than raising -- ``meta_learners.cawpe_combine``
    handles a partial mapping gracefully.
    """
    ridge_families = {
        block.family for block in meta.blocks
        if block.learner == "ridge" and block.family in FAMILIES
    }
    calibrated = {}
    for family in sorted(ridge_families):
        try:
            X_train, _ = load_raw_family_features(dataset, resample, family, "train")
            X_test, _ = load_raw_family_features(dataset, resample, family, "test")
            calibrated[family] = calibrate_ridge_family(
                family, X_train, meta.y_train_enc, X_test, n_classes=meta.n_classes, random_state=random_state,
            )
        except Exception:  # noqa: BLE001 - one family's calibration must not sink CAWPE entirely
            continue
    return calibrated


def _run_methods(
    meta, dataset: str, resample: int, tabiclv2_ecoc: bool, random_state: int,
    tabiclv2_backbone_kwargs: Optional[dict] = None,
    tabiclv2_group_size: int = 100,
    tabiclv2_stratify_groups_by_class: bool = True,
    tabiclv2_ecoc_backend: str = "many_class",
    tabiclv2_ecoc_alphabet_size: int = 10,
    tabiclv2_ecoc_redundancy: int = 4,
    tabiclv2_ecoc_code_size: float = 1.5,
):
    """Run every meta-method on an already-assembled ``MetaFeatures``,
    returning ``(npz_payload, manifest_rows)`` -- see this module's
    docstring for exactly what's saved. Each method is independently
    timed and wrapped so one failing does not stop the others.
    """
    npz_payload = {}
    manifest_rows = []
    n_classes, n_train, n_test = meta.n_classes, meta.X_train.shape[0], meta.X_test.shape[0]

    def _record(method: str, start: float, proba=None, pred_enc=None, error: Optional[str] = None, extra=None):
        elapsed = time.perf_counter() - start
        row = {
            "Dataset": dataset, "Resample": resample, "Method": method,
            "NClasses": n_classes, "NTrain": n_train, "NTest": n_test,
            "FitSeconds": elapsed, "Error": error, "Accuracy": None,
        }
        if extra:
            row.update(extra)
        if error is None:
            if proba is not None:
                npz_payload[f"{method}__test_proba"] = proba
                pred_enc = proba.argmax(axis=1)
                row["Accuracy"] = accuracy_from_proba(proba, meta.y_test_enc)
            npz_payload[f"{method}__test_pred_enc"] = pred_enc
            if row["Accuracy"] is None:
                row["Accuracy"] = accuracy_score(meta.y_test_enc, pred_enc)
        else:
            npz_payload[f"{method}__error"] = np.asarray(error)
        manifest_rows.append(row)

    # 1. hard_majority_vote -- no probabilistic output.
    start = time.perf_counter()
    try:
        pred_enc = hard_majority_vote(meta)
        _record("hard_majority_vote", start, pred_enc=pred_enc)
    except Exception as exc:  # noqa: BLE001
        _record("hard_majority_vote", start, error=repr(exc))

    # 2. logistic_regression
    start = time.perf_counter()
    try:
        proba = fit_predict_logistic_regression(meta, random_state=random_state)
        _record("logistic_regression", start, proba=proba)
    except Exception as exc:  # noqa: BLE001
        _record("logistic_regression", start, error=repr(exc))

    # 3. lightgbm
    start = time.perf_counter()
    try:
        proba = fit_predict_lightgbm(meta, random_state=random_state)
        _record("lightgbm", start, proba=proba)
    except Exception as exc:  # noqa: BLE001
        _record("lightgbm", start, error=repr(exc))

    # 4. cawpe
    start = time.perf_counter()
    try:
        ridge_calibrated = _cawpe_test_probabilities(dataset, resample, meta, random_state)
        test_proba_by_block = build_cawpe_test_probabilities(meta, ridge_calibrated)
        proba = cawpe_combine(meta, test_proba_by_block)
        n_ridge_total = sum(1 for b in meta.blocks if b.learner == "ridge")
        _record(
            "cawpe", start, proba=proba,
            extra={"RidgeFamiliesCalibrated": f"{len(ridge_calibrated)}/{n_ridge_total}"},
        )
    except Exception as exc:  # noqa: BLE001
        _record("cawpe", start, error=repr(exc))

    # 5. cawpe_hard_vote -- weighted hard-voting counterpart to cawpe,
    # dev-subset-only candidate (see meta_learners.cawpe_hard_vote's own
    # docstring). No probabilistic output, same convention as
    # hard_majority_vote.
    start = time.perf_counter()
    try:
        pred_enc = cawpe_hard_vote(meta)
        _record("cawpe_hard_vote", start, pred_enc=pred_enc)
    except Exception as exc:  # noqa: BLE001
        _record("cawpe_hard_vote", start, error=repr(exc))

    # 6. tabiclv2
    start = time.perf_counter()
    try:
        proba = fit_predict_tabiclv2(
            meta, ecoc=tabiclv2_ecoc, group_size=tabiclv2_group_size,
            stratify_groups_by_class=tabiclv2_stratify_groups_by_class,
            ecoc_backend=tabiclv2_ecoc_backend, ecoc_alphabet_size=tabiclv2_ecoc_alphabet_size,
            ecoc_redundancy=tabiclv2_ecoc_redundancy, ecoc_code_size=tabiclv2_ecoc_code_size,
            random_state=random_state, backbone_kwargs=tabiclv2_backbone_kwargs,
        )
        _record(
            "tabiclv2", start, proba=proba,
            extra={
                "EcocUsed": tabiclv2_ecoc, "EcocBackend": tabiclv2_ecoc_backend if tabiclv2_ecoc else None,
                "GroupSize": tabiclv2_group_size, "StratifyGroupsByClass": tabiclv2_stratify_groups_by_class,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _record("tabiclv2", start, error=repr(exc))

    # 7. meta_oracle: diagnostic-only upper bound over the six methods
    # above (never oracle_cv/oracle below -- see meta_learners.py's own
    # docstring's "meta_oracle" entry). Picks whichever of those six
    # succeeded with the best TEST accuracy for this job.
    start = time.perf_counter()
    try:
        candidate_methods = ("hard_majority_vote", "logistic_regression", "lightgbm", "cawpe", "cawpe_hard_vote", "tabiclv2")
        candidates = [r for r in manifest_rows if r["Method"] in candidate_methods and r["Error"] is None]
        if not candidates:
            raise ValueError("meta_oracle: no meta-method succeeded for this job -- nothing to pick from.")
        best_row = max(candidates, key=lambda r: r["Accuracy"])
        best_method = best_row["Method"]
        proba = npz_payload.get(f"{best_method}__test_proba")
        pred_enc = None if proba is not None else npz_payload[f"{best_method}__test_pred_enc"]
        _record(
            "meta_oracle", start, proba=proba, pred_enc=pred_enc,
            extra={"SelectedMethod": best_method},
        )
    except Exception as exc:  # noqa: BLE001
        _record("meta_oracle", start, error=repr(exc))

    # 8. oracle_cv (leakage-free: picks the OOF-best single base learner)
    start = time.perf_counter()
    try:
        block = best_block_by_oof_accuracy(meta)
        proba = oracle_by_oof_accuracy(meta)
        _record(
            "oracle_cv", start, proba=proba,
            extra={"SelectedFamily": block.family, "SelectedLearner": block.learner},
        )
    except Exception as exc:  # noqa: BLE001
        _record("oracle_cv", start, error=repr(exc))

    # 9. oracle (diagnostic-only: picks the TEST-best single base learner)
    start = time.perf_counter()
    try:
        block = best_block_by_test_accuracy(meta)
        proba = oracle_best_single_learner(meta)
        _record(
            "oracle", start, proba=proba,
            extra={"SelectedFamily": block.family, "SelectedLearner": block.learner},
        )
    except Exception as exc:  # noqa: BLE001
        _record("oracle", start, error=repr(exc))

    return npz_payload, manifest_rows


def _process_dataset_resample_job(
    dataset: str, resample: int, tabiclv2_ecoc: bool, tabiclv2_backbone_kwargs: Optional[dict] = None,
    tabiclv2_group_size: int = 100, tabiclv2_stratify_groups_by_class: bool = True,
    tabiclv2_ecoc_backend: str = "many_class",
    tabiclv2_ecoc_alphabet_size: int = 10, tabiclv2_ecoc_redundancy: int = 4,
    tabiclv2_ecoc_code_size: float = 1.5,
) -> dict:
    """Worker entry point: load one (dataset, resample)'s scores.npz,
    run every meta-method, save predictions.npz + this job's manifest
    fragment + DONE marker. Mirrors build_stacking_features.py's own
    per-job error handling: any exception escaping the whole job (as
    opposed to one *method* failing inside it, handled by ``_run_methods``)
    is caught and recorded to ``ERROR.txt``.
    """
    job_dir, scores_path, predictions_path = _paths(dataset, resample)
    job_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        if not scores_path.is_file():
            raise FileNotFoundError(
                f"{scores_path} not found -- run build_stacking_features.py for "
                f"{dataset}/resample{resample} first."
            )
        with np.load(scores_path, allow_pickle=False) as scores:
            scores_dict = dict(scores.items())

        meta = assemble_meta_features(scores_dict)
        npz_payload, manifest_rows = _run_methods(
            meta, dataset, resample, tabiclv2_ecoc=tabiclv2_ecoc, random_state=RANDOM_STATE,
            tabiclv2_backbone_kwargs=tabiclv2_backbone_kwargs,
            tabiclv2_group_size=tabiclv2_group_size,
            tabiclv2_stratify_groups_by_class=tabiclv2_stratify_groups_by_class,
            tabiclv2_ecoc_backend=tabiclv2_ecoc_backend,
            tabiclv2_ecoc_alphabet_size=tabiclv2_ecoc_alphabet_size,
            tabiclv2_ecoc_redundancy=tabiclv2_ecoc_redundancy,
            tabiclv2_ecoc_code_size=tabiclv2_ecoc_code_size,
        )
        npz_payload["classes_"] = meta.classes_
        npz_payload["y_train_enc"] = meta.y_train_enc
        npz_payload["y_test_enc"] = meta.y_test_enc
        np.savez_compressed(predictions_path, **npz_payload)

        pd.DataFrame(manifest_rows).to_csv(job_dir / "manifest.csv", index=False)
        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        n_errors = sum(1 for row in manifest_rows if row["Error"] is not None)
        return {
            "dataset": dataset, "resample": resample, "status": "done",
            "elapsed": time.perf_counter() - start, "n_method_errors": n_errors,
        }
    except Exception as exc:  # noqa: BLE001 - one bad job must not abort the whole run
        (job_dir / "ERROR.txt").write_text(f"{exc!r}\n\n{traceback.format_exc()}")
        return {
            "dataset": dataset, "resample": resample, "status": "error",
            "elapsed": time.perf_counter() - start, "error": repr(exc),
        }


def finalize() -> None:
    """Aggregate every completed job's manifest fragment into one
    top-level ``manifest.csv``. Safe to call standalone.
    """
    fragments = sorted(RESULTS_DIR.glob("units/*/resample*/manifest.csv"))
    if not fragments:
        print("No completed jobs found; nothing to aggregate.")
        return
    combined = pd.concat([pd.read_csv(p) for p in fragments], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    print(f"Aggregated {len(fragments)} completed job(s) -> {RESULTS_DIR / 'manifest.csv'}")


@dataclass
class Job:
    dataset: str
    resample: int


def build_jobs(datasets: List[str]) -> List[Job]:
    return [
        Job(dataset=dataset, resample=resample)
        for dataset in datasets
        for resample in range(NUM_RESAMPLES)
        if not _is_done(dataset, resample)
    ]


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
            f"{DEFAULT_MAX_TABICLV2_WORKERS}) -- every job here includes a TabICLv2 fit, which is "
            "heavy enough that this default is capped well below cpu_count; see "
            "compare_tabiclv2_multiclass_handling.py's docstring's 'Parallelism and memory' section "
            "for why -- a BrokenProcessPool there was the reason this cap exists)."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu"), default="auto",
        help=(
            "'cpu' forces TabICLv2 onto CPU (backbone_kwargs={'device': 'cpu'}) instead of leaving "
            "device selection to tabicl/torch's own auto-detection. 'auto' (default) doesn't pass "
            "backbone_kwargs at all."
        ),
    )
    parser.add_argument(
        "--tabiclv2-ecoc", choices=("auto", "native", "ecoc"), default="auto",
        help=(
            "Which of TabICLv2's many-class handling modes to use. 'auto' (default) "
            "reads compare_tabiclv2_multiclass_handling.py's own aggregated manifest.csv "
            "and picks whichever mode it measured as more accurate, falling back to "
            "'native' if that harness hasn't been run yet -- see this module's docstring. "
            "'native'/'ecoc' force one mode explicitly, ignoring the harness's results."
        ),
    )
    parser.add_argument(
        "--group-size", type=int, default=100,
        help=(
            "Max columns per TabICLv2 feature-group (GroupedICLClassifier's own default is 100). "
            "Only affects the tabiclv2 method -- see this module's docstring's 'Parallelism and "
            "memory' section."
        ),
    )
    parser.add_argument(
        "--no-stratify-groups-by-class", dest="stratify_groups_by_class", action="store_false", default=True,
        help=(
            "Disable class-stratified TabICLv2 group assignment (see meta_learners.fit_predict_tabiclv2's "
            "and backbone.py's 'Grouping' docstring section) -- reverts to the original random group "
            "assignment. Also restricts which StratifyGroupsByClass rows --tabiclv2-ecoc auto's manifest "
            "read considers -- see _auto_select_tabiclv2_ecoc's docstring. Default: stratify (on)."
        ),
    )
    parser.add_argument(
        "--ecoc-backend", choices=("many_class", "sklearn"), default="many_class",
        help=(
            "Which ECOC implementation to use when tabiclv2-ecoc resolves to ECOC. 'many_class' "
            "(default) is tabpfn_extensions.many_class.ManyClassClassifier -- believed to be the "
            "actual 'TabPFNv2 ECOC wrapper' the TabICLv2 paper's own text says it supports (requires "
            "the tabpfn-extensions package). 'sklearn' is the original scikit-learn "
            "OutputCodeClassifier implementation, kept for comparison -- see "
            "compare_tabiclv2_multiclass_handling.py's and backbone.py's docstrings. Also restricts "
            "which EcocBackend rows --tabiclv2-ecoc auto's manifest read considers -- see "
            "_auto_select_tabiclv2_ecoc's docstring."
        ),
    )
    parser.add_argument(
        "--ecoc-alphabet-size", type=int, default=10,
        help=(
            "'many_class' backend only: the base estimator's own max classes-per-fit limit. "
            "Default 10 matches TabICLv2's own pretrained max_classes."
        ),
    )
    parser.add_argument(
        "--ecoc-redundancy", type=int, default=4,
        help="'many_class' backend only: forwarded to ManyClassClassifier's own n_estimators_redundancy (its own default is 4).",
    )
    parser.add_argument(
        "--ecoc-code-size", type=float, default=1.5,
        help=(
            "'sklearn' backend only: forwarded to OutputCodeClassifier(code_size=...) when "
            "tabiclv2-ecoc resolves to ECOC (sklearn's own default is 1.5) -- see this module's "
            "docstring's 'Parallelism and memory' section."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Forwarded to TabICLClassifier(batch_size=...) (tabicl's own default is 8). Default (unset) leaves tabicl's own default.",
    )
    parser.add_argument(
        "--offload-mode", choices=("auto", "gpu", "cpu", "disk"), default=None,
        help=(
            "Forwarded to TabICLClassifier(offload_mode=...). 'disk' memory-maps the dominant "
            "embedding tensor to --disk-offload-dir instead of RAM -- see this module's docstring's "
            "'Parallelism and memory' section. Default (unset) leaves tabicl's own default ('auto')."
        ),
    )
    parser.add_argument(
        "--disk-offload-dir", type=str, default=None,
        help="Forwarded to TabICLClassifier(disk_offload_dir=...). Required when --offload-mode disk is set.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned job list, then exit without running anything.",
    )
    args = parser.parse_args(argv)
    if args.offload_mode == "disk" and not args.disk_offload_dir:
        parser.error("--offload-mode disk requires --disk-offload-dir.")
    tabiclv2_backbone_kwargs = {}
    if args.device == "cpu":
        tabiclv2_backbone_kwargs["device"] = "cpu"
    if args.batch_size is not None:
        tabiclv2_backbone_kwargs["batch_size"] = args.batch_size
    if args.offload_mode is not None:
        tabiclv2_backbone_kwargs["offload_mode"] = args.offload_mode
    if args.disk_offload_dir is not None:
        tabiclv2_backbone_kwargs["disk_offload_dir"] = args.disk_offload_dir
    tabiclv2_backbone_kwargs = tabiclv2_backbone_kwargs or None

    if args.tabiclv2_ecoc == "auto":
        tabiclv2_ecoc, ecoc_detail = _auto_select_tabiclv2_ecoc(
            ecoc_backend=args.ecoc_backend, group_size=args.group_size,
            stratify_groups_by_class=args.stratify_groups_by_class,
        )
        if tabiclv2_ecoc is None:
            tabiclv2_ecoc = False  # native -- see _auto_select_tabiclv2_ecoc's docstring
    else:
        tabiclv2_ecoc = args.tabiclv2_ecoc == "ecoc"
        ecoc_detail = f"forced via --tabiclv2-ecoc {args.tabiclv2_ecoc}."

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cpu_count = detect_cpu_count()
    n_workers = args.max_workers or min(max(1, cpu_count - RESERVED_CORES), DEFAULT_MAX_TABICLV2_WORKERS)

    datasets = dev_subset_datasets()
    jobs = build_jobs(datasets)
    n_total = len(datasets) * NUM_RESAMPLES
    print(
        f"Detected {cpu_count} CPU core(s) -> {n_workers} worker(s) (capped for TabICLv2's memory "
        "footprint -- see compare_tabiclv2_multiclass_handling.py's docstring; override with "
        "--max-workers if you've checked memory headroom).\n"
        f"{len(jobs)} job(s) to run ({n_total - len(jobs)} already done, resuming) "
        f"across {len(datasets)} dataset(s) x {NUM_RESAMPLES} resample(s).\n"
        f"TabICLv2 ECOC: {tabiclv2_ecoc} ({ecoc_detail}) | device: {args.device} | "
        f"group_size: {args.group_size} | stratify_groups_by_class: {args.stratify_groups_by_class} | "
        f"ecoc_backend: {args.ecoc_backend} | "
        f"ecoc_alphabet_size: {args.ecoc_alphabet_size} | ecoc_redundancy: {args.ecoc_redundancy} | "
        f"ecoc_code_size: {args.ecoc_code_size} | batch_size: {args.batch_size} | "
        f"offload_mode: {args.offload_mode} | disk_offload_dir: {args.disk_offload_dir}"
    )
    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        for job in jobs:
            print(f"  {job.dataset} resample{job.resample}")
        print("--dry-run: exiting without running anything.")
        return

    import concurrent.futures as cf
    import multiprocessing

    mp_context = multiprocessing.get_context("spawn")
    completed, errored, total_method_errors = 0, 0, 0

    with cf.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp_context, initializer=_worker_init
    ) as executor:
        start_time = time.perf_counter()
        futures = {
            executor.submit(
                _process_dataset_resample_job, job.dataset, job.resample, tabiclv2_ecoc, tabiclv2_backbone_kwargs,
                args.group_size, args.stratify_groups_by_class, args.ecoc_backend, args.ecoc_alphabet_size,
                args.ecoc_redundancy, args.ecoc_code_size,
            ): job
            for job in jobs
        }
        for future in cf.as_completed(futures):
            job = futures[future]
            result = future.result()
            if result["status"] == "done":
                completed += 1
                total_method_errors += result["n_method_errors"]
                extra = f" ({result['n_method_errors']} method error(s))" if result["n_method_errors"] else ""
                print(
                    f"[{completed + errored}/{len(jobs)}] OK   {job.dataset} resample{job.resample} "
                    f"({result['elapsed']:.1f}s){extra}"
                )
            else:
                errored += 1
                print(
                    f"[{completed + errored}/{len(jobs)}] FAIL {job.dataset} resample{job.resample}: "
                    f"{result.get('error')}"
                )
        elapsed = time.perf_counter() - start_time
        print(
            f"\nDone in {elapsed / 60:.1f} min: {completed} succeeded ({total_method_errors} total "
            f"method-level error(s) among them), {errored} job(s) failed outright."
        )

    finalize()


if __name__ == "__main__":
    main()
