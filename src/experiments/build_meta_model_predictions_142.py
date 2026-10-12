"""Build and cache the meta-model layer's final predictions across the
full 142-dataset x 30-resample archive -- the 142-scale sibling of
``build_meta_model_predictions.py`` (which only covers the 20-dataset dev
subset). Runs the same eight meta-methods that module's docstring
documents (``hard_majority_vote``, ``logistic_regression``, ``lightgbm``,
``cawpe``, ``tabiclv2``, ``meta_oracle``, ``oracle_cv``, ``oracle`` -- see
``src.tsfm_ensemble.meta_learners``'s module docstring for each one's
rationale); everything about *how* each method is fit/scored is unchanged
and reused directly from that module. What's different here is only where
the base-learner scores and (for CAWPE) raw features come from.

``tabiclv2`` uses ``fit_predict_tabiclv2``'s default (``backbone="tabicl"``,
strict): if the real TabICLv2 backbone can't be imported or fit, that one
method's job-level try/except records an ``Error`` for it (see
``_run_methods`` below) rather than the whole (dataset, resample) job
failing -- but it never silently substitutes a placeholder classifier, so
every non-errored ``tabiclv2`` row in the manifest is genuinely TabICLv2.

No manual analysis step needed in between
--------------------------------------------
This is meant to run right after ``cv_ablation_142.py`` finishes, with no
one available to look at its results first (see
``analyze_cv_ablation_142.py``'s own docstring for the full reasoning).
Concretely, this script reads TWO upstream artifacts instead of
``build_stacking_features.py``'s single ``scores.npz``:

1. ``cv_ablation_142.py``'s per-(dataset, resample, family) fragment
   ``.npz`` files (``{learner}__{arm}__oof_train``/``__test`` keys, one
   arm's worth of scores per learner -- see that module's docstring's
   "Output layout"), instead of ``build_stacking_features.py``'s single
   "cv"-arm-only ``scores.npz``.
2. ``analyze_cv_ablation_142.py``'s ``chosen_arms.json`` -- which arm
   ("cv" vs "bagB15"/"bagB25" for ridge/naive_bayes, "cv" vs "nativeoob"
   for extra_trees) to actually pull from each family's fragment, per
   (family, learner), decided mechanically from OOF-honesty, no human
   judgment call. If this file doesn't exist yet (e.g. run out of order),
   every (family, learner) falls back to the "cv" arm -- the same arm
   every other script already used before this ablation existed -- and
   says so loudly, matching this project's "auditable default, never a
   silent one" convention (see ``build_meta_model_predictions.py``'s own
   ``_auto_select_tabiclv2_ecoc``).

``_load_unit_scores`` does the actual reshaping: each family's fragment
carries ``{learner}__{arm}__oof_train``/``__test``; this rebuilds the
``{family}__{learner}__oof_train``/``__test`` keys
``meta_learners.assemble_meta_features`` expects, picking each learner's
chosen arm per family, so downstream everything (``assemble_meta_features``,
``hard_majority_vote``, ``fit_predict_logistic_regression``, etc.) is used
completely unmodified.

CAWPE needs no raw feature cache at 142 scale
--------------------------------------------------
Unlike the dev-subset version, this script's CAWPE does NOT depend on
``save_142_transforms.py`` (or any raw feature cache) at all. CAWPE
itself is just a weighted sum of each base learner's probabilities
(``meta_learners.cawpe_combine``); the only reason ridge's raw
``decision_function`` margins ever needed a raw feature matrix was to
refit a fresh classifier and calibrate it into a genuine probability
(``calibrate_ridge_family``, ``CalibratedClassifierCV`` on raw X). That
refit turns out to be unnecessary: Platt/sigmoid calibration only needs
(score, true label) pairs, and cv_ablation_142.py's fragments already
carry honest OOF/test ridge scores -- see
``meta_learners.calibrate_ridge_family_from_scores``'s docstring for the
full reasoning. So this script never touches raw features, never needs
``save_142_transforms.py`` to have been run, and has no multi-terabyte
raw-cache disk cost. A ridge block whose per-class calibration fails is
simply left out of CAWPE's combination for that job (same "one bad
component doesn't sink the whole method" convention as everywhere else).

y_test_enc can be genuinely absent for a unit
-------------------------------------------------
``cv_ablation_142.py``'s own ``_run_unit`` skips saving ``y_test_enc``
for the rare (dataset, resample) where a test-split label was never seen
in training (``LabelEncoder.transform`` would raise) -- OOF-side honesty
can still be measured there, but nothing test-side can. Since that
depends only on the (dataset, resample)'s label split, not on family, if
one family's fragment lacks it, every family's fragment for that same
(dataset, resample) does too. This script cannot score meta-learners at
all for such a unit (there is no true test accuracy to evaluate against)
-- it's recorded as a whole-job error (``ERROR.txt``), not silently
skipped, matching this project's "one bad job doesn't take down the
whole run, but doesn't disappear either" convention.

Output layout, under RESULTS_DIR (results/meta_model_predictions_142/)
----------------------------------------------------------------------
Identical to build_meta_model_predictions.py's own (see that module's
docstring): ``<dataset>/resample<k>/predictions.npz``,
``units/<dataset>/resample<k>/{DONE,ERROR.txt,manifest.csv}``,
top-level ``manifest.csv`` via ``finalize()``.

Usage
-----
    python -m src.experiments.build_meta_model_predictions_142 --dry-run
    python -m src.experiments.build_meta_model_predictions_142 --max-workers 8
    python -m src.experiments.build_meta_model_predictions_142 --device cpu
    python -m src.experiments.build_meta_model_predictions_142 --datasets OliveOil,PigCVP --resamples 2
    python -m src.experiments.build_meta_model_predictions_142 --finalize-only

    # The full unattended chain, no manual step in between, no raw
    # feature cache needed (see "CAWPE needs no raw feature cache" above):
    python -m src.experiments.cv_ablation_142 --max-workers 40 && \\
        python -m src.experiments.analyze_cv_ablation_142 && \\
        python -m src.experiments.build_meta_model_predictions_142 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.metrics import accuracy_score  # noqa: E402

from src.experiments.common import RESAMPLE_INDICES_DIR, detect_cpu_count  # noqa: E402
from src.experiments.compare_tabiclv2_multiclass_handling import (  # noqa: E402
    DEFAULT_MAX_TABICLV2_WORKERS,
    RESULTS_DIR as COMPARE_TABICLV2_RESULTS_DIR,
)
from src.experiments.build_meta_model_predictions import _auto_select_tabiclv2_ecoc  # noqa: E402
from src.tsfm_ensemble.base_learners import FAMILIES, LEARNER_NAMES, RANDOM_STATE  # noqa: E402
from src.tsfm_ensemble.meta_learners import (  # noqa: E402
    accuracy_from_proba,
    assemble_meta_features,
    best_block_by_oof_accuracy,
    best_block_by_test_accuracy,
    build_cawpe_test_probabilities,
    calibrate_ridge_family_from_scores,
    cawpe_combine,
    fit_predict_lightgbm,
    fit_predict_logistic_regression,
    fit_predict_tabiclv2,
    hard_majority_vote,
    oracle_best_single_learner,
    oracle_by_oof_accuracy,
)

CV_ABLATION_142_RESULTS_DIR = Path(os.environ.get("CV_ABLATION_142_RESULTS_DIR", str(Path("results") / "cv_ablation_142")))
FRAGMENTS_DIR = CV_ABLATION_142_RESULTS_DIR / "fragments"
CHOSEN_ARMS_PATH = CV_ABLATION_142_RESULTS_DIR / "chosen_arms.json"
RESULTS_DIR = Path(
    os.environ.get("META_MODEL_PREDICTIONS_142_RESULTS_DIR", str(Path("results") / "meta_model_predictions_142"))
)

NUM_RESAMPLES = 30
RESERVED_CORES = 1
DEFAULT_ARM = "cv"  # matches analyze_cv_ablation_142.py's own DEFAULT_ARM/fallback baseline.


def list_142_datasets(resample_indices_dir: Path = RESAMPLE_INDICES_DIR) -> List[str]:
    """Duplicated from cv_ablation_142.py / save_142_transforms.py -- see
    either module's docstring for why.
    """
    return sorted(p.name for p in resample_indices_dir.iterdir() if p.is_dir())


def _load_chosen_arms(path: Path = CHOSEN_ARMS_PATH) -> Tuple[Dict[str, Dict[str, str]], str]:
    """Return ``({family: {learner: fragment_suffix}}, detail)``.

    Falls back to ``DEFAULT_ARM`` ("cv") for every (family, learner) if
    ``path`` doesn't exist -- e.g. this script run before
    ``analyze_cv_ablation_142.py`` -- rather than raising, so the
    unattended chain degrades to "use the pre-ablation baseline arm"
    instead of stalling entirely; ``detail`` says so loudly either way
    (see this module's docstring's "No manual analysis step needed"
    section and ``build_meta_model_predictions.py``'s own
    ``_auto_select_tabiclv2_ecoc`` for the same auditable-fallback
    convention this mirrors).
    """
    if not path.is_file():
        fallback = {family: {learner: DEFAULT_ARM for learner in LEARNER_NAMES} for family in FAMILIES}
        return fallback, (
            f"{path} not found -- run analyze_cv_ablation_142.py first for a measured choice; "
            f"falling back to {DEFAULT_ARM!r} for every (family, learner)."
        )
    decisions = json.loads(path.read_text())
    resolved = {
        family: {learner: decisions[family][learner]["fragment_suffix"] for learner in LEARNER_NAMES}
        for family in FAMILIES
    }
    chosen_str = ", ".join(f"{f}/{l}={resolved[f][l]}" for f in FAMILIES for l in LEARNER_NAMES)
    return resolved, f"loaded from {path}: {chosen_str}"


def _load_unit_scores(dataset: str, resample: int, chosen_arms: Dict[str, Dict[str, str]]) -> Dict:
    """Reshape cv_ablation_142.py's per-family fragment ``.npz`` files for
    one (dataset, resample) into the wide-matrix-ready dict
    ``meta_learners.assemble_meta_features`` expects (see this module's
    docstring's "No manual analysis step needed" section). A family with
    no fragment file at all (that unit's whole family failed in
    cv_ablation_142.py) or a (learner, chosen arm) pair with no matching
    key in its fragment (that specific arm errored for that unit) is
    simply omitted -- assemble_meta_features's own ``missing`` handling
    (see its docstring) takes it from there. Raises ``ValueError`` only
    when nothing usable exists at all: no fragment carries
    ``y_train_enc``, or none carries ``y_test_enc`` (see this module's
    docstring's "y_test_enc can be genuinely absent" section) -- both
    caught by the caller as a whole-job error.
    """
    combined: Dict = {}
    y_train_enc = y_test_enc = classes_ = None
    missing_families = []
    for family in FAMILIES:
        fragment_path = FRAGMENTS_DIR / dataset / f"resample{resample}" / f"{family}.npz"
        if not fragment_path.is_file():
            missing_families.append(family)
            continue
        with np.load(fragment_path, allow_pickle=False) as frag:
            if y_train_enc is None and "y_train_enc" in frag.files:
                y_train_enc = frag["y_train_enc"]
                classes_ = frag["classes_"]
            if y_test_enc is None and "y_test_enc" in frag.files:
                y_test_enc = frag["y_test_enc"]
            for learner in LEARNER_NAMES:
                arm = chosen_arms[family][learner]
                oof_key, test_key = f"{learner}__{arm}__oof_train", f"{learner}__{arm}__test"
                if oof_key in frag.files and test_key in frag.files:
                    combined[f"{family}__{learner}__oof_train"] = frag[oof_key]
                    combined[f"{family}__{learner}__test"] = frag[test_key]

    if y_train_enc is None:
        raise ValueError(
            f"No cv_ablation_142.py fragment with y_train_enc found for {dataset}/resample{resample} "
            f"(missing families: {missing_families}) -- has cv_ablation_142.py been run for this unit?"
        )
    if y_test_enc is None:
        raise ValueError(
            f"No cv_ablation_142.py fragment carries y_test_enc for {dataset}/resample{resample} -- "
            "a test-split label was likely unseen in training (see cv_ablation_142.py's _run_unit); "
            "this unit cannot be scored here."
        )
    combined["y_train_enc"] = y_train_enc
    combined["y_test_enc"] = y_test_enc
    combined["classes_"] = classes_
    return combined


def _cawpe_test_probabilities(dataset: str, resample: int, meta, random_state: int):
    """142-scale CAWPE ridge calibration -- unlike
    build_meta_model_predictions.py's own dev-subset version, this needs
    no raw feature cache at all: it calibrates each family's ridge block
    directly from the OOF/test decision_function scores already sitting
    in ``meta`` (assembled from cv_ablation_142.py's fragments), via
    ``meta_learners.calibrate_ridge_family_from_scores``. See that
    function's docstring for why this is equivalent to (and actually
    more consistent than) refitting ridge from raw features, and this
    module's docstring for why the raw-cache approach
    (``save_142_transforms.py``) turned out to be unnecessary.
    ``dataset``/``resample``/``random_state`` are unused here (kept in
    the signature to match the dev-subset version's call shape).
    """
    ridge_families = {
        block.family for block in meta.blocks
        if block.learner == "ridge" and block.family in FAMILIES
    }
    calibrated = {}
    for block in meta.blocks:
        if block.learner != "ridge" or block.family not in ridge_families:
            continue
        try:
            oof_scores = meta.X_train[:, block.start:block.stop]
            test_scores = meta.X_test[:, block.start:block.stop]
            calibrated[block.family] = calibrate_ridge_family_from_scores(
                oof_scores, meta.y_train_enc, test_scores, n_classes=meta.n_classes,
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
    """Identical to build_meta_model_predictions.py's own ``_run_methods``
    -- duplicated (not imported) since it closes over this module's own
    ``_cawpe_test_probabilities``.
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

    start = time.perf_counter()
    try:
        pred_enc = hard_majority_vote(meta)
        _record("hard_majority_vote", start, pred_enc=pred_enc)
    except Exception as exc:  # noqa: BLE001
        _record("hard_majority_vote", start, error=repr(exc))

    start = time.perf_counter()
    try:
        proba = fit_predict_logistic_regression(meta, random_state=random_state)
        _record("logistic_regression", start, proba=proba)
    except Exception as exc:  # noqa: BLE001
        _record("logistic_regression", start, error=repr(exc))

    start = time.perf_counter()
    try:
        proba = fit_predict_lightgbm(meta, random_state=random_state)
        _record("lightgbm", start, proba=proba)
    except Exception as exc:  # noqa: BLE001
        _record("lightgbm", start, error=repr(exc))

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

    # meta_oracle: diagnostic-only upper bound over the five methods just
    # above (never oracle_cv/oracle below -- see meta_learners.py's own
    # docstring's "meta_oracle" entry). Picks whichever of those five
    # succeeded with the best TEST accuracy for this job and reports it;
    # never deployable, same caveat as "oracle" below, one level up.
    start = time.perf_counter()
    try:
        candidate_methods = ("hard_majority_vote", "logistic_regression", "lightgbm", "cawpe", "tabiclv2")
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


def _paths(dataset: str, resample: int):
    job_dir = RESULTS_DIR / "units" / dataset / f"resample{resample}"
    predictions_path = RESULTS_DIR / dataset / f"resample{resample}" / "predictions.npz"
    return job_dir, predictions_path


def _is_done(dataset: str, resample: int) -> bool:
    job_dir, predictions_path = _paths(dataset, resample)
    return (job_dir / "DONE").is_file() and predictions_path.is_file()


def _process_dataset_resample_job(
    dataset: str, resample: int, chosen_arms: Dict[str, Dict[str, str]], tabiclv2_ecoc: bool,
    tabiclv2_backbone_kwargs: Optional[dict] = None,
    tabiclv2_group_size: int = 100, tabiclv2_stratify_groups_by_class: bool = True,
    tabiclv2_ecoc_backend: str = "many_class",
    tabiclv2_ecoc_alphabet_size: int = 10, tabiclv2_ecoc_redundancy: int = 4,
    tabiclv2_ecoc_code_size: float = 1.5,
) -> dict:
    """Worker entry point: assemble one (dataset, resample)'s meta
    features from cv_ablation_142.py's fragments (per ``chosen_arms``),
    run every meta-method, save predictions.npz + this job's manifest
    fragment + DONE marker. Any exception escaping the whole job (as
    opposed to one *method* failing inside it) is caught and recorded to
    ERROR.txt.
    """
    job_dir, predictions_path = _paths(dataset, resample)
    job_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        scores_dict = _load_unit_scores(dataset, resample, chosen_arms)
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


def build_jobs(datasets: List[str], resamples: int) -> List[Job]:
    return [
        Job(dataset=dataset, resample=resample)
        for dataset in datasets
        for resample in range(resamples)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help=(
            f"Override the default worker count (min(cpu_count - {RESERVED_CORES}, "
            f"{DEFAULT_MAX_TABICLV2_WORKERS}) -- every job here includes a TabICLv2 fit; see "
            "build_meta_model_predictions.py's docstring's 'Parallelism and memory' section for why "
            "this default is capped well below cpu_count."
        ),
    )
    parser.add_argument("--resamples", type=int, default=NUM_RESAMPLES)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated subset (default: all 142).")
    parser.add_argument(
        "--device", choices=("auto", "cpu"), default="auto",
        help="'cpu' forces TabICLv2 onto CPU. 'auto' (default) leaves device selection to tabicl/torch.",
    )
    parser.add_argument(
        "--tabiclv2-ecoc", choices=("auto", "native", "ecoc"), default="auto",
        help=(
            "Which of TabICLv2's many-class handling modes to use -- 'auto' reads "
            "compare_tabiclv2_multiclass_handling.py's manifest exactly as "
            "build_meta_model_predictions.py's own --tabiclv2-ecoc auto does (this project hasn't "
            "run that comparison separately at 142 scale; the dev-subset comparison is reused)."
        ),
    )
    parser.add_argument("--group-size", type=int, default=100)
    parser.add_argument("--no-stratify-groups-by-class", dest="stratify_groups_by_class", action="store_false", default=True)
    parser.add_argument("--ecoc-backend", choices=("many_class", "sklearn"), default="many_class")
    parser.add_argument("--ecoc-alphabet-size", type=int, default=10)
    parser.add_argument("--ecoc-redundancy", type=int, default=4)
    parser.add_argument("--ecoc-code-size", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--offload-mode", choices=("auto", "gpu", "cpu", "disk"), default=None)
    parser.add_argument("--disk-offload-dir", type=str, default=None)
    parser.add_argument("--finalize-every", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.offload_mode == "disk" and not args.disk_offload_dir:
        parser.error("--offload-mode disk requires --disk-offload-dir.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        finalize()
        return

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
            results_dir=COMPARE_TABICLV2_RESULTS_DIR, ecoc_backend=args.ecoc_backend, group_size=args.group_size,
            stratify_groups_by_class=args.stratify_groups_by_class,
        )
        if tabiclv2_ecoc is None:
            tabiclv2_ecoc = False
    else:
        tabiclv2_ecoc = args.tabiclv2_ecoc == "ecoc"
        ecoc_detail = f"forced via --tabiclv2-ecoc {args.tabiclv2_ecoc}."

    chosen_arms, chosen_arms_detail = _load_chosen_arms()
    print(f"Base-learner arm selection: {chosen_arms_detail}")

    cpu_count = detect_cpu_count()
    n_workers = args.max_workers or min(max(1, cpu_count - RESERVED_CORES), DEFAULT_MAX_TABICLV2_WORKERS)

    datasets = args.datasets.split(",") if args.datasets else list_142_datasets()
    jobs = build_jobs(datasets, args.resamples)
    n_total = len(datasets) * args.resamples
    print(
        f"Detected {cpu_count} CPU core(s) -> {n_workers} worker(s) (capped for TabICLv2's memory footprint).\n"
        f"{len(jobs)} job(s) to run ({n_total - len(jobs)} already done, resuming) "
        f"across {len(datasets)} dataset(s) x {args.resamples} resample(s).\n"
        f"TabICLv2 ECOC: {tabiclv2_ecoc} ({ecoc_detail}) | device: {args.device}"
    )
    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
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
        in_flight = {}
        remaining = list(jobs)
        n_finished = 0

        while remaining or in_flight:
            while remaining and len(in_flight) < n_workers:
                job = remaining.pop(0)
                future = executor.submit(
                    _process_dataset_resample_job, job.dataset, job.resample, chosen_arms, tabiclv2_ecoc,
                    tabiclv2_backbone_kwargs, args.group_size, args.stratify_groups_by_class, args.ecoc_backend,
                    args.ecoc_alphabet_size, args.ecoc_redundancy, args.ecoc_code_size,
                )
                in_flight[future] = job

            done_futures, _ = cf.wait(in_flight.keys(), return_when=cf.FIRST_COMPLETED)
            for future in done_futures:
                job = in_flight.pop(future)
                result = future.result()
                n_finished += 1
                if result["status"] == "done":
                    completed += 1
                    total_method_errors += result["n_method_errors"]
                    extra = f" ({result['n_method_errors']} method error(s))" if result["n_method_errors"] else ""
                    print(f"[{n_finished}/{len(jobs)}] OK   {job.dataset} resample{job.resample} ({result['elapsed']:.1f}s){extra}")
                else:
                    errored += 1
                    print(f"[{n_finished}/{len(jobs)}] FAIL {job.dataset} resample{job.resample}: {result.get('error')}")
                if args.finalize_every and n_finished % args.finalize_every == 0:
                    finalize()

        elapsed = time.perf_counter() - start_time
        print(
            f"\nDone in {elapsed / 60:.1f} min: {completed} succeeded ({total_method_errors} total "
            f"method-level error(s) among them), {errored} job(s) failed outright."
        )

    finalize()


if __name__ == "__main__":
    main()
