"""Split TIGER's own deployed meta-classification stage's combined
fit+predict timing into separately-timed fit and predict blocks, at full
142-dataset x 30-resample scale -- the piece missing from
``results/meta_model_predictions_142/manifest.csv``'s ``FitSeconds``
(every method there, ``tabiclv2`` and ``cawpe_hard_vote`` alike, is timed
as one combined ``fit_predict_*`` block; see
``build_meta_model_predictions_142.py``'s own ``_run_methods``/``_record``),
needed to report an exact training-only / inference-only / both split for
TIGER in sn-article.tex Table~\\ref{tab:runtimes}.

Deliberately narrow in scope: only the two methods TIGER's own adaptive
rule (``meta_learner_selection.select_meta_learner``) ever actually
deploys -- ``cawpe_hard_vote`` (weighted hard majority vote) and
``tabiclv2`` -- are re-timed here, not the other six candidate methods
``build_meta_model_predictions_142.py`` also runs (``hard_majority_vote``,
``logistic_regression``, ``lightgbm``, ``cawpe``, ``meta_oracle``,
``oracle_cv``, ``oracle``) for its own broader meta-method comparison
(Section 5.2's Table~\\ref{tab:meta_method_comparison}-type analysis) --
those are never deployed, so their timing isn't needed for this table.
``cawpe_hard_vote`` costs next to nothing either way (~0.01s per unit,
already established in sn-article.tex's current Runtimes section), so
essentially all of this script's wall-clock cost is ``tabiclv2`` -- the
expensive half of the estimate in this project's own paper-drafting
notes (~171 core-hours serial; NOT ~171/40 hours at 40 workers, since
TabICLv2's own memory footprint caps real parallelism well below core
count -- see ``DEFAULT_MAX_TABICLV2_WORKERS`` below and this script's own
``--max-workers`` default).

Inputs are read exactly as ``build_meta_model_predictions_142.py`` reads
them (same ``_load_chosen_arms``/``_load_unit_scores``/
``assemble_meta_features`` call chain, same ``--tabiclv2-ecoc auto``
resolution via ``_auto_select_tabiclv2_ecoc``, imported directly rather
than duplicated -- unlike ``cv_ablation_142.py``'s sibling-script
convention, exact parameter consistency with TabICLv2's own ECOC/group-
size configuration matters too much here to risk re-deriving it), so
this script's ``tabiclv2``/``cawpe_hard_vote`` accuracy numbers should
reproduce ``results/meta_model_predictions_142/manifest.csv``'s own
(and ``results/meta_model_predictions_142/manifest.csv``'s own
``cawpe_hard_vote``/``auto_selected`` rows, added by
``apply_meta_learner_selection_142.py``) almost exactly -- useful as a
built-in sanity check while this runs.

``cawpe_hard_vote``'s own "fit" (``meta_learners.learner_weights`` --
OOF-accuracy-derived weights, needs only training labels and each base
learner's already-cached OOF scores, no test data) and "predict"
(``meta_learners._cawpe_hard_vote_tally`` -- the weighted-tally vote
over test-set argmax predictions) are two functions specifically
extracted from ``meta_learners.cawpe_hard_vote`` for this purpose (see
that module's own docstring); ``tabiclv2``'s fit/predict split needs no
such extraction -- ``GroupedICLClassifier.fit``/``.predict_proba`` were
already two separate calls inside ``meta_learners.fit_predict_tabiclv2``,
just timed together there. This script duplicates that function's body
(not imported -- ``fit_predict_tabiclv2`` returns only ``proba``, no
timing split, and every other caller of it needs to keep working
unchanged) with the two calls timed separately instead.

NOTE: this script imports ``tabicl``/``tabpfn_extensions``/``torch``
(via ``GroupedICLClassifier``) and was written and smoke-tested for its
``cawpe_hard_vote`` path only -- those packages are not installed in the
environment this was developed in, so the ``tabiclv2`` path could not be
exercised end-to-end here. Run a small ``--datasets <one>,--resamples 1``
smoke test on the actual TabICLv2-capable machine before launching the
full 142-dataset run.

Usage (from the repo root):
    python3 -m src.experiments.measure_meta_classification_split_142 --dry-run
    python3 -m src.experiments.measure_meta_classification_split_142 --datasets ACSF1 --resamples 1
    python3 -m src.experiments.measure_meta_classification_split_142
    python3 -m src.experiments.measure_meta_classification_split_142 --max-workers 16  # only if profiled safe
"""
from __future__ import annotations

import os  # noqa: E402

for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import argparse  # noqa: E402
import multiprocessing  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from src.experiments.build_meta_model_predictions_142 import (  # noqa: E402
    _auto_select_tabiclv2_ecoc,
    _load_chosen_arms,
    _load_unit_scores,
    NUM_RESAMPLES,
    RESERVED_CORES,
    list_142_datasets,
)
from src.experiments.compare_tabiclv2_multiclass_handling import (  # noqa: E402
    DEFAULT_MAX_TABICLV2_WORKERS,
    RESULTS_DIR as COMPARE_TABICLV2_RESULTS_DIR,
)
from src.tsfm_ensemble.base_learners import RANDOM_STATE  # noqa: E402
from src.tsfm_ensemble.meta_learners import (  # noqa: E402
    _cawpe_hard_vote_tally,
    _class_group_key,
    _realign_columns,
    accuracy_from_proba,
    assemble_meta_features,
    learner_weights,
)

RESULTS_DIR = Path(os.environ.get(
    "MEASURE_META_CLASSIFICATION_SPLIT_142_RESULTS_DIR", str(Path("results") / "measure_meta_classification_split_142")
))
UNITS_DIR = RESULTS_DIR / "units"


def _unit_dir(dataset: str, resample: int) -> Path:
    return UNITS_DIR / dataset / f"resample{resample}"


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


def _run_cawpe_hard_vote(meta) -> Dict:
    t0 = time.perf_counter()
    weights = learner_weights(meta)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_enc = _cawpe_hard_vote_tally(meta, weights)
    predict_seconds = time.perf_counter() - t0

    accuracy = accuracy_score(meta.y_test_enc, pred_enc)
    return {"FitSeconds": fit_seconds, "PredictSeconds": predict_seconds, "Accuracy": accuracy, "Error": None}


def _run_tabiclv2(
    meta, tabiclv2_ecoc: bool, group_size: int, stratify_groups_by_class: bool,
    ecoc_backend: str, ecoc_alphabet_size: int, ecoc_redundancy: int, ecoc_code_size: float,
    backbone_kwargs: Optional[dict], random_state: int,
) -> Dict:
    """Same model construction and ``.fit``/``.predict_proba`` calls as
    ``meta_learners.fit_predict_tabiclv2`` (duplicated, not imported --
    see this module's own docstring), fit and predict now timed
    separately instead of as one combined block.
    """
    from src.tsfm_ensemble.backbone import GroupedICLClassifier

    model = GroupedICLClassifier(
        group_size=group_size, backbone="tabicl", backbone_kwargs=backbone_kwargs,
        ecoc=tabiclv2_ecoc, ecoc_backend=ecoc_backend, ecoc_alphabet_size=ecoc_alphabet_size,
        ecoc_redundancy=ecoc_redundancy, ecoc_code_size=ecoc_code_size, random_state=random_state,
    )
    class_group_key = _class_group_key(meta) if stratify_groups_by_class else None

    t0 = time.perf_counter()
    model.fit(meta.X_train, meta.y_train_enc, class_group_key=class_group_key)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = model.predict_proba(meta.X_test)
    predict_seconds = time.perf_counter() - t0

    proba = _realign_columns(proba, model.classes_, meta.n_classes)
    accuracy = accuracy_from_proba(proba, meta.y_test_enc)
    return {"FitSeconds": fit_seconds, "PredictSeconds": predict_seconds, "Accuracy": accuracy, "Error": None}


def _run_unit(
    dataset: str, resample: int, chosen_arms: Dict[str, Dict[str, str]], tabiclv2_ecoc: bool,
    tabiclv2_backbone_kwargs: Optional[dict], tabiclv2_group_size: int, tabiclv2_stratify_groups_by_class: bool,
    tabiclv2_ecoc_backend: str, tabiclv2_ecoc_alphabet_size: int, tabiclv2_ecoc_redundancy: int,
    tabiclv2_ecoc_code_size: float,
) -> Dict:
    job_dir = _unit_dir(dataset, resample)
    job_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest_rows: List[Dict] = []

    try:
        scores_dict = _load_unit_scores(dataset, resample, chosen_arms)
        meta = assemble_meta_features(scores_dict)
        n_classes, n_train, n_test = meta.n_classes, meta.X_train.shape[0], meta.X_test.shape[0]
        base_row = {"Dataset": dataset, "Resample": resample, "NClasses": n_classes, "NTrain": n_train, "NTest": n_test}

        try:
            result = _run_cawpe_hard_vote(meta)
        except Exception as exc:  # noqa: BLE001 - one method's failure must not sink the whole unit
            result = {"FitSeconds": None, "PredictSeconds": None, "Accuracy": None, "Error": repr(exc)}
        manifest_rows.append({**base_row, "Method": "cawpe_hard_vote", **result})

        try:
            result = _run_tabiclv2(
                meta, tabiclv2_ecoc=tabiclv2_ecoc, group_size=tabiclv2_group_size,
                stratify_groups_by_class=tabiclv2_stratify_groups_by_class, ecoc_backend=tabiclv2_ecoc_backend,
                ecoc_alphabet_size=tabiclv2_ecoc_alphabet_size, ecoc_redundancy=tabiclv2_ecoc_redundancy,
                ecoc_code_size=tabiclv2_ecoc_code_size, backbone_kwargs=tabiclv2_backbone_kwargs,
                random_state=RANDOM_STATE,
            )
        except Exception as exc:  # noqa: BLE001
            result = {"FitSeconds": None, "PredictSeconds": None, "Accuracy": None, "Error": repr(exc)}
        manifest_rows.append({**base_row, "Method": "tabiclv2", **result})

        pd.DataFrame(manifest_rows).to_csv(job_dir / "manifest.csv", index=False)
        (job_dir / "DONE").touch()
        error_marker = job_dir / "ERROR.txt"
        if error_marker.is_file():
            error_marker.unlink()
        return {"dataset": dataset, "resample": resample, "status": "done", "elapsed": time.perf_counter() - start}
    except Exception:  # noqa: BLE001 - one bad job (e.g. missing fragment) must not abort the whole run
        (job_dir / "ERROR.txt").write_text(traceback.format_exc())
        return {"dataset": dataset, "resample": resample, "status": "error", "elapsed": time.perf_counter() - start}


# Distinct from cv_ablation_142.py's JOB_SHUFFLE_SEED (20260915) and
# measure_base_learner_final_split_142.py's (20260929), since this is yet
# another job list; has zero effect on any result, only on which unit runs
# when. Dataset-interleaved round-robin order matters here too: in plain
# dataset-major order, all --max-workers workers would pull consecutive
# resamples of whichever dataset sorts first, so if that dataset happens to
# be large, every worker hits TabICLv2's memory-bound path on it at once --
# the same failure mode documented in cv_ablation_142.py's own build_jobs.
JOB_SHUFFLE_SEED = 20260930


@dataclass(frozen=True)
class Job:
    dataset: str
    resample: int


def build_jobs(datasets: List[str], resamples: int) -> List[Job]:
    by_dataset: Dict[str, List[Job]] = {}
    for dataset in datasets:
        for resample in range(resamples):
            if not (_unit_dir(dataset, resample) / "DONE").is_file():
                by_dataset.setdefault(dataset, []).append(Job(dataset, resample))

    scheduler_rng = random.Random(JOB_SHUFFLE_SEED)
    dataset_order = list(by_dataset)
    scheduler_rng.shuffle(dataset_order)
    for dataset in dataset_order:
        scheduler_rng.shuffle(by_dataset[dataset])

    queues = [by_dataset[dataset] for dataset in dataset_order]
    jobs: List[Job] = []
    while queues:
        next_round = []
        for queue in queues:
            jobs.append(queue.pop(0))
            if queue:
                next_round.append(queue)
        queues = next_round
    return jobs


def finalize() -> pd.DataFrame:
    rows = []
    for manifest_path in UNITS_DIR.glob("*/*/manifest.csv"):
        if (manifest_path.parent / "DONE").is_file():
            rows.append(pd.read_csv(manifest_path))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help=(
            f"Defaults to min(cpu_count - {RESERVED_CORES}, {DEFAULT_MAX_TABICLV2_WORKERS}) -- every job "
            "here includes a real TabICLv2 fit; see build_meta_model_predictions.py's docstring's "
            "'Parallelism and memory' section for why this default is capped well below cpu_count, and "
            "profile_tabiclv2_memory.py for measuring whether a higher value is safe on your machine."
        ),
    )
    parser.add_argument("--resamples", type=int, default=NUM_RESAMPLES)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated subset (default: all 142).")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--tabiclv2-ecoc", choices=("auto", "native", "ecoc"), default="auto")
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
    UNITS_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        df = finalize()
        print(f"Finalized {len(df)} completed unit(s) -> {RESULTS_DIR / 'manifest.csv'}")
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
    print(f"TabICLv2 ECOC: {tabiclv2_ecoc} ({ecoc_detail})")

    chosen_arms, chosen_arms_detail = _load_chosen_arms()
    print(f"Base-learner arm selection: {chosen_arms_detail}")

    datasets = args.datasets.split(",") if args.datasets else list_142_datasets()
    jobs = build_jobs(datasets, args.resamples)
    n_total = len(datasets) * args.resamples
    print(f"{len(jobs)} job(s) to run ({n_total - len(jobs)} already done, resuming) "
          f"across {len(datasets)} dataset(s) x {args.resamples} resample(s).")

    if not jobs:
        print("Nothing to do.")
        finalize()
        return
    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    import concurrent.futures as cf

    cpu_count = os.cpu_count() or 1
    n_workers = args.max_workers or min(max(1, cpu_count - RESERVED_CORES), DEFAULT_MAX_TABICLV2_WORKERS)
    print(f"Detected {cpu_count} CPU core(s) -> {n_workers} worker(s) (capped for TabICLv2's memory footprint).")

    mp_context = multiprocessing.get_context("spawn")
    completed, errored = 0, 0
    start_time = time.perf_counter()

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context, initializer=_worker_init) as executor:
        in_flight = {}
        remaining = list(jobs)
        n_finished = 0

        while remaining or in_flight:
            while remaining and len(in_flight) < n_workers:
                job = remaining.pop(0)
                future = executor.submit(
                    _run_unit, job.dataset, job.resample, chosen_arms, tabiclv2_ecoc,
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
                else:
                    errored += 1
                elapsed = time.perf_counter() - start_time
                rate = n_finished / elapsed if elapsed > 0 else 0.0
                eta = (len(jobs) - n_finished) / rate if rate > 0 else float("nan")
                print(f"[{n_finished}/{len(jobs)}] {completed} done, {errored} errored "
                      f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min remaining) -- "
                      f"last: {job.dataset} resample{job.resample} ({result['status']}, {result['elapsed']:.1f}s)")
                if args.finalize_every and n_finished % args.finalize_every == 0:
                    finalize()

    print(f"\nRun complete: {completed} done, {errored} errored, {time.perf_counter() - start_time:.1f}s total.")
    df = finalize()
    print(f"Finalized {len(df)} completed unit(s) -> {RESULTS_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
