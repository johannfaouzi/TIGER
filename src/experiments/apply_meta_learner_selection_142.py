"""Apply the dev-subset-derived automatic meta-learner selection rule
(``tsfm_ensemble.meta_learner_selection.select_meta_learner``) to the full
142-dataset x 30-resample benchmark, and add it as a new ``auto_selected``
method row in ``results/meta_model_predictions_142/manifest.csv``.

This is the intended use of the rule: it was designed using only the 20
development datasets and never touched this 142-dataset benchmark while
being derived (see ``meta_learner_selection``'s own module docstring). This
script evaluates it, completely unmodified, on this held-out set -- it does
NOT re-derive or re-tune anything here.

Two things this script does, per (dataset, resample) job, mirroring
``build_meta_model_predictions_142.py``'s own ``_run_methods``/``_record``
convention exactly (same row schema, same npz-payload key convention) so
the new rows slot into the existing manifest/predictions.npz files as if
they'd been produced by the original run:

1. Computes ``cawpe_hard_vote`` fresh -- it was never part of the original
   142-dataset run (deliberately dev-subset-only when first added; see
   ``meta_learners.cawpe_hard_vote``'s own docstring). This is cheap (no
   model fitting, just a weighted vote over already-cached base-learner
   scores reloaded via this pipeline's own ``_load_unit_scores``/
   ``_load_chosen_arms``), so unlike the TabICLv2 patch this runs directly
   in this sandbox -- no ``tabicl``/``torch`` dependency.
2. Computes this job's NTrain/NClasses, applies
   ``select_meta_learner(NTrain, NClasses)`` to pick between
   ``cawpe_hard_vote`` and ``tabiclv2`` (the rule's only two possible
   outputs), and records that choice as a new ``auto_selected`` method row
   -- exactly like ``meta_oracle`` already does for its own pick: looks up
   the selected method's ``{method}__test_proba``/``{method}__test_pred_enc``
   from the (possibly just-updated) npz payload and re-records it under
   ``auto_selected__*``, with ``SelectedMethod`` noting which one it was.

Resumable like every other per-job pipeline in this project: each job's
manifest fragment is read, has its ``cawpe_hard_vote``/``auto_selected``
rows replaced (idempotent), and rewritten -- a job already carrying both is
just overwritten with the same result on a rerun, so this script can be
run repeatedly across several time-budgeted invocations.

Usage (from the repo root):
    python3 -m src.experiments.apply_meta_learner_selection_142 [--time-budget SECONDS]
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.experiments.build_meta_model_predictions_142 import (
    NUM_RESAMPLES,
    _is_done,
    _load_chosen_arms,
    _load_unit_scores,
    _paths,
    finalize,
    list_142_datasets,
)
from src.tsfm_ensemble.meta_learner_selection import select_meta_learner
from src.tsfm_ensemble.meta_learners import (
    accuracy_from_proba,
    assemble_meta_features,
    cawpe_hard_vote,
)


def _record(manifest_rows, npz_payload, meta, dataset, resample, n_classes, n_train, n_test,
            method: str, elapsed: float, proba=None, pred_enc=None,
            error: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    """Identical in shape to ``build_meta_model_predictions_142.py``'s own
    ``_run_methods``'s inner ``_record`` closure, reproduced here (not
    imported -- it's a closure, not a top-level function) so new rows use
    exactly the same schema and npz-key convention as the rest of this
    pipeline's manifest/predictions.npz files.
    """
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
    return row


def _patch_one(dataset: str, resample: int, chosen_arms: dict) -> dict:
    if not _is_done(dataset, resample):
        return {"dataset": dataset, "resample": resample, "status": "skip",
                "reason": "original job not done"}

    job_dir, predictions_path = _paths(dataset, resample)
    manifest_fragment_path = job_dir / "manifest.csv"
    if not manifest_fragment_path.is_file():
        return {"dataset": dataset, "resample": resample, "status": "skip",
                "reason": "manifest.csv fragment missing"}

    scores_dict = _load_unit_scores(dataset, resample, chosen_arms)
    meta = assemble_meta_features(scores_dict)
    n_train, n_test, n_classes = meta.X_train.shape[0], meta.X_test.shape[0], meta.n_classes

    with np.load(predictions_path, allow_pickle=False) as existing:
        npz_payload = dict(existing.items())
    old_fragment = pd.read_csv(manifest_fragment_path)
    other_rows = old_fragment[~old_fragment["Method"].isin(["cawpe_hard_vote", "auto_selected"])].to_dict("records")

    manifest_rows = []

    # 1. cawpe_hard_vote: fresh, never part of the original 142-scale run.
    start = time.perf_counter()
    try:
        pred_enc = cawpe_hard_vote(meta)
        _record(manifest_rows, npz_payload, meta, dataset, resample, n_classes, n_train, n_test,
                "cawpe_hard_vote", time.perf_counter() - start, pred_enc=pred_enc)
    except Exception as exc:  # noqa: BLE001 -- one bad job must not abort the whole run
        _record(manifest_rows, npz_payload, meta, dataset, resample, n_classes, n_train, n_test,
                "cawpe_hard_vote", time.perf_counter() - start, error=repr(exc))

    # 2. auto_selected: select_meta_learner's pick, re-recorded from
    #    whichever of cawpe_hard_vote (just computed above) / tabiclv2
    #    (already present in the original run's npz payload) it names --
    #    exactly mirroring how meta_oracle already re-records its own pick.
    start = time.perf_counter()
    try:
        selected_method = select_meta_learner(n_train, n_classes)
        selected_row = next(
            (r for r in (other_rows + manifest_rows) if r["Method"] == selected_method), None,
        )
        if selected_row is None:
            raise ValueError(f"auto_selected: no manifest row found for selected method {selected_method!r}.")
        if selected_row["Error"] is not None and not (isinstance(selected_row["Error"], float) and pd.isna(selected_row["Error"])):
            raise ValueError(f"auto_selected: selected method {selected_method!r} itself errored: {selected_row['Error']}")
        proba = npz_payload.get(f"{selected_method}__test_proba")
        pred_enc = None if proba is not None else npz_payload[f"{selected_method}__test_pred_enc"]
        _record(manifest_rows, npz_payload, meta, dataset, resample, n_classes, n_train, n_test,
                "auto_selected", time.perf_counter() - start, proba=proba, pred_enc=pred_enc,
                extra={"SelectedMethod": selected_method})
    except Exception as exc:  # noqa: BLE001
        _record(manifest_rows, npz_payload, meta, dataset, resample, n_classes, n_train, n_test,
                "auto_selected", time.perf_counter() - start, error=repr(exc))

    np.savez_compressed(predictions_path, **npz_payload)
    pd.DataFrame(other_rows + manifest_rows).to_csv(manifest_fragment_path, index=False)

    chv_acc = next((r["Accuracy"] for r in manifest_rows if r["Method"] == "cawpe_hard_vote"), None)
    auto_row = next(r for r in manifest_rows if r["Method"] == "auto_selected")
    return {"dataset": dataset, "resample": resample, "status": "done",
            "cawpe_hard_vote_acc": chv_acc, "selected_method": auto_row.get("SelectedMethod"),
            "auto_selected_acc": auto_row["Accuracy"], "auto_selected_error": auto_row["Error"]}


def _already_patched(dataset: str, resample: int) -> bool:
    job_dir, _ = _paths(dataset, resample)
    manifest_fragment_path = job_dir / "manifest.csv"
    if not manifest_fragment_path.is_file():
        return False
    methods = set(pd.read_csv(manifest_fragment_path, usecols=["Method"])["Method"])
    return {"cawpe_hard_vote", "auto_selected"}.issubset(methods)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget", type=float, default=150.0,
                         help="Stop after this many seconds (default: 150) so this fits one call.")
    args = parser.parse_args()

    chosen_arms, chosen_arms_detail = _load_chosen_arms()
    print(f"Base-learner arm selection: {chosen_arms_detail}")

    datasets = list_142_datasets()
    all_jobs = [(ds, r) for ds in datasets for r in range(NUM_RESAMPLES)]
    remaining = [(ds, r) for ds, r in all_jobs if not _already_patched(ds, r)]
    print(f"{len(remaining)} job(s) remaining out of {len(all_jobs)} total "
          f"({len(datasets)} datasets x {NUM_RESAMPLES} resamples).")

    start = time.perf_counter()
    n_done = n_skip = n_error = 0
    for dataset, resample in remaining:
        if time.perf_counter() - start > args.time_budget:
            print(f"Time budget reached after {n_done + n_skip + n_error} job(s) this run.")
            break
        try:
            r = _patch_one(dataset, resample, chosen_arms)
        except Exception as exc:  # noqa: BLE001 -- one bad job must not abort the whole run
            n_error += 1
            print(f"  ERROR {dataset} resample{resample}: {exc!r}")
            continue
        if r["status"] == "done":
            n_done += 1
        else:
            n_skip += 1
            print(f"  SKIP  {dataset} resample{resample}: {r.get('reason')}")

    print(f"\nThis invocation: {n_done} patched, {n_skip} skipped, {n_error} errored "
          f"in {time.perf_counter() - start:.1f}s.")
    finalize()


if __name__ == "__main__":
    main()
