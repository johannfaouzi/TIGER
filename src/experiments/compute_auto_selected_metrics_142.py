"""Compute BalancedAccuracy/LogLoss(NLL)/ROCAUC(AUROC)/F1 for the
``auto_selected`` method (TIGER's deployed adaptive meta-classification
rule) on the full 142-dataset x 30-resample benchmark, and patch them into
``results/meta_model_predictions_142/manifest.csv``.

Why this exists
----------------
``build_meta_model_predictions_142.py``/``apply_meta_learner_selection_142.py``
only ever recorded ``Accuracy`` for every method, including
``auto_selected`` -- no BalancedAccuracy/LogLoss/ROCAUC/F1 anywhere in this
pipeline (see those two modules' own ``_record`` closures). The paper's
Table 4 (``tab:multi-metric-142``) and Appendix Table A1
(``tab:multi-metric-tabiclv2-subset``) both report AUROC/NLL for TIGER, but
no script in this repository ever computed them -- confirmed by grepping
the whole ``src/`` tree for ``roc_auc_score``/``log_loss``: the only hits
are in ``run_142_original_algorithms.py``, which computes these metrics
for the four *literature* algorithms (RDST, QUANT, WEASEL2.0, KG-MTP)
only, never for this project's own stacking ensemble. This script closes
that gap for ``auto_selected`` specifically (the method actually reported
as "TIGER" in the paper), using exactly the same metric formulas as
``run_142_original_algorithms.py``'s own ``compute_classification_metrics``
(reproduced verbatim below, not imported, for the same reason that module
gives for reproducing it from the sibling ``moment-quant`` project: small,
self-contained, and this keeps every comparison in the paper on one
formula without an awkward cross-script import).

The one-hot convention when WHMV is selected
----------------------------------------------
``auto_selected``'s own npz payload (see ``build_meta_model_predictions_142.
_run_methods``'s ``_record`` closure) only ever stores a
``auto_selected__test_proba`` array when the adaptive rule selected
``tabiclv2`` for that unit -- ``cawpe_hard_vote`` (weighted hard majority
vote, WHMV) has no probabilistic output at all (confirmed directly: no
``cawpe_hard_vote__test_proba`` key exists in any unit's predictions.npz,
by construction of ``meta_learners.cawpe_hard_vote``, a pure vote with no
calibrated score), so on the 2,550 of 4,260 units where WHMV was selected,
only ``auto_selected__test_pred_enc`` (the hard label) is available.

This script fills in AUROC/NLL for exactly those units the same way the
rest of this paper already handles a no-probability classifier -- see
Appendix A's own prose (``whenever TIGER's adaptive rule selects weighted
hard majority vote, we likewise assign a probability of 1 to its
predicted class and 0 to every other class``) and
``run_142_original_algorithms.predict_with_proba``'s identical one-hot
fallback for ``RidgeClassifierCV`` (no averaging or recombination of the
twelve base learners' own scores is attempted -- a hard vote has already
discarded that information by the time it produces a single predicted
label, so there is nothing left to combine).

Resumability
-------------
Same per-job-fragment convention as ``apply_meta_learner_selection_142.py``:
each completed unit's ``results/meta_model_predictions_142/units/<dataset>/
resample<k>/manifest.csv`` fragment is read, has its ``auto_selected`` row's
BalancedAccuracy/LogLoss/ROCAUC/F1 columns added or overwritten in place
(every other row/method in that fragment is left untouched), and
rewritten -- safe to rerun. ``finalize()`` (imported from
``build_meta_model_predictions_142``) then rebuilds the top-level
manifest.csv from all fragments.

As a sanity check, this script also recomputes Accuracy from the same
y_pred/y_proba it builds and asserts it matches the already-recorded
Accuracy column for that row (to 1e-9) -- if that assertion ever fails, the
one-hot/proba reconstruction here has diverged from what originally
produced the cached predictions, and that is a bug to fix, not a
discrepancy to silently average over.

Usage (from the repo root):
    python3 -m src.experiments.compute_auto_selected_metrics_142
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from src.experiments.build_meta_model_predictions_142 import (
    NUM_RESAMPLES,
    _is_done,
    _paths,
    finalize,
    list_142_datasets,
)

METHOD = "auto_selected"


def _one_hot(pred_enc: np.ndarray, n_classes: int) -> np.ndarray:
    """Probability-1-for-the-predicted-class, 0-elsewhere fallback, used
    whenever ``auto_selected`` deferred to WHMV (``cawpe_hard_vote``) for
    this unit -- see this module's own docstring's "one-hot convention"
    section.
    """
    proba = np.zeros((len(pred_enc), n_classes))
    proba[np.arange(len(pred_enc)), pred_enc] = 1.0
    return proba


def compute_classification_metrics(y_true, y_pred, y_proba, n_classes):
    """Verbatim copy of ``run_142_original_algorithms.
    compute_classification_metrics`` -- see that function's own docstring.
    Reproduced here (not imported) so every algorithm in the paper's
    multi-metric tables, literature or this project's own, is scored by
    exactly the same formula, without an awkward cross-script import
    between two otherwise-independent ``src/experiments/`` scripts.
    """
    labels = np.arange(n_classes)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_proba, labels=labels)
    if n_classes == 2:
        auroc = roc_auc_score(y_true, y_proba[:, 1], labels=labels, average="weighted", multi_class="ovr")
    else:
        auroc = roc_auc_score(y_true, y_proba, labels=labels, average="weighted", multi_class="ovr")
    unique, counts = np.unique(y_true, return_counts=True)
    sort_order = np.argsort(unique)
    unique, counts = unique[sort_order], counts[sort_order]
    minority_class = unique[np.flatnonzero(counts == counts.min())[0]]
    if n_classes == 2:
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=minority_class, zero_division=0.0)
    else:
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0.0)
    return dict(BalancedAccuracy=balanced_accuracy, LogLoss=logloss, ROCAUC=auroc, F1=f1, Accuracy=accuracy)


def _process_unit(dataset: str, resample: int) -> dict:
    if not _is_done(dataset, resample):
        return {"dataset": dataset, "resample": resample, "status": "skip", "reason": "original job not done"}

    job_dir, predictions_path = _paths(dataset, resample)
    manifest_fragment_path = job_dir / "manifest.csv"
    if not manifest_fragment_path.is_file():
        return {"dataset": dataset, "resample": resample, "status": "skip", "reason": "manifest fragment missing"}

    with np.load(predictions_path, allow_pickle=False) as npz:
        y_test_enc = npz["y_test_enc"]
        n_classes = len(npz["classes_"])
        pred_enc = npz[f"{METHOD}__test_pred_enc"]
        proba_key = f"{METHOD}__test_proba"
        if proba_key in npz.files:
            proba = npz[proba_key]
            source = "tabiclv2 (genuine proba)"
        else:
            proba = _one_hot(pred_enc, n_classes)
            source = "cawpe_hard_vote (one-hot fallback)"

    metrics = compute_classification_metrics(y_test_enc, pred_enc, proba, n_classes)

    fragment = pd.read_csv(manifest_fragment_path)
    row_mask = fragment["Method"] == METHOD
    if row_mask.sum() != 1:
        raise ValueError(f"{manifest_fragment_path}: expected exactly 1 {METHOD!r} row, found {row_mask.sum()}.")

    existing_accuracy = float(fragment.loc[row_mask, "Accuracy"].iloc[0])
    if not np.isclose(existing_accuracy, metrics["Accuracy"], atol=1e-9):
        raise AssertionError(
            f"{dataset} resample{resample}: recomputed Accuracy ({metrics['Accuracy']!r}) does not match "
            f"already-cached Accuracy ({existing_accuracy!r}) -- proba/pred_enc reconstruction has diverged "
            f"from what originally produced this unit's predictions."
        )

    for col in ("BalancedAccuracy", "LogLoss", "ROCAUC", "F1"):
        fragment.loc[row_mask, col] = metrics[col]
    fragment.to_csv(manifest_fragment_path, index=False)

    return {
        "dataset": dataset, "resample": resample, "status": "done", "source": source,
        "BalancedAccuracy": metrics["BalancedAccuracy"], "LogLoss": metrics["LogLoss"],
        "ROCAUC": metrics["ROCAUC"], "F1": metrics["F1"],
    }


def main() -> None:
    datasets = list_142_datasets()
    all_units = [(ds, r) for ds in datasets for r in range(NUM_RESAMPLES)]
    print(f"Processing {len(all_units)} unit(s) ({len(datasets)} datasets x {NUM_RESAMPLES} resamples)...")

    start = time.perf_counter()
    n_done = n_skip = 0
    n_onehot = n_genuine = 0
    for dataset, resample in all_units:
        r = _process_unit(dataset, resample)
        if r["status"] == "done":
            n_done += 1
            if "one-hot" in r["source"]:
                n_onehot += 1
            else:
                n_genuine += 1
        else:
            n_skip += 1
            print(f"  SKIP {dataset} resample{resample}: {r.get('reason')}")

    print(
        f"\n{n_done} unit(s) patched ({n_genuine} genuine TabICLv2 proba, "
        f"{n_onehot} one-hot WHMV fallback), {n_skip} skipped, "
        f"in {time.perf_counter() - start:.1f}s."
    )
    finalize()


if __name__ == "__main__":
    main()
