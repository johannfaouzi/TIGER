"""Try loading every dataset in the 142-dataset UCR benchmark via aeon's
default download-and-cache path (``extract_path=None``), and report which
ones fail -- a quick audit for "which datasets does this server actually
have working network/cache access to" before running the real,
expensive experiment scripts (``run_142_ucr_datasets.py``,
``save_dev_subset_transforms.py``), both of which now prefer a local
data folder (``src.experiments.common.LOCAL_AEON_DATA_DIR``) but still
fall back to this same ``extract_path=None`` path for anything not
found there -- so this script's failures are exactly the datasets that
fallback can't rescue either, and are candidates for adding to that
local folder by hand.

Deliberately does *not* use ``common.load_classification_with_fallback``
-- the whole point here is to test the ``extract_path=None`` path in
isolation, not to have a local copy silently paper over a dataset that's
actually broken via the default path.

Usage
-----
    python -m src.experiments.check_dataset_loading
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import redux, univariate_equal_length

RESULTS_DIR = Path("results")


def list_142_datasets() -> list:
    """The full 142-dataset list -- ``run_142_ucr_datasets.py``'s own
    ``list_datasets()`` (with ``DEVELOPMENT = False``) computes this same
    ``sorted(list(univariate_equal_length) + list(redux))`` expression,
    but then slices it to ``[:2]``, which looks like a leftover smoke-test
    truncation rather than intentional (its own docstring says "the 142
    UCR univariate datasets", and this expression's un-sliced length
    should match the 142 directories under ``PythonResampleIndices/``) --
    reproduced here without that slice, since this script's whole purpose
    is checking all 142, not 2.
    """
    return sorted(list(univariate_equal_length) + list(redux))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = list_142_datasets()
    print(f"Checking {len(datasets)} dataset(s) via extract_path=None...\n")

    rows = []
    failed = []
    for i, dataset in enumerate(datasets):
        start = time.perf_counter()
        try:
            load_classification(dataset, split="train", extract_path=None, load_equal_length=True, load_no_missing=True)
            load_classification(dataset, split="test", extract_path=None, load_equal_length=True, load_no_missing=True)
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not abort the check
            elapsed = time.perf_counter() - start
            print(f"[{i + 1:>3}/{len(datasets)}] FAIL {dataset:32s} ({elapsed:5.1f}s): {exc!r}")
            rows.append({"Dataset": dataset, "Status": "fail", "Seconds": elapsed, "Error": repr(exc)})
            failed.append(dataset)
        else:
            elapsed = time.perf_counter() - start
            print(f"[{i + 1:>3}/{len(datasets)}] OK   {dataset:32s} ({elapsed:5.1f}s)")
            rows.append({"Dataset": dataset, "Status": "ok", "Seconds": elapsed, "Error": ""})

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "dataset_loading_check.csv", index=False)

    print(f"\n{len(datasets) - len(failed)}/{len(datasets)} loaded successfully.")
    if failed:
        print(f"\n{len(failed)} dataset(s) FAILED to load via extract_path=None:")
        for dataset in failed:
            print(f"  {dataset}")
    else:
        print("\nAll datasets loaded successfully.")
    print(f"\nFull results -> {RESULTS_DIR / 'dataset_loading_check.csv'}")


if __name__ == "__main__":
    main()
