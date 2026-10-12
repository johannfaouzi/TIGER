"""Shared helpers for the ``src/experiments/`` scripts -- kept dependency-
light (no aeon/sklearn imports at module level) so it stays importable
regardless of which aeon version happens to be installed (see
``run_142_ucr_datasets.py``'s own module docstring history: its top-level
``from aeon.datasets.tsc_datasets import redux`` import is aeon-version-
sensitive, which this module deliberately avoids depending on).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np

# Two known-benign sklearn UserWarnings, silenced here (not just in
# tsfm_ensemble.base_learners) so every src/experiments/ script gets them
# regardless of whether it happens to import base_learners -- in
# particular save_dev_subset_transforms.py, which never imports
# base_learners.py (it only does feature *extraction*, not classifier
# fitting) but still triggers the first warning below whenever a family
# extractor's own internal machinery is label-aware on a many-class
# dataset (e.g. DictionaryFamily's chi2_top_k feature selection, or
# shapelet/RDST's supervised shapelet selection, both of which pass `y`
# to sklearn/aeon internals). Registering the filter only in
# base_learners.py left extraction-time occurrences of this warning
# uncovered -- this module is the one thing every src/experiments/
# script already imports, so it's the right place for filters meant to
# apply project-wide rather than to one specific script.
#
#   1. "The number of unique classes is greater than 50% of the number
#      of samples. `y` could represent a regression problem, not a
#      classification problem." -- sklearn's type_of_target heuristic
#      for catching a mislabeled regression target. A false positive for
#      this project's actual data: UCR/UEA TSC datasets routinely have
#      dozens of genuine classes against a modest per-resample sample
#      count (this project's own dev subset includes datasets with 52
#      classes at 2 training instances each). Verified this doesn't
#      swallow this project's own deliberate warnings (_make_cv's
#      "cannot stratify", run_family_base_learners's per-learner
#      OOF-failure warning) -- those use different message text and stay
#      fully visible; also verified (fitting a 200-tree
#      ExtraTreesClassifier with n_jobs=-1 on data that triggers this
#      201 times, once per tree plus once at the forest level) that this
#      filter, registered once at import time, does still apply under
#      sklearn's own internal (thread-based) parallelism -- 0 escapes.
#   2. "`sklearn.utils.parallel.delayed` should be used with
#      `sklearn.utils.parallel.Parallel` to make it possible to
#      propagate the scikit-learn configuration of the current thread to
#      the joblib workers." -- not from this project's own code: aeon's
#      WEASELTransformerV2 (aeon/classification/dictionary_based/
#      _weasel_v2.py) imports Parallel/delayed directly from joblib
#      rather than sklearn.utils.parallel's wrapped versions (confirmed
#      by reading that file's imports), which is exactly the mismatch
#      this warning flags. Harmless for this project: the only effect of
#      the missing propagation is that joblib workers spawned inside
#      that mismatch don't inherit sklearn's thread-local config
#      (working_memory, assume_finite, etc.), none of which this project
#      customizes away from sklearn's own defaults.
for _message in (
    r"The number of unique classes is greater than 50% of the number of samples\.",
    r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`",
):
    warnings.filterwarnings("ignore", message=_message, category=UserWarning)

RESAMPLE_INDICES_DIR = Path("PythonResampleIndices")

# A local folder with aeon-format datasets already extracted, one
# subfolder per dataset (aeon's own extract_path convention:
# <dir>/<dataset>/<dataset>_TRAIN.ts etc.) -- checked first, before
# falling back to aeon's own default download-and-cache location. This
# sidesteps two real problems with relying on the default cache alone:
# some servers can't reach timeseriesclassification.com at all (no
# network egress), and even when they can, a previously-interrupted
# download can leave a corrupted/truncated .ts file silently cached
# there, which surfaces later as a confusing parse error (e.g. aeon's
# "Inconsistent number of dimensions in case N" -- a malformed/blank
# line in a partially-downloaded file, not a problem with the dataset
# itself). Defaults to the ``datasets/`` folder at the repo root
# (alongside ``src/``) -- like RESAMPLE_INDICES_DIR above, a plain
# relative path, since these scripts are always run from the repo root
# (``python -m src.experiments.<script>``). Override via the
# LOCAL_AEON_DATA_DIR env var if needed.
LOCAL_AEON_DATA_DIR = Path(os.environ.get("LOCAL_AEON_DATA_DIR", "datasets"))


def detect_cpu_count() -> int:
    """Cores actually available to this process -- ``os.sched_getaffinity``
    (Linux-only) respects cgroup/container/SLURM CPU limits, which
    ``os.cpu_count()`` does not (it reports the *machine's* total core
    count even when this process is confined to a subset of them, e.g.
    inside a Docker container with ``--cpus`` set, or a shared cluster
    node). Falls back to ``os.cpu_count()`` on platforms without
    ``sched_getaffinity`` (e.g. macOS), then to 1 as a last resort.

    Shared by every ``src/experiments/`` script that schedules its own
    worker processes (``save_dev_subset_transforms.py``,
    ``build_stacking_features.py``) so they agree on what "available
    cores" means.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def load_resample(
    dataset: str,
    resample: int,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test_full: np.ndarray,
    y_test_full: np.ndarray,
    resample_indices_dir: Path = RESAMPLE_INDICES_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild one resample's train/test split from the provided indices.

    Resample 0 is always the dataset's original train/test split, as
    published; resamples 1+ pool train+test together and re-split
    according to ``PythonResampleIndices/<dataset>/resample<k>Indices_*.txt``
    -- the standard UCR-archive resampling scheme, so results are directly
    comparable with ``comparisons/``'s published per-resample scores.
    """
    if resample == 0:
        return X_train_full, y_train_full, X_test_full, y_test_full

    X_pooled = np.concatenate([X_train_full, X_test_full], axis=0)
    y_pooled = np.concatenate([y_train_full, y_test_full], axis=0)
    indices_pooled = np.arange(X_pooled.shape[0])

    train_indices_path = resample_indices_dir / dataset / f"resample{resample}Indices_TRAIN.txt"
    if not train_indices_path.is_file():
        raise FileNotFoundError(f"Could not find resample indices file {train_indices_path}.")
    indices_train = np.loadtxt(train_indices_path, dtype=np.int64)
    indices_test = np.setdiff1d(indices_pooled, indices_train)

    X_train, y_train = X_pooled[indices_train], y_pooled[indices_train]
    X_test, y_test = X_pooled[indices_test], y_pooled[indices_test]
    return X_train, y_train, X_test, y_test


def load_classification_with_fallback(
    dataset: str,
    split: str,
    load_equal_length: bool = True,
    load_no_missing: bool = True,
    local_dir: Path = LOCAL_AEON_DATA_DIR,
):
    """Load one split of ``dataset`` via ``aeon.datasets.load_classification``,
    preferring a local pre-populated data folder over aeon's own
    download-and-cache behaviour.

    Tries ``extract_path=local_dir`` first -- if ``dataset`` is there,
    this never touches the network and can't hit a corrupted-cache parse
    error. Falls back to ``extract_path=None`` (aeon's own default) on
    *any* failure from the local attempt, covering both "not present
    locally" and "present but something's wrong with it" -- if the
    fallback also fails, that (aeon's own) exception propagates normally,
    still a real, actionable error.
    """
    from aeon.datasets import load_classification

    try:
        return load_classification(
            dataset, split=split, extract_path=str(local_dir),
            load_equal_length=load_equal_length, load_no_missing=load_no_missing,
        )
    except Exception as exc:
        print(
            f"  [{dataset}/{split}] not loadable from {local_dir} ({exc!r}); "
            "falling back to aeon's default (extract_path=None)."
        )
        return load_classification(
            dataset, split=split, extract_path=None,
            load_equal_length=load_equal_length, load_no_missing=load_no_missing,
        )


def dev_subset_datasets(path: Path = Path("results") / "family_strengths" / "dev_subset.csv") -> list:
    """The current development-subset dataset names (see
    ``analyze_family_strengths.py``), in file order.
    """
    import pandas as pd  # local import: keep this module's own top-level imports minimal

    return pd.read_csv(path, usecols=[0])["Dataset"].tolist()


def load_features_npz(path: Path):
    """Load one .npz written by this project's own dense-or-sparse
    feature-saving convention (an ``is_sparse`` bool flag; CSR components
    -- ``data``/``indices``/``indptr``/``shape`` -- if sparse, else a
    plain ``X`` array; ``y`` either way) back into ``(X, y)``.

    Shared by ``save_dev_subset_transforms.py`` (writes these, see
    ``_save_features`` there) and any later reader of its raw/ cache
    (e.g. ``tsfm_ensemble.base_learners``) -- kept here rather than
    duplicated in each so both stay in exact agreement about the file
    format.
    """
    from scipy import sparse as sp  # local import: keep this module dependency-light at import time

    with np.load(path, allow_pickle=False) as data:
        y = data["y"]
        if bool(data["is_sparse"]):
            X = sp.csr_matrix(
                (data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"])
            )
        else:
            X = data["X"]
    return X, y


# Must match save_dev_subset_transforms.py's own RESULTS_DIR default --
# duplicated here (rather than imported from that module) so this stays
# importable without pulling in that script's own thread-limiting env-var
# setup at import time (which deliberately runs before numpy is touched
# at all -- see that module's own top-of-file comment).
DEV_SUBSET_TRANSFORMS_RESULTS_DIR = Path(
    os.environ.get("DEV_SUBSET_TRANSFORMS_RESULTS_DIR", str(Path("results") / "dev_subset_transforms"))
)


def load_raw_family_features(
    dataset: str, resample: int, family: str, split: str,
    results_dir: Path = DEV_SUBSET_TRANSFORMS_RESULTS_DIR,
):
    """Load one family's cached raw (pre-reduction) extractor output for
    one (dataset, resample, split) -- a convenience wrapper around
    ``load_features_npz`` that builds the path per
    ``save_dev_subset_transforms.py``'s own
    ``raw/<dataset>/resample<k>/<family>_{train,test}.npz`` layout (see
    that module's docstring for the full on-disk format).
    """
    path = results_dir / "raw" / dataset / f"resample{resample}" / f"{family}_{split}.npz"
    return load_features_npz(path)
