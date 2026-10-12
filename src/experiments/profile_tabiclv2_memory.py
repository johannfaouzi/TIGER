"""Measure real TabICLv2 peak RAM on *this* machine, on *this* project's
own meta-feature shapes -- the calibration step the other two TabICLv2-
invoking scripts' "Parallelism and memory" docstring sections point at
instead of guessing ``--max-workers``/``--ecoc-code-size``/
``--offload-mode`` by feel.

Why this has to run on the real server, not in development
-------------------------------------------------------------
``compare_tabiclv2_multiclass_handling.py``'s and
``build_meta_model_predictions.py``'s current ``DEFAULT_MAX_TABICLV2_WORKERS
= 8`` cap is an explicitly-flagged, *unmeasured* placeholder: this
project's own sandboxed development environment had no network route to
TabICLv2's Hugging Face Hub checkpoint, so real ``tabicl`` fits could never
actually be run and measured there -- only reasoned about from the
installed ``tabicl``/``scikit-learn`` packages' own source (see those two
scripts' docstrings for exactly what that source-level reasoning found:
ECOC mode's ``int(n_classes * ecoc_code_size)`` retained backbone
instances per group, and the ``(batch_size, n_rows, n_columns, embed_dim)``
embedding tensor tabicl's own ``offload_mode`` docstring calls "the main
memory bottleneck"). This script is meant to be run once here, wherever
the checkpoint download *is* reachable, to turn that structural reasoning
into real numbers for your own machine, your own ``tabicl``/torch build,
and your own dev-subset dataset shapes.

What it does
------------
For one or more (dataset, resample) jobs -- by default, the single
*heaviest* qualifying job it can find under ``STACKING_FEATURES_RESULTS_DIR``
(largest ``(n_train + n_test) * n_features``, a proxy for the dominant
embedding tensor's total size) -- runs a real ``fit_predict_tabiclv2`` call
for each requested mode (native and/or ECOC) in its own freshly-spawned
child process, and records that process's peak resident set size
(``resource.getrusage(RUSAGE_SELF).ru_maxrss``, platform-aware: Linux
reports kilobytes, macOS reports bytes -- see ``_peak_rss_mb``) alongside
its wall-clock time. One child process per measurement, run sequentially
(never in parallel with each other) -- that isolates each measurement to
exactly one job's own peak RAM, uninflated by any other concurrent fit,
which is exactly the number ``--max-workers`` capacity planning needs
("how much RAM does *one* worker actually use").

Every row is appended to ``RESULTS_DIR/memory_profile.csv`` (safe to
re-run / extend with different flags across multiple invocations -- rows
are appended, not overwritten). If ``--total-ram-gb`` is given, also
prints a suggested ``--max-workers`` per mode:
``floor((total_ram_gb - reserved_ram_gb) / peak_rss_gb_for_that_mode)``,
capped by this machine's own core count -- a real, measured-input version
of the same capacity question ``DEFAULT_MAX_TABICLV2_WORKERS`` currently
answers with a guess.

This does NOT change either script's own default -- it only measures and
prints/saves. Update ``DEFAULT_MAX_TABICLV2_WORKERS`` in
``compare_tabiclv2_multiclass_handling.py`` yourself (or just keep passing
``--max-workers`` explicitly) once you trust a number from here.

Usage
-----
    # Check which job(s)/mode(s) --device cpu (no other flags) would actually
    # measure, and their real shapes, before running any real (potentially
    # slow) TabICLv2 fits:
    python -m src.experiments.profile_tabiclv2_memory --device cpu --dry-run

    # Profile the single heaviest qualifying dev-subset job, both modes, on CPU:
    python -m src.experiments.profile_tabiclv2_memory --device cpu

    # Profile a specific (dataset, resample) job instead of auto-picking:
    python -m src.experiments.profile_tabiclv2_memory --device cpu \\
        --dataset PigAirwayPressure --resample 0

    # Profile every qualifying job (slower, but a full picture rather than a single worst case):
    python -m src.experiments.profile_tabiclv2_memory --device cpu --all

    # Also get a suggested --max-workers for this machine's real RAM:
    python -m src.experiments.profile_tabiclv2_memory --device cpu \\
        --total-ram-gb 503 --reserved-ram-gb 32

    # Measure the disk-offload lever's effect on the same job:
    python -m src.experiments.profile_tabiclv2_memory --device cpu \\
        --offload-mode disk --disk-offload-dir /tmp/tabicl_offload
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# Same reasoning as the other src/experiments/ scripts' identical block --
# must happen before numpy (or anything that transitively imports it) is
# imported.
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

from src.experiments.common import dev_subset_datasets, detect_cpu_count  # noqa: E402
from src.experiments.compare_tabiclv2_multiclass_handling import (  # noqa: E402
    NUM_RESAMPLES,
    DEFAULT_MIN_CLASSES,
    _scores_path,
    qualifying_jobs,
)
from src.tsfm_ensemble.base_learners import RANDOM_STATE  # noqa: E402

RESULTS_DIR = Path(
    os.environ.get("PROFILE_TABICLV2_MEMORY_RESULTS_DIR", str(Path("results") / "tabiclv2_memory_profile"))
)


def _peak_rss_mb() -> float:
    """This process's own peak resident set size, in MB, platform-aware:
    Linux's ``ru_maxrss`` is kilobytes; macOS's is bytes -- both
    documented behaviour of the underlying ``getrusage(2)`` syscall (not
    a Python-level inconsistency), easy to silently get 1024x wrong if
    not handled explicitly.
    """
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def _job_shape(dataset: str, resample: int) -> Optional[dict]:
    """This (dataset, resample) job's real meta-feature shape (loads its
    already-built ``scores.npz`` and assembles the wide meta-feature
    matrix, the same input every meta-method -- including TabICLv2 --
    actually fits on), or ``None`` if that scores.npz doesn't exist yet.
    Cheap: no model fitting, just array concatenation.
    """
    from src.tsfm_ensemble.meta_learners import assemble_meta_features

    path = _scores_path(dataset, resample)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as scores:
        scores_dict = dict(scores.items())
    meta = assemble_meta_features(scores_dict)
    return {
        "dataset": dataset, "resample": resample,
        "n_train": meta.X_train.shape[0], "n_test": meta.X_test.shape[0],
        "n_features": meta.X_train.shape[1], "n_classes": meta.n_classes,
    }


def _select_jobs(
    datasets: List[str], min_classes: int, dataset: Optional[str], resample: Optional[int], profile_all: bool,
) -> List[dict]:
    """Real shapes (see ``_job_shape``) for whichever job(s) this run
    should profile -- an explicit ``--dataset``/``--resample`` always
    wins; otherwise every qualifying (``>= min_classes`` classes, so ECOC
    is actually exercised -- see ``compare_tabiclv2_multiclass_handling.py``'s
    own ``qualifying_jobs``) job with a built ``scores.npz``, either all
    of them (``--all``) or just the heaviest one by
    ``(n_train + n_test) * n_features`` -- a proxy for the dominant
    per-instance embedding tensor's total size (rows x columns), and (for
    ECOC) ``n_features`` also drives the retained-instance count via
    ``n_groups`` -- so this is a genuine "likely worst case for this
    project's own dev subset", not an arbitrary pick.
    """
    if dataset is not None:
        resamples = [resample] if resample is not None else list(range(NUM_RESAMPLES))
        shapes = [s for s in (_job_shape(dataset, r) for r in resamples) if s is not None]
        return shapes

    all_jobs = qualifying_jobs(datasets, min_classes)
    shapes = [s for s in (_job_shape(job.dataset, job.resample) for job in all_jobs) if s is not None]
    if not shapes or profile_all:
        return shapes
    heaviest = max(shapes, key=lambda s: (s["n_train"] + s["n_test"]) * s["n_features"])
    return [heaviest]


def _measure_one(shape: dict, ecoc: bool, group_size: int, ecoc_code_size: float, backbone_kwargs: Optional[dict]):
    """Run one real ``fit_predict_tabiclv2`` call and report this
    process's own peak RSS afterwards. Meant to be run inside its own
    freshly-spawned child process (see ``main``) so the reported peak RSS
    reflects exactly one job's own memory use, not this launcher script's
    or any sibling measurement's.
    """
    from src.tsfm_ensemble.meta_learners import assemble_meta_features, fit_predict_tabiclv2

    with np.load(_scores_path(shape["dataset"], shape["resample"]), allow_pickle=False) as scores:
        scores_dict = dict(scores.items())
    meta = assemble_meta_features(scores_dict)

    t0 = time.perf_counter()
    error = None
    try:
        fit_predict_tabiclv2(
            meta, ecoc=ecoc, group_size=group_size, ecoc_code_size=ecoc_code_size,
            random_state=RANDOM_STATE, backbone_kwargs=backbone_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - report the failure, don't crash the profiling run
        error = repr(exc)
        traceback.print_exc()
    elapsed = time.perf_counter() - t0

    return {
        "Dataset": shape["dataset"], "Resample": shape["resample"], "Ecoc": ecoc,
        "NClasses": shape["n_classes"], "NTrain": shape["n_train"], "NTest": shape["n_test"],
        "NFeatures": shape["n_features"], "GroupSize": group_size, "EcocCodeSize": ecoc_code_size,
        "BatchSize": (backbone_kwargs or {}).get("batch_size"),
        "OffloadMode": (backbone_kwargs or {}).get("offload_mode"),
        "PeakRssMb": _peak_rss_mb(), "ElapsedSeconds": elapsed, "Error": error,
    }


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


def _child_entrypoint(conn, shape, ecoc, group_size, ecoc_code_size, backbone_kwargs) -> None:
    """Child-process target for ``_run_isolated``. Must be a module-level
    function (not a closure) -- the ``spawn`` start method pickles the
    target, and a closure over ``_run_isolated``'s own locals isn't
    picklable.
    """
    _worker_init()
    result = _measure_one(shape, ecoc, group_size, ecoc_code_size, backbone_kwargs)
    conn.send(result)
    conn.close()


def _run_isolated(shape, ecoc, group_size, ecoc_code_size, backbone_kwargs) -> dict:
    """Run ``_measure_one`` inside a single freshly-spawned child process
    and return its result dict -- one process per measurement, joined
    before the next one starts, so peak-RSS readings never mix two
    measurements together (see the module docstring's "What it does").
    """
    import multiprocessing

    mp_context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = mp_context.Pipe(duplex=False)

    proc = mp_context.Process(
        target=_child_entrypoint, args=(child_conn, shape, ecoc, group_size, ecoc_code_size, backbone_kwargs),
    )
    proc.start()
    child_conn.close()
    try:
        result = parent_conn.recv()
    except EOFError:
        result = None
    proc.join()
    if result is None:
        result = {
            "Dataset": shape["dataset"], "Resample": shape["resample"], "Ecoc": ecoc,
            "NClasses": shape["n_classes"], "NTrain": shape["n_train"], "NTest": shape["n_test"],
            "NFeatures": shape["n_features"], "GroupSize": group_size, "EcocCodeSize": ecoc_code_size,
            "BatchSize": (backbone_kwargs or {}).get("batch_size"),
            "OffloadMode": (backbone_kwargs or {}).get("offload_mode"),
            "PeakRssMb": None, "ElapsedSeconds": None,
            "Error": f"child process exited without a result (exit code {proc.exitcode})",
        }
    return result


def _suggest_max_workers(df: pd.DataFrame, total_ram_gb: float, reserved_ram_gb: float, cpu_count: int) -> None:
    usable_gb = total_ram_gb - reserved_ram_gb
    if usable_gb <= 0:
        print(f"--total-ram-gb ({total_ram_gb}) <= --reserved-ram-gb ({reserved_ram_gb}); nothing to suggest.")
        return
    valid = df[df["Error"].isna() & df["PeakRssMb"].notna()]
    if valid.empty:
        print("No successful measurement to base a --max-workers suggestion on.")
        return
    print(
        f"\nSuggested --max-workers for {total_ram_gb:.0f}GB total RAM "
        f"({reserved_ram_gb:.0f}GB reserved -> {usable_gb:.0f}GB usable), by mode "
        "(measured peak RSS of one worker, this run's rows only):"
    )
    for ecoc, group in valid.groupby("Ecoc"):
        worst_mb = group["PeakRssMb"].max()
        worst_gb = worst_mb / 1024.0
        by_memory = int(usable_gb // worst_gb) if worst_gb > 0 else cpu_count
        suggested = max(1, min(by_memory, cpu_count - 1))
        mode = "ecoc" if ecoc else "native"
        print(
            f"  {mode:6s}: worst-case measured peak RSS = {worst_mb:.0f}MB/worker -> "
            f"floor({usable_gb:.0f}GB / {worst_gb:.2f}GB) = {by_memory}, capped at cpu_count-1={cpu_count - 1} "
            f"-> suggest --max-workers {suggested}"
        )
    print(
        "This is only as representative as the job(s) profiled -- re-run with --all if you want every "
        "qualifying dev-subset job covered rather than just the heaviest one."
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default=None, help="Profile this dataset instead of auto-picking.")
    parser.add_argument(
        "--resample", type=int, default=None,
        help="Profile this resample only (requires --dataset). Default: every resample 0..4 for that dataset.",
    )
    parser.add_argument(
        "--min-classes", type=int, default=DEFAULT_MIN_CLASSES,
        help=f"Only auto-pick among (dataset, resample) jobs with at least this many classes (default {DEFAULT_MIN_CLASSES}), matching compare_tabiclv2_multiclass_handling.py's own ECOC-qualifying threshold.",
    )
    parser.add_argument(
        "--all", action="store_true", dest="profile_all",
        help="Profile every qualifying job instead of just the single heaviest one (slower; each job is measured once per --modes entry).",
    )
    parser.add_argument(
        "--modes", type=str, default="native,ecoc",
        help="Comma-separated subset of {native,ecoc} to measure (default: both).",
    )
    parser.add_argument("--group-size", type=int, default=100, help="Same meaning as the other two scripts' --group-size.")
    parser.add_argument("--ecoc-code-size", type=float, default=1.5, help="Same meaning as the other two scripts' --ecoc-code-size (ECOC mode only).")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto", help="Same meaning as the other two scripts' --device.")
    parser.add_argument("--batch-size", type=int, default=None, help="Same meaning as the other two scripts' --batch-size.")
    parser.add_argument("--offload-mode", choices=("auto", "gpu", "cpu", "disk"), default=None, help="Same meaning as the other two scripts' --offload-mode.")
    parser.add_argument("--disk-offload-dir", type=str, default=None, help="Required when --offload-mode disk is set.")
    parser.add_argument(
        "--total-ram-gb", type=float, default=None,
        help="If given, print a suggested --max-workers per mode from this run's measured peak RSS.",
    )
    parser.add_argument(
        "--reserved-ram-gb", type=float, default=16.0,
        help="Headroom subtracted from --total-ram-gb for the OS/other processes before suggesting --max-workers (default 16).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"CSV path to append results to (default {RESULTS_DIR / 'memory_profile.csv'}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print which job(s)/mode(s) would be profiled (and their real shapes), then exit without running anything -- each real measurement is a genuine TabICLv2 fit, worth checking the selection first.",
    )
    args = parser.parse_args(argv)

    if args.resample is not None and args.dataset is None:
        parser.error("--resample requires --dataset.")
    if args.offload_mode == "disk" and not args.disk_offload_dir:
        parser.error("--offload-mode disk requires --disk-offload-dir.")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("native", "ecoc"):
            parser.error(f"--modes: unknown mode {m!r} (expected 'native' and/or 'ecoc').")
    if not modes:
        parser.error("--modes: at least one of native/ecoc is required.")

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

    output_path = Path(args.output) if args.output else RESULTS_DIR / "memory_profile.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = dev_subset_datasets()
    shapes = _select_jobs(datasets, args.min_classes, args.dataset, args.resample, args.profile_all)
    if not shapes:
        print(
            "No qualifying (dataset, resample) job found with a built scores.npz -- run "
            "build_stacking_features.py first (or pass --dataset/--resample directly)."
        )
        return

    print(f"Profiling {len(shapes)} job(s) x {len(modes)} mode(s) = {len(shapes) * len(modes)} measurement(s):")
    for s in shapes:
        print(f"  {s['dataset']} resample{s['resample']}: n_train={s['n_train']} n_test={s['n_test']} n_features={s['n_features']} n_classes={s['n_classes']}")
    print(
        f"Config: device={args.device} group_size={args.group_size} ecoc_code_size={args.ecoc_code_size} "
        f"batch_size={args.batch_size} offload_mode={args.offload_mode} disk_offload_dir={args.disk_offload_dir}"
    )
    print("Each measurement runs in its own freshly-spawned process, sequentially -- this can be slow; that's expected.")
    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    rows = []
    n_done = 0
    n_total = len(shapes) * len(modes)
    for s in shapes:
        for mode in modes:
            ecoc = mode == "ecoc"
            start = time.perf_counter()
            result = _run_isolated(s, ecoc, args.group_size, args.ecoc_code_size, backbone_kwargs)
            elapsed = time.perf_counter() - start
            n_done += 1
            rows.append(result)
            status = "FAIL" if result.get("Error") else "OK"
            peak = f"{result['PeakRssMb']:.0f}MB" if result.get("PeakRssMb") is not None else "n/a"
            print(
                f"[{n_done}/{n_total}] {status} {s['dataset']} resample{s['resample']} mode={mode}: "
                f"peak_rss={peak} wall={elapsed:.1f}s" + (f" error={result['Error']}" if result.get("Error") else "")
            )

    df = pd.DataFrame(rows)
    header = not output_path.is_file()
    df.to_csv(output_path, mode="a", header=header, index=False)
    print(f"\nAppended {len(df)} row(s) -> {output_path}")

    if args.total_ram_gb is not None:
        cpu_count = detect_cpu_count()
        _suggest_max_workers(df, args.total_ram_gb, args.reserved_ram_gb, cpu_count)


if __name__ == "__main__":
    main()
