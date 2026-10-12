"""
Build the training-time, inference-time, and combined (sum) runtime
tables for all six algorithms (TIGER plus the five literature algorithms
of Section 5.2), over the 142-data-set benchmark, each data set's own
resamples first collapsed to their median before taking total/median/min/
max across the 142 resulting per-data-set values -- same convention as
the original combined-runtime table this supersedes.

Per-algorithm training/inference definitions:
    TIGER     : results/tiger_train_inference_split_142/manifest.csv
                (TrainingSeconds, InferenceSeconds columns -- see
                compute_train_inference_split_142.py for how these were
                built from the base-learner, CV-ablation, and
                meta-classification manifests).
    KG-MTP, QUANT, RDST, WEASEL 2.0:
                results/run_142_original_algorithms/manifest.csv.
                Training = TrainTransformSeconds + TrainClassifySeconds
                (feature extraction on the training split + classifier
                fit). Inference = TestTransformSeconds + TestClassifySeconds
                (feature extraction on the test split + predict).
    HC2       : results/run_142_hc2/manifest.csv. Training = TrainSeconds
                (hc2.fit, which bundles each of its 4 modules' own
                transform + classifier internally -- HC2 is used here as
                a single opaque estimator, not broken into stages).
                Inference = InferenceSeconds (hc2.predict_proba).
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path.home() / "mnt" / "tsc-with-tfm" / "results"


def per_dataset_medians(df: pd.DataFrame, train_col: str, infer_col: str) -> pd.DataFrame:
    """Given a (Dataset, Resample)-level frame with Training/Inference
    seconds columns, collapse resamples to each data set's median."""
    out = df.groupby("Dataset")[[train_col, infer_col]].median()
    out.columns = ["Training", "Inference"]
    out["Sum"] = df.assign(Sum=df[train_col] + df[infer_col]).groupby("Dataset")["Sum"].median()
    assert len(out) == 142, f"expected 142 data sets, got {len(out)}"
    return out


def load_tiger() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "tiger_train_inference_split_142" / "manifest.csv")
    return per_dataset_medians(df, "TrainingSeconds", "InferenceSeconds")


def load_original_algorithms() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(RESULTS_DIR / "run_142_original_algorithms" / "manifest.csv")
    df["Training"] = df["TrainTransformSeconds"] + df["TrainClassifySeconds"]
    df["Inference"] = df["TestTransformSeconds"] + df["TestClassifySeconds"]
    out = {}
    for alg, label in [("KG-MTP", "KG-MTP"), ("QUANT", "QUANT"), ("RDST", "RDST"), ("WEASEL2.0", "WEASEL~2.0")]:
        sub = df[df["Algorithm"] == alg]
        out[label] = per_dataset_medians(sub, "Training", "Inference")
    return out


def load_hc2() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "run_142_hc2" / "manifest.csv")
    return per_dataset_medians(df, "TrainSeconds", "InferenceSeconds")


def summary_row(per_dataset: pd.DataFrame, col: str) -> tuple[float, float, float, float]:
    s = per_dataset[col]
    return s.sum() / 3600.0, s.median(), s.min(), s.max()


def main() -> None:
    algorithms = {}
    algorithms.update(load_original_algorithms())
    algorithms["HC2"] = load_hc2()
    algorithms["TIGER"] = load_tiger()

    for table_col, table_name in [("Training", "Training"), ("Inference", "Inference"), ("Sum", "Sum (train+test)")]:
        print(f"=== {table_name} ===")
        rows = []
        for alg, per_dataset in algorithms.items():
            total_h, med, mn, mx = summary_row(per_dataset, table_col)
            rows.append((alg, total_h, med, mn, mx))
        rows.sort(key=lambda r: r[2])  # order by median, fastest first
        for alg, total_h, med, mn, mx in rows:
            print(f"  {alg:10s}  total={total_h:8.2f}h  median={med:9.1f}s  min={mn:8.1f}s  max={mx:10.1f}s")
        print()


if __name__ == "__main__":
    main()
