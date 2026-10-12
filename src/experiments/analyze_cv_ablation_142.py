"""Pick, per (family, learner), which of ``cv_ablation_142.py``'s scored
arms ("cv" vs "bagB15"/"bagB25" for ridge/naive_bayes; "cv" vs
"nativeoob" for extra_trees) gives the more honest OOF signal, and write
that decision to a small audited JSON file -- so a downstream 142-scale
meta-learner run can consume the *right* base-learner scores for each
(family, learner) without a human first eyeballing
``results/cv_ablation_142/manifest.csv``.

This exists specifically so the pipeline can run unattended: the user
asked whether a follow-up 142x30-scale meta-learner evaluation could
kick off right after ``cv_ablation_142.py`` finishes, without a manual
analysis step in between (they won't be able to look at the results for
a few days). The arm-selection question turns out to be fully mechanical
-- ``cv_ablation_142.py``'s own docstring already nails down the metric
("Score representation caveat" section): Accuracy-based OOF-honesty,
``|OOF acc - true test acc|``, is the one metric that's apples-to-apples
across every arm (LogLoss/ROCAUC are not, for the bagging arms -- see
that docstring). So "pick the arm with the smallest mean |diff|, per
(family, learner)" requires no judgment call, only arithmetic over a
manifest that already exists -- the same shape of decision
``build_meta_model_predictions.py``'s own ``_auto_select_tabiclv2_ecoc``
already makes for TabICLv2's native-vs-ECOC choice, including that
function's own discipline of logging the decision loudly (never a silent
default) so it stays auditable whenever a human does get to look.

Usage
-----
    python -m src.experiments.analyze_cv_ablation_142
    python -m src.experiments.analyze_cv_ablation_142 --manifest results/cv_ablation_142/manifest.csv
    python -m src.experiments.analyze_cv_ablation_142 --min-rows 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.tsfm_ensemble.base_learners import FAMILIES, LEARNER_NAMES

RESULTS_DIR = Path("results") / "cv_ablation_142"

# Candidate arms per learner, and each arm's manifest column prefix /
# fragment .npz key suffix -- see cv_ablation_142.py's own module
# docstring ("Scored arms per (dataset, resample, family, learner)") and
# its _run_unit for where these exact names come from. extra_trees never
# gets bagB15/bagB25 (it's already an ensemble -- see that module's
# "Extra Trees is already an ensemble" section); ridge/naive_bayes never
# get nativeoob (sklearn's bagging-OOB machinery is extra_trees-specific).
ARM_COLUMN_PREFIX = {"cv": "Cv", "bagB15": "Bag15", "bagB25": "Bag25", "nativeoob": "NativeOOB"}
CANDIDATE_ARMS = {
    "ridge": ("cv", "bagB15", "bagB25"),
    "naive_bayes": ("cv", "bagB15", "bagB25"),
    "extra_trees": ("cv", "nativeoob"),
}

# Fragment .npz key suffix for each arm -- identical to the arm names
# above except "cv" itself, kept as an explicit mapping (not assumed
# equal to the dict key) so a future arm-naming change in
# cv_ablation_142.py can't silently desync this from that script's own
# npz_payload keys without this module also needing an update.
ARM_FRAGMENT_SUFFIX = {"cv": "cv", "bagB15": "bagB15", "bagB25": "bagB25", "nativeoob": "nativeoob"}

DEFAULT_ARM = "cv"  # cv_ablation_142.py's own baseline -- what every downstream script used before this ablation existed.


def _arm_honesty(manifest: pd.DataFrame, family: str, learner: str, arm: str) -> Dict:
    """Mean/median/count of Accuracy-based OOF-honesty (``|OOF acc - test
    acc|``) for one (family, learner, arm), over every manifest row where
    that arm scored successfully (its own ``{Prefix}Error`` column is
    NaN) and a true test accuracy was available (``{Prefix}TestAcc`` not
    NaN -- absent for the rare unit where a test-split label was never
    seen in training; see cv_ablation_142.py's ``_run_unit``). Returns
    ``{"mean": nan, "median": nan, "n_rows": 0}`` when nothing qualifies,
    so a caller can still print/compare it without a branch for "missing
    entirely".
    """
    prefix = ARM_COLUMN_PREFIX[arm]
    sub = manifest[(manifest["Family"] == family) & (manifest["Learner"] == learner)]
    error_col = f"{prefix}Error"
    oof_col, test_col = f"{prefix}OOFAcc", f"{prefix}TestAcc"
    if error_col not in sub.columns:
        sub = sub.assign(**{error_col: np.nan})
    if oof_col not in sub.columns or test_col not in sub.columns:
        return {"mean": float("nan"), "median": float("nan"), "n_rows": 0}
    valid = sub[sub[error_col].isna() & sub[oof_col].notna() & sub[test_col].notna()]
    if valid.empty:
        return {"mean": float("nan"), "median": float("nan"), "n_rows": 0}
    diff = (valid[oof_col] - valid[test_col]).abs()
    return {"mean": float(diff.mean()), "median": float(diff.median()), "n_rows": int(len(diff))}


def choose_arms(manifest: pd.DataFrame, min_rows: int = 1) -> Dict[str, Dict[str, Dict]]:
    """Return ``{family: {learner: decision}}`` for every (family,
    learner) in ``FAMILIES x LEARNER_NAMES``, where ``decision`` has
    ``"chosen_arm"``, ``"fragment_suffix"``, ``"reason"`` (always a
    human-readable, print-safe string -- see this module's docstring on
    why this is logged rather than silently decided), and
    ``"candidates"`` (every candidate arm's honesty stats, so the full
    comparison is on record, not just the winner).

    A (family, learner) with fewer than ``min_rows`` qualifying rows for
    EVERY candidate arm falls back to ``DEFAULT_ARM`` ("cv" -- the
    pre-ablation baseline every other script already used) rather than
    picking a winner off too little data; the reason string says so
    explicitly.
    """
    decisions: Dict[str, Dict[str, Dict]] = {}
    for family in FAMILIES:
        decisions[family] = {}
        for learner in LEARNER_NAMES:
            arms = CANDIDATE_ARMS[learner]
            stats = {arm: _arm_honesty(manifest, family, learner, arm) for arm in arms}
            qualifying = {arm: s for arm, s in stats.items() if s["n_rows"] >= min_rows}
            if not qualifying:
                decisions[family][learner] = {
                    "chosen_arm": DEFAULT_ARM,
                    "fragment_suffix": ARM_FRAGMENT_SUFFIX[DEFAULT_ARM],
                    "candidates": stats,
                    "reason": (
                        f"no candidate arm had >= {min_rows} qualifying row(s) for "
                        f"{family}/{learner} -- falling back to {DEFAULT_ARM!r} (the pre-ablation "
                        "baseline) rather than choosing off too little data."
                    ),
                }
                continue
            chosen_arm = min(qualifying, key=lambda arm: qualifying[arm]["mean"])
            ranked = ", ".join(
                f"{arm}={stats[arm]['mean']:.4f} (n={stats[arm]['n_rows']})"
                if stats[arm]["n_rows"] else f"{arm}=n/a (n=0)"
                for arm in arms
            )
            decisions[family][learner] = {
                "chosen_arm": chosen_arm,
                "fragment_suffix": ARM_FRAGMENT_SUFFIX[chosen_arm],
                "candidates": stats,
                "reason": (
                    f"{family}/{learner}: chose {chosen_arm!r} -- smallest mean Accuracy-based "
                    f"OOF-honesty |OOF acc - test acc| among arms with >= {min_rows} qualifying "
                    f"row(s). Compared: {ranked}."
                ),
            }
    return decisions


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default=str(RESULTS_DIR / "manifest.csv"))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "chosen_arms.json"))
    parser.add_argument(
        "--min-rows", type=int, default=100,
        help=(
            "Minimum qualifying (dataset, resample) rows a candidate arm needs to be eligible at "
            "all -- guards against picking a 'winner' off a handful of units if cv_ablation_142.py "
            "was interrupted early. Default 100 (out of up to 142x30=4260 possible per family/learner)."
        ),
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run cv_ablation_142.py (or its --finalize-only) first."
        )
    manifest = pd.read_csv(manifest_path)
    print(f"Loaded {len(manifest)} row(s) from {manifest_path}.")

    decisions = choose_arms(manifest, min_rows=args.min_rows)

    print("\nArm selection, per (family, learner):")
    for family in FAMILIES:
        for learner in LEARNER_NAMES:
            print(f"  {decisions[family][learner]['reason']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decisions, indent=2, sort_keys=True))
    print(f"\nWrote arm decisions -> {output_path}")
    print(
        "This file is read automatically by build_meta_model_predictions_142.py "
        "(see its docstring) -- no manual step needed to apply it."
    )


if __name__ == "__main__":
    main()
