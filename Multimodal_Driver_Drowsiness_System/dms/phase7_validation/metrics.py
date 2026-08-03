"""
Phase 7 - Real-World Validation: Metrics
==========================================
Computes the metrics needed to validate the system per the Phase 7
deliverable: calibrate detection thresholds, reduce false positives and
false negatives, and measure overall model accuracy/robustness across
conditions (daytime, nighttime, adverse weather).

Requires: numpy, scikit-learn
    pip install numpy scikit-learn --break-system-packages
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    confusion_matrix, f1_score,
)


@dataclass
class ValidationRun:
    condition: str            # "daytime", "nighttime", "rain", "fog", etc.
    y_true: np.ndarray        # ground truth: 1 = drowsy/anomalous event, 0 = normal
    y_score: np.ndarray       # model's continuous score in [0, 1]


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Finds the threshold that maximizes F1, and reports the false
    positive / false negative rate at that operating point -- this is
    the core Phase 7 "calibrate detection thresholds" task."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    best_f1, best_threshold = -1.0, 0.5
    for t in np.unique(y_score):
        preds = (y_score >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    preds = (y_score >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else None,
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
    }


def validate_across_conditions(runs: List[ValidationRun]) -> dict:
    """Per-condition metrics + a robustness summary (how much
    performance degrades in nighttime/adverse weather vs daytime --
    directly answers the Phase 7 'robustness' deliverable)."""
    results = {}
    for run in runs:
        results[run.condition] = find_optimal_threshold(run.y_true, run.y_score)

    aucs = {c: r["auc"] for c, r in results.items() if r["auc"] is not None}
    if "daytime" in aucs and len(aucs) > 1:
        baseline = aucs["daytime"]
        degradation = {
            c: round(baseline - auc, 4)
            for c, auc in aucs.items() if c != "daytime"
        }
        results["_robustness_summary"] = {
            "baseline_condition": "daytime",
            "baseline_auc": baseline,
            "auc_degradation_vs_daytime": degradation,
        }
    return results


if __name__ == "__main__":
    # Synthetic validation data across conditions, to be replaced with
    # real Phase 7 vehicle-test recordings + ground-truth annotations
    # (observer-coded drowsiness episodes, or KSS self-report).
    rng = np.random.default_rng(42)

    def synth_run(condition: str, n: int = 500, noise: float = 0.15) -> ValidationRun:
        y_true = rng.integers(0, 2, size=n)
        y_score = np.clip(y_true * 0.7 + rng.normal(0, noise, size=n) + 0.15, 0, 1)
        return ValidationRun(condition, y_true, y_score)

    runs = [
        synth_run("daytime", noise=0.15),
        synth_run("nighttime", noise=0.25),   # more noise -> expect lower AUC
        synth_run("rain", noise=0.30),
        synth_run("fog", noise=0.35),
    ]

    results = validate_across_conditions(runs)
    for condition, metrics in results.items():
        print(f"\n=== {condition} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
