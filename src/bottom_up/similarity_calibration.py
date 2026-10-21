"""Offline bottom-up threshold selection from annotated HLG/HLG-prime pairs.

This utility does not read a ground truth or run inside the bottom-up cycle.
Labels must describe equivalence of the two supplied texts, not their validity
against project documentation. Results are in-sample, not held-out validation.
"""

import math


def evaluate_threshold(records: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for record in records:
        predicted = record["score"] >= threshold
        actual = record["equivalent"]
        tp += int(predicted and actual)
        fp += int(predicted and not actual)
        tn += int(not predicted and not actual)
        fn += int(not predicted and actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "balanced_accuracy": (recall + specificity) / 2,
            "margin": min(abs(record["score"] - threshold) for record in records)}


def calibrate_parent_threshold(records: list[dict]) -> dict:
    """Maximize equivalent-class F1 on a 0.01 grid.

    Break ties by balanced accuracy, then distance from the nearest observed
    score. The grid and margin avoid a brittle cut between nearly equal scores.
    This policy favors recovering paraphrases while reporting false matches.
    """
    if not records or {record.get("equivalent") for record in records} != {True, False}:
        raise ValueError("Calibration requires both equivalent and non-equivalent pairs")
    for record in records:
        if type(record["equivalent"]) is not bool:
            raise ValueError("Equivalence annotations must be booleans")
        score = record.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not -1 <= score <= 1):
            raise ValueError("Calibration requires finite cosine scores in [-1, 1]")
    grid = [evaluate_threshold(records, round(i / 100, 2)) for i in range(101)]
    best = max(grid, key=lambda row: (row["f1"], row["balanced_accuracy"], row["margin"]))
    return {"selected_threshold": best["threshold"], "metrics": best,
            "selection": "max F1; ties: balanced accuracy, then margin; grid step 0.01",
            "pair_count": len(records), "equivalent_count": sum(r["equivalent"] for r in records),
            "held_out_pair_count": 0, "validated": False, "grid": grid}
