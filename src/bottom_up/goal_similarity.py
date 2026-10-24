"""Numerical comparisons for bottom-up review; no LLM or evaluation imports."""

from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys


import numpy as np

from src.data_model import BottomUpHighLevelGoal, HighLevelGoal, HighLevelGoals, indexed_high_level_goals


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_REVISION = os.getenv("BOTTOM_UP_EMBEDDING_REVISION", "main")
DEFAULT_HLG_SIMILARITY_THRESHOLD = 0.68
_configured_thresholds = [float(value) for name in (
    "BOTTOM_UP_SIMILARITY_THRESHOLD", "BOTTOM_UP_HLG_SIMILARITY_THRESHOLD") if (value := os.getenv(name)) is not None]
if len(set(_configured_thresholds)) > 1:
    raise ValueError("Parent and reconstructed-HLG comparisons require one similarity threshold")
SIMILARITY_THRESHOLD = (_configured_thresholds[0] if _configured_thresholds
                        else DEFAULT_HLG_SIMILARITY_THRESHOLD)
if not 0 <= SIMILARITY_THRESHOLD <= 1:
    raise ValueError("Bottom-up similarity threshold must be between 0 and 1")
MAX_SIMILARITY_ATTEMPTS = 3
SIMILARITY_TECHNICAL_ATTEMPTS = MAX_SIMILARITY_ATTEMPTS


@lru_cache(maxsize=1)
def _runtime_import_error() -> str | None:
    """Check native dependencies outside the notebook kernel before importing.

    Broken binary packages can terminate Python without raising an exception.
    Keep that failure explicit and avoid killing the user's notebook process.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from sentence_transformers import SentenceTransformer"],
            capture_output=True, text=True, timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            return (f"sentence-transformers import failed (exit {result.returncode}). "
                    f"Check the embedding environment and its native dependencies. {detail}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Cannot validate embedding runtime: {exc}"
    return None


@lru_cache(maxsize=1)
def _embedding_model():
    error = _runtime_import_error()
    if error:
        raise RuntimeError(error)
    from sentence_transformers import SentenceTransformer

    # Keep GPU memory available to the local generator and critics.
    return SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu")


def similarity_configuration(*, parent_threshold=None, duplicate_threshold=None) -> dict:
    threshold = SIMILARITY_THRESHOLD if parent_threshold is None else parent_threshold
    if isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1")
    if duplicate_threshold is not None and duplicate_threshold != threshold:
        raise ValueError("Parent and duplicate comparisons use the same fixed similarity threshold")
    parent_threshold = threshold
    return {
        "embedding_model": EMBEDDING_MODEL,
        "requested_revision": EMBEDDING_REVISION,
        "similarity_threshold": threshold,
        "threshold_policy": "prime_against_all_original_hlgs",
        "parent_threshold": parent_threshold,
        "duplicate_threshold": threshold,
        "thresholds_validated": False,
        "parent_threshold_calibration": {
            "status": "shared_threshold",
            "annotation": "user-directed provisional value; not independently validated",
            "selected_default": DEFAULT_HLG_SIMILARITY_THRESHOLD,
            "active_threshold_matches_default": parent_threshold == DEFAULT_HLG_SIMILARITY_THRESHOLD,
        },
        "duplicate_comparisons_method": "prime_original_cosine_then_documentary_review",
        "parent_comparisons_scope": "all_current_original_hlgs",
        "duplicate_comparisons_scope": "all_current_original_hlgs",
        "actor_filter": "casefold_and_normalize_whitespace_on_actor_name",
        "actor_filter_scope": "duplicate_removal_candidates_only; numerical comparisons cover all actors",
        "parent_text": "description",
        "preprocessing": "none",
        "device": "cpu",
    }


@lru_cache(maxsize=4096)
def _encode_text(text: str) -> tuple[float, ...]:
    if not text.strip():
        raise ValueError("Cannot embed empty goal text")
    model = _embedding_model()
    tokens = model.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    if len(tokens) > model.max_seq_length:
        raise ValueError(f"Goal has {len(tokens)} tokens; embedding limit is {model.max_seq_length}")
    vector = np.asarray(model.encode(text, normalize_embeddings=True), dtype=float)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding vector")
    return tuple(vector / norm)


def _resolved_revision() -> str | None:
    if not _embedding_model.cache_info().currsize:
        return None
    transformer = _embedding_model()[0]
    revision = getattr(getattr(getattr(transformer, "auto_model", None), "config", None),
                       "_commit_hash", None)
    if revision:
        return revision
    # Some Transformers versions do not expose _commit_hash on loaded configs.
    # Resolve the already cached snapshot without a network request.
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(EMBEDDING_MODEL, "config.json", revision=EMBEDDING_REVISION)
    if isinstance(cached, str):
        parts = Path(cached).parts
        if "snapshots" in parts:
            index = parts.index("snapshots") + 1
            if index < len(parts):
                return parts[index]
    return None


def _cosine(left, right) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0):
        raise ValueError("Invalid vectors for cosine similarity")
    return float(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))


def _actor_key(goal: HighLevelGoal) -> str:
    return " ".join(goal.actor.name.casefold().split())


def _analyze_similarities_once(branches: dict[str, HighLevelGoal],
                         reconstructions: dict[str, BottomUpHighLevelGoal], *,
                         parent_threshold=None, duplicate_threshold=None) -> dict:
    """Compare each reconstructed HLG with every current original HLG."""
    config = similarity_configuration(parent_threshold=parent_threshold, duplicate_threshold=duplicate_threshold)
    threshold = config["similarity_threshold"]
    records, vectors, originals, reconstruction_errors, original_errors = {}, {}, {}, {}, {}
    ids = list(branches)
    # Every original HLG remains a column, including parents without children.
    for goal_id, parent in branches.items():
        try:
            originals[goal_id] = _encode_text(parent.description)
        except Exception as exc:
            original_errors[goal_id] = f"{type(exc).__name__}: {exc}"
        if goal_id in reconstructions:
            try:
                vectors[goal_id] = _encode_text(reconstructions[goal_id].reconstructed_high_level_goal)
            except Exception as exc:
                reconstruction_errors[goal_id] = f"{type(exc).__name__}: {exc}"

    pairs = []
    for goal_id in ids:
        comparisons = []
        for other in ids:
            row = {"other_branch_id": other, "score": None}
            if goal_id not in reconstructions:
                row.update(status="NOT_RECONSTRUCTABLE")
            elif goal_id in reconstruction_errors or other in original_errors:
                row.update(status="ERROR", error=reconstruction_errors.get(goal_id) or original_errors[other])
            else:
                try:
                    row.update(status="VALID", score=_cosine(vectors[goal_id], originals[other]))
                except Exception as exc:
                    row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            comparisons.append(row)
        own = comparisons[ids.index(goal_id)]
        record = {"status": own["status"], "score": own["score"],
                  "relationship": ("EQUIVALENT" if own["score"] >= threshold else "NOT_EQUIVALENT")
                                  if own["status"] == "VALID" else None,
                  "reconstructable": goal_id in reconstructions,
                  "duplicate_label": None, "duplicate_candidates": [],
                  "original_hlg_comparisons": comparisons,
                  "original_hlg_comparisons_complete": all(row["status"] == "VALID" for row in comparisons)}
        if own.get("error"):
            record["error"] = own["error"]
        elif own["status"] == "NOT_RECONSTRUCTABLE":
            record["error"] = "No reconstruction available for numerical comparison"
        records[goal_id] = record

    for goal_id in ids:
        record = records[goal_id]
        for row in record["original_hlg_comparisons"]:
            other = row["other_branch_id"]
            if other == goal_id:
                continue
            candidate = (row["status"] == "VALID" and row["score"] >= threshold
                         and record["status"] == "VALID" and records[other]["status"] == "VALID"
                         and _actor_key(branches[goal_id]) == _actor_key(branches[other]))
            pairs.append({"branch_id": goal_id, **row, "candidate": candidate})
            if candidate:
                record["duplicate_candidates"].append(other)
        if record["status"] == "VALID":
            record["duplicate_label"] = "POTENTIAL_DUPLICATE" if record["duplicate_candidates"] else "NO_DUPLICATE_MATCH"
    config["parent_hlg_comparison_count"] = len(reconstructions)
    config["original_hlg_comparison_count"] = len(reconstructions) * len(branches)
    config["resolved_revision"] = _resolved_revision()
    return {"configuration": config, "branches": records, "pairs": pairs}


def analyze_similarities(branches: dict[str, HighLevelGoal] | HighLevelGoals,
                         reconstructions: dict[str, BottomUpHighLevelGoal], *,
                         parent_threshold=None, duplicate_threshold=None) -> dict:
    """Recover failed numerical work before allowing documentary review.

    Successful text vectors stay cached. Errors are never cached by lru_cache;
    model/import failures can be retried after resetting their cached state.
    No text is shortened and no score or threshold is substituted.
    Missing reconstructions are unavailable inputs, not retryable numerical errors.
    """
    if isinstance(branches, HighLevelGoals):
        branches = indexed_high_level_goals(branches)
    history = []
    for attempt in range(1, SIMILARITY_TECHNICAL_ATTEMPTS + 1):
        analysis = _analyze_similarities_once(branches, reconstructions,
            parent_threshold=parent_threshold, duplicate_threshold=duplicate_threshold)
        errors = {}
        for bid, record in analysis["branches"].items():
            failed = [row for row in record.get("original_hlg_comparisons", [])
                      if row.get("status") == "ERROR"]
            if record["status"] == "ERROR" or failed:
                errors[bid] = {"parent_error": record.get("error"),
                               "comparison_errors": failed}
        history.append({"attempt": attempt, "accepted": not errors, "errors": errors})
        if not errors:
            break
        print(f"[bottom-up] similarity recovery | attempt={attempt}/"
              f"{SIMILARITY_TECHNICAL_ATTEMPTS} | failed_branches={len(errors)}", flush=True)
        if attempt < SIMILARITY_TECHNICAL_ATTEMPTS:
            _embedding_model.cache_clear()
            _runtime_import_error.cache_clear()
    analysis["technical_attempts"] = history
    analysis["configuration"]["similarity_technical_attempts"] = SIMILARITY_TECHNICAL_ATTEMPTS
    return analysis


def build_cosine_payload(branches: dict[str, HighLevelGoal],
                         reconstructions: dict[str, BottomUpHighLevelGoal],
                         analysis: dict, *, iteration: int | None = None) -> dict:
    """Expose the full reconstructed-HLG versus original-HLG comparisons."""
    return {
        "iteration": iteration,
        "configuration": {key: value for key, value in analysis["configuration"].items()
                          if key != "parent_threshold_calibration"},
        "branches": [{"branch_id": goal_id, "actor": parent.actor.name,
            "hlg": {"name": parent.name, "description": parent.description},
            "hlg_prime": (reconstructions[goal_id].reconstructed_high_level_goal
                          if goal_id in reconstructions else None),
            **analysis["branches"][goal_id]} for goal_id, parent in branches.items()],
        "pairs": analysis["pairs"],
    }


def build_review_payload(payload: dict) -> dict:
    """Expose categorical similarity signals; keep numerical values in diagnostics."""
    branches = payload["branches"]
    positions = {goal["branch_id"]: i for i, goal in enumerate(branches)}
    errors, rows = [], []
    for i, goal in enumerate(branches):
        labels = [None] * len(branches)
        for comparison in goal.get("original_hlg_comparisons", []):
            j = positions[comparison["other_branch_id"]]
            if comparison["status"] == "VALID":
                if i == j:
                    labels[j] = goal["relationship"]
                elif goal["status"] == "VALID" and branches[j]["status"] == "VALID":
                    labels[j] = ("POTENTIAL_DUPLICATE" if comparison["other_branch_id"]
                                 in goal["duplicate_candidates"] else "NO_DUPLICATE_MATCH")
                else:
                    errors.append([i, j, "NOT_ELIGIBLE_FOR_DUPLICATE_REVIEW"])
            else:
                errors.append([i, j, comparison["status"]])
        rows.append(labels)
    return {
        "requested_branch_ids": [g["branch_id"] for g in branches if g["status"] == "VALID"],
        "branches": [{"branch_id": g["branch_id"], "actor": g["actor"],
            "hlg": g["hlg"], "hlg_prime": g["hlg_prime"], "status": g["status"],
            "reconstructable": g["reconstructable"], "relationship": g["relationship"],
            "original_hlg_comparisons_complete": g["original_hlg_comparisons_complete"],
            "duplicate_candidates": g["duplicate_candidates"], "duplicate_label": g["duplicate_label"]}
            for g in branches],
        "original_labels": rows, "comparison_statuses": errors,
        "duplicate_review": "prime_original_similarity_labels_same_actor_documentary_confirmation",
    }
