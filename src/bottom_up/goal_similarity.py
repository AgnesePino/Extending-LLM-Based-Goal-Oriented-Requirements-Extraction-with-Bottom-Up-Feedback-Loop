"""Numerical comparisons for bottom-up review; no LLM or evaluation imports."""

from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.data_model import BottomUpHighLevelGoal, GoalBranch


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_REVISION = os.getenv("BOTTOM_UP_EMBEDDING_REVISION", "main")
DEFAULT_HLG_SIMILARITY_THRESHOLD = 0.68
_configured_thresholds = [float(value) for name in (
    "BOTTOM_UP_SIMILARITY_THRESHOLD", "BOTTOM_UP_HLG_SIMILARITY_THRESHOLD",
    "BOTTOM_UP_DUPLICATE_SIMILARITY_THRESHOLD") if (value := os.getenv(name)) is not None]
if len(set(_configured_thresholds)) > 1:
    raise ValueError("Equivalence and duplicate candidates require one shared similarity threshold")
SIMILARITY_THRESHOLD = (_configured_thresholds[0] if _configured_thresholds
                        else DEFAULT_HLG_SIMILARITY_THRESHOLD)
if not 0 <= SIMILARITY_THRESHOLD <= 1:
    raise ValueError("Bottom-up similarity threshold must be between 0 and 1")
# Compatibility names refer to the same value; callers cannot set them separately.
HLG_SIMILARITY_THRESHOLD = DUPLICATE_SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD


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
    supplied = [value for value in (parent_threshold, duplicate_threshold) if value is not None]
    if len(set(supplied)) > 1:
        raise ValueError("Equivalence and duplicate candidates require one shared similarity threshold")
    threshold = supplied[0] if supplied else SIMILARITY_THRESHOLD
    if isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1")
    parent_threshold = duplicate_threshold = threshold
    return {
        "embedding_model": EMBEDDING_MODEL,
        "requested_revision": EMBEDDING_REVISION,
        "similarity_threshold": threshold,
        "threshold_policy": "shared_equivalence_and_duplicate_candidates",
        "parent_threshold": parent_threshold,
        "duplicate_threshold": duplicate_threshold,
        "thresholds_validated": False,
        "parent_threshold_calibration": {
            "status": "provisional_shared_threshold",
            "annotation": "user-directed provisional value; not independently validated",
            "selected_default": DEFAULT_HLG_SIMILARITY_THRESHOLD,
            "active_threshold_matches_default": parent_threshold == DEFAULT_HLG_SIMILARITY_THRESHOLD,
        },
        "duplicate_threshold_calibrated": False,
        "original_hlg_comparisons_scope": "all_original_hlgs_including_own",
        "original_hlg_actor_filter": "none",
        "duplicate_comparisons_scope": "same_actor_reconstructed_hlgs",
        "actor_filter": "casefold_and_normalize_whitespace_on_actor_name",
        "actor_filter_scope": "duplicate_comparisons_only",
        "original_hlg_comparisons_usage": "documentary_review",
        "parent_text": "name + newline + description",
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


def _actor_key(branch: GoalBranch) -> str:
    return " ".join(branch.high_level_goal.actor.name.casefold().split())


def analyze_similarities(branches: list[GoalBranch],
                         reconstructions: dict[str, BottomUpHighLevelGoal], *,
                         parent_threshold=None, duplicate_threshold=None) -> dict:
    """Compare primes to all original HLGs; restrict prime pairs to one actor.

    An embedding error makes only the affected branch inconclusive. Empty
    decompositions have no score and cannot be proposed as duplicate candidates.
    """
    config = similarity_configuration(parent_threshold=parent_threshold, duplicate_threshold=duplicate_threshold)
    parent_threshold = config["parent_threshold"]
    duplicate_threshold = config["duplicate_threshold"]
    records, vectors = {}, {}
    for branch in branches:
        record = {"status": "VALID", "score": None, "relationship": "NOT_EQUIVALENT",
                  "reconstructable": bool(branch.low_level_goals),
                  "duplicate_label": None, "duplicate_candidates": []}
        records[branch.branch_id] = record
        if not branch.low_level_goals:
            continue
        try:
            original = f"{branch.high_level_goal.name}\n{branch.high_level_goal.description}"
            rebuilt = reconstructions[branch.branch_id].reconstructed_high_level_goal
            parent_vector, prime_vector = _encode_text(original), _encode_text(rebuilt)
            score = _cosine(parent_vector, prime_vector)
            vectors[branch.branch_id] = prime_vector
            record.update(score=score, relationship=(
                "EQUIVALENT" if score >= parent_threshold else "NOT_EQUIVALENT"))
        except Exception as exc:
            record.update(status="ERROR", relationship=None, error=f"{type(exc).__name__}: {exc}")
    pairs = []
    for i, left in enumerate(branches):
        for right in branches[i + 1:]:
            if (_actor_key(left) != _actor_key(right)
                    or left.branch_id not in vectors or right.branch_id not in vectors):
                continue
            score = _cosine(vectors[left.branch_id], vectors[right.branch_id])
            pair = {"branch_id": left.branch_id, "other_branch_id": right.branch_id,
                    "score": score, "candidate": score >= duplicate_threshold}
            pairs.append(pair)
            if pair["candidate"]:
                records[left.branch_id]["duplicate_label"] = "DUPLICATE"
                records[right.branch_id]["duplicate_label"] = "DUPLICATE"
                records[left.branch_id]["duplicate_candidates"].append(right.branch_id)
                records[right.branch_id]["duplicate_candidates"].append(left.branch_id)

    # Every reconstructed goal versus every original HLG, regardless of actor.
    # The actor filter above applies only to prime/prime duplicate candidates.
    original_vectors, original_errors = {}, {}
    if any(branch.low_level_goals for branch in branches):
        for original in branches:
            try:
                text = f"{original.high_level_goal.name}\n{original.high_level_goal.description}"
                original_vectors[original.branch_id] = _encode_text(text)
            except Exception as exc:
                original_errors[original.branch_id] = f"{type(exc).__name__}: {exc}"
    for branch in branches:
        comparisons = []
        prime_vector, prime_error = None, None
        if branch.low_level_goals:
            try:
                prime_vector = _encode_text(reconstructions[branch.branch_id].reconstructed_high_level_goal)
            except Exception as exc:
                prime_error = f"{type(exc).__name__}: {exc}"
        for original in branches:
            own_parent = original.branch_id == branch.branch_id
            comparison = {"original_branch_id": original.branch_id,
                          "original_hlg_name": original.high_level_goal.name,
                          "original_hlg_actor": original.high_level_goal.actor.name,
                          "is_own_parent": own_parent, "score": None}
            if not branch.low_level_goals:
                comparison["status"] = "NOT_RECONSTRUCTABLE"
            elif prime_error or original.branch_id in original_errors:
                comparison.update(status="ERROR", error=prime_error or original_errors[original.branch_id])
            else:
                try:
                    # Reuse the exact existing score on the matrix diagonal.
                    own_score = records[branch.branch_id]["score"]
                    score = own_score if own_parent and own_score is not None else _cosine(
                        prime_vector, original_vectors[original.branch_id])
                    comparison.update(status="VALID", score=score)
                except Exception as exc:
                    comparison.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            comparisons.append(comparison)
        records[branch.branch_id]["original_hlg_comparisons"] = comparisons
        records[branch.branch_id]["original_hlg_comparisons_complete"] = all(
            item["status"] == "VALID" for item in comparisons)
        other_matches = [item["original_branch_id"] for item in comparisons
                         if not item["is_own_parent"] and item["status"] == "VALID"
                         and item["score"] >= parent_threshold]
        record = records[branch.branch_id]
        record["matching_other_original_hlg_ids"] = other_matches
    config["original_hlg_comparison_count"] = len(branches) ** 2
    config["resolved_revision"] = _resolved_revision()
    return {"configuration": config, "branches": records, "pairs": pairs}


def _compact_original_comparison(row: dict, positions: dict[str, int]) -> dict:
    compact = {key: value for key, value in row.items()
               if key not in {"original_hlg_name", "original_hlg_actor", "is_own_parent"}}
    if "original_branch_id" in compact:
        target = compact.pop("original_branch_id")
        compact = {"original_branch_index": positions[target], **compact}
    if compact.get("status") == "VALID":
        compact.pop("status")
    return compact


def build_cosine_payload(branches: list[GoalBranch],
                         reconstructions: dict[str, BottomUpHighLevelGoal],
                         analysis: dict, *, iteration: int | None = None) -> dict:
    """Build the one JSON object shared by disk output and all three critics."""
    positions = {branch.branch_id: index for index, branch in enumerate(branches)}
    return {
        "iteration": iteration,
        # Keep provenance, thresholds and every score, but omit the offline
        # calibration narrative from the LLM input. Full config is in the trace.
        "configuration": {
            **{key: value for key, value in analysis["configuration"].items()
               if key != "parent_threshold_calibration"},
            "original_comparison_reference": "zero_based_index_in_branches",
            "original_comparison_default_status": "VALID",
        },
        "branches": [{
            "branch_id": branch.branch_id,
            "actor": branch.high_level_goal.actor.name,
            "hlg": {"name": branch.high_level_goal.name, "description": branch.high_level_goal.description},
            "hlg_prime": reconstructions[branch.branch_id].reconstructed_high_level_goal,
            **analysis["branches"][branch.branch_id],
            # Targets are already defined in branches. Repeating their name,
            # actor and diagonal flag for every matrix cell grows quadratically.
            "original_hlg_comparisons": [
                _compact_original_comparison(row, positions)
                for row in analysis["branches"][branch.branch_id].get("original_hlg_comparisons", [])
            ],
        } for branch in branches],
        "pairs": analysis["pairs"],
    }
