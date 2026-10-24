"""Fresh prime/parent cosine pairs for manual review, without ground truth."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "Assegno Unico Universale - SIA Project 24 25",
    "Calculator",
    "Fitness Tracker",
    "Genome Nexus",
    "Planning Poker",
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def active_experiment_runs():
    """Avoid model leases that could unload another process's active model."""
    active = []
    for path in (ROOT / "output/bottom_up_experiments").glob("*/datasets/*.json"):
        if time.time() - path.stat().st_mtime > 180:
            continue
        data = json.loads(path.read_bytes())
        active.extend(f"{data.get('name', path.stem)} / {key}" for key, run in
                      data.get("runs", {}).items() if run.get("status") == "running")
    return active


def prepare_document():
    from src.data_model import HighLevelGoals, LowLevelGoals, indexed_high_level_goals, low_level_goals_for_hlg
    from src.bottom_up import goal_reconstructor, goal_similarity
    from src.llm_clients import QWEN_GENERATOR_MODEL

    files = {}
    for path in (ROOT / "output/top_down").glob("*_ZERO_SHOT_feedback_original_top_down_top_down.json"):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload["name"] in DATASETS:
            if payload["name"] in files:
                raise ValueError(f"Duplicate baseline: {payload['name']}")
            files[payload["name"]] = (path, raw, payload)
    if set(files) != set(DATASETS):
        raise ValueError(f"Missing datasets: {set(DATASETS) - set(files)}")
    datasets = []
    for name in DATASETS:
        path, raw, payload = files[name]
        parents = HighLevelGoals.model_validate(payload["structuredHighLevelGoals"])
        children = LowLevelGoals.model_validate(payload["structuredLowLevelGoals"])
        pairs = []
        assigned_children = 0
        for bid, parent in indexed_high_level_goals(parents).items():
            llgs = low_level_goals_for_hlg(parent, children)
            assigned_children += len(llgs)
            pairs.append({
                "branch_id": bid, "actor": parent.actor.model_dump(mode="json"),
                "hlg_original": parent.model_dump(mode="json"),
                "llg_descriptions": [child.description for child in llgs],
                "llgs": [child.model_dump(mode="json") for child in llgs],
                "hlg_prime": None, "cosine_similarity": None,
                "relationship_at_reference_threshold": None,
                "status": "pending" if llgs else "not_reconstructable",
                "human_review": {"equivalent": None, "differences": "", "notes": ""},
            })
        if assigned_children != len(children.low_level_goals):
            raise ValueError(f"Unmatched or multiply assigned LLGs in {name}")
        datasets.append({"dataset": name, "baseline_file": str(path.relative_to(ROOT)),
            "baseline_sha256": digest(raw), "project_description": payload["description"],
            "actors": payload.get("structuredActors"), "pairs": pairs})
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared", "dataset_count": len(datasets),
        "pair_count": sum(len(d["pairs"]) for d in datasets),
        "selection": "Five explicitly selected projects with different domains and hierarchy sizes; not a random sample.",
        "generator_model": QWEN_GENERATOR_MODEL,
        "reconstruction": "Fresh HLG prime from LLG descriptions only; current pipeline prompt; no evaluator, repair or ground truth.",
        "reference_threshold": 0.68, "selected_threshold": None,
        "threshold_status": "awaiting_human_annotations",
        "instructions": "Read each HLG original and fresh HLG prime alongside cosine_similarity. Set human_review.equivalent to true or false and describe differences. The cosine is a pair score, not a calculated optimal threshold. The 0.68 label is provisional, not a human annotation. No threshold is selected automatically.",
        "embedding_configuration": goal_similarity.similarity_configuration(parent_threshold=0.68),
        "code_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in (
            Path(__file__), Path(goal_reconstructor.__file__), Path(goal_similarity.__file__))},
        "datasets": datasets,
    }


def run(path, *, prepare_only=False):
    document = json.loads(path.read_bytes()) if path.exists() else prepare_document()
    save(path, document)
    print(f"JSON: {path}", flush=True)
    if prepare_only:
        return document
    from src.bottom_up import goal_reconstructor, goal_similarity
    from src.data_model import LowLevelGoal
    from src.llm_clients import loaded_local_model, QWEN_GENERATOR_MODEL, reset_model_clients

    for relative, expected in document["code_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError(f"Code changed since preparation: {relative}; prepare a new JSON")
    active = active_experiment_runs()
    if active:
        document["status"] = "waiting_for_experiment_server"
        save(path, document)
        raise RuntimeError("An experiment is still running on the shared server: " + "; ".join(active))
    document["status"] = "running"
    document.pop("error", None)
    save(path, document)
    try:
        with loaded_local_model(QWEN_GENERATOR_MODEL):
            for dataset in document["datasets"]:
                for pair in dataset["pairs"]:
                    if pair["status"] in {"completed", "not_reconstructable"}:
                        continue
                    print(f"[reconstruct] {dataset['dataset']} / {pair['branch_id']}", flush=True)
                    if pair["hlg_prime"] is None:
                        result = goal_reconstructor.reconstruct_high_level_goal(
                            [LowLevelGoal.model_validate(child) for child in pair["llgs"]])
                        pair["hlg_prime"] = result.reconstructed_high_level_goal
                        pair["status"] = "reconstructed"
                        save(path, document)
                    score = goal_similarity._cosine(
                        goal_similarity._encode_text(pair["hlg_prime"]),
                        goal_similarity._encode_text(pair["hlg_original"]["description"]))
                    pair["cosine_similarity"] = score
                    pair["relationship_at_reference_threshold"] = (
                        "EQUIVALENT" if score >= document["reference_threshold"] else "NOT_EQUIVALENT")
                    pair["status"] = "completed"
                    save(path, document)
                    print(f"[cosine] {score:.6f}", flush=True)
        document["status"] = "awaiting_human_review"
        document["resolved_embedding_revision"] = goal_similarity._resolved_revision()
        document["completed_pairs"] = sum(pair["status"] == "completed"
            for dataset in document["datasets"] for pair in dataset["pairs"])
        save(path, document)
    except BaseException as exc:
        document["status"] = "interrupted" if not isinstance(exc, Exception) else "failed"
        document["error"] = f"{type(exc).__name__}: {exc}"
        save(path, document)
        raise
    finally:
        reset_model_clients()
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), prepare_only=args.prepare_only)
