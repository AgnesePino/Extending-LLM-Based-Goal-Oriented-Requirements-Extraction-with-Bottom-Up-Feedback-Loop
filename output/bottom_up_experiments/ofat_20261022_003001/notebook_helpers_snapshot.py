from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import time
import traceback


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str = "reference"
    experiment: str = "reference"
    evaluator_few_shot_examples: int = 4
    parent_threshold: float = 0.68
    duplicate_threshold: float = 0.80
    evaluator_temperature: float = 0.2

    def cycle_options(self):
        return {key: value for key, value in asdict(self).items()
                if key not in {"config_id", "experiment"}}


def experiment_configurations():
    base = ExperimentConfig()
    return [base] + [
        replace(base, config_id=f"E1_{count}_examples", experiment="E1_few_shot",
                evaluator_few_shot_examples=count) for count in (0, 2, 6)
    ] + [
        replace(base, config_id=f"E2_{label}", experiment="E2_thresholds",
                parent_threshold=parent, duplicate_threshold=duplicate)
        for label, parent, duplicate in (("permissive", .64, .76),
                                        ("restrictive", .72, .84), ("very_restrictive", .76, .88))
    ] + [
        replace(base, config_id=f"E3_temperature_{temperature:.1f}", experiment="E3_temperature",
                evaluator_temperature=temperature) for temperature in (0., .4, .6)
    ]


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slug(name):
    label = re.sub(r'[^\w.-]+', '_', name).strip('._')[:100]
    if not label:
        raise ValueError("Empty dataset identifier")
    return label + "_" + _digest(name.encode())[:8]


def runtime_configuration():
    from src import llm_clients as clients
    from src.bottom_up import global_goal_evaluator as evaluator
    from src.bottom_up import goal_cycle_orchestrator as cycle
    from src.bottom_up import goal_similarity as similarity
    from src.self_critique import refine_response as reflection
    packages = {}
    for name in ("openai", "pydantic", "numpy", "sentence-transformers", "torch", "transformers", "pandas", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "policy": evaluator.PIPELINE_POLICY_VERSION,
        "generator_and_reconstructor": clients.QWEN_GENERATOR_MODEL,
        "evaluator_models": list(evaluator.BOTTOM_UP_EVALUATOR_MODELS),
        "repair_critic": clients.configured_critic_model(),
        "quorum": evaluator.EVALUATOR_VOTING_QUORUM,
        "embedding_model": similarity.EMBEDDING_MODEL,
        "embedding_revision": similarity.EMBEDDING_REVISION,
        "generation_mode": "ZERO_SHOT", "generation_evaluator_ablation": False,
        "repair_critic_examples": len(reflection._bottom_up_few_shot_examples(reflection.BottomUpEvalMode.HIGH_LEVEL)),
        "repair_quality_threshold": reflection.BOTTOM_UP_QUALITY_THRESHOLD,
        "repair_max_attempts": reflection.BOTTOM_UP_MAX_ATTEMPTS,
        "max_hlg_rewrites": cycle.MAX_HLG_REWRITES_PER_BRANCH,
        "endpoint": clients.LM_STUDIO_BASE_URL,
        "context_length": clients.LM_STUDIO_EVALUATOR_CONTEXT_LENGTH,
        "evaluator_max_tokens": clients.BOTTOM_UP_CRITIC_MAX_TOKENS,
        "generator_max_tokens": clients.QWEN_GENERATOR_MAX_TOKENS,
        "vote_attempts": evaluator.BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS,
        "request_timeout_seconds": clients.REQUEST_TIMEOUT_SECONDS,
        "api_max_retries": clients.API_MAX_RETRIES,
        "python": platform.python_version(),
        "packages": packages,
        "seed_policy": "No seed forced; repetitions are independent calls, not guaranteed deterministic.",
        "temperature_support": "Requested temperature recorded in each generation metadata; no silent fallback.",
    }




def notebook_helpers_source():
    notebook_path = PROJECT_ROOT / "notebook" / "03_bottom_up_experiments.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if "experiment_helpers" in cell.get("metadata", {}).get("tags", [])
    )


def prepare_suite(project_root, baseline_files, suite_id, *, repetitions=1,
                  configurations=None, max_iterations=25, expected_datasets=12):
    """Freeze all baselines before the first model call; refuse changed resumes."""
    from src.bottom_up.global_goal_evaluator import evaluator_examples, SYSTEM_PROMPT
    from src.data_model import HighLevelGoals, LowLevelGoals
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", suite_id):
        raise ValueError("Use a simple suite_id without path separators")
    if type(repetitions) is not int or repetitions < 1 or not 1 <= max_iterations <= 25:
        raise ValueError("Require positive integer repetitions and 1..25 iterations")
    root = Path(project_root).resolve()
    directory = root / "output" / "bottom_up_experiments" / suite_id
    configs = list(configurations if configurations is not None else experiment_configurations())
    allowed = {c.config_id: c for c in experiment_configurations()}
    if not configs or len({c.config_id for c in configs}) != len(configs):
        raise ValueError("Require distinct experiment configurations")
    if any(allowed.get(c.config_id) != c for c in configs):
        raise ValueError("Configurations must match the declared OFAT design")
    baselines, frozen = [], {}
    for path in baseline_files:
        path = Path(path).resolve()
        raw = path.read_bytes()
        data = json.loads(raw)
        HighLevelGoals.model_validate(data["structuredHighLevelGoals"])
        LowLevelGoals.model_validate(data["structuredLowLevelGoals"])
        if not data.get("name") or not data.get("description"):
            raise ValueError(f"Missing project name/description: {path}")
        slug = _slug(data["name"])
        if slug in frozen:
            raise ValueError(f"Duplicate dataset: {data['name']}")
        # Freeze source documents and initial hierarchies only, never references.
        frozen[slug] = raw
        baselines.append({"dataset": data["name"], "dataset_id": slug,
                          "source_file": str(path), "sha256": _digest(raw),
                          "snapshot": f"baselines/{slug}.json"})
    if len(baselines) != expected_datasets:
        raise ValueError(f"Expected {expected_datasets} datasets, found {len(baselines)}")
    code_files = sorted((root / "src").rglob("*.py"))
    manifest = {
        "schema_version": 1, "suite_id": suite_id,
        "configurations": [asdict(c) for c in configs], "repetitions": repetitions,
        "max_iterations": max_iterations, "baselines": baselines,
        "planned_runs": len(configs) * len(baselines) * repetitions,
        "fixed_runtime": runtime_configuration(),
        "code_sha256": {str(p.relative_to(root)): _digest(p.read_bytes()) for p in code_files},
        "notebook_helpers_sha256": _digest(notebook_helpers_source().encode("utf-8")),
        "examples_by_count": {str(n): evaluator_examples(n) for n in (0, 2, 4, 6)},
        "evaluator_system_prompt": SYSTEM_PROMPT,
        "design": "OFAT; E2 varies the threshold pair together; reference run once per repetition.",
        "ground_truth_usage": "Post-hoc evaluation only; never passed to the cycle.",
    }
    path = directory / "manifest.json"
    if path.exists():
        existing = _read(path)
        if {k: v for k, v in existing.items() if k != "created_at"} != manifest:
            raise ValueError("Suite inputs/code/settings changed. Choose a new suite_id.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Nonempty suite directory without manifest")
        for item in baselines:
            destination = directory / item["snapshot"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(frozen[item["dataset_id"]])
        (directory / "notebook_helpers_snapshot.py").write_text(notebook_helpers_source(), encoding="utf-8")
        _write_json(path, {**manifest, "created_at": datetime.now(timezone.utc).isoformat()})
    for item in baselines:
        if _digest((directory / item["snapshot"]).read_bytes()) != item["sha256"]:
            raise ValueError(f"Frozen baseline altered: {item['dataset']}")
    return directory


def _job_directory(directory, config, repetition, baseline):
    return (directory / "runs" / config["config_id"] / f"repeat_{repetition:03d}"
            / baseline["dataset_id"])


def _latest(job):
    files = sorted(job.glob("attempt_*/run.json"))
    return files[-1] if files else None


def _verify_suite(directory):
    manifest = _read(directory / "manifest.json")
    root = directory.parents[2]
    if manifest.get("notebook_helpers_sha256") != _digest(notebook_helpers_source().encode("utf-8")):
        raise ValueError("Notebook helper code changed or suite uses the previous module layout; prepare a new suite")
    if manifest["fixed_runtime"] != runtime_configuration():
        raise ValueError("Runtime settings changed; prepare a new suite")
    for relative, digest in manifest["code_sha256"].items():
        if _digest((root / relative).read_bytes()) != digest:
            raise ValueError(f"Code changed since suite preparation: {relative}")
    for baseline in manifest["baselines"]:
        if _digest((directory / baseline["snapshot"]).read_bytes()) != baseline["sha256"]:
            raise ValueError(f"Frozen baseline changed: {baseline['dataset']}")
    return manifest


def _completed_result_error(record, payload):
    """A final result, including normal limit stops, must have a full trace."""
    from src.data_model import GlobalGoalCycleResult
    if record.get("status") != "completed" or record.get("error"):
        return "Run failed or interrupted before completion"
    try:
        cycle = GlobalGoalCycleResult.model_validate(payload["globalGoalCycle"])
    except (KeyError, ValueError, TypeError) as exc:
        return f"Invalid final result: {exc}"
    traces = cycle.iterations
    if not traces or len(traces) != cycle.completed_iterations:
        return "Missing completed iteration trace"
    if [t.iteration for t in traces] != list(range(1, len(traces) + 1)):
        return "Incomplete iteration sequence"
    if any(not t.checkpoint_complete for t in traces):
        return "Incomplete iteration checkpoint"
    last = traces[-1]
    if (last.high_level_goals != cycle.final_high_level_goals
            or last.low_level_goals != cycle.final_low_level_goals
            or payload.get("finalHighLevelGoals") != cycle.final_high_level_goals.model_dump(mode="json")
            or payload.get("finalLowLevelGoals") != cycle.final_low_level_goals.model_dump(mode="json")):
        return "Final hierarchy does not match the last completed iteration"
    if cycle.stop_reason.value not in {
            "ALL_BRANCHES_CONFIRMED", "MAX_ITERATIONS_REACHED", "NO_ACTIONS_REMAIN_WITH_WARNINGS"}:
        return "Unexpected stop reason"
    if cycle.converged != (cycle.stop_reason.value == "ALL_BRANCHES_CONFIRMED"):
        return "Inconsistent convergence status"
    if any("SIMILARITY_UNAVAILABLE" in w or "EVALUATOR_CONTEXT_LENGTH_EXCEEDED" in w for w in cycle.warnings):
        return "Run stopped because comparison or model context was unavailable"
    return None


def _pipeline_fingerprint(manifest):
    # These former modules only organized experiments/evaluation; the actual
    # inference code is required to be identical even after moving helpers here.
    bookkeeping = {"src/bottom_up/experiments.py", "src/bottom_up/iteration_exports.py",
                   "src/evaluation/configuration_comparison.py"}
    return {path.replace(chr(92), "/"): checksum
            for path, checksum in manifest["code_sha256"].items()
            if path.replace(chr(92), "/") not in bookkeeping}


def _compatible_previous_suites(directory, manifest):
    compatible = []
    for path in sorted(directory.parent.glob("*/manifest.json"), reverse=True):
        if path.parent.resolve() == directory.resolve():
            continue
        other = _read(path)
        if (other.get("fixed_runtime") == manifest["fixed_runtime"]
                and other.get("max_iterations") == manifest["max_iterations"]
                and other.get("examples_by_count") == manifest["examples_by_count"]
                and other.get("evaluator_system_prompt") == manifest["evaluator_system_prompt"]
                and _pipeline_fingerprint(other) == _pipeline_fingerprint(manifest)):
            compatible.append((path.parent, other))
    return compatible


def _find_completed_run(previous_suites, config, repetition, baseline):
    for directory, manifest in previous_suites:
        if config not in manifest["configurations"] or repetition > manifest["repetitions"]:
            continue
        for candidate in manifest["baselines"]:
            if candidate["dataset"] != baseline["dataset"] or candidate["sha256"] != baseline["sha256"]:
                continue
            if _digest((directory / candidate["snapshot"]).read_bytes()) != baseline["sha256"]:
                continue
            job = _job_directory(directory, config, repetition, candidate)
            for run_path in sorted(job.glob("attempt_*/run.json"), reverse=True):
                record = _read(run_path)
                result_path = run_path.parent / "result.json"
                if record.get("status") != "completed" or not result_path.is_file():
                    continue
                if _digest(result_path.read_bytes()) != record.get("result_sha256"):
                    continue
                payload = _read(result_path)
                if (payload.get("name") != baseline["dataset"]
                        or payload.get("baselineSha256") != baseline["sha256"]
                        or payload.get("experiment") != config or payload.get("repetition") != repetition):
                    continue
                actual = payload.get("globalGoalCycle", {}).get("configuration", {})
                thresholds = actual.get("similarity", {})
                if (actual.get("evaluator_temperature") != config["evaluator_temperature"]
                        or actual.get("evaluator_few_shot_examples") != config["evaluator_few_shot_examples"]
                        or thresholds.get("parent_threshold") != config["parent_threshold"]
                        or thresholds.get("duplicate_threshold") != config["duplicate_threshold"]):
                    continue
                if _completed_result_error(record, payload) is None:
                    return run_path.parent, record
    return None


def run_suite(directory, *, selected_config_ids=None, verbose=True, retry_failed=True, reuse_completed=True):
    """Resume completed jobs; failed/interrupted jobs get a new attempt directory.

    Run sequentially in one kernel. KeyboardInterrupt preserves checkpoints and
    propagates immediately. This function never loads ground truths.
    """
    from src import llm_clients
    from src.bottom_up.goal_cycle_orchestrator import run_global_goal_cycle
    from src.data_model import Actors, HighLevelGoals, LowLevelGoals
    from src.examples.shot_learning import ShotPromptingMode
    directory = Path(directory).resolve()
    manifest = _verify_suite(directory)
    previous_suites = _compatible_previous_suites(directory, manifest) if reuse_completed else []
    available = {c["config_id"] for c in manifest["configurations"]}
    selected = available if selected_config_ids is None else set(selected_config_ids)
    if not selected <= available:
        raise ValueError(f"Unknown configurations: {selected - available}")
    for config in manifest["configurations"]:
        if config["config_id"] not in selected:
            continue
        for repetition in range(1, manifest["repetitions"] + 1):
            for baseline in manifest["baselines"]:
                job = _job_directory(directory, config, repetition, baseline)
                previous = _latest(job)
                if previous:
                    record = _read(previous)
                    if record["status"] == "completed":
                        result_path = previous.parent / "result.json"
                        if not result_path.exists() or _digest(result_path.read_bytes()) != record["result_sha256"]:
                            raise ValueError(f"Completed result missing/altered: {result_path}")
                        completion_error = _completed_result_error(record, _read(result_path))
                        if completion_error is None:
                            print(f"[skip completed] {config['config_id']} / {repetition} / {baseline['dataset']}")
                            continue
                        print(f"[rerun incomplete result] {baseline['dataset']}: {completion_error}")
                    if record["status"] == "failed" and not retry_failed:
                        continue
                attempts = sorted(job.glob("attempt_*"))
                number = max((int(p.name.split('_')[-1]) for p in attempts), default=0) + 1
                attempt = job / f"attempt_{number:04d}"
                reusable = _find_completed_run(previous_suites, config, repetition, baseline)
                if reusable is not None:
                    import shutil
                    source_attempt, source_record = reusable
                    shutil.copytree(source_attempt, attempt)
                    if _digest((attempt / "result.json").read_bytes()) != source_record["result_sha256"]:
                        raise ValueError("Result changed while copying a completed run")
                    origin = source_record.get("origin_run_directory", str(source_attempt.resolve()))
                    reused_record = {**source_record, "reused_from": str(source_attempt.resolve()),
                                     "origin_run_directory": origin,
                                     "original_elapsed_seconds": source_record.get("original_elapsed_seconds", source_record.get("elapsed_seconds")),
                                     "elapsed_seconds": 0.0,
                                     "reused_at": datetime.now(timezone.utc).isoformat()}
                    _write_json(attempt / "run.json", reused_record)
                    print(f"[skip reused result] {config['config_id']} / {repetition} / {baseline['dataset']} "
                          f"| source={source_attempt}", flush=True)
                    export_operational_tables(directory)
                    continue
                attempt.mkdir(parents=True, exist_ok=False)
                record = {"config_id": config["config_id"], "experiment": config["experiment"],
                          "repetition": repetition, "dataset": baseline["dataset"],
                          "baseline_sha256": baseline["sha256"], "configuration": config,
                          "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
                _write_json(attempt / "run.json", record)
                print(f"\n=== {config['config_id']} | repeat {repetition} | {baseline['dataset']} ===", flush=True)
                started = time.perf_counter()
                try:
                    data = _read(directory / baseline["snapshot"])
                    llm_clients.reset_model_clients()
                    result = run_global_goal_cycle(
                        project_description=data["description"],
                        actors=Actors.model_validate(data["structuredActors"]) if data.get("structuredActors") else None,
                        initial_high_level_goals=HighLevelGoals.model_validate(data["structuredHighLevelGoals"]),
                        initial_low_level_goals=LowLevelGoals.model_validate(data["structuredLowLevelGoals"]),
                        mode=ShotPromptingMode.ZERO_SHOT, evaluator_ablation=False,
                        max_iterations=manifest["max_iterations"],
                        evaluation_output_directory=attempt / "iterations",
                        output_file_prefix=baseline["dataset"], verbose=verbose,
                        **ExperimentConfig(**config).cycle_options())
                    payload = {"name": data["name"], "baselineSourceFile": baseline["snapshot"],
                               "baselineSha256": baseline["sha256"], "experiment": config,
                               "repetition": repetition, "globalGoalCycle": result.model_dump(mode="json"),
                               "finalHighLevelGoals": result.final_high_level_goals.model_dump(mode="json"),
                               "finalLowLevelGoals": result.final_low_level_goals.model_dump(mode="json")}
                    _write_json(attempt / "result.json", payload)
                    record.update(status="completed", converged=result.converged,
                                  stop_reason=result.stop_reason.value,
                                  completed_iterations=result.completed_iterations,
                                  result_sha256=_digest((attempt / "result.json").read_bytes()))
                except BaseException as exc:
                    record.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                                  error=f"{type(exc).__name__}: {exc}")
                    (attempt / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    if not isinstance(exc, Exception):
                        raise
                finally:
                    record.update(elapsed_seconds=time.perf_counter() - started,
                                  finished_at=datetime.now(timezone.utc).isoformat())
                    _write_json(attempt / "run.json", record)
                    llm_clients.reset_model_clients()
                    export_operational_tables(directory)
    return export_operational_tables(directory)




def export_operational_tables(directory):
    """Export counts with explicit denominators; never count retry history per branch."""
    import pandas as pd
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    runs, iterations, decisions, attempts = [], [], [], []
    for config in manifest["configurations"]:
        for repetition in range(1, manifest["repetitions"] + 1):
            for baseline in manifest["baselines"]:
                job = _job_directory(directory, config, repetition, baseline)
                path = _latest(job)
                identity = {"config_id": config["config_id"], "experiment": config["experiment"],
                            "repetition": repetition, "dataset": baseline["dataset"]}
                if path is None:
                    runs.append({**identity, "status": "not_started"})
                    continue
                record = _read(path)
                runs.append({**identity, **{k: v for k, v in record.items() if k != "configuration"},
                             "attempt_directory": str(path.parent.relative_to(directory))})
                if record["status"] != "completed":
                    continue
                result_path = path.parent / "result.json"
                if _digest(result_path.read_bytes()) != record["result_sha256"]:
                    raise ValueError(f"Result integrity check failed: {result_path}")
                cycle = _read(result_path)["globalGoalCycle"]
                for trace in cycle["iterations"]:
                    key = {**identity, "iteration": trace["iteration"]}
                    quorum_count, unanimous_count, vote_count, valid_count = 0, 0, 0, 0
                    for bid, evaluation in trace["evaluations"].items():
                        votes = evaluation.get("critic_votes", {}).get("documentary", [])
                        valid = [v for v in votes if v["status"] == "VALID"]
                        winning = max(Counter(v["decision"] for v in valid).values(), default=0)
                        quorum_count += winning >= 2
                        unanimous_count += winning == 3
                        vote_count += len(votes)
                        valid_count += len(valid)
                        decisions.append({**key, "branch_id": bid, "decision": evaluation["final_decision"],
                            "action_outcome": trace["structural_action_outcomes"].get(bid),
                            "rationale": evaluation["rationale"], "valid_votes": len(valid),
                            "winning_votes": winning,
                            "relationship": evaluation.get("similarity", {}).get("relationship"),
                            "duplicate_label": evaluation.get("similarity", {}).get("duplicate_label")})
                    for vote in trace.get("critic1_global_votes", []):
                        for history in vote.get("attempt_history", []):
                            try:
                                raw = json.loads(history.get("raw_response") or "")
                                json_valid = isinstance(raw, dict)
                            except (ValueError, TypeError):
                                json_valid = False
                            attempts.append({**key, "model": vote["model"], "attempt": history["attempt"],
                                "json_object_valid": json_valid, "response_contract_valid": not bool(history["errors"]),
                                "requested_branches": len(history.get("requested_branch_ids", [])),
                                "errors": json.dumps(history["errors"], ensure_ascii=False),
                                "generation": json.dumps(history.get("generation", {}), ensure_ascii=False)})
                    analysis = trace["similarity_analysis"]
                    labels = Counter(b.get("relationship") for b in analysis["branches"].values())
                    n = len(trace["evaluations"])
                    iterations.append({**key, "branches": n, "quorum_branches": quorum_count,
                        "quorum_rate": quorum_count / n if n else None,
                        "unanimous_branches": unanimous_count,
                        "unanimity_rate": unanimous_count / n if n else None,
                        "valid_branch_votes": valid_count, "branch_votes": vote_count,
                        "valid_branch_vote_rate": valid_count / vote_count if vote_count else None,
                        "equivalent": labels["EQUIVALENT"], "not_equivalent": labels["NOT_EQUIVALENT"],
                        "duplicate_pairs": sum(p["candidate"] for p in analysis["pairs"]),
                        "warnings": json.dumps(trace["warnings"], ensure_ascii=False)})
    summaries = []
    for config in manifest["configurations"]:
        local = [r for r in runs if r["config_id"] == config["config_id"]]
        complete = [r for r in local if r["status"] == "completed"]
        status = Counter(r["status"] for r in local)
        traces = [r for r in iterations if r["config_id"] == config["config_id"]]
        requests = [r for r in attempts if r["config_id"] == config["config_id"]]
        n_branches = sum(r["branches"] for r in traces)
        n_votes = sum(r["branch_votes"] for r in traces)
        summaries.append({**config, "planned_runs": len(local), "completed_runs": len(complete),
            "failed_runs": status["failed"], "not_started_runs": status["not_started"],
            "interrupted_runs": status["interrupted"], "running_runs": status["running"],
            "converged_runs": sum(r["converged"] for r in complete),
            "nonconverged_runs": sum(not r["converged"] for r in complete),
            "convergence_rate_completed": sum(r["converged"] for r in complete) / len(complete) if complete else None,
            "mean_iterations_completed": sum(r["completed_iterations"] for r in complete) / len(complete) if complete else None,
            "branch_evaluations": n_branches, "branch_votes": n_votes,
            "quorum_rate": sum(r["quorum_branches"] for r in traces) / n_branches if n_branches else None,
            "unanimity_rate": sum(r["unanimous_branches"] for r in traces) / n_branches if n_branches else None,
            "valid_branch_vote_rate": sum(r["valid_branch_votes"] for r in traces) / n_votes if n_votes else None,
            "evaluator_requests": len(requests),
            "json_object_valid_rate": sum(r["json_object_valid"] for r in requests) / len(requests) if requests else None,
            "response_contract_valid_rate": sum(r["response_contract_valid"] for r in requests) / len(requests) if requests else None})
    decision_table = pd.DataFrame(decisions)
    action_counts = (decision_table.groupby(["config_id", "decision", "action_outcome"], dropna=False)
                     .size().rename("count").reset_index() if decisions else pd.DataFrame(
                         columns=["config_id", "decision", "action_outcome", "count"]))
    tables = {"runs": pd.DataFrame(runs), "iterations": pd.DataFrame(iterations),
              "decisions": decision_table, "evaluator_attempts": pd.DataFrame(attempts),
              "configurations": pd.DataFrame(manifest["configurations"]),
              "operational_summary": pd.DataFrame(summaries), "action_counts": action_counts}
    destination = directory / "comparison"
    destination.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig")
    # External reviewers fill a separate file; re-exporting never overwrites it.
    review = destination / "qualitative_review_template.csv"
    if (not review.exists() and decisions
            and all(run["status"] in {"completed", "failed"} for run in runs)):
        frame = pd.DataFrame(decisions)
        frame = frame[frame["action_outcome"] != "CONFIRM_BRANCH"].copy()
        for column in ("reviewer", "source_evidence", "correction_correct", "lost_requirements", "notes"):
            frame[column] = ""
        frame.to_csv(review, index=False, encoding="utf-8-sig")
    return tables




def evaluate_suite(directory, ground_truth_by_name, score_goals, *, metric_configuration):
    """Post-hoc soft P/R/F1; callback returns precision, recall, f1 and matched count.

    Failed/missing runs remain in runs.csv, never masquerading as zero-quality
    hierarchies. Nonconvergent completed runs are evaluated with their status.
    """
    import pandas as pd
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    tables = export_operational_tables(directory)
    references = {b["dataset"]: ground_truth_by_name[b["dataset"]] for b in manifest["baselines"]}
    rows, matches = [], []

    def texts(collection, key):
        return [(item.get("name") or item.get("description") or "").strip()
                for item in collection[key]]

    def add(identity, hlg, llg):
        reference = references[identity["dataset"]]
        for level, generated, refs in (("HLG", texts(hlg, "goals"), reference["highLevelGoals"]),
                                       ("LLG", texts(llg, "low_level_goals"), reference["lowLevelGoals"])):
            score = dict(score_goals(generated, refs))
            assignments = score.pop("matches", [])
            rows.append({**identity, "level": level, "generated_goal_count": len(generated),
                         "reference_goal_count": len(refs), **score})
            matches.extend({**identity, "level": level, **match} for match in assignments)

    baselines = {b["dataset"]: b for b in manifest["baselines"]}
    for name, baseline in baselines.items():
        source = directory / baseline["snapshot"]
        if _digest(source.read_bytes()) != baseline["sha256"]:
            raise ValueError(f"Baseline altered: {source}")
        data = _read(source)
        add({"dataset": name, "config_id": "top_down", "repetition": 0, "iteration": 0,
             "state": "baseline", "baseline_sha256": baseline["sha256"]},
            data["structuredHighLevelGoals"], data["structuredLowLevelGoals"])
    for run in tables["runs"].to_dict("records"):
        if run["status"] != "completed":
            continue
        payload = _read(directory / run["attempt_directory"] / "result.json")
        if payload["baselineSha256"] != baselines[run["dataset"]]["sha256"]:
            raise ValueError("Result does not use the frozen baseline")
        identity = {k: run[k] for k in ("dataset", "config_id", "repetition", "converged", "stop_reason")}
        identity["baseline_sha256"] = payload["baselineSha256"]
        for trace in payload["globalGoalCycle"]["iterations"]:
            if trace.get("checkpoint_complete") is False:
                continue
            add({**identity, "iteration": trace["iteration"], "state": "iteration"},
                trace["high_level_goals"], trace["low_level_goals"])
        add({**identity, "iteration": payload["globalGoalCycle"]["completed_iterations"], "state": "final"},
            payload["finalHighLevelGoals"], payload["finalLowLevelGoals"])
    metrics = pd.DataFrame(rows)
    baseline = metrics[metrics.state == "baseline"]
    final = metrics[metrics.state == "final"]
    values = ["precision", "recall", "f1"]
    deltas = final.merge(baseline[["dataset", "level", *values]], on=["dataset", "level"],
                         suffixes=("", "_top_down"), validate="many_to_one")
    for value in values:
        deltas[f"delta_{value}_top_down"] = deltas[value] - deltas[f"{value}_top_down"]
    reference = final[final.config_id == "reference"]
    paired = final[final.config_id != "reference"].merge(
        reference[["dataset", "repetition", "level", *values]],
        on=["dataset", "repetition", "level"], suffixes=("", "_reference"), validate="many_to_one")
    for value in values:
        paired[f"delta_{value}_reference"] = paired[value] - paired[f"{value}_reference"]
    # Average repetitions within each dataset before macro-averaging datasets.
    per_dataset = final.groupby(["config_id", "dataset", "level"])[values].mean().reset_index()
    macro = per_dataset.groupby(["config_id", "level"])[values].agg(["mean", "std", "count"])
    macro.columns = ["_".join(column) for column in macro.columns]
    paired_columns = [f"delta_{value}_reference" for value in values]
    paired_dataset = paired.groupby(["config_id", "dataset", "level"])[paired_columns].mean().reset_index()
    paired_macro = paired_dataset.groupby(["config_id", "level"])[paired_columns].agg(["mean", "std", "count"])
    paired_macro.columns = ["_".join(column) for column in paired_macro.columns]
    tables.update(metrics=metrics, baseline_deltas=deltas, paired_reference_deltas=paired,
                  macro_metrics=macro.reset_index(), paired_macro=paired_macro.reset_index(),
                  matches=pd.DataFrame(matches))
    for name, table in tables.items():
        table.to_csv(directory / "comparison" / f"{name}.csv", index=False, encoding="utf-8-sig")
    _write_json(directory / "comparison" / "evaluation_manifest.json", {
        "metric": metric_configuration, "ground_truth_sha256": _digest(_json_bytes(references)),
        "ground_truth_snapshot": references, "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregation": "Mean repetitions within dataset, then macro mean across datasets. Counts exported.",
        "pairing": "Alternative/reference matched on dataset, repetition, level; missing pairs excluded explicitly.",
        "missing_reference_pairs": len(final[final.config_id != "reference"]) - len(paired),
        "status_policy": "Evaluate all completed results, including nonconvergent; failures excluded, retained in runs.csv.",
    })
    return tables


from contextlib import contextmanager
import json
from pathlib import Path
import re
from threading import Event, Thread


def _dataset_folder(name):
    folder = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip().rstrip('. ')
    if not folder or folder in {'.', '..'}:
        raise ValueError("Dataset must have a usable directory name")
    if folder.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        folder = '_' + folder
    return folder


def sync_iteration_exports(suite_directory, *, _signatures=None):
    """Copy current and historical attempts without deleting or moving sources."""
    suite = Path(suite_directory).resolve()
    manifest = json.loads((suite / 'manifest.json').read_text(encoding='utf-8'))
    target_root = suite / 'bottom_up_iterations'
    mapping = {b['dataset_id']: _dataset_folder(b['dataset']) for b in manifest['baselines']}
    if len({name.casefold() for name in mapping.values()}) != len(mapping):
        raise ValueError("Dataset names collide after Windows filename normalization")
    signatures = {} if _signatures is None else _signatures
    copied = 0
    index = []
    for config in manifest['configurations']:
        for baseline in manifest['baselines']:
            dataset_name = mapping[baseline['dataset_id']]
            for repeat in range(1, manifest['repetitions'] + 1):
                job = suite / 'runs' / config['config_id'] / f'repeat_{repeat:03d}' / baseline['dataset_id']
                readable_job = target_root / config['config_id'] / dataset_name / f'repeat_{repeat:03d}'
                for attempt in sorted(job.glob('attempt_*')):
                    destination = readable_job / attempt.name
                    index.append({'configuration': config['config_id'], 'dataset': baseline['dataset'],
                                  'repetition': repeat, 'attempt': attempt.name,
                                  'source': str((attempt / 'iterations').relative_to(suite)),
                                  'directory': str(destination.relative_to(suite))})
                    for source in (attempt / 'iterations').glob('*.json'):
                        stat = source.stat()
                        signature = (stat.st_mtime_ns, stat.st_size)
                        target = destination / source.name
                        if signatures.get(source) == signature and target.exists():
                            continue
                        raw = source.read_bytes()
                        try:
                            json.loads(raw)
                        except (ValueError, UnicodeDecodeError):
                            # The orchestrator can be updating a checkpoint right now.
                            # Retain the last valid exported snapshot and retry later.
                            continue
                        destination.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix('.json.tmp')
                        temporary.write_bytes(raw)
                        temporary.replace(target)
                        signatures[source] = signature
                        copied += 1
    target_root.mkdir(parents=True, exist_ok=True)
    index_path = target_root / 'index.json'
    content = json.dumps(index, ensure_ascii=False, indent=2)
    if not index_path.exists() or index_path.read_text(encoding='utf-8') != content:
        index_path.write_text(content, encoding='utf-8')
    return {'directory': target_root, 'copied_files': copied, 'attempts': len(index)}


@contextmanager
def live_iteration_exports(suite_directory, *, interval_seconds=2):
    """Keep readable folders updated while run_suite executes in the notebook."""
    if interval_seconds <= 0:
        raise ValueError("Export polling interval must be positive")
    signatures, errors = {}, []
    stop = Event()
    sync_iteration_exports(suite_directory, _signatures=signatures)

    def monitor():
        while not stop.wait(interval_seconds):
            try:
                sync_iteration_exports(suite_directory, _signatures=signatures)
            except Exception as exc:
                errors.append(exc)
                return

    worker = Thread(target=monitor, name='experiment-iteration-exports', daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        sync_iteration_exports(suite_directory, _signatures=signatures)
        if errors:
            raise RuntimeError(f"Readable iteration export failed: {errors[0]}") from errors[0]
