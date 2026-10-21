"""Snapshot-based numerical review and transactional bottom-up repairs."""
import hashlib
import io
import json
import re
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from src.bottom_up.global_goal_evaluator import (
    BOTTOM_UP_EVALUATOR_MODELS, PIPELINE_POLICY_VERSION,
    BOTTOM_UP_EVALUATOR_TEMPERATURE, evaluate_all_branches,
    CRITIC1_FEW_SHOT_EXAMPLE_COUNT, evaluator_examples,
)
from src.bottom_up.goal_similarity import (
    analyze_similarities, similarity_configuration, build_cosine_payload,
)
from src.bottom_up.goal_reconstructor import reconstruct_all_branches
from src.data_model import (
    Actors, BottomUpHighLevelGoal, GlobalGoalCycleIteration, GlobalGoalCycleResult,
    GlobalGoalCycleStopReason, GlobalGoalEvaluationDecision, GlobalGoalEvaluationResult,
    GoalBranch, HighLevelGoal, HighLevelGoalDecision, HighLevelGoals,
    LowLevelGoal, LowLevelGoalRegenerationRequest, LowLevelGoals,
)
from src.examples.shot_learning import ShotPromptingMode
from src.extraction.extractor import generate_high_level_goals, generate_bottom_up_low_level_goals
from src.self_critique.refine_response import (
    BottomUpEvalMode, RepairCandidateRejected, generate_bottom_up_response_with_reflection,
)

DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS = 25
MAX_HLG_REWRITES_PER_BRANCH = 3
BOTTOM_UP_CRITIC_MEMORY_TURNS = 0
BOTTOM_UP_CRITIC_STATE_UPDATES = 0


def _model_payload(value):
    """Return plain data so Pydantic can cross module-reload boundaries."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _coerce_model(value, model_type, *, context: str):
    """Rebuild a model with the schema class currently used by this module."""
    try:
        return model_type.model_validate(_model_payload(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def _stable_hlg_key(goal: HighLevelGoal) -> str:
    """Return a technical fingerprint without making a semantic judgment."""
    payload = json.dumps(
        goal.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _branches_from_embedded_parents(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> list[GoalBranch]:
    """Build one branch per HLG, including HLGs with no generated children."""
    fingerprint_occurrences: dict[str, int] = {}
    branches = []
    for goal in high_level_goals.goals:
        fingerprint = _stable_hlg_key(goal)
        occurrence = fingerprint_occurrences.get(fingerprint, 0) + 1
        fingerprint_occurrences[fingerprint] = occurrence
        branches.append(GoalBranch(
            branch_id=f"hlg_{fingerprint}_{occurrence:03d}",
            high_level_goal=goal,
            low_level_goals=[],
        ))
    for llg in low_level_goals.low_level_goals:
        parent = llg.high_level_associated
        branch = next(
            (item for item in branches if item.high_level_goal == parent),
            None,
        )
        if branch is None:
            raise ValueError(
                "An LLG references a parent that is absent from the current "
                f"HLG state: {parent.actor.name}::{parent.name}."
            )
        branch.low_level_goals.append(llg)
    return branches


def _validate_embedded_parents(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> None:
    """Reject a baseline whose embedded LLG parents are not official HLGs."""
    unexpected = {
        f"{parent.actor.name}::{parent.name}"
        for parent in (
            goal.high_level_associated
            for goal in low_level_goals.low_level_goals
        )
        if parent not in high_level_goals.goals
    }
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(
            "The top-down baseline is inconsistent: these embedded LLG "
            f"parents are absent from structuredHighLevelGoals: {names}. "
            "Regenerate the top-down baseline before running bottom-up."
        )


def _actors_from_goals(high_level_goals: HighLevelGoals) -> Actors:
    actors = []
    seen = set()
    for goal in high_level_goals.goals:
        actor_key = _key(goal.actor.name)
        if actor_key not in seen:
            seen.add(actor_key)
            actors.append(goal.actor)
    return Actors(actors=actors)


def _merge_actors(*actor_sets: Actors) -> Actors:
    actors = []
    seen = set()
    for actor_set in actor_sets:
        for actor in actor_set.actors:
            actor_key = _key(actor.name)
            if actor_key not in seen:
                seen.add(actor_key)
                actors.append(actor)
    return Actors(actors=actors)


def _regenerate_llgs_with_top_down(
    request: LowLevelGoalRegenerationRequest,
    mode: ShotPromptingMode,
    evaluator_ablation: bool,
) -> LowLevelGoals:
    actors = _actors_from_goals(request.high_level_goals)
    guidance = "\n".join(request.guidance_by_parent_name.values())
    focused_description = (
        "WHY THIS DECOMPOSITION IS BEING REGENERATED (rationale; verify against source):\n" + request.rationale
        + ("\n\nOPERATIONAL REPAIR GUIDANCE:\n" + guidance
           if guidance.strip() and guidance.strip() != request.rationale.strip() else "")
        + "\n\nPROJECT SOURCE (authoritative):\n" + request.project_description
        + "\n\nCURRENT PARENT SCOPE:\n" + request.high_level_goals.model_dump_json()
        + "\n\nEXISTING CHILDREN, including full descriptions:\n"
        + request.existing_low_level_goals.model_dump_json()
        + "\n\nOTHER HLG BOUNDARIES:\n" + request.sibling_high_level_goals.model_dump_json()
        + "\n\nReturn the entire corrected, non-redundant decomposition of the supplied parent. "
        "Preserve documented valid actions and details. Remove unsupported or redundant "
        "children and add only essential documented missing interactions. Do not import "
        "other HLG responsibilities or conventional features. The current parent is fixed; "
        "do not penalize these children for a defect of an earlier parent."
    )

    def generate_with_context(high_level_goals, feedback=None, mode=mode):
        return generate_bottom_up_low_level_goals(
            high_level_goals,
            feedback=feedback,
            mode=mode,
            generation_context=focused_description,
        )

    result, _, _ = generate_bottom_up_response_with_reflection(
        target_type="Low Level Goals",
        call_function=generate_with_context,
        define_args=(request.high_level_goals,),
        eval_mode=BottomUpEvalMode.LOW_LEVEL,
        eval_args=(focused_description, actors, request.high_level_goals),
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
        focused_scope=(
            "Evaluate only the supplied actor, the listed parent HLGs, and "
            "their LLG decomposition. Do not require other project goals."
        ),
    )
    return _coerce_model(
        result,
        LowLevelGoals,
        context="Low-Level Goal generator output",
    )


def _deduplicate_llg_candidates(goals: list[LowLevelGoal]) -> list[LowLevelGoal]:
    """Remove exact repeated names/descriptions for the same parent."""
    kept = []
    seen = set()
    for goal in goals:
        identity = (
            _stable_hlg_key(goal.high_level_associated),
            _key(goal.name),
            _key(goal.description),
        )
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(goal)
    return kept


def _accept_regenerated_llgs(
    *,
    existing: list[LowLevelGoal],
    generated: list[LowLevelGoal],
) -> tuple[list[LowLevelGoal], str | None]:
    """Reject empty replacements and preserve the old branch."""
    candidates = _deduplicate_llg_candidates(generated)
    preserved = list(existing)
    if not candidates:
        return preserved, "LLG_REGENERATION_REJECTED_EMPTY"
    return candidates, None


def _generate_hlgs_with_top_down(branch, description, guidance, mode, evaluator_ablation):
    focused = (f"Project source:\n{description}\n\nActor:\n{branch.high_level_goal.actor.model_dump_json()}"
               f"\nOld parent:\n{branch.high_level_goal.model_dump_json()}\nRequired replacement:\n{guidance}"
               "\nReturn exactly one source-supported HLG for this actor, following the reconstructed scope.")
    actors = Actors(actors=[branch.high_level_goal.actor])
    result, _, _ = generate_bottom_up_response_with_reflection(
        target_type="High Level Goals", call_function=generate_high_level_goals,
        define_args=(focused, actors), eval_mode=BottomUpEvalMode.HIGH_LEVEL,
        eval_args=(focused, actors), shotPromptingMode=mode, llama_ablation=evaluator_ablation,
        focused_scope="Evaluate only this replacement outcome and actor against the project source.")
    goals = _coerce_model(result, HighLevelGoals, context="HLG repair")
    if len(goals.goals) != 1:
        raise ValueError("HLG generation must produce exactly one candidate")
    if goals.goals[0].actor != branch.high_level_goal.actor:
        raise ValueError("Generated HLG must preserve the exact actor")
    return goals.goals[0]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_iteration(directory, trace, file_prefix=""):
    if directory is None:
        return
    root = Path(directory)
    _write_json(root / f"{file_prefix}iteration_{trace.iteration:03d}.json", trace.model_dump(mode="json"))


def _mark_unresolved(result, message):
    return result.model_copy(update={
        "final_decision": GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
        "evaluation_status": "REPAIR_UNRESOLVED", "rationale": result.rationale + " " + message})


def run_global_goal_cycle(
    *, project_description: str, initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals, mode=ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation=False, evaluation_output_directory=None,
    max_iterations=DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS, actors: Actors | None = None,
    verbose: bool = False, output_file_prefix: str | None = None,
    evaluator_few_shot_examples: int | None = None,
    evaluator_temperature: float | None = None,
    parent_threshold: float | None = None, duplicate_threshold: float | None = None,
) -> GlobalGoalCycleResult:
    if not 1 <= max_iterations <= DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS:
        raise ValueError("max_iterations must be between 1 and 25")
    evaluator_few_shot_examples = (CRITIC1_FEW_SHOT_EXAMPLE_COUNT if evaluator_few_shot_examples is None
                                   else evaluator_few_shot_examples)
    evaluator_examples(evaluator_few_shot_examples)
    evaluator_temperature = BOTTOM_UP_EVALUATOR_TEMPERATURE if evaluator_temperature is None else evaluator_temperature
    if not 0 <= evaluator_temperature <= 2:
        raise ValueError("Evaluator temperature must be between 0 and 2")
    similarity_options = {"parent_threshold": parent_threshold, "duplicate_threshold": duplicate_threshold}
    active_similarity = similarity_configuration(**similarity_options)
    file_prefix = ""
    if output_file_prefix is not None:
        safe_name = re.sub(r"[^\w.-]+", "_", output_file_prefix).strip("._")
        if not safe_name:
            raise ValueError("Output file prefix must contain a project name")
        file_prefix = safe_name + "_"
    current_hlgs = _coerce_model(initial_high_level_goals, HighLevelGoals, context="initial HLGs").model_copy(deep=True)
    current_llgs = _coerce_model(initial_low_level_goals, LowLevelGoals, context="initial LLGs").model_copy(deep=True)
    _validate_embedded_parents(current_hlgs, current_llgs)
    known_actors = _merge_actors(actors or Actors(actors=[]), _actors_from_goals(current_hlgs))
    rewrites, regenerations, rejections = {}, {}, {}
    traces, warnings = [], []
    config = {"pipeline_policy_version": PIPELINE_POLICY_VERSION,
              "output_file_prefix": file_prefix,
              "evaluator_models": list(BOTTOM_UP_EVALUATOR_MODELS),
              "evaluator_temperature": evaluator_temperature,
              "evaluator_few_shot_examples": evaluator_few_shot_examples,
              "evaluator_response_format": "json_schema",
              "evaluator_endpoint": "/v1/chat/completions",
              "similarity": active_similarity, "completeness_enabled": False,
              "new_goal_discovery": "disabled",
              "direct_llg_audit_enabled": False,
              "max_global_iterations": max_iterations,
              "max_hlg_rewrites_per_lineage": MAX_HLG_REWRITES_PER_BRANCH,
              "output_verbosity": "detailed" if verbose else "iteration_summary"}

    def finish(reason, converged):
        return GlobalGoalCycleResult(
            converged=converged, stop_reason=reason, completed_iterations=len(traces),
            final_high_level_goals=current_hlgs, final_low_level_goals=current_llgs,
            iterations=traces, hlg_rewrite_counts=rewrites, llg_regeneration_counts=regenerations,
            repair_rejection_counts=rejections, warnings=warnings, configuration=config,
            confirmed_branch_ids=traces[-1].confirmed_branch_ids if traces else [])

    for iteration in range(1, max_iterations + 1):
        print(f"[bottom-up] iteration {iteration}/{max_iterations} | full snapshot review", flush=True)
        stage_logs = {}

        def run_stage(name, function, *args, **kwargs):
            if verbose:
                return function(*args, **kwargs)
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    return function(*args, **kwargs)
            finally:
                if stdout.getvalue() or stderr.getvalue():
                    stage_logs[name] = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}

        branches = _branches_from_embedded_parents(current_hlgs, current_llgs)
        reconstructions = {k: _coerce_model(v, BottomUpHighLevelGoal, context="reconstruction")
                           for k, v in run_stage("reconstruction", reconstruct_all_branches, branches).items()}
        analysis = run_stage("similarity", analyze_similarities, branches, reconstructions, **similarity_options)
        config["similarity"] = analysis["configuration"]
        cosine_payload = build_cosine_payload(branches, reconstructions, analysis, iteration=iteration)
        if evaluation_output_directory is not None:
            _write_json(Path(evaluation_output_directory) / f"{file_prefix}cosine_{iteration:03d}.json", cosine_payload)
        if verbose:
            print(json.dumps(cosine_payload, ensure_ascii=False, indent=2), flush=True)
        global_votes = []

        def save_global_vote(record):
            global_votes.append(record)
            checkpoint(0, len(branches), {})

        def checkpoint(done, total, evaluations):
            trace = GlobalGoalCycleIteration(
                iteration=iteration, reconstructions=reconstructions, evaluations=evaluations,
                high_level_goals=current_hlgs, low_level_goals=current_llgs,
                critic1_global_votes=list(global_votes), similarity_analysis=analysis,
                stage_logs=dict(stage_logs),
                input_hierarchy={"high_level_goals": current_hlgs.model_dump(mode="json"),
                                 "low_level_goals": current_llgs.model_dump(mode="json")},
                checkpoint_complete=False, completed_branch_count=done, total_branch_count=total)
            _save_iteration(evaluation_output_directory, trace, file_prefix)

        checkpoint(0, len(branches), {})
        evaluations = run_stage("documentary_review", evaluate_all_branches,
            project_description, branches, reconstructions, actors=known_actors,
            on_global_vote=save_global_vote,
            on_evaluation=checkpoint, similarity_analysis=analysis, cosine_payload=cosine_payload,
            few_shot_example_count=evaluator_few_shot_examples, evaluator_temperature=evaluator_temperature)
        evaluations = {k: _coerce_model(v, GlobalGoalEvaluationResult, context="global review")
                       for k, v in evaluations.items()}
        start_hlgs, start_llgs = current_hlgs, current_llgs
        next_goals = {b.branch_id: b.high_level_goal for b in branches}
        next_children = {b.branch_id: list(b.low_level_goals) for b in branches}
        outcomes, iteration_warnings = {}, []
        changed = False
        structural_requested = any(e.final_decision in {
            GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL,
            GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL} for e in evaluations.values())
        protected = {e.high_level_evaluation.covered_by_branch_id for e in evaluations.values()
                     if e.high_level_evaluation.covered_by_branch_id}
        for branch in branches:
            result = evaluations[branch.branch_id]
            if result.final_decision != GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL:
                continue
            if branch.branch_id in protected:
                evaluations[branch.branch_id] = _mark_unresolved(result, "Conflicting survivor removal.")
                continue
            next_goals.pop(branch.branch_id)
            next_children.pop(branch.branch_id)
            outcomes[branch.branch_id] = "REMOVED_WITH_CHILDREN"
            changed = True

        for branch in branches:
            bid = branch.branch_id
            result = evaluations[bid]
            rewrite = result.final_decision == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
            regenerate = result.final_decision == GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
            if not (rewrite or regenerate) or bid not in next_goals:
                continue
            if regenerate and structural_requested:
                evaluations[bid] = _mark_unresolved(result, "LLG_REPAIR_DEFERRED_STRUCTURAL_ACTIONS")
                outcomes[bid] = "DEFERRED"
                continue
            counts = rewrites if rewrite else regenerations
            if rewrite and rewrites.get(bid, 0) >= MAX_HLG_REWRITES_PER_BRANCH:
                message = "HLG_REWRITE_LIMIT_REACHED"
                evaluations[bid] = _mark_unresolved(result, message)
                iteration_warnings.append(f"{bid}: {message}")
                outcomes[bid] = "LIMIT_REACHED"
                continue
            counts[bid] = counts.get(bid, 0) + 1
            try:
                parent = branch.high_level_goal
                if rewrite:
                    guidance = (f"Documented reconstructed intention:\n{reconstructions[bid].reconstructed_high_level_goal}"
                                f"\nRequested correction:\n{result.high_level_evaluation.rewriting_focus}")
                    parent = run_stage(f"hlg_repair_{bid}", _generate_hlgs_with_top_down,
                                       branch, project_description, guidance, mode, evaluator_ablation)
                    if parent == branch.high_level_goal:
                        raise ValueError("Rewrite returned the unchanged parent")
                    if any(g == parent for key, g in next_goals.items() if key != bid):
                        raise ValueError("Rewritten parent exactly duplicates an existing parent")
                feedback = result.high_level_evaluation.rewriting_focus if rewrite else result.regeneration_feedback
                generated = run_stage(f"llg_repair_{bid}", _regenerate_llgs_with_top_down, LowLevelGoalRegenerationRequest(
                    high_level_goals=HighLevelGoals(goals=[parent]), project_description=project_description,
                    rationale=result.rationale, guidance_by_parent_name={parent.name: feedback or result.rationale},
                    existing_low_level_goals=LowLevelGoals(low_level_goals=branch.low_level_goals),
                    sibling_high_level_goals=HighLevelGoals(goals=[g for key, g in next_goals.items() if key != bid])),
                    mode, evaluator_ablation)
                generated = _coerce_model(generated, LowLevelGoals, context="LLG repair")
                if any(child.high_level_associated != parent for child in generated.low_level_goals):
                    raise ValueError("Generated LLG did not preserve the exact parent")
                accepted, rejection = _accept_regenerated_llgs(
                    existing=branch.low_level_goals, generated=generated.low_level_goals)
                if rejection:
                    raise ValueError(rejection)
                if not rewrite and accepted == branch.low_level_goals:
                    raise ValueError("Regeneration returned the unchanged decomposition")
                next_goals[bid], next_children[bid] = parent, accepted
                if rewrite:
                    new_bid = _branches_from_embedded_parents(
                        HighLevelGoals(goals=[parent]), LowLevelGoals(low_level_goals=accepted))[0].branch_id
                    rewrites[new_bid] = rewrites[bid]
                    regenerations[new_bid] = regenerations.get(bid, 0)
                    rejections[new_bid] = rejections.get(bid, 0)
                outcomes[bid] = "REWRITTEN_WITH_CHILDREN" if rewrite else "LLGS_REGENERATED"
                changed = True
            except Exception as exc:
                rejections[bid] = rejections.get(bid, 0) + 1
                message = f"REPAIR_REJECTED: {type(exc).__name__}: {exc}"
                evaluations[bid] = _mark_unresolved(result, message)
                outcomes[bid] = "REJECTED_STATE_PRESERVED"
                iteration_warnings.append(f"{bid}: {message}")

        current_hlgs = HighLevelGoals(goals=list(next_goals.values()))
        current_llgs = LowLevelGoals(low_level_goals=[child for children in next_children.values() for child in children])
        confirmed = [bid for bid, e in evaluations.items()
                     if e.final_decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH]
        for bid, result in evaluations.items():
            outcomes.setdefault(bid, result.final_decision.value)
            if verbose:
                print(f"[bottom-up] applied | branch={bid} | {outcomes[bid]}", flush=True)
        context_blocked = not changed and any(
            result.evaluation_status == "CONTEXT_LENGTH_EXCEEDED" for result in evaluations.values()
        )
        if context_blocked:
            iteration_warnings.append("EVALUATOR_CONTEXT_LENGTH_EXCEEDED: unchanged snapshot will not be retried")
        trace = GlobalGoalCycleIteration(
            iteration=iteration, reconstructions=reconstructions, evaluations=evaluations,
            critic1_global_votes=global_votes, similarity_analysis=analysis,
            stage_logs=stage_logs,
            high_level_goals=current_hlgs, low_level_goals=current_llgs,
            input_hierarchy={"high_level_goals": start_hlgs.model_dump(mode="json"),
                             "low_level_goals": start_llgs.model_dump(mode="json")},
            output_hierarchy={"high_level_goals": current_hlgs.model_dump(mode="json"),
                              "low_level_goals": current_llgs.model_dump(mode="json")},
            structural_action_outcomes=outcomes,
            hlg_structure_summary=dict(Counter(e.final_decision.value for e in evaluations.values())),
            llg_regeneration_counts=dict(regenerations), hlg_rewrite_counts=dict(rewrites),
            repair_rejection_counts=dict(rejections), confirmed_branch_ids=confirmed,
            warnings=iteration_warnings, checkpoint_complete=True,
            completed_branch_count=len(evaluations), total_branch_count=len(branches))
        traces.append(trace)
        warnings.extend(iteration_warnings)
        _save_iteration(evaluation_output_directory, trace, file_prefix)
        print(f"[bottom-up] iteration {iteration} complete | branches={len(branches)} | "
              f"changed={changed} | decisions={dict(Counter(e.final_decision.value for e in evaluations.values()))}",
              flush=True)
        if evaluation_output_directory is not None:
            print(f"  saved: {file_prefix}iteration_{iteration:03d}.json, {file_prefix}cosine_{iteration:03d}.json", flush=True)
        if not changed and len(confirmed) == len(branches):
            return finish(GlobalGoalCycleStopReason.ALL_BRANCHES_CONFIRMED, True)
        if context_blocked:
            return finish(GlobalGoalCycleStopReason.NO_ACTIONS_REMAIN_WITH_WARNINGS, False)
        if branches and all(analysis["branches"][b.branch_id]["status"] == "ERROR" for b in branches):
            warnings.append("SIMILARITY_UNAVAILABLE: no LLM verdict can replace a numerical comparison")
            return finish(GlobalGoalCycleStopReason.NO_ACTIONS_REMAIN_WITH_WARNINGS, False)
        if not changed:
            actionable = [b for b in branches if evaluations[b.branch_id].evaluation_status == "REPAIR_UNRESOLVED"]
            if actionable and all(outcomes.get(b.branch_id) == "LIMIT_REACHED" for b in actionable):
                return finish(GlobalGoalCycleStopReason.NO_ACTIONS_REMAIN_WITH_WARNINGS, False)
    return finish(GlobalGoalCycleStopReason.MAX_ITERATIONS_REACHED, False)
