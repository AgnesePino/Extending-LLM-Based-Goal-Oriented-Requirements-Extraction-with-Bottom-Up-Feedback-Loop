"""Snapshot-based numerical review and transactional bottom-up repairs."""
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
from src.bottom_up.result_storage import atomic_write, compact_cycle
from src.data_model import (
    Actors, BottomUpHighLevelGoal, GlobalGoalCycleIteration, GlobalGoalCycleResult,
    GlobalGoalCycleStopReason, GlobalGoalEvaluationDecision, GlobalGoalEvaluationResult,
    GoalBranch, HighLevelGoal, HighLevelGoals, LowLevelGoals, group_low_level_goals_by_hlg,
)
from src.examples.shot_learning import ShotPromptingMode
from src.extraction.extractor import generate_high_level_goals, generate_low_level_goals
from src.self_critique.refine_response import (
    EvalMode, Feedback, MAX_ATTEMPTS, QUALITY_THRESHOLD, generate_response_with_reflection,
)

DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS = 25
MAX_HLG_COLLECTION_REPAIR_PASSES = 3
# Compatibility for external callers that recorded the former branch limit.
MAX_HLG_REWRITES_PER_BRANCH = MAX_HLG_COLLECTION_REPAIR_PASSES
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


def _branches_from_embedded_parents(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> list[GoalBranch]:
    """Read groups directly from the HLG references carried by generated LLGs."""
    return group_low_level_goals_by_hlg(low_level_goals)


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


def _adapt_bottom_up_feedback(call_function, initial_feedback):
    """Supply the bottom-up request once; later attempts belong to top-down."""
    first_call = True

    def generate(*args, feedback=None, **kwargs):
        nonlocal first_call
        if first_call and feedback is None:
            feedback = initial_feedback
        first_call = False
        return call_function(*args, feedback=feedback, **kwargs)

    return generate


def _run_original_top_down(*, initial_feedback, records, stage, **kwargs):
    """Record all original reflection results for the transactional handoff."""
    kwargs["call_function"] = _adapt_bottom_up_feedback(
        kwargs["call_function"], initial_feedback)
    result, score, critique, attempts, history = generate_response_with_reflection(
        max_attempts=MAX_ATTEMPTS, **kwargs)
    records[stage] = {
        "score": score, "critique": critique, "attempts": attempts,
        "score_history": history,
        "candidate": _model_payload(result),
    }
    return result


def _repair_collections_with_top_down(
    *, description, actors, high_level_goals, low_level_goals,
    hlg_requests, records,
):
    """Start at HLG; replace the old hierarchy only after the LLG phase succeeds."""
    candidates = _run_original_top_down(
        target_type="High Level Goals", call_function=generate_high_level_goals,
        define_args=(description, actors), eval_mode=EvalMode.HIGH_LEVEL,
        eval_args=(description, actors),
        initial_feedback=Feedback(previous_output=high_level_goals,
                                  critique="\n\n".join(hlg_requests)),
        records=records, stage="highLevelGoals",
        shotPromptingMode=ShotPromptingMode.ZERO_SHOT, llama_ablation=False)
    candidates = _coerce_model(candidates, HighLevelGoals, context="HLG repair")
    if not candidates.goals:
        raise ValueError("Top-down returned an empty HLG collection")
    score = records["highLevelGoals"]["score"]
    if score is None or score < QUALITY_THRESHOLD:
        raise ValueError("HLG repair was not approved by the top-down critic; old HLGs and LLGs preserved")
    # This is the ordinary top-down LLG phase, not a separate bottom-up action.
    # It receives the approved HLGs; no orchestration metadata is inserted into
    # generator feedback, and old children remain live until this phase succeeds.
    children = _run_original_top_down(
        target_type="Low Level Goals", call_function=generate_low_level_goals,
        define_args=(candidates,), eval_mode=EvalMode.LOW_LEVEL,
        eval_args=(description, actors, candidates), initial_feedback=None,
        records=records, stage="lowLevelGoals",
        shotPromptingMode=ShotPromptingMode.ZERO_SHOT, llama_ablation=False)
    children = _coerce_model(children, LowLevelGoals, context="LLG repair")
    if not children.low_level_goals:
        raise ValueError("Top-down returned an empty LLG collection")
    candidate_names = {goal.name for goal in candidates.goals}
    if any(child.high_level_associated.name not in candidate_names for child in children.low_level_goals):
        raise ValueError("Top-down LLGs reference an HLG absent from the approved collection")
    return candidates, children


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
    on_state=None,
) -> GlobalGoalCycleResult:
    if mode != ShotPromptingMode.ZERO_SHOT or evaluator_ablation:
        raise ValueError("Bottom-up repairs require original zero-shot generation and an enabled critic")
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
    input_groups = group_low_level_goals_by_hlg(current_llgs)
    current_hlgs = HighLevelGoals(goals=[group.high_level_goal for group in input_groups])
    known_actors = _merge_actors(actors or Actors(actors=[]), _actors_from_goals(current_hlgs))
    rewrites, rejections = {}, {}
    traces, warnings = [], []
    hlg_repair_passes = 0
    config = {"pipeline_policy_version": PIPELINE_POLICY_VERSION,
              "output_file_prefix": file_prefix,
              "evaluator_models": list(BOTTOM_UP_EVALUATOR_MODELS),
              "evaluator_temperature": evaluator_temperature,
              "evaluator_few_shot_examples": evaluator_few_shot_examples,
              "evaluator_response_format": "json_schema",
              "evaluator_endpoint": "/v1/chat/completions",
              "similarity": active_similarity, "completeness_enabled": False,
              "branch_grouping_policy": "direct_llg_embedded_parent_name_and_description_actor_ignored",
              "new_goal_discovery": "no_explicit_bottom_up_addition; top_down_cardinality_unrestricted",
              "direct_llg_audit_enabled": False,
              "max_global_iterations": max_iterations,
              "repair_policy": "original_top_down_start_at_hlg_then_llgs",
              "repair_feedback_policy": "join_agreeing_feedback_critique_only",
              "repair_acceptance": "hlg_critic_approval_then_llg_structural_validation_atomic_commit",
              "repair_reflection_max_attempts": MAX_ATTEMPTS,
              "repair_quality_threshold": QUALITY_THRESHOLD,
              "repair_prompting_mode": "ZERO_SHOT",
              "repair_critic_enabled": True,
              "max_hlg_collection_repair_passes": MAX_HLG_COLLECTION_REPAIR_PASSES,
              "output_verbosity": "detailed" if verbose else "iteration_summary"}

    def publish(pending=None, result=None):
        if result is not None:
            state = result.model_dump(mode="json", exclude_defaults=True)
        else:
            state = dict(configuration=config, completed_iterations=len(traces),
                         final_high_level_goals=current_hlgs.model_dump(mode="json"),
                         final_low_level_goals=current_llgs.model_dump(mode="json"),
                         iterations=[trace.model_dump(mode="json", exclude_defaults=True)
                                     for trace in traces + ([pending] if pending is not None else [])])
        if evaluation_output_directory is not None:
            atomic_write(Path(evaluation_output_directory) / f"{file_prefix}results.json", compact_cycle(state))
        if on_state is not None:
            on_state(state)

    def finish(reason, converged):
        result = GlobalGoalCycleResult(
            converged=converged, stop_reason=reason, completed_iterations=len(traces),
            final_high_level_goals=current_hlgs, final_low_level_goals=current_llgs,
            iterations=traces, hlg_rewrite_counts=rewrites,
            repair_rejection_counts=rejections, warnings=warnings, configuration=config,
            confirmed_branch_ids=traces[-1].confirmed_branch_ids if traces else [])
        publish(result=result)
        return result

    publish()

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
            publish(pending=trace)

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

        # Approved removals are committed before the transactional handoff.
        current_hlgs = HighLevelGoals(goals=list(next_goals.values()))
        current_llgs = LowLevelGoals(low_level_goals=[child for children in next_children.values() for child in children])
        hlg_requests, requested_ids = [], []
        top_down_reflections = {}
        for branch in branches:
            bid = branch.branch_id
            if bid not in next_goals:
                continue
            result = evaluations[bid]
            rewrite = result.final_decision == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
            if not rewrite:
                continue
            if hlg_repair_passes >= MAX_HLG_COLLECTION_REPAIR_PASSES:
                message = "HLG_COLLECTION_REPAIR_LIMIT_REACHED"
                evaluations[bid] = _mark_unresolved(result, message)
                iteration_warnings.append(f"{bid}: {message}")
                outcomes[bid] = "LIMIT_REACHED"
                continue
            instruction = result.feedback.critique if result.feedback else None
            if not instruction or not instruction.strip():
                evaluations[bid] = _mark_unresolved(result, "Missing repair instruction")
                outcomes[bid] = "REJECTED_STATE_PRESERVED"
                iteration_warnings.append(f"{bid}: Missing repair instruction")
                continue
            hlg_requests.append(instruction)
            requested_ids.append(bid)
            rewrites[bid] = rewrites.get(bid, 0) + 1
        handoff_completed = False
        if requested_ids:
            hlg_repair_passes += 1
            try:
                candidate_hlgs, candidate_llgs = run_stage(
                    "top_down_repair", _repair_collections_with_top_down,
                    description=project_description, actors=known_actors,
                    high_level_goals=current_hlgs, low_level_goals=current_llgs,
                    hlg_requests=hlg_requests,
                    records=top_down_reflections)
                state_changed = candidate_hlgs != current_hlgs or candidate_llgs != current_llgs
                current_hlgs, current_llgs = candidate_hlgs, candidate_llgs
                changed = changed or state_changed
                handoff_completed = True
                for bid in requested_ids:
                    outcomes[bid] = "HLG_LLG_HANDOFF_COMPLETED"
            except Exception as exc:
                message = f"REPAIR_REJECTED: {type(exc).__name__}: {exc}"
                for bid in requested_ids:
                    rejections[bid] = rejections.get(bid, 0) + 1
                    evaluations[bid] = _mark_unresolved(evaluations[bid], message)
                    outcomes[bid] = "REJECTED_STATE_PRESERVED"
                iteration_warnings.append(message)
        config["hlg_collection_repair_passes"] = hlg_repair_passes
        confirmed = [] if handoff_completed else [bid for bid, e in evaluations.items()
                     if bid in next_goals and e.final_decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH]
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
            top_down_reflections=top_down_reflections,
            critic1_global_votes=global_votes, similarity_analysis=analysis,
            stage_logs=stage_logs,
            high_level_goals=current_hlgs, low_level_goals=current_llgs,
            input_hierarchy={"high_level_goals": start_hlgs.model_dump(mode="json"),
                             "low_level_goals": start_llgs.model_dump(mode="json")},
            output_hierarchy={"high_level_goals": current_hlgs.model_dump(mode="json"),
                              "low_level_goals": current_llgs.model_dump(mode="json")},
            structural_action_outcomes=outcomes,
            hlg_structure_summary=dict(Counter(e.final_decision.value for e in evaluations.values())),
            hlg_rewrite_counts=dict(rewrites),
            repair_rejection_counts=dict(rejections), confirmed_branch_ids=confirmed,
            warnings=iteration_warnings, checkpoint_complete=True,
            completed_branch_count=len(evaluations), total_branch_count=len(branches))
        traces.append(trace)
        warnings.extend(iteration_warnings)
        publish()
        print(f"[bottom-up] iteration {iteration} complete | branches={len(branches)} | "
              f"changed={changed} | decisions={dict(Counter(e.final_decision.value for e in evaluations.values()))}",
              flush=True)
        if evaluation_output_directory is not None:
            print(f"  updated: {file_prefix}results.json", flush=True)
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
