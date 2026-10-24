"""Function-based bottom-up loop over the original top-down goal collections."""
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
    BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS,
)
from src.bottom_up.goal_similarity import (
    analyze_similarities, similarity_configuration, build_cosine_payload,
    SIMILARITY_TECHNICAL_ATTEMPTS,
)
from src.bottom_up.goal_reconstructor import reconstruct_all_branches
from src.bottom_up.result_storage import atomic_write, compact_cycle
from src.data_model import (
    Actors, BottomUpHighLevelGoal, GlobalGoalCycleIteration, GlobalGoalCycleResult,
    GlobalGoalCycleStopReason, HighLevelGoalDecision, GlobalGoalEvaluationResult, RESULT_SCHEMA_VERSION,
    HighLevelGoals, LowLevelGoals, indexed_high_level_goals, low_level_goals_for_hlg,
)
from src.examples.shot_learning import ShotPromptingMode
from src.extraction.extractor import generate_high_level_goals, generate_low_level_goals
from src.self_critique.refine_response import (
    EvalMode, Feedback, MAX_ATTEMPTS, QUALITY_THRESHOLD, generate_response_with_reflection,
)

# Fixed global bound; iteration counters increase without changing this value.
MAX_ITERATIONS = 25



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


def _reconcile_parent_references(high_level_goals, low_level_goals, *, strict, adjustments, warnings):
    """Recover only unmatched references with one exact description match."""
    parents_by_description = {}
    for parent in high_level_goals.goals:
        parents_by_description.setdefault(parent.description, []).append(parent)
    reconciled = []
    for index, child in enumerate(low_level_goals.low_level_goals):
        reference = child.high_level_associated
        if reference in high_level_goals.goals:
            reconciled.append(child)
            continue
        matches = parents_by_description.get(reference.description, [])
        if len(matches) != 1:
            reason = "absent from" if not matches else "ambiguous in"
            message = f"Top-down LLG parent description is {reason} the approved collection"
            if strict:
                raise ValueError(message)
            warnings.append(f"UNRESOLVED_PARENT_REFERENCE: LLG {index} ({child.name}): {message}")
            reconciled.append(child)
            continue
        parent = matches[0]
        adjustments.append({"llg_index": index, "llg_name": child.name,
            "matched_description": parent.description,
            "generated_parent_reference": reference.model_dump(mode="json"),
            "approved_parent_reference": parent.model_dump(mode="json")})
        reconciled.append(child.model_copy(update={"high_level_associated": parent}))
    return LowLevelGoals(low_level_goals=reconciled)


def _repair_collections_with_top_down(
    *, description, actors, high_level_goals, low_level_goals,
    hlg_requests, records,
):
    """Forward all approved rewrite suggestions to the original top-down generator."""
    if not hlg_requests:
        raise ValueError("A repair requires at least one branch suggestion")
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
    adjustments = []
    records["lowLevelGoals"]["parent_reference_reconciliations"] = adjustments
    children = _reconcile_parent_references(
        candidates, children, strict=True, adjustments=adjustments, warnings=[])
    return candidates, children


def _mark_unresolved(result, message):
    return GlobalGoalEvaluationResult(
        branch_id=result.branch_id, decision=None, diagnostic=" ".join(filter(None, (result.diagnostic, message))),
        evaluation_status="INCONCLUSIVE")


def apply_review_decisions(
    *, project_description: str, known_actors: Actors,
    current_hlgs: HighLevelGoals, current_llgs: LowLevelGoals,
    branches, evaluations, repair,
):
    """Remove approved branches, then send all rewrite feedback to top-down.

    Inputs are the existing goal models and evaluator outputs. Return updated
    collections and trace data; do not constrain the generator's intervention.
    """
    next_goals = dict(branches)
    removed_parents = []
    outcomes, iteration_warnings = {}, []
    changed = False
    protected = {e.surviving_branch_id for e in evaluations.values()
                 if e.surviving_branch_id}
    for branch in branches:
        if (evaluations[branch].decision in {HighLevelGoalDecision.REMOVE, HighLevelGoalDecision.DUPLICATE_REMOVAL}
                and branch in protected):
            evaluations[branch] = _mark_unresolved(evaluations[branch], "Conflicting survivor removal.")
    for branch in branches:
        result = evaluations[branch]
        if result.decision not in {HighLevelGoalDecision.REMOVE, HighLevelGoalDecision.DUPLICATE_REMOVAL}:
            continue
        next_goals.pop(branch)
        removed_parents.append(branches[branch])
        outcomes[branch] = "REMOVED_WITH_CHILDREN"
        changed = True

    # Apply all approved removals before repairing the surviving collection.
    current_hlgs = HighLevelGoals(goals=list(next_goals.values()))
    current_llgs = LowLevelGoals(low_level_goals=[
        child for child in current_llgs.low_level_goals
        if not any(child.high_level_associated.model_dump(mode="json") == parent.model_dump(mode="json")
                   for parent in removed_parents)
        or any(child.high_level_associated.model_dump(mode="json") == parent.model_dump(mode="json")
               for parent in next_goals.values())])
    hlg_requests, requested_ids = [], []
    top_down_reflections = {}
    for branch in branches:
        bid = branch
        if bid not in next_goals:
            continue
        result = evaluations[bid]
        rewrite = result.decision == HighLevelGoalDecision.REWRITE
        if not rewrite:
            continue
        instruction = result.feedback.critique if result.feedback else None
        if not instruction or not instruction.strip():
            evaluations[bid] = _mark_unresolved(result, "Missing repair instruction")
            outcomes[bid] = "REJECTED_STATE_PRESERVED"
            iteration_warnings.append(f"{bid}: Missing repair instruction")
            continue
        hlg_requests.append(instruction)
        requested_ids.append(bid)
    handoff_completed = False
    if requested_ids:
        try:
            candidate_hlgs, candidate_llgs = repair(
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
                outcomes[bid] = "REJECTED_STATE_PRESERVED"
            iteration_warnings.append(message)
    
    confirmed = [] if handoff_completed else [bid for bid, e in evaluations.items()
                 if bid in next_goals and e.decision == HighLevelGoalDecision.KEEP]
    for bid, result in evaluations.items():
        outcomes.setdefault(bid, result.decision.value if result.decision else "INCONCLUSIVE")

    return dict(high_level_goals=current_hlgs, low_level_goals=current_llgs,
                evaluations=evaluations, outcomes=outcomes, changed=changed,
                reflections=top_down_reflections, warnings=iteration_warnings,
                confirmed=confirmed)


def run_global_goal_cycle(
    *, project_description: str, initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals, mode=ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation=False, evaluation_output_directory=None,
    max_iterations=MAX_ITERATIONS, actors: Actors | None = None,
    verbose: bool = False, output_file_prefix: str | None = None,
    evaluator_few_shot_examples: int | None = None,
    evaluator_temperature: float | None = None,
    parent_threshold: float | None = None, duplicate_threshold: float | None = None,
    on_state=None,
) -> GlobalGoalCycleResult:
    """Review original top-down goals using prime versus all original HLG cosine."""
    if mode != ShotPromptingMode.ZERO_SHOT or evaluator_ablation:
        raise ValueError("Bottom-up repairs require original zero-shot generation and an enabled critic")
    if type(max_iterations) is not int or not 1 <= max_iterations <= MAX_ITERATIONS:
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
    known_actors = _merge_actors(actors or Actors(actors=[]), _actors_from_goals(current_hlgs))
    traces, warnings = [], []
    initial_adjustments = []
    current_llgs = _reconcile_parent_references(
        current_hlgs, current_llgs, strict=False, adjustments=initial_adjustments, warnings=warnings)
    config = {"pipeline_policy_version": PIPELINE_POLICY_VERSION,
              "execution_structure": "functions_over_top_down_models",
              "reconstruction_history_policy": "fresh_messages_per_call_and_retry_no_prior_responses",
              "output_file_prefix": file_prefix,
              "evaluator_models": list(BOTTOM_UP_EVALUATOR_MODELS),
              "evaluator_temperature": evaluator_temperature,
              "evaluator_few_shot_examples": evaluator_few_shot_examples,
              "evaluator_response_format": "json_schema",
              "evaluator_endpoint": "/v1/chat/completions",
              "evaluator_retry_policy": "whole_global_response_per_model",
              "evaluator_attempts_per_iteration": BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS,
              "context_recovery_policy": "reload_affected_model_double_context",
              "similarity_technical_attempts": SIMILARITY_TECHNICAL_ATTEMPTS,
              "review_payload_format": "categorical_matrix_v5_prime_original_labels",
              "response_length_policy": "rewrite_feedback_500_character_target_no_length_rejection",
              "similarity": active_similarity,
              "hierarchy_input_policy": "original_top_down_hlg_and_llg_collections",
              "goal_counter_scope": "position_within_iteration_snapshot",
              "max_global_iterations": max_iterations,
              "repair_policy": "original_top_down_start_at_hlg_then_llgs",
              "repair_feedback_policy": "join_agreeing_feedback_critique_only",
              "repair_acceptance": "hlg_critic_approval_then_llg_structural_validation_atomic_commit",
              "parent_reference_policy": "exact_reference_first_then_unique_exact_description",
              "initial_parent_reference_reconciliations": initial_adjustments,
              "repair_reflection_max_attempts": MAX_ATTEMPTS,
              "repair_quality_threshold": QUALITY_THRESHOLD,
              "repair_prompting_mode": "ZERO_SHOT",
              "repair_critic_enabled": True,
              "hlg_collection_repair_policy": "one_batch_pass_per_iteration_until_global_stop",
              "removal_schedule": "all_approved_removals_before_batch_rewrites",
              "generator_intervention_policy": "full_collection_generation_no_bottom_up_freeze",
              "action_schedule": "all_approved_removals_then_all_surviving_rewrite_suggestions",
              "output_verbosity": "detailed" if verbose else "iteration_summary"}

    def publish(pending=None, result=None):
        if result is not None:
            state = result.model_dump(mode="json", exclude_defaults=True)
        else:
            state = dict(schema_version=RESULT_SCHEMA_VERSION, configuration=config, completed_iterations=len(traces),
                         warnings=list(warnings),
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
            schema_version=RESULT_SCHEMA_VERSION,
            converged=converged, stop_reason=reason, completed_iterations=len(traces),
            final_high_level_goals=current_hlgs, final_low_level_goals=current_llgs,
            iterations=traces, warnings=warnings, configuration=config)
        publish(result=result)
        return result

    publish()

    for iteration in range(1, max_iterations + 1):
        print(f"[bottom-up] iteration {iteration}/{max_iterations} | full snapshot review", flush=True)

        def run_stage(name, function, *args, **kwargs):
            if verbose:
                return function(*args, **kwargs)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return function(*args, **kwargs)

        branches = indexed_high_level_goals(current_hlgs)
        children_by_hlg = {bid: low_level_goals_for_hlg(parent, current_llgs)
                           for bid, parent in branches.items()}
        branch_warnings = [f"HLG_WITHOUT_CHILDREN: branch {bid} ({branches[bid].name})"
                           for bid, children in children_by_hlg.items() if not children]
        if verbose:
            for message in branch_warnings:
                print(f"[bottom-up] {message}", flush=True)
        reconstructions = {k: _coerce_model(v, BottomUpHighLevelGoal, context="reconstruction")
                           for k, v in run_stage("reconstruction", reconstruct_all_branches,
                               {bid: children for bid, children in children_by_hlg.items() if children}).items()}
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
                evaluator_responses=list(global_votes), similarity_analysis=analysis,
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
        update = apply_review_decisions(
            project_description=project_description, known_actors=known_actors,
            current_hlgs=current_hlgs, current_llgs=current_llgs,
            branches=branches, evaluations=evaluations,
            repair=lambda **kwargs: run_stage(
                "top_down_repair", _repair_collections_with_top_down, **kwargs))
        current_hlgs, current_llgs = update["high_level_goals"], update["low_level_goals"]
        evaluations, outcomes = update["evaluations"], update["outcomes"]
        changed, confirmed = update["changed"], update["confirmed"]
        top_down_reflections, iteration_warnings = update["reflections"], update["warnings"]
        iteration_warnings.extend(branch_warnings)
        if verbose:
            for bid, outcome in outcomes.items():
                print(f"[bottom-up] applied | branch={bid} | {outcome}", flush=True)
        context_blocked = not changed and any(
            result.evaluation_status == "CONTEXT_LENGTH_EXCEEDED" for result in evaluations.values()
        )
        if context_blocked:
            iteration_warnings.append("EVALUATOR_CONTEXT_LENGTH_EXCEEDED: iteration attempt budget exhausted; remaining valid actions applied; review continues next iteration")
        if branches and all(analysis["branches"][b]["status"] == "ERROR" for b in branches):
            iteration_warnings.append("SIMILARITY_UNAVAILABLE: iteration attempt budget exhausted; review continues next iteration")
        trace = GlobalGoalCycleIteration(
            iteration=iteration, reconstructions=reconstructions, evaluations=evaluations,
            top_down_reflections=top_down_reflections,
            evaluator_responses=global_votes, similarity_analysis=analysis,
            high_level_goals=current_hlgs, low_level_goals=current_llgs,
            input_hierarchy={"high_level_goals": start_hlgs.model_dump(mode="json"),
                             "low_level_goals": start_llgs.model_dump(mode="json")},
            structural_action_outcomes=outcomes,
            warnings=iteration_warnings, checkpoint_complete=True,
            completed_branch_count=len(evaluations), total_branch_count=len(branches))
        traces.append(trace)
        warnings.extend(iteration_warnings)
        publish()
        print(f"[bottom-up] iteration {iteration} complete | branches={len(branches)} | "
              f"changed={changed} | decisions={dict(Counter(e.decision.value if e.decision else 'INCONCLUSIVE' for e in evaluations.values()))}",
              flush=True)
        if evaluation_output_directory is not None:
            print(f"  updated: {file_prefix}results.json", flush=True)
        if not changed and len(confirmed) == len(branches):
            return finish(GlobalGoalCycleStopReason.ALL_BRANCHES_CONFIRMED, True)
    return finish(GlobalGoalCycleStopReason.MAX_ITERATIONS_REACHED, False)
