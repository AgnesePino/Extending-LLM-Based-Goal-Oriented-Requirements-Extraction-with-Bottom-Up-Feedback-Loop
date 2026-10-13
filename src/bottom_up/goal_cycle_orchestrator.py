"""Bottom-up validation and repair with an optional, subsequent HLG discovery."""

import hashlib
import json
from pathlib import Path

from src.bottom_up.global_goal_evaluator import (
    evaluate_all_branches,
    evaluate_missing_high_level_goals,
)
from src.bottom_up.goal_reconstructor import reconstruct_all_branches
from src.data_model import (
    Actor,
    Actors,
    BottomUpHighLevelGoal,
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    GlobalGoalCycleStopReason,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    GoalBranch,
    HighLevelGoal,
    HighLevelGoalRemovalBasis,
    HighLevelGoalGenerationAction,
    HighLevelGoalGenerationRequest,
    HighLevelGoalGeneratorInput,
    HighLevelGoals,
    LowLevelGoal,
    LowLevelGoalRegenerationRequest,
    LowLevelGoals,
    MissingHighLevelGoalDecision,
    MissingHighLevelGoalEvaluation,
)
from src.examples.shot_learning import ShotPromptingMode
from src.extraction.extractor import (
    generate_high_level_goals,
    generate_bottom_up_low_level_goals,
)
from src.self_critique.refine_response import (
    BottomUpEvalMode,
    generate_bottom_up_response_with_reflection,
)
from src.llm_clients import EvaluatorConversation


DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS = 5
MAX_LLG_REGENERATIONS_PER_BRANCH = 2
MAX_HLG_REWRITES_PER_BRANCH = 2
# Keep each iteration conservative: outstanding repairs are reconsidered on
# the next pass against the newly updated hierarchy.
MAX_HLG_REWRITES_PER_ITERATION = 2
MAX_HLG_REMOVALS_PER_ITERATION: int | None = 2
MAX_LLG_REGENERATIONS_PER_ITERATION = 2
MAX_REPLACEMENT_HLGS_PER_REWRITE = 1
BOTTOM_UP_CRITIC_MEMORY_TURNS = 4
BOTTOM_UP_CRITIC_STATE_UPDATES = 2


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


def _embedded_parent_key(
    parent: HighLevelGoal,
    expected_parents: dict[str, HighLevelGoal],
) -> str | None:
    """Validate a generated embedded parent without canonicalizing it."""
    matches = [
        key for key, expected in expected_parents.items()
        if parent == expected
    ]
    return matches[0] if len(matches) == 1 else None


def _generation_request(
    request_id: str,
    action: HighLevelGoalGenerationAction,
    description: str,
    actor: Actor,
    rationale: str,
    target_branch_id: str | None = None,
) -> HighLevelGoalGenerationRequest:
    return HighLevelGoalGenerationRequest(
        request_id=request_id,
        action=action,
        generator_input=HighLevelGoalGeneratorInput(
            project_description=description,
            actors=Actors(actors=[actor]),
        ),
        rationale=rationale,
        target_branch_id=target_branch_id,
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


def _actor_for_request(request_actor: str, actors: Actors) -> Actor | None:
    wanted = _key(request_actor)
    for actor in actors.actors:
        if _key(actor.name) == wanted:
            return actor
    return None


def _should_add_missing_hlg(
    *,
    missing_hlg_evaluation: MissingHighLevelGoalEvaluation | None,
    replacements: dict[str, list[HighLevelGoal]],
    removed_branch_ids: set[str],
) -> bool:
    """Apply an optional discovery verdict only to a stable validated state."""
    if (
        missing_hlg_evaluation is None
        or missing_hlg_evaluation.decision
        != MissingHighLevelGoalDecision.FOUND
    ):
        return False
    if replacements or removed_branch_ids:
        return False
    return True


def _generate_hlgs_with_top_down(
    request: HighLevelGoalGenerationRequest,
    mode: ShotPromptingMode,
    evaluator_ablation: bool,
) -> HighLevelGoals:
    focused_description = (
        "Evaluator rationale (use this as the primary correction guidance):\n"
        f"{request.rationale}\n\n"
        f"{request.generator_input.project_description}"
    )
    result, _, _ = generate_bottom_up_response_with_reflection(
        target_type="High Level Goals",
        call_function=generate_high_level_goals,
        define_args=(
            focused_description,
            request.generator_input.actors,
        ),
        eval_mode=BottomUpEvalMode.HIGH_LEVEL,
        eval_args=(
            focused_description,
            request.generator_input.actors,
        ),
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
        focused_scope=(
            "Evaluate only the supplied actor and the single functional "
            "intention described by this replacement request."
        ),
    )
    result = _coerce_model(
        result,
        HighLevelGoals,
        context="High-Level Goal generator output",
    )
    if not result.goals:
        raise ValueError("The top-down pipeline generated no High-Level Goals.")
    allowed_actors = {
        _key(actor.name): actor
        for actor in request.generator_input.actors.actors
    }
    if not allowed_actors:
        raise ValueError("HLG generation requires at least one known actor.")
    normalized_goals = []
    for goal in result.goals:
        actor = allowed_actors.get(_key(goal.actor.name))
        if actor is None:
            if len(allowed_actors) != 1:
                raise ValueError(
                    f"Generated HLG '{goal.name}' uses unknown actor "
                    f"'{goal.actor.name}'."
                )
            actor = next(iter(allowed_actors.values()))
        normalized_goals.append(goal.model_copy(update={"actor": actor}))
    return HighLevelGoals(goals=normalized_goals)


def _regenerate_llgs_with_top_down(
    request: LowLevelGoalRegenerationRequest,
    mode: ShotPromptingMode,
    evaluator_ablation: bool,
) -> LowLevelGoals:
    actors = _actors_from_goals(request.high_level_goals)
    scope = "\n".join(
        f"- {goal.actor.name}: {goal.description}"
        for goal in request.high_level_goals.goals
    )
    feedback = "\n".join(request.guidance_by_parent_name.values())
    existing_by_parent: dict[str, list[LowLevelGoal]] = {}
    for llg in request.existing_low_level_goals.low_level_goals:
        existing_by_parent.setdefault(
            _stable_hlg_key(llg.high_level_associated), []
        ).append(llg)
    constraints = []
    for goal in request.high_level_goals.goals:
        parent_key = _stable_hlg_key(goal)
        existing = existing_by_parent.get(parent_key, [])
        constraints.append(
            f"- {goal.name}: keep valid existing goals and return the "
            "complete, non-redundant LLG decomposition at atomic functional "
            "interaction granularity; there is no fixed numeric cap. Keep "
            "distinct documented actions separate instead of collapsing them "
            f"into a generic operation. Existing LLGs: {[item.name for item in existing]}"
        )
    focused_description = (
        "Evaluator rationale (use this as the primary correction guidance):\n"
        f"{feedback or 'No additional rationale provided.'}\n\n"
        "Generate only the decomposition of these documented stakeholder "
        f"intentions:\n{scope}\n\n"
        "Repair constraints:\n"
        f"{chr(10).join(constraints)}\n"
        "Preserve already valid LLGs. Do not add generic CRUD variants, UI "
        "steps, notifications, confirmations, or sibling-HLG responsibilities "
        "unless the evaluator explicitly identifies them as essential. Cover "
        "every distinct operation explicitly required by the documentation, "
        "using a complete and non-redundant decomposition."
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
    """Remove exact semantic labels emitted more than once for one parent."""
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


def _save_iteration(directory: str | Path | None, trace: GlobalGoalCycleIteration):
    if directory is None:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"iteration_{trace.iteration:03d}.json").write_text(
        trace.model_dump_json(indent=2), encoding="utf-8"
    )


def _with_decision(
    evaluation: GlobalGoalEvaluationResult,
    decision: GlobalGoalEvaluationDecision,
    rationale_suffix: str,
) -> GlobalGoalEvaluationResult:
    return evaluation.model_copy(
        update={
            "final_decision": decision,
            "rationale": f"{evaluation.rationale} {rationale_suffix}",
        }
    )


def _select_safe_hlg_removals(
    removal_branches: list[GoalBranch],
    evaluations: dict[str, GlobalGoalEvaluationResult],
) -> tuple[set[str], list[str]]:
    """Apply critic removals by technical branch ID and protect merge targets."""
    selected: set[str] = set()
    protected: set[str] = set()
    warnings: list[str] = []

    for branch in removal_branches:
        if (
            MAX_HLG_REMOVALS_PER_ITERATION is not None
            and len(selected) >= MAX_HLG_REMOVALS_PER_ITERATION
        ):
            break
        branch_key = branch.branch_id
        hlg_evaluation = evaluations[branch.branch_id].high_level_evaluation
        if branch_key in protected:
            warnings.append(
                f"{branch.branch_id}: HLG_REMOVAL_SKIPPED_COVER_TARGET"
            )
            continue

        if (
            hlg_evaluation.removal_basis
            == HighLevelGoalRemovalBasis.FULLY_REDUNDANT
        ):
            target_key = hlg_evaluation.covered_by_branch_id
            if not target_key:
                warnings.append(
                    f"{branch.branch_id}: HLG_REMOVAL_SKIPPED_MISSING_COVER_ID"
                )
                continue
            if target_key in selected:
                warnings.append(
                    f"{branch.branch_id}: HLG_REMOVAL_SKIPPED_CIRCULAR_COVER"
                )
                continue
            protected.add(target_key)

        selected.add(branch_key)

    if len(removal_branches) > len(selected):
        warnings.append(
            "HLG_REMOVAL_LIMIT_OR_DEPENDENCY_GUARD "
            f"({len(removal_branches)} requested, {len(selected)} applied)"
        )
    return selected, warnings


def _schedule_redundant_hlg_merges(
    selected_branch_ids: set[str],
    branches: list[GoalBranch],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    llg_regenerations: dict[
        str, tuple[HighLevelGoal, str, list[LowLevelGoal]]
    ],
) -> None:
    """Preserve both decompositions when the critic removes a redundant HLG."""
    branches_by_id = {branch.branch_id: branch for branch in branches}
    for branch_id in selected_branch_ids:
        evaluation = evaluations[branch_id].high_level_evaluation
        if evaluation.removal_basis != HighLevelGoalRemovalBasis.FULLY_REDUNDANT:
            continue
        target_id = evaluation.covered_by_branch_id
        source = branches_by_id[branch_id]
        target = branches_by_id.get(target_id or "")
        if target is None or target.branch_id in selected_branch_ids:
            continue
        previous = llg_regenerations.get(target.branch_id)
        preserved = list(target.low_level_goals)
        if previous is not None:
            preserved = list(previous[2])
        llg_regenerations[target.branch_id] = (
            target.high_level_goal,
            (
                f"{evaluation.rationale} Preserve every valid capability from "
                "both HLG decompositions while producing one complete LLG set."
            ),
            _deduplicate_llg_candidates([
                *preserved,
                *source.low_level_goals,
            ]),
        )


def _llg_names_by_parent(
    low_level_goals: LowLevelGoals,
) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for goal in low_level_goals.low_level_goals:
        parent_key = _stable_hlg_key(goal.high_level_associated)
        names.setdefault(parent_key, set()).add(goal.name)
    return names


def _critic_state_summary(
    *,
    iteration: int,
    previous_hlgs: HighLevelGoals,
    previous_llgs: LowLevelGoals,
    current_hlgs: HighLevelGoals,
    current_llgs: LowLevelGoals,
) -> str:
    previous_keys = {_stable_hlg_key(goal) for goal in previous_hlgs.goals}
    current_keys = {_stable_hlg_key(goal) for goal in current_hlgs.goals}
    added = current_keys - previous_keys
    removed = previous_keys - current_keys

    previous_llg_names = _llg_names_by_parent(previous_llgs)
    current_llg_names = _llg_names_by_parent(current_llgs)
    changed_llg_branches = 0
    added_llg_count = 0
    removed_llg_count = 0
    for key in sorted(previous_llg_names.keys() | current_llg_names.keys()):
        added_llgs = sorted(
            current_llg_names.get(key, set())
            - previous_llg_names.get(key, set())
        )
        removed_llgs = sorted(
            previous_llg_names.get(key, set())
            - current_llg_names.get(key, set())
        )
        if added_llgs or removed_llgs:
            changed_llg_branches += 1
            added_llg_count += len(added_llgs)
            removed_llg_count += len(removed_llgs)
    return (
        f"Iteration {iteration} applied hierarchy updates. "
        f"HLGs added: {len(added)}; removed or replaced: {len(removed)}. "
        f"LLG branches changed: {changed_llg_branches}; "
        f"LLGs added: {added_llg_count}; removed: {removed_llg_count}. "
        f"Current totals: {len(current_hlgs.goals)} HLGs and "
        f"{len(current_llgs.low_level_goals)} LLGs."
    )


def run_global_goal_cycle(
    *,
    project_description: str,
    initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals,
    mode: ShotPromptingMode = ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation: bool = False,
    evaluation_output_directory: str | Path | None = None,
    max_iterations: int = DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS,
    enable_discovery: bool = False,
) -> GlobalGoalCycleResult:
    initial_high_level_goals = _coerce_model(
        initial_high_level_goals,
        HighLevelGoals,
        context="initial High-Level Goals",
    )
    initial_low_level_goals = _coerce_model(
        initial_low_level_goals,
        LowLevelGoals,
        context="initial Low-Level Goals",
    )
    _validate_embedded_parents(
        initial_high_level_goals,
        initial_low_level_goals,
    )
    current_hlgs = initial_high_level_goals.model_copy(deep=True)
    current_llgs = initial_low_level_goals.model_copy(deep=True)
    initial_branches = _branches_from_embedded_parents(current_hlgs, current_llgs)
    critic_conversation = EvaluatorConversation(
        max_turns=BOTTOM_UP_CRITIC_MEMORY_TURNS,
        max_state_updates=BOTTOM_UP_CRITIC_STATE_UPDATES,
    )
    critic_conversation.remember_state(
        "Initial input before iteration 1; no previous iteration exists. "
        "Current HLGs: "
        f"{sorted(branch.branch_id for branch in initial_branches)}. "
        f"Total LLGs: {len(current_llgs.low_level_goals)}."
    )
    llg_regeneration_counts: dict[str, int] = {}
    confirmed_branch_ids: set[str] = set()
    hlg_rewrite_counts: dict[str, int] = {}
    retired_hlgs: list[HighLevelGoal] = []
    traces: list[GlobalGoalCycleIteration] = []
    warnings: list[str] = []
    known_actors = _actors_from_goals(initial_high_level_goals)

    for iteration in range(1, max_iterations + 1):
        iteration_start_hlgs = current_hlgs.model_copy(deep=True)
        iteration_start_llgs = current_llgs.model_copy(deep=True)
        iteration_warnings: list[str] = []
        print(
            f"[bottom-up] iteration {iteration}/{max_iterations} starting",
            flush=True,
        )
        branches = _branches_from_embedded_parents(current_hlgs, current_llgs)

        # A confirmed branch is skipped while the HLG structure is unchanged.
        # Structural changes clear this set after the iteration and trigger a
        # fresh audit. Branch IDs are technical references to complete HLGs.
        active_branches = [
            branch
            for branch in branches
            if branch.branch_id not in confirmed_branch_ids
        ]
        print(
            f"[bottom-up] full pass: {len(active_branches)} active branch(es) "
            f"out of {len(branches)} total",
            flush=True,
        )

        raw_reconstructions = reconstruct_all_branches(active_branches)
        reconstructions = {
            branch_id: _coerce_model(
                reconstruction,
                BottomUpHighLevelGoal,
                context=f"reconstruction for {branch_id}",
            )
            for branch_id, reconstruction in raw_reconstructions.items()
        }
        raw_evaluations = evaluate_all_branches(
            project_description,
            active_branches,
            reconstructions,
            current_hlgs,
            critic_conversation,
            branches,
        )
        evaluations = {
            branch_id: _coerce_model(
                evaluation,
                GlobalGoalEvaluationResult,
                context=f"evaluation for {branch_id}",
            )
            for branch_id, evaluation in raw_evaluations.items()
        }
        confirmed_branch_ids.update(
            branch.branch_id
            for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        )
        validation_pass_complete = all(
            branch.branch_id in confirmed_branch_ids for branch in branches
        )

        missing_hlg_evaluation = None
        if enable_discovery and validation_pass_complete:
            known_actors = _merge_actors(
                known_actors, _actors_from_goals(current_hlgs)
            )
            missing_hlg_evaluation = _coerce_model(
                evaluate_missing_high_level_goals(
                    project_description,
                    current_hlgs,
                    known_actors,
                    critic_conversation,
                    HighLevelGoals(goals=list(retired_hlgs)),
                ),
                MissingHighLevelGoalEvaluation,
                context="missing High-Level Goal evaluation",
            )
            if (
                missing_hlg_evaluation.decision
                == MissingHighLevelGoalDecision.INCONCLUSIVE
            ):
                iteration_warnings.append(
                    "MISSING_HLG_EVALUATION_INCONCLUSIVE"
                )

        removed_branch_ids: set[str] = set()
        llg_regenerations: dict[
            str, tuple[HighLevelGoal, str, list[LowLevelGoal]]
        ] = {}
        replacements: dict[str, list[HighLevelGoal]] = {}
        additions: list[HighLevelGoal] = []
        # Every verdict is collected against the same immutable iteration
        # snapshot. Apply all independent actions of the highest safe priority
        # together: a small batch of rewrites, otherwise safe removals,
        # otherwise a small batch of LLG repairs. Deferred requests are
        # reconsidered on the next pass against the updated hierarchy.
        rewrite_branches = [
            branch for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        ]
        if len(rewrite_branches) > MAX_HLG_REWRITES_PER_ITERATION:
            iteration_warnings.append(
                "HLG_REWRITE_BATCH_LIMIT "
                f"({len(rewrite_branches)} requested, "
                f"{MAX_HLG_REWRITES_PER_ITERATION} scheduled)"
            )
            rewrite_branches = rewrite_branches[
                :MAX_HLG_REWRITES_PER_ITERATION
            ]
        for branch in rewrite_branches:
            evaluation = evaluations[branch.branch_id]
            original_key = branch.branch_id
            rewrite_count = hlg_rewrite_counts.get(original_key, 0)
            if rewrite_count >= MAX_HLG_REWRITES_PER_BRANCH:
                evaluations[branch.branch_id] = _with_decision(
                    evaluation,
                    GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
                    "HLG rewrite limit reached; retaining the current HLG.",
                )
                iteration_warnings.append(
                    f"{branch.branch_id}: HLG_REWRITE_LIMIT_REACHED"
                )
                continue

            request = evaluation.replacement_request
            generated = _generate_hlgs_with_top_down(
                _generation_request(
                    f"{branch.branch_id}_rewrite",
                    HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL,
                    request.generation_project_description,
                    request.actor,
                    request.rationale,
                    branch.branch_id,
                ),
                mode,
                evaluator_ablation,
            )
            replacement_goals = []
            # A focused rewrite request yields one candidate at most;
            # accepting every candidate here caused HLG proliferation.
            for goal in generated.goals[:MAX_REPLACEMENT_HLGS_PER_REWRITE]:
                if not goal.name.strip():
                    continue
                replacement_goals.append(goal)
                llg_regenerations[f"{branch.branch_id}:replacement"] = (
                    goal,
                    request.rationale,
                    list(branch.low_level_goals),
                )
            if replacement_goals:
                hlg_rewrite_counts[original_key] = rewrite_count + 1
                replacements[original_key] = replacement_goals
            else:
                iteration_warnings.append(
                    f"{branch.branch_id}: HLG_REGENERATION_EMPTY_IGNORED"
                )
        if not replacements:
            removal_branches = [
                branch for branch in active_branches
                if evaluations[branch.branch_id].final_decision
                == GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
            ]
            removed_branch_ids, removal_warnings = _select_safe_hlg_removals(
                removal_branches,
                evaluations,
            )
            iteration_warnings.extend(removal_warnings)
            _schedule_redundant_hlg_merges(
                removed_branch_ids,
                branches,
                evaluations,
                llg_regenerations,
            )

        if not replacements and not removed_branch_ids:
            regeneration_branches = [
                branch for branch in active_branches
                if evaluations[branch.branch_id].final_decision
                == GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
            ]
            if (
                len(regeneration_branches)
                > MAX_LLG_REGENERATIONS_PER_ITERATION
            ):
                iteration_warnings.append(
                    "LLG_REGENERATION_BATCH_LIMIT "
                    f"({len(regeneration_branches)} requested, "
                    f"{MAX_LLG_REGENERATIONS_PER_ITERATION} scheduled)"
                )
                regeneration_branches = regeneration_branches[
                    :MAX_LLG_REGENERATIONS_PER_ITERATION
                ]
            for branch in regeneration_branches:
                evaluation = evaluations[branch.branch_id]
                stable_key = branch.branch_id
                count = llg_regeneration_counts.get(stable_key, 0)
                if count >= MAX_LLG_REGENERATIONS_PER_BRANCH:
                    evaluations[branch.branch_id] = _with_decision(
                        evaluation,
                        GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
                        "LLG regeneration limit reached; keeping the latest LLGs.",
                    )
                    llg_evaluation = evaluation.low_level_evaluation
                    unsupported = (
                        llg_evaluation.unsupported_or_misleading_llg_ids
                        if llg_evaluation else []
                    )
                    missing = (
                        llg_evaluation.missing_essential_capabilities
                        if llg_evaluation else []
                    )
                    iteration_warnings.append(
                        f"{branch.branch_id}: LLG_REGENERATION_LIMIT_REACHED "
                        f"(unsupported={unsupported or ['none']}; "
                        f"missing={missing or ['none']})"
                    )
                    continue
                llg_regeneration_counts[stable_key] = count + 1
                llg_regenerations[stable_key] = (
                    branch.high_level_goal,
                    evaluation.low_level_evaluation.regeneration_feedback
                    or evaluation.rationale,
                    list(branch.low_level_goals),
                )

        unresolved_decisions = {
            evaluation.final_decision
            for evaluation in evaluations.values()
            if evaluation.final_decision
            in {
                GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
                GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
            }
        }
        if GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED in unresolved_decisions:
            iteration_warnings.append(
                "UNRESOLVED_LLG_REGENERATION_LIMIT_REACHED"
            )
        if GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE in unresolved_decisions:
            iteration_warnings.append("UNRESOLVED_EVALUATION_INCONCLUSIVE")

        # Apply branch-level validation and repair decisions. Optional discovery
        # is evaluated only after every current branch has been confirmed.
        next_goals: list[HighLevelGoal] = []
        for branch in branches:
            if branch.branch_id in removed_branch_ids:
                retired_hlgs.append(branch.high_level_goal)
                continue
            if branch.branch_id in replacements:
                retired_hlgs.append(branch.high_level_goal)
            next_goals.extend(
                replacements.get(branch.branch_id, [branch.high_level_goal])
            )
        changed_branch_ids = (
            removed_branch_ids | set(replacements) | set(llg_regenerations)
        )
        next_llgs = [
            llg
            for branch in branches
            if branch.branch_id not in changed_branch_ids
            for llg in branch.low_level_goals
        ]

        # The branch evaluator can request LLG repair, but the global HLG
        # coverage check is deliberately a separate, single project-wide call.
        if llg_regenerations:
            for parent_key, item in llg_regenerations.items():
                parent, guidance, existing = item
                normalized_existing = [
                    llg
                    if llg.high_level_associated == parent
                    else llg.model_copy(
                        update={"high_level_associated": parent}
                    )
                    for llg in existing
                ]
                try:
                    generated_llgs = _regenerate_llgs_with_top_down(
                        LowLevelGoalRegenerationRequest(
                            high_level_goals=HighLevelGoals(goals=[parent]),
                            guidance_by_parent_name={parent_key: guidance},
                            existing_low_level_goals=LowLevelGoals(
                                low_level_goals=normalized_existing
                            ),
                        ),
                        mode,
                        evaluator_ablation,
                    )
                except Exception as exc:
                    next_llgs.extend(normalized_existing)
                    iteration_warnings.append(
                        f"{parent.name}: LLG_REGENERATION_FAILED_OLD_STATE_KEPT "
                        f"({type(exc).__name__}: {exc})"
                    )
                    continue

                generated_for_parent = []
                for llg in generated_llgs.low_level_goals:
                    if llg.high_level_associated != parent:
                        iteration_warnings.append(
                            f"LLG '{llg.name}' did not preserve the exact "
                            f"generated parent '{parent.name}'; skipped."
                        )
                        continue
                    generated_for_parent.append(llg)
                accepted, rejection = _accept_regenerated_llgs(
                    existing=normalized_existing,
                    generated=generated_for_parent,
                )
                next_llgs.extend(accepted)
                if rejection:
                    iteration_warnings.append(
                        f"{parent.name}: {rejection}"
                    )

        current_hlgs = HighLevelGoals(goals=next_goals)
        current_llgs = LowLevelGoals(low_level_goals=next_llgs)

        if enable_discovery:
            known_actors = _merge_actors(
                known_actors, _actors_from_goals(current_hlgs)
            )
        actors = known_actors

        if _should_add_missing_hlg(
            missing_hlg_evaluation=missing_hlg_evaluation,
            replacements=replacements,
            removed_branch_ids=removed_branch_ids,
        ):
            for request_index, request in enumerate(
                missing_hlg_evaluation.missing_goal_requests,
                start=1,
            ):
                actor = _actor_for_request(request.actor, actors)
                if actor is None:
                    iteration_warnings.append(
                        f"{request.actor}: UNKNOWN_ACTOR_REQUEST_IGNORED"
                    )
                    continue
                generated = _generate_hlgs_with_top_down(
                    _generation_request(
                        f"iteration_{iteration}_missing_{request_index}",
                        HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL,
                        request.generation_project_description,
                        actor,
                        missing_hlg_evaluation.rationale,
                    ),
                    mode,
                    evaluator_ablation,
                )
                new_goals = []
                for goal in generated.goals[:1]:
                    if not goal.name.strip():
                        continue
                    new_goals.append(goal)
                    additions.append(goal)

            if additions:
                current_hlgs = HighLevelGoals(
                    goals=[*current_hlgs.goals, *additions]
                )
                # Generate each decomposition independently so equal HLG values
                # never collapse into one dictionary entry.
                for addition_index, parent in enumerate(additions, start=1):
                    operation_id = f"addition_{addition_index:03d}"
                    generated_llgs = _regenerate_llgs_with_top_down(
                        LowLevelGoalRegenerationRequest(
                            high_level_goals=HighLevelGoals(goals=[parent]),
                            guidance_by_parent_name={
                                operation_id: missing_hlg_evaluation.rationale
                            },
                        ),
                        mode,
                        evaluator_ablation,
                    )
                    associated = [
                        llg for llg in generated_llgs.low_level_goals
                        if llg.high_level_associated == parent
                    ]
                    for llg in generated_llgs.low_level_goals:
                        if llg.high_level_associated != parent:
                            iteration_warnings.append(
                                f"LLG '{llg.name}' did not preserve an exact "
                                "generated parent HLG; skipped."
                            )
                    accepted, rejection = _accept_regenerated_llgs(
                        existing=[],
                        generated=associated,
                    )
                    current_llgs.low_level_goals.extend(accepted)
                    if rejection:
                        iteration_warnings.append(
                            f"{parent.name}: {rejection}"
                        )

        # An HLG-level change alters the project-wide responsibility map. Audit
        # every branch again on the next iteration instead of trusting a KEEP
        # decision made against the previous HLG set.
        if replacements or removed_branch_ids or additions:
            confirmed_branch_ids.clear()

        critic_conversation.remember_state(
            _critic_state_summary(
                iteration=iteration,
                previous_hlgs=iteration_start_hlgs,
                previous_llgs=iteration_start_llgs,
                current_hlgs=current_hlgs,
                current_llgs=current_llgs,
            )
        )

        # Only real confirmations permit convergence. An evaluator failure is
        # transient and is retried on the next iteration; limit-related
        # inconclusive decisions remain terminal warnings.
        permanent_request_size_markers = (
            "request too large",
            "input tokens per minute",
            "maximum context length",
            "context_length_exceeded",
        )
        retryable_branch_inconclusive = any(
            evaluation.final_decision
            == GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
            and "evaluation failed:" in evaluation.rationale.casefold()
            and not any(
                marker in evaluation.rationale.casefold()
                for marker in permanent_request_size_markers
            )
            for evaluation in evaluations.values()
        )
        permanent_size_failures = [
            evaluation.branch_id
            for evaluation in evaluations.values()
            if evaluation.final_decision
            == GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
            and any(
                marker in evaluation.rationale.casefold()
                for marker in permanent_request_size_markers
            )
        ]
        if permanent_size_failures:
            iteration_warnings.append(
                "NON_RETRYABLE_EVALUATOR_REQUEST_TOO_LARGE: "
                + ", ".join(permanent_size_failures)
            )
        retryable_coverage_inconclusive = (
            missing_hlg_evaluation is not None
            and missing_hlg_evaluation.decision
            == MissingHighLevelGoalDecision.INCONCLUSIVE
        )
        current_branches = _branches_from_embedded_parents(
            current_hlgs, current_llgs
        )
        empty_branches = [
            branch for branch in current_branches
            if not branch.low_level_goals
        ]
        if empty_branches:
            iteration_warnings.append(
                "HLGS_WITHOUT_LOW_LEVEL_GOALS: "
                + ", ".join(
                    f"{branch.branch_id} ({branch.high_level_goal.actor.name}::"
                    f"{branch.high_level_goal.name})"
                    for branch in empty_branches
                )
            )
        # Convergence is a property of the complete hierarchy, not merely of
        # the subset evaluated in this iteration.  Iteration 1 always contains
        # every branch because the confirmed set starts empty.  Corrections are
        # applied only after that full sequential pass; any applied structural
        # change forces another iteration.
        all_confirmed = validation_pass_complete and not (
            replacements or removed_branch_ids or additions
        )
        discovery_complete = (
            missing_hlg_evaluation is None
            or missing_hlg_evaluation.decision
            == MissingHighLevelGoalDecision.NO_MISSING
        )

        trace = GlobalGoalCycleIteration(
            iteration=iteration,
            reconstructions=reconstructions,
            evaluations=evaluations,
            high_level_goals=current_hlgs,
            low_level_goals=current_llgs,
            missing_hlg_evaluation=missing_hlg_evaluation,
            missing_hlg_decision=(
                None
                if missing_hlg_evaluation is None
                else (
                    GlobalGoalEvaluationDecision.MISSING_HIGH_LEVEL_GOALS_FOUND
                    if missing_hlg_evaluation.decision
                    == MissingHighLevelGoalDecision.FOUND
                    else (
                        GlobalGoalEvaluationDecision.NO_MISSING_HIGH_LEVEL_GOALS
                        if missing_hlg_evaluation.decision
                        == MissingHighLevelGoalDecision.NO_MISSING
                        else GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
                    )
                )
            ),
            llg_regeneration_counts=dict(llg_regeneration_counts),
            warnings=list(iteration_warnings),
        )
        traces.append(trace)
        _save_iteration(evaluation_output_directory, trace)
        warnings.extend(iteration_warnings)

        if all_confirmed and discovery_complete:
            return GlobalGoalCycleResult(
                converged=True,
                stop_reason=GlobalGoalCycleStopReason.ALL_BRANCHES_CONFIRMED,
                completed_iterations=iteration,
                final_high_level_goals=current_hlgs,
                final_low_level_goals=current_llgs,
                iterations=traces,
                llg_regeneration_counts=dict(llg_regeneration_counts),
                warnings=warnings,
            )

        actions_remain = bool(
            (
                missing_hlg_evaluation is not None
                and missing_hlg_evaluation.decision
                == MissingHighLevelGoalDecision.FOUND
            )
            or retryable_branch_inconclusive
            or retryable_coverage_inconclusive
            or any(
                evaluation.final_decision
                in {
                    GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS,
                    GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL,
                    GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL,
                }
                for evaluation in evaluations.values()
            )
        )
        if discovery_complete and not actions_remain and iteration_warnings:
            return GlobalGoalCycleResult(
                converged=False,
                stop_reason=GlobalGoalCycleStopReason.NO_ACTIONS_REMAIN_WITH_WARNINGS,
                completed_iterations=iteration,
                final_high_level_goals=current_hlgs,
                final_low_level_goals=current_llgs,
                iterations=traces,
                llg_regeneration_counts=dict(llg_regeneration_counts),
                warnings=warnings,
            )

    return GlobalGoalCycleResult(
        converged=False,
        stop_reason=GlobalGoalCycleStopReason.MAX_ITERATIONS_REACHED,
        completed_iterations=max_iterations,
        final_high_level_goals=current_hlgs,
        final_low_level_goals=current_llgs,
        iterations=traces,
        llg_regeneration_counts=dict(llg_regeneration_counts),
        warnings=warnings,
    )
