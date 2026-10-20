"""Bottom-up validation and repair with mandatory final HLG completeness analysis."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.bottom_up.global_goal_evaluator import (
    BOTTOM_UP_EVALUATOR_MODELS,
    evaluate_all_branches,
    evaluate_missing_high_level_goals,
    evaluation_status,
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
    HighLevelGoalEvaluation,
    LowLevelGoalEvaluation,
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
    RepairCandidateRejected,
    generate_bottom_up_response_with_reflection,
)


DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS = 5
MAX_LLG_REGENERATIONS_PER_BRANCH = 2
MAX_HLG_REWRITES_PER_BRANCH = 3
# Keep each iteration conservative: outstanding repairs are reconsidered on
# the next pass against the newly updated hierarchy.
MAX_HLG_REWRITES_PER_ITERATION = 2
MAX_HLG_REMOVALS_PER_ITERATION: int | None = 2
MAX_DUPLICATE_HLG_REMOVALS_PER_ITERATION: int | None = None
MAX_LLG_REGENERATIONS_PER_ITERATION = 2
MAX_REPLACEMENT_HLGS_PER_REWRITE = 1
MAX_REJECTED_REPAIR_CYCLES = 2
MAX_NEW_HLGS_PER_ITERATION = 3
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
    *, original_high_level_goal: HighLevelGoal | None = None,
    existing_high_level_goals: HighLevelGoals | None = None,
    existing_low_level_goals: LowLevelGoals | None = None,
    generator_guidance: str | None = None,
    consolidated_high_level_goals: HighLevelGoals | None = None,
) -> HighLevelGoalGenerationRequest:
    return HighLevelGoalGenerationRequest(
        request_id=request_id,
        action=action,
        generator_input=HighLevelGoalGeneratorInput(
            project_description=description,
            actors=Actors(actors=[actor]),
        ),
        rationale=rationale,
        generator_guidance=generator_guidance,
        target_branch_id=target_branch_id,
        original_high_level_goal=original_high_level_goal,
        consolidated_high_level_goals=consolidated_high_level_goals or HighLevelGoals(goals=[]),
        existing_high_level_goals=existing_high_level_goals or HighLevelGoals(goals=[]),
        existing_low_level_goals=existing_low_level_goals or LowLevelGoals(low_level_goals=[]),
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
    """Apply a positive completeness verdict only to a stable validated state."""
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
        "WHY THIS HLG GENERATION IS REQUESTED (rationale; verify against source):\n" + request.rationale
        + "\n\nREQUESTED ACTION:\n" + request.action.value
        + "\n\nOPERATIONAL GUIDANCE:\n" + (request.generator_guidance or "Follow the requested scope below.")
        + "\n\nSOURCE AND REQUESTED SCOPE:\n" + request.generator_input.project_description
        + "\n\nOriginal target HLG (null for a new outcome):\n"
        + (request.original_high_level_goal.model_dump_json() if request.original_high_level_goal else "null")
        + "\n\nOriginal HLGs to consolidate into ONE outcome (empty for ordinary rewrite):\n"
        + request.consolidated_high_level_goals.model_dump_json()
        + "\n\nExisting children to preserve where supported:\n" + request.existing_low_level_goals.model_dump_json()
        + "\n\nCurrent HLG boundaries (do not duplicate their outcomes):\n"
        + request.existing_high_level_goals.model_dump_json()
        + "\nCorrect only the requested outcome. Source evidence takes precedence over feedback. "
        "Preserve supported capabilities; do not add conventional features or rewrite for style."
        " Prefer a coherent stakeholder outcome at HLG level. Keep individual "
        "operations, field checks, screens and workflow steps in the LLG "
        "decomposition. Generalize only within the documented scope, preserving "
        "required capabilities and clear actor responsibility; avoid vague "
        "umbrella goals and do not merge independent outcomes."
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
            "Evaluate only the supplied actor and the coherent functional "
            "outcome described by this replacement/consolidation request. Preserve "
            "the supported capabilities of every supplied consolidation member. Evaluate the current "
            "candidate against the source. Do not treat an old diagnosis as proof "
            "of a remaining defect or require an atomic LLG list inside the HLG."
            if request.action != HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL
            else "Evaluate only the supplied actor and the merged completeness "
            "requests. Allow multiple distinct HLGs when multiple independent "
            "outcomes are requested, within the explicit remaining batch budget. "
            "Judge the selected requested outcomes only; other missing outcomes "
            "may be deferred to a later iteration. Do not require all requests "
            "to fit this batch or every responsibility in the full project. "
            "Judge HLG stakeholder outcomes; do not require an LLG decomposition."
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


def _critic_folder(model: str) -> str:
    folders = dict(zip(BOTTOM_UP_EVALUATOR_MODELS, ("ministral", "granite", "gemma")))
    return folders.get(model, "critic_" + hashlib.sha256(model.encode()).hexdigest()[:12])


def _save_critic_vote(directory, iteration, branch, task, record):
    """Persist an individual vote immediately, before loading the next voter."""
    if directory is None:
        return
    folder = Path(directory) / _critic_folder(record.model) / f"iteration_{iteration:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{branch.branch_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if task != "hlg" and path.exists() else {}
    payload.update(iteration=iteration, branch_id=branch.branch_id, model=record.model,
                   high_level_goal=branch.high_level_goal.model_dump(mode="json"),
                   low_level_goals=[g.model_dump(mode="json") for g in branch.low_level_goals],
                   checkpoint_complete=False)
    payload[task] = record.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_iteration(directory: str | Path | None, trace: GlobalGoalCycleIteration,
                    branches: list[GoalBranch] | None = None):
    if directory is None:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"iteration_{trace.iteration:03d}.json").write_text(
        trace.model_dump_json(indent=2), encoding="utf-8"
    )
    for branch in branches or []:
        evaluation = trace.evaluations.get(branch.branch_id)
        for model in BOTTOM_UP_EVALUATOR_MODELS:
            folder = path / _critic_folder(model) / f"iteration_{trace.iteration:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{branch.branch_id}.json"
            payload = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
            payload.update(iteration=trace.iteration, branch_id=branch.branch_id, model=model,
                           high_level_goal=branch.high_level_goal.model_dump(mode="json"),
                           low_level_goals=[g.model_dump(mode="json") for g in branch.low_level_goals],
                           checkpoint_complete=trace.checkpoint_complete)
            if evaluation is not None:
                for task in ("hlg", "reconstruction", "llg"):
                    vote = next((v for v in evaluation.critic_votes.get(task, []) if v.model == model), None)
                    if vote is not None and task not in payload:
                        payload[task] = vote.model_dump(mode="json")
                payload["branch_status"] = "EVALUATED"
                payload["aggregated_branch_decision"] = evaluation.final_decision.value
                payload["llg_status"] = "EVALUATED" if payload.get("llg") else (
                    "SKIPPED" if trace.checkpoint_complete else "PENDING")
            else:
                payload["branch_status"] = "PENDING"
                payload["llg_status"] = "PENDING"
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if trace.missing_hlg_evaluation is not None:
        for vote in trace.missing_hlg_evaluation.evaluator_votes:
            folder = path / _critic_folder(vote.model) / f"iteration_{trace.iteration:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "completeness.json").write_text(json.dumps(dict(
                iteration=trace.iteration, task="completeness", model=vote.model,
                vote=vote.model_dump(mode="json"), checkpoint_complete=trace.checkpoint_complete,
            ), ensure_ascii=False, indent=2), encoding="utf-8")


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
    confirmed_branch_ids: set[str] | None = None,
) -> tuple[set[str], list[str]]:
    """Apply critic removals by technical branch ID and protect merge targets."""
    selected: set[str] = set()
    protected: set[str] = set()
    warnings: list[str] = []
    ordinary_removals = 0
    duplicate_removals = 0

    for branch in removal_branches:
        branch_key = branch.branch_id
        hlg_evaluation = evaluations[branch.branch_id].high_level_evaluation
        redundant = hlg_evaluation.removal_basis in {
            HighLevelGoalRemovalBasis.FULLY_REDUNDANT,
            HighLevelGoalRemovalBasis.OVER_SPECIFIC,
        }
        limit = MAX_DUPLICATE_HLG_REMOVALS_PER_ITERATION if redundant else MAX_HLG_REMOVALS_PER_ITERATION
        count = duplicate_removals if redundant else ordinary_removals
        if limit is not None and count >= limit:
            continue
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
        duplicate_removals += int(redundant)
        ordinary_removals += int(not redundant)

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
    confirmed_branch_ids: set[str] | None = None,
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
def _select_llg_repair_batch(branches, evaluations, counts, warnings):
    """Exclude exhausted branches first, then fairly schedule eligible repairs."""
    eligible = []
    for branch in branches:
        count = counts.get(branch.branch_id, 0)
        if count >= MAX_LLG_REGENERATIONS_PER_BRANCH:
            evaluations[branch.branch_id] = _with_decision(evaluations[branch.branch_id],
                GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
                "LLG regeneration budget exhausted; unresolved critic verdict retained.")
            warnings.append(f"{branch.branch_id}: LLG_REGENERATION_LIMIT_REACHED")
        else:
            eligible.append(branch)
    eligible.sort(key=lambda branch: counts.get(branch.branch_id, 0))
    selected = eligible[:MAX_LLG_REGENERATIONS_PER_ITERATION]
    if len(eligible) > len(selected):
        warnings.append(f"LLG_REGENERATION_BATCH_LIMIT ({len(eligible)} eligible, {len(selected)} scheduled)")
    return selected


def _select_hlg_rewrite_jobs(branches, evaluations, frozen_ids, pending, counts, warnings):
    """Schedule disjoint atomic rewrites from the current pass's decisions."""
    by_id = {b.branch_id: b for b in branches}
    eligible, reserved = [], set()
    for branch in branches:
        verdict = evaluations[branch.branch_id]
        if verdict.final_decision != GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL:
            continue
        member_ids = [branch.branch_id, *verdict.high_level_evaluation.consolidate_with_branch_ids]
        if (len(set(member_ids)) != len(member_ids)
                or any(identifier not in by_id for identifier in member_ids)
                or any(_key(by_id[identifier].high_level_goal.actor.name)
                       != _key(branch.high_level_goal.actor.name) for identifier in member_ids)
                or any(identifier in pending and identifier != branch.branch_id for identifier in member_ids)
                or (len(member_ids) > 1 and any(counts.get(identifier, 0) >= MAX_HLG_REWRITES_PER_BRANCH
                                               for identifier in member_ids))):
            warnings.append(f"{branch.branch_id}: CONSOLIDATION_INELIGIBLE_MEMBERS")
            continue
        members = [by_id[identifier] for identifier in member_ids]
        eligible.append((branch, members))
        reserved.update(member_ids)
    selected, claimed = [], set()
    for branch, members in eligible:
        member_ids = {member.branch_id for member in members}
        if member_ids & claimed:
            warnings.append(f"{branch.branch_id}: HLG_REWRITE_OVERLAP_DEFERRED")
            continue
        if len(selected) >= MAX_HLG_REWRITES_PER_ITERATION:
            warnings.append(f"{branch.branch_id}: HLG_REWRITE_BATCH_LIMIT")
            continue
        selected.append((branch, members))
        claimed.update(member_ids)
    return selected, reserved


def run_global_goal_cycle(
    *,
    project_description: str,
    initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals,
    mode: ShotPromptingMode = ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation: bool = False,
    evaluation_output_directory: str | Path | None = None,
    max_iterations: int = DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS,
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
    _validate_embedded_parents(initial_high_level_goals, initial_low_level_goals)
    current_hlgs = initial_high_level_goals.model_copy(deep=True)
    current_llgs = initial_low_level_goals.model_copy(deep=True)
    critic_conversation = None  # No conversational history between judgments.
    llg_regeneration_counts: dict[str, int] = {}
    pending_llg_audit_ids: set[str] = set()
    confirmed_branch_ids: set[str] = set()
    hlg_rewrite_counts: dict[str, int] = {}
    traces: list[GlobalGoalCycleIteration] = []
    warnings: list[str] = []
    known_actors = _actors_from_goals(initial_high_level_goals)
    repair_rejection_counts: dict[str, int] = {}
    forced_confirmed_branch_ids: set[str] = set()
    pending_repairs: dict[str, GlobalGoalEvaluationResult] = {}
    completeness_rejection_counts: dict[str, int] = {}
    pending_completeness = {}

    def reject_repair(branch_id, verdict, exc):
        count = repair_rejection_counts.get(branch_id, 0) + 1
        repair_rejection_counts[branch_id] = count
        if count >= MAX_REJECTED_REPAIR_CYCLES:
            forced_confirmed_branch_ids.add(branch_id)
            confirmed_branch_ids.add(branch_id)
            pending_repairs.pop(branch_id, None)
            marker = "CONFIRM_BRANCH_AFTER_TWO_REJECTED_REPAIRS"
            decision = GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        else:
            pending_repairs[branch_id] = verdict
            marker = "REPAIR_REJECTED_RETRY_NEXT_ITERATION"
            decision = GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
        evaluations[branch_id] = _with_decision(verdict, decision, f"{marker}: {exc}")
        iteration_warnings.append(f"{branch_id}: {marker}: {exc}")

    for iteration in range(1, max_iterations + 1):
        iteration_start_hlgs = current_hlgs.model_copy(deep=True)
        iteration_start_llgs = current_llgs.model_copy(deep=True)
        iteration_warnings: list[str] = []
        print(
            f"[bottom-up] iteration {iteration}/{max_iterations} starting",
            flush=True,
        )
        branches = _branches_from_embedded_parents(current_hlgs, current_llgs)

        # A confirmation describes one pass; every surviving branch is re-evaluated.
        active_branches = list(branches)
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
        critic1_global_votes = []

        def save_global_vote(record):
            critic1_global_votes.append(record)
            if evaluation_output_directory is not None:
                folder = (Path(evaluation_output_directory) / _critic_folder(record.model)
                          / f"iteration_{iteration:03d}")
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "critic1_global.json").write_text(
                    json.dumps({"iteration": iteration, **record.model_dump(mode="json")},
                               ensure_ascii=False, indent=2), encoding="utf-8")

        def save_branch_checkpoint(
            completed_count: int,
            total_count: int,
            partial_evaluations: dict[str, GlobalGoalEvaluationResult],
        ) -> None:
            """Persist each completed branch decision before the next call."""
            checkpoint = GlobalGoalCycleIteration(
                iteration=iteration,
                reconstructions=reconstructions,
                evaluations=partial_evaluations,
                critic1_global_votes=list(critic1_global_votes),
                high_level_goals=iteration_start_hlgs,
                low_level_goals=iteration_start_llgs,
                llg_regeneration_counts=dict(llg_regeneration_counts),
                pending_llg_audit_ids=sorted(pending_llg_audit_ids),
                hlg_rewrite_counts=dict(hlg_rewrite_counts),
                confirmed_branch_ids=sorted(confirmed_branch_ids),
                repair_rejection_counts=dict(repair_rejection_counts),
                forced_confirmed_branch_ids=sorted(forced_confirmed_branch_ids),
                completeness_rejection_counts=dict(completeness_rejection_counts),
                warnings=["IN_PROGRESS_BRANCH_CHECKPOINT"],
                checkpoint_complete=False,
                completed_branch_count=completed_count,
                total_branch_count=total_count,
            )
            _save_iteration(evaluation_output_directory, checkpoint, branches)

        def save_critic_vote(branch, task, records):
            for record in records:
                _save_critic_vote(evaluation_output_directory, iteration, branch, task, record)

        known_actors = _merge_actors(known_actors, _actors_from_goals(current_hlgs))
        raw_evaluations = evaluate_all_branches(
            project_description,
            active_branches,
            reconstructions,
            current_hlgs,
            critic_conversation,
            branches,
            on_evaluation=save_branch_checkpoint,
            on_vote=save_critic_vote,
            on_global_vote=save_global_vote,
            blocked_hlg_repair_ids=set(forced_confirmed_branch_ids),
            hlg_rewrite_counts=dict(hlg_rewrite_counts),
            max_hlg_rewrites=MAX_HLG_REWRITES_PER_BRANCH,
            confirmed_branch_ids=set(confirmed_branch_ids),
            required_llg_audit_ids=set(pending_llg_audit_ids),
        )
        evaluations = {
            branch_id: _coerce_model(
                evaluation,
                GlobalGoalEvaluationResult,
                context=f"evaluation for {branch_id}",
            )
            for branch_id, evaluation in raw_evaluations.items()
        }
        # Fresh judgments supersede old confirmations and pending diagnoses.
        # A deferred or inconclusive audit must not consume the obligation.
        pending_llg_audit_ids.difference_update(
            branch_id for branch_id, verdict in evaluations.items()
            if verdict.low_level_evaluation is not None
        )
        confirmed_branch_ids = set()
        for branch in active_branches:
            verdict = evaluations[branch.branch_id]
            if verdict.final_decision in {
                GlobalGoalEvaluationDecision.CONFIRM_BRANCH,
                GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL,
            }:
                pending_repairs.pop(branch.branch_id, None)
            elif branch.branch_id in pending_repairs:
                pending_repairs[branch.branch_id] = verdict
            if (branch.branch_id in forced_confirmed_branch_ids
                    and verdict.final_decision in {
                        GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL,
                        GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS,
                    }):
                evaluations[branch.branch_id] = _with_decision(
                    verdict, GlobalGoalEvaluationDecision.CONFIRM_BRANCH,
                    "REPAIR_REJECTION_LIMIT_RETAINED_AFTER_FRESH_EVALUATION")
                pending_repairs.pop(branch.branch_id, None)
        rewrite_jobs, rewrite_reserved_ids = _select_hlg_rewrite_jobs(
            active_branches, evaluations, confirmed_branch_ids, pending_repairs,
            hlg_rewrite_counts, iteration_warnings)
        # Apply voted structural removals before generating rewrites of this state.
        if any(
            evaluation.final_decision == GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
            for evaluation in evaluations.values()
        ):
            rewrite_jobs, rewrite_reserved_ids = [], set()
        # Audit the last generated decomposition before declaring its budget
        # exhausted. Exhaustion unlocks coverage analysis, not approval.
        exhausted_branches = [
            branch for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
            and branch.branch_id not in pending_repairs
            and llg_regeneration_counts.get(branch.branch_id, 0)
            >= MAX_LLG_REGENERATIONS_PER_BRANCH
        ]
        _select_llg_repair_batch(exhausted_branches, evaluations,
                                 llg_regeneration_counts, iteration_warnings)
        confirmed_branch_ids.update(
            branch.branch_id
            for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            and branch.branch_id not in rewrite_reserved_ids
        )
        exhausted_branch_ids = {
            branch.branch_id for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED
        }
        validation_pass_complete = all(
            branch.branch_id in confirmed_branch_ids | exhausted_branch_ids
            for branch in branches
        )

        missing_hlg_evaluation = None
        if validation_pass_complete:
            if exhausted_branch_ids:
                iteration_warnings.append(
                    "COMPLETENESS_WITH_UNRESOLVED_BRANCHES: "
                    + ", ".join(sorted(exhausted_branch_ids)))
            known_actors = _merge_actors(
                known_actors, _actors_from_goals(current_hlgs)
            )
            missing_hlg_evaluation = _coerce_model(
                evaluate_missing_high_level_goals(
                    project_description, current_hlgs, known_actors),
                MissingHighLevelGoalEvaluation,
                context="missing High-Level Goal evaluation",
            )
            if pending_completeness:
                requests_by_actor = {
                    _key(r.actor): r for r in missing_hlg_evaluation.missing_goal_requests
                }
                requests_by_actor.update({key: item[0] for key, item in pending_completeness.items()})
                missing_hlg_evaluation = missing_hlg_evaluation.model_copy(update={
                    "decision": MissingHighLevelGoalDecision.FOUND,
                    "quality_score": 1,
                    "missing_goal_requests": list(requests_by_actor.values()),
                    "rationale": missing_hlg_evaluation.rationale + "\nRetry previously rejected completeness: "
                        + "\n".join(item[1] for item in pending_completeness.values()),
                })
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
        accepted_llg_repair_ids: set[str] = set()
        replacements: dict[str, list[HighLevelGoal]] = {}
        consolidation_sources: dict[str, list[GoalBranch]] = {}
        additions: list[HighLevelGoal] = []
        addition_guidance: dict[str, str] = {}
        # Every verdict is collected against the same immutable iteration
        # snapshot. Apply all independent actions of the highest safe priority
        # together: a small batch of rewrites, otherwise safe removals,
        # otherwise a small batch of LLG repairs. Deferred requests are
        # reconsidered on the next pass against the updated hierarchy.
        for branch, members in rewrite_jobs:
            evaluation = evaluations[branch.branch_id]
            original_key = branch.branch_id
            request = evaluation.replacement_request
            member_ids = {member.branch_id for member in members}
            existing_children = [child for member in members for child in member.low_level_goals]
            consolidation_goals = HighLevelGoals(goals=[member.high_level_goal for member in members])
            try:
                generated = _generate_hlgs_with_top_down(
                    _generation_request(
                        f"{branch.branch_id}_rewrite",
                        HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL,
                        request.generation_project_description,
                        request.actor,
                        request.rationale,
                        branch.branch_id,
                        original_high_level_goal=branch.high_level_goal,
                        existing_high_level_goals=HighLevelGoals(goals=[
                            b.high_level_goal for b in branches if b.branch_id not in member_ids
                        ]),
                        existing_low_level_goals=LowLevelGoals(low_level_goals=existing_children),
                        consolidated_high_level_goals=(consolidation_goals if len(members) > 1 else None),
                        generator_guidance=evaluation.high_level_evaluation.rewriting_focus,
                    ),
                    mode,
                    evaluator_ablation,
                )
            except RepairCandidateRejected as exc:
                reject_repair(branch.branch_id, evaluation, exc)
                continue
            replacement_goals = []
            # A focused rewrite request yields one candidate at most;
            # accepting every candidate here caused HLG proliferation.
            for goal in generated.goals[:MAX_REPLACEMENT_HLGS_PER_REWRITE]:
                if not goal.name.strip():
                    continue
                if any(b.branch_id in confirmed_branch_ids and b.high_level_goal == goal
                       for b in branches):
                    iteration_warnings.append(
                        f"{branch.branch_id}: HLG_REWRITE_REJECTED_FROZEN_PARENT_COLLISION")
                    continue
                replacement_goals.append(goal)
                llg_regenerations[f"{branch.branch_id}:replacement"] = (
                    goal,
                    request.rationale + "\nGenerate the full decomposition for the rewritten parent. "
                    + "Preserve source-supported capabilities. Original rewrite instructions: "
                    + (evaluation.high_level_evaluation.rewriting_focus or request.generation_project_description),
                    existing_children,
                )
            if replacement_goals:
                replacements[original_key] = replacement_goals
                consolidation_sources[original_key] = [member for member in members
                                                       if member.branch_id != original_key]
                removed_branch_ids.update(member.branch_id for member in consolidation_sources[original_key])
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
                confirmed_branch_ids,
            )
            iteration_warnings.extend(removal_warnings)
            _schedule_redundant_hlg_merges(
                removed_branch_ids,
                branches,
                evaluations,
                llg_regenerations,
                confirmed_branch_ids,
            )

        if not replacements and not removed_branch_ids:
            regeneration_branches = [
                branch for branch in active_branches
                if evaluations[branch.branch_id].final_decision
                == GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
            ]
            regeneration_branches = _select_llg_repair_batch(
                regeneration_branches, evaluations,
                {key: (0 if key in pending_repairs else count)
                 for key, count in llg_regeneration_counts.items()}, iteration_warnings)
            for branch in regeneration_branches:
                evaluation = evaluations[branch.branch_id]
                stable_key = branch.branch_id
                count = llg_regeneration_counts.get(stable_key, 0)
                llg_regeneration_counts[stable_key] = count + 1
                llg_regenerations[stable_key] = (
                    branch.high_level_goal,
                    evaluation.low_level_evaluation.rationale
                    + "\n\nOperational repair request:\n"
                    + (evaluation.low_level_evaluation.regeneration_feedback or evaluation.rationale),
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
        inconclusive_statuses = {
            evaluation_status(evaluation) for evaluation in evaluations.values()
            if evaluation.final_decision == GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
        }
        if inconclusive_statuses - {"DEFERRED_STRUCTURAL_ACTIONS"}:
            iteration_warnings.append("UNRESOLVED_EVALUATION_INCONCLUSIVE")
            iteration_warnings.extend(sorted(
                "UNRESOLVED_" + status for status in inconclusive_statuses
                if status != "DEFERRED_STRUCTURAL_ACTIONS"))

        # Apply branch-level validation and repair decisions. Mandatory completeness analysis
        # is evaluated after all branches are confirmed or audited at their limit.
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
        # A consolidation is committed only with its accepted full child set.
        # Several removed sources can depend on the same target transaction.
        merge_sources = {}
        for source in branches:
            if source.branch_id not in removed_branch_ids:
                continue
            verdict = evaluations[source.branch_id].high_level_evaluation
            if (verdict.removal_basis == HighLevelGoalRemovalBasis.FULLY_REDUNDANT
                    and verdict.covered_by_branch_id in llg_regenerations):
                merge_sources.setdefault(verdict.covered_by_branch_id, []).append(source)
        for anchor_id, sources in consolidation_sources.items():
            merge_sources[f"{anchor_id}:replacement"] = sources

        def restore_repair(parent_key, semantic_rejection=None):
            origin_id = parent_key.removesuffix(":replacement")
            origin = next(b for b in branches if b.branch_id == origin_id)
            next_llgs.extend(origin.low_level_goals)
            replacements.pop(origin_id, None)
            sources = merge_sources.get(parent_key, [])
            for source in sources:
                removed_branch_ids.discard(source.branch_id)
                next_llgs.extend(source.low_level_goals)
            if sources:
                iteration_warnings.append(f"{parent_key}: CONSOLIDATION_ROLLED_BACK")
            if semantic_rejection is not None:
                # Retry the original removal request, not a target-only repair
                # that would forget the source decomposition.
                retry_branches = [origin] if parent_key.endswith(":replacement") else sources or [origin]
                for branch in retry_branches:
                    reject_repair(branch.branch_id, evaluations[branch.branch_id], semantic_rejection)

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
                            rationale=guidance,
                            project_description=project_description,
                            sibling_high_level_goals=HighLevelGoals(goals=[
                                b.high_level_goal for b in branches
                                if b.high_level_goal != parent
                                and b.branch_id != parent_key.removesuffix(":replacement")
                                and b.branch_id not in removed_branch_ids
                            ]),
                            guidance_by_parent_name={parent_key: guidance},
                            existing_low_level_goals=LowLevelGoals(
                                low_level_goals=normalized_existing
                            ),
                        ),
                        mode,
                        evaluator_ablation,
                    )
                except RepairCandidateRejected as exc:
                    restore_repair(parent_key, semantic_rejection=exc)
                    continue
                except Exception as exc:
                    restore_repair(parent_key)
                    iteration_warnings.append(
                        f"{parent.name}: LLG_REGENERATION_FAILED_OLD_STATE_KEPT "
                        f"({type(exc).__name__}: {exc})"
                    )
                    continue

                generated_for_parent = []
                invalid_parent = False
                for llg in generated_llgs.low_level_goals:
                    if llg.high_level_associated != parent:
                        invalid_parent = True
                        iteration_warnings.append(
                            f"LLG '{llg.name}' did not preserve the exact "
                            f"generated parent '{parent.name}'; entire replacement rejected."
                        )
                        continue
                    generated_for_parent.append(llg)
                accepted, rejection = _accept_regenerated_llgs(
                    existing=normalized_existing,
                    generated=generated_for_parent,
                )
                if invalid_parent:
                    rejection = "LLG_REGENERATION_REJECTED_PARENT_MISMATCH"
                if rejection:
                    restore_repair(parent_key)
                    iteration_warnings.append(
                        f"{parent.name}: {rejection}"
                    )
                else:
                    next_llgs.extend(accepted)
                    accepted_llg_repair_ids.add(parent_key.removesuffix(":replacement"))
                    pending_repairs.pop(parent_key.removesuffix(":replacement"), None)
                    for source in merge_sources.get(parent_key, []):
                        pending_repairs.pop(source.branch_id, None)

        next_goals: list[HighLevelGoal] = []
        next_origins: list[GoalBranch] = []
        for branch in branches:
            if branch.branch_id in removed_branch_ids:
                continue
            descendants = replacements.get(branch.branch_id, [branch.high_level_goal])
            next_goals.extend(descendants)
            next_origins.extend([branch] * len(descendants))
        current_hlgs = HighLevelGoals(goals=next_goals)
        current_llgs = LowLevelGoals(low_level_goals=next_llgs)
        # Carry budgets and current-pass confirmations by lineage, including ID changes
        # caused by a rewrite or by the occurrence index of an identical HLG.
        next_branches = _branches_from_embedded_parents(current_hlgs, current_llgs)
        next_confirmed_ids: set[str] = set()
        next_forced_ids: set[str] = set()
        next_llg_audit_ids: set[str] = set()
        previous_rewrite_counts = dict(hlg_rewrite_counts)
        for origin, descendant in zip(next_origins, next_branches):
            applied_rewrite = origin.branch_id in replacements
            if (origin.branch_id in accepted_llg_repair_ids
                    or (origin.branch_id in pending_llg_audit_ids and not applied_rewrite)):
                next_llg_audit_ids.add(descendant.branch_id)
            hlg_rewrite_counts[descendant.branch_id] = (
                max(previous_rewrite_counts.get(member_id, 0) for member_id in
                    [origin.branch_id, *[b.branch_id for b in consolidation_sources.get(origin.branch_id, [])]])
                + 1 if applied_rewrite else previous_rewrite_counts.get(origin.branch_id, 0)
            )
            if (origin.branch_id in confirmed_branch_ids and not applied_rewrite
                    and origin.branch_id not in llg_regenerations
                    and descendant.branch_id not in next_llg_audit_ids):
                next_confirmed_ids.add(descendant.branch_id)
            if origin.branch_id in forced_confirmed_branch_ids:
                next_forced_ids.add(descendant.branch_id)
        confirmed_branch_ids = next_confirmed_ids
        forced_confirmed_branch_ids = next_forced_ids
        pending_llg_audit_ids = next_llg_audit_ids

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
                remaining_slots = MAX_NEW_HLGS_PER_ITERATION - len(additions)
                if remaining_slots <= 0:
                    iteration_warnings.append("COMPLETENESS_REQUESTS_DEFERRED_BATCH_LIMIT")
                    break
                actor = _actor_for_request(request.actor, actors)
                if actor is None:
                    iteration_warnings.append(
                        f"{request.actor}: UNKNOWN_ACTOR_REQUEST_IGNORED"
                    )
                    continue
                completeness_key = _key(request.actor)
                if completeness_rejection_counts.get(completeness_key, 0) >= MAX_REJECTED_REPAIR_CYCLES:
                    iteration_warnings.append(f"{request.actor}: COMPLETENESS_REPAIR_LIMIT_REACHED")
                    continue
                try:
                    generated = _generate_hlgs_with_top_down(
                        _generation_request(
                            f"iteration_{iteration}_missing_{request_index}",
                            HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL,
                            "Completeness requests for this actor (generate one HLG "
                            "per selected independent intention, within the batch budget):\n"
                            f"{request.generation_project_description}\n\n"
                            "Existing HLGs (do not duplicate covered outcomes):\n"
                            f"{current_hlgs.model_dump_json()}\n\n"
                            f"Original project documentation:\n{project_description}",
                            actor,
                            missing_hlg_evaluation.rationale,
                            generator_guidance=(
                                f"Return at most {remaining_slots} new HLGs in this call. "
                                f"The global limit is {MAX_NEW_HLGS_PER_ITERATION} additions per iteration, "
                                "across all actors. Prioritize documented independent gaps; "
                                "leave remaining outcomes for a later completeness analysis.\n"
                                + request.generation_project_description
                            ),
                        ),
                        mode,
                        evaluator_ablation,
                    )
                except RepairCandidateRejected as exc:
                    completeness_rejection_counts[completeness_key] = completeness_rejection_counts.get(completeness_key, 0) + 1
                    if completeness_rejection_counts[completeness_key] < MAX_REJECTED_REPAIR_CYCLES:
                        pending_completeness[completeness_key] = (request, missing_hlg_evaluation.rationale)
                    else:
                        pending_completeness.pop(completeness_key, None)
                    iteration_warnings.append(f"{request.actor}: COMPLETENESS_HLG_REJECTED: {exc}")
                    continue
                pending_completeness.pop(completeness_key, None)
                for goal in generated.goals:
                    if not goal.name.strip():
                        continue
                    if goal in current_hlgs.goals or goal in additions:
                        continue
                    if len(additions) >= MAX_NEW_HLGS_PER_ITERATION:
                        iteration_warnings.append("COMPLETENESS_GENERATED_GOALS_DEFERRED_BATCH_LIMIT")
                        break
                    additions.append(goal)
                    addition_guidance[_stable_hlg_key(goal)] = (
                        missing_hlg_evaluation.rationale
                        + "\nGenerate the full LLG decomposition for this approved new parent. "
                        + "Missing-outcome instructions for this actor:\n"
                        + request.generation_project_description
                    )

            if additions:
                current_hlgs = HighLevelGoals(
                    goals=[*current_hlgs.goals, *additions]
                )
                # Generate each decomposition independently so equal HLG values
                # never collapse into one dictionary entry.
                for addition_index, parent in enumerate(additions, start=1):
                    operation_id = f"addition_{addition_index:03d}"
                    generation_rationale = addition_guidance[_stable_hlg_key(parent)]
                    try:
                        generated_llgs = _regenerate_llgs_with_top_down(
                            LowLevelGoalRegenerationRequest(
                                high_level_goals=HighLevelGoals(goals=[parent]),
                                rationale=generation_rationale,
                                project_description=project_description,
                                sibling_high_level_goals=HighLevelGoals(goals=[g for g in current_hlgs.goals if g != parent]),
                                guidance_by_parent_name={
                                    operation_id: generation_rationale
                                },
                            ),
                            mode,
                            evaluator_ablation,
                        )
                    except RepairCandidateRejected as exc:
                        branch = _branches_from_embedded_parents(
                            HighLevelGoals(goals=[parent]), LowLevelGoals(low_level_goals=[])
                        )[0]
                        verdict = GlobalGoalEvaluationResult(
                            branch_id=branch.branch_id, rationale=str(exc),
                            final_decision=GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS,
                            high_level_evaluation=HighLevelGoalEvaluation(
                                decision="KEEP_ORIGINAL_HIGH_LEVEL_GOAL", quality_score=4,
                                rationale="Completeness parent approved; decomposition missing.",
                                reconstruction_relationship="NOT_RECONSTRUCTABLE", requires_llg_evaluation=True,
                            ),
                            low_level_evaluation=LowLevelGoalEvaluation(
                                decision="REGENERATE_LOW_LEVEL_GOALS", quality_score=2,
                                rationale=str(exc), regeneration_feedback=generation_rationale,
                                missing_essential_capabilities=["Entire decomposition of the approved completeness HLG"],
                            ),
                        )
                        reject_repair(branch.branch_id, verdict, exc)
                        continue
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

        # New/replaced branches are unconfirmed; all surviving branches are audited next pass.

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
        all_confirmed = all(
            branch.branch_id in confirmed_branch_ids for branch in current_branches
        ) and not (replacements or removed_branch_ids or additions or pending_llg_audit_ids)
        all_settled = all(
            branch.branch_id in confirmed_branch_ids | exhausted_branch_ids
            for branch in current_branches
        ) and not (replacements or removed_branch_ids or additions or pending_llg_audit_ids)
        completeness_complete = (
            (missing_hlg_evaluation is not None
                and missing_hlg_evaluation.decision == MissingHighLevelGoalDecision.NO_MISSING)
        )

        completeness_exhausted = bool(
            missing_hlg_evaluation is not None
            and missing_hlg_evaluation.decision == MissingHighLevelGoalDecision.FOUND
            and not additions and not pending_completeness
            and all(completeness_rejection_counts.get(_key(r.actor), 0) >= MAX_REJECTED_REPAIR_CYCLES
                    for r in missing_hlg_evaluation.missing_goal_requests)
        )
        structural_action_outcomes = {}
        for branch in branches:
            ev = evaluations[branch.branch_id]
            proposal = (ev.high_level_evaluation.structural_decision or {}).get("decision")
            if branch.branch_id in removed_branch_ids:
                outcome = "REMOVED"
            elif branch.branch_id in replacements:
                outcome = "REWRITTEN"
            elif proposal in {"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL", "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"}:
                reasons = [w for w in iteration_warnings
                           if branch.branch_id in w or branch.high_level_goal.name in w
                           or "LIMIT" in w or "DEPENDENCY_GUARD" in w]
                outcome = "DEFERRED_OR_REJECTED: " + ("; ".join(reasons) or ev.rationale)
            else:
                outcome = ev.final_decision.value
            structural_action_outcomes[branch.branch_id] = outcome
        trace = GlobalGoalCycleIteration(
            iteration=iteration,
            reconstructions=reconstructions,
            evaluations=evaluations,
            critic1_global_votes=list(critic1_global_votes),
            hlg_structure_summary={
                "proposed_actions": dict(Counter(
                    (ev.high_level_evaluation.structural_decision or {}).get("decision", "INCONCLUSIVE")
                    for ev in evaluations.values())),
                "proposed_removal_bases": dict(Counter(
                    ev.high_level_evaluation.removal_basis.value
                    for ev in evaluations.values() if ev.high_level_evaluation.removal_basis is not None)),
                "applied_removals": len(removed_branch_ids),
                "applied_rewrites": len(replacements),
            },
            structural_action_outcomes=structural_action_outcomes,
            high_level_goals=current_hlgs,
            low_level_goals=current_llgs,
            hlg_consolidations={anchor: [b.branch_id for b in members]
                for anchor, members in consolidation_sources.items() if members and anchor in replacements},
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
            pending_llg_audit_ids=sorted(pending_llg_audit_ids),
            hlg_rewrite_counts=dict(hlg_rewrite_counts),
            confirmed_branch_ids=sorted(confirmed_branch_ids),
            repair_rejection_counts=dict(repair_rejection_counts),
            forced_confirmed_branch_ids=sorted(forced_confirmed_branch_ids),
            completeness_rejection_counts=dict(completeness_rejection_counts),
            warnings=list(iteration_warnings),
            checkpoint_complete=True,
            completed_branch_count=len(evaluations),
            total_branch_count=len(active_branches),
        )
        traces.append(trace)
        _save_iteration(evaluation_output_directory, trace, branches)
        warnings.extend(iteration_warnings)

        if all_settled and (completeness_complete or completeness_exhausted):
            converged = all_confirmed and not completeness_exhausted
            return GlobalGoalCycleResult(
                converged=converged,
                stop_reason=(GlobalGoalCycleStopReason.NO_ACTIONS_REMAIN_WITH_WARNINGS
                             if not converged else GlobalGoalCycleStopReason.ALL_BRANCHES_CONFIRMED),
                completed_iterations=iteration,
                final_high_level_goals=current_hlgs,
                final_low_level_goals=current_llgs,
                iterations=traces,
                llg_regeneration_counts=dict(llg_regeneration_counts),
                pending_llg_audit_ids=sorted(pending_llg_audit_ids),
                hlg_rewrite_counts=dict(hlg_rewrite_counts),
                confirmed_branch_ids=sorted(confirmed_branch_ids),
                repair_rejection_counts=dict(repair_rejection_counts),
                forced_confirmed_branch_ids=sorted(forced_confirmed_branch_ids),
                completeness_rejection_counts=dict(completeness_rejection_counts),
                warnings=warnings,
            )

        actions_remain = bool(
            replacements or removed_branch_ids or additions
            # A successful full regeneration must receive a fresh audit even
            # when that application consumed the last available repair slot.
            or current_llgs != iteration_start_llgs
            or (
                missing_hlg_evaluation is not None
                and missing_hlg_evaluation.decision
                == MissingHighLevelGoalDecision.FOUND
            )
            or bool(pending_repairs)
            or bool(pending_llg_audit_ids)
            or (all_settled and missing_hlg_evaluation is None)
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
        if not actions_remain and iteration_warnings:
            return GlobalGoalCycleResult(
                converged=False,
                stop_reason=GlobalGoalCycleStopReason.NO_ACTIONS_REMAIN_WITH_WARNINGS,
                completed_iterations=iteration,
                final_high_level_goals=current_hlgs,
                final_low_level_goals=current_llgs,
                iterations=traces,
                llg_regeneration_counts=dict(llg_regeneration_counts),
                pending_llg_audit_ids=sorted(pending_llg_audit_ids),
                hlg_rewrite_counts=dict(hlg_rewrite_counts),
                confirmed_branch_ids=sorted(confirmed_branch_ids),
                repair_rejection_counts=dict(repair_rejection_counts),
                forced_confirmed_branch_ids=sorted(forced_confirmed_branch_ids),
                completeness_rejection_counts=dict(completeness_rejection_counts),
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
        pending_llg_audit_ids=sorted(pending_llg_audit_ids),
        hlg_rewrite_counts=dict(hlg_rewrite_counts),
        confirmed_branch_ids=sorted(confirmed_branch_ids),
        repair_rejection_counts=dict(repair_rejection_counts),
        forced_confirmed_branch_ids=sorted(forced_confirmed_branch_ids),
        completeness_rejection_counts=dict(completeness_rejection_counts),
        warnings=warnings,
    )
