"""Shared models for the original top-down and bottom-up pipelines."""

import hashlib
import json
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentDescription(BaseModel):
    description: str


class Actor(BaseModel):
    name: str
    description: str


class Actors(BaseModel):
    actors: list[Actor]


class HighLevelGoal(BaseModel):
    name: str
    description: str
    actor: Actor


class HighLevelGoals(BaseModel):
    goals: list[HighLevelGoal]


class LowLevelGoal(BaseModel):
    name: str
    description: str
    high_level_associated: HighLevelGoal


class LowLevelGoals(BaseModel):
    low_level_goals: list[LowLevelGoal]


class API(BaseModel):
    api_name: str
    api_path: str
    description: str
    request_type: str


class APIMapping(BaseModel):
    APIs: list[API]
    low_level_goal: LowLevelGoal


class Critique(BaseModel):
    score: float
    comment: str


class GoalBranch(BaseModel):
    branch_id: str
    high_level_goal: HighLevelGoal
    low_level_goals: list[LowLevelGoal]


def serialize_top_down_hierarchy(actors: Actors, high_level_goals: HighLevelGoals,
                                 low_level_goals: LowLevelGoals) -> dict:
    """Store explicit generator references verbatim, without resolving identities.

    An embedded parent different from every HLG-stage output is a separate
    observed variant. Both versions remain visible; no actor or goal is rewritten.
    """
    branches: list[GoalBranch] = []
    sources: dict[str, dict] = {}
    fingerprints: dict[str, list[int]] = {}

    def fingerprint(model):
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))

    def append_parent(parent, source, index):
        key = fingerprint(parent)
        positions = fingerprints.setdefault(key, [])
        bid = f"hlg_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}_{len(positions) + 1:03d}"
        positions.append(len(branches))
        branches.append(GoalBranch(branch_id=bid, high_level_goal=parent, low_level_goals=[]))
        sources[bid] = {"source": source, "source_index": index}
        return branches[-1]

    for index, parent in enumerate(high_level_goals.goals):
        append_parent(parent, "high_level_generator", index)
    for index, child in enumerate(low_level_goals.low_level_goals):
        positions = fingerprints.get(fingerprint(child.high_level_associated))
        branch = (branches[positions[0]] if positions else
                  append_parent(child.high_level_associated, "low_level_generator_parent_reference", index))
        branch.low_level_goals.append(child)

    hierarchy = []
    actor_positions = {}
    for actor in actors.actors:
        key = fingerprint(actor)
        if key not in actor_positions:
            actor_positions[key] = len(hierarchy)
            hierarchy.append({"actor": actor.model_dump(mode="json"), "goals": []})
    for branch in branches:
        actor = branch.high_level_goal.actor
        key = fingerprint(actor)
        if key not in actor_positions:
            actor_positions[key] = len(hierarchy)
            hierarchy.append({"actor": actor.model_dump(mode="json"), "goals": []})
        hierarchy[actor_positions[key]]["goals"].append(branch.model_dump(mode="json"))
    return {"hierarchySchemaVersion": 2,
            "hierarchyPolicy": "preserve_exact_generator_associations",
            "actorHierarchy": hierarchy,
            "goalBranches": [branch.model_dump(mode="json") for branch in branches],
            "hierarchyProvenance": sources}


def group_low_level_goals_by_hlg(low_level_goals: LowLevelGoals) -> list[GoalBranch]:
    """Group the generated LLGs by their embedded parent text, ignoring actor.

    GoalBranch is only the existing transport container for a group. No child
    reference is rewritten and no group is synthesized from the HLG collection.
    """
    groups = {}
    for child in low_level_goals.low_level_goals:
        parent = child.high_level_associated
        key = (parent.name, parent.description)
        if key not in groups:
            serialized = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
            groups[key] = GoalBranch(branch_id=f"hlg_{fingerprint}_001",
                high_level_goal=parent, low_level_goals=[])
        groups[key].low_level_goals.append(child)
    return list(groups.values())


def group_goals_by_parent(high_level_goals: HighLevelGoals,
                         low_level_goals: LowLevelGoals) -> list[GoalBranch]:
    """Compatibility entry point: group directly from generated LLG references."""
    return group_low_level_goals_by_hlg(low_level_goals)


class BottomUpHighLevelGoalLLMOutput(BaseModel):
    reconstructed_high_level_goal: str


class BottomUpHighLevelGoal(BottomUpHighLevelGoalLLMOutput):
    branch_id: str
    source_low_level_goal_ids: list[str]
    source_low_level_goal_map: dict[str, str] | None = None


class HighLevelGoalDecision(str, Enum):
    KEEP = "KEEP_ORIGINAL_HIGH_LEVEL_GOAL"
    REWRITE = "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"
    REMOVE = "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL"
    DUPLICATE_REMOVAL = "DUPLICATE_REMOVAL"
    ADD = "ADD_NEW_HIGH_LEVEL_GOAL"


class ReconstructionRelationship(str, Enum):
    EQUIVALENT = "EQUIVALENT"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"


class EvaluatorVoteRecord(BaseModel):
    model: str
    status: str
    decision: str | None = None
    raw_response: str | None = None
    parsed_response: dict | None = None
    error: str | None = None
    attempt_history: list[dict] = Field(default_factory=list)


class HighLevelGoalEvaluation(BaseModel):
    rationale: str
    decision: HighLevelGoalDecision
    reconstruction_relationship: ReconstructionRelationship | None = None
    structural_decision: dict | None = None
    requires_llg_evaluation: bool
    rewrite_limit_reached: bool = False
    post_repair_llg_audit_required: bool = False
    rewriting_focus: str | None = None
    consolidate_with_branch_ids: list[str] = Field(default_factory=list)
    covered_by_branch_id: str | None = None
    covered_by_high_level_goal_name: str | None = None
    voter_count: int = 1
    valid_vote_count: int = 1
    winning_vote_count: int = 1
    vote_distribution: dict[str, int] = Field(default_factory=dict)
    evaluator_votes: list[EvaluatorVoteRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_rewriting_focus(self):
        if self.consolidate_with_branch_ids:
            if self.decision != HighLevelGoalDecision.REWRITE:
                raise ValueError("Only REWRITE may request consolidation")
            if (len(set(self.consolidate_with_branch_ids)) != len(self.consolidate_with_branch_ids)
                    or any(not item.strip() for item in self.consolidate_with_branch_ids)):
                raise ValueError("Consolidation IDs must be distinct non-empty references")
        if self.rewrite_limit_reached and (
            self.decision != HighLevelGoalDecision.KEEP
            or not self.requires_llg_evaluation
        ):
            raise ValueError("Rewrite limit requires operational KEEP and LLG evaluation")
        if self.post_repair_llg_audit_required and (
            self.decision != HighLevelGoalDecision.KEEP
            or not self.requires_llg_evaluation
        ):
            raise ValueError("Post-repair audit requires operational KEEP and LLG evaluation")
        if self.decision == HighLevelGoalDecision.REWRITE and not self.rewriting_focus:
            raise ValueError("REWRITE_ORIGINAL_HIGH_LEVEL_GOAL requires rewriting_focus")
        if self.decision != HighLevelGoalDecision.REWRITE and self.rewriting_focus:
            raise ValueError(
                "Only REWRITE_ORIGINAL_HIGH_LEVEL_GOAL may define rewriting_focus"
            )
        if self.decision == HighLevelGoalDecision.DUPLICATE_REMOVAL and (
            not self.covered_by_branch_id
            or not self.covered_by_high_level_goal_name
        ):
            raise ValueError("DUPLICATE_REMOVAL requires an exact surviving branch")
        if self.decision != HighLevelGoalDecision.DUPLICATE_REMOVAL and (
            self.covered_by_branch_id is not None
            or self.covered_by_high_level_goal_name is not None
        ):
            raise ValueError(
                "Only DUPLICATE_REMOVAL may define a surviving branch"
            )
        if (
            self.decision != HighLevelGoalDecision.KEEP
            and self.requires_llg_evaluation
        ):
            raise ValueError(
                "Only KEEP_ORIGINAL_HIGH_LEVEL_GOAL may require LLG evaluation"
            )
        if (
            self.decision == HighLevelGoalDecision.KEEP
            and self.reconstruction_relationship
            == ReconstructionRelationship.EQUIVALENT
            and self.requires_llg_evaluation
            and not self.rewrite_limit_reached
            and not self.post_repair_llg_audit_required
        ):
            raise ValueError(
                "An equivalent reconstruction must not trigger LLG evaluation"
            )
        return self


class HlgStructuralDecision(BaseModel):
    """Critic 1's decision on one original parent, independent of reconstruction."""

    rationale: str = Field(min_length=1)
    decision: HighLevelGoalDecision
    branch_id: str = Field(min_length=1)
    outcome_justification: str = Field(min_length=1)
    rewriting_focus: str | None = None
    consolidate_with_branch_ids: list[str] = Field(default_factory=list)
    covered_by_branch_id: str | None = None
    covered_by_high_level_goal_name: str | None = None
    voter_count: int = 1
    valid_vote_count: int = 1
    winning_vote_count: int = 1
    vote_distribution: dict[str, int] = Field(default_factory=dict)
    evaluator_votes: list[EvaluatorVoteRecord] = Field(default_factory=list)

    @field_validator("branch_id", "rationale", "outcome_justification")
    @classmethod
    def require_structural_text(cls, value):
        if not value.strip():
            raise ValueError("Structural decisions require non-empty evidence")
        return value.strip()

    @model_validator(mode="after")
    def validate_action_fields(self):
        HighLevelGoalEvaluation(
            decision=self.decision, rationale=self.rationale,
            requires_llg_evaluation=self.decision == HighLevelGoalDecision.KEEP,
            rewriting_focus=self.rewriting_focus,
            consolidate_with_branch_ids=self.consolidate_with_branch_ids,
            covered_by_branch_id=self.covered_by_branch_id,
            covered_by_high_level_goal_name=self.covered_by_high_level_goal_name,
        )
        return self


class GlobalHlgReviewResponse(BaseModel):
    # Individual entries are validated separately: one faulty decision must not
    # discard other well-formed decisions from the same complete-list response.
    decisions: list[dict] = Field(default_factory=list)
    completeness: dict | None = None


class HlgReconstructionDiagnosis(BaseModel):
    rationale: str = Field(min_length=1)
    reconstruction_relationship: ReconstructionRelationship

    @field_validator("rationale")
    @classmethod
    def require_diagnostic_evidence(cls, value):
        if not value.strip():
            raise ValueError("Reconstruction diagnosis requires evidence")
        return value.strip()


class HighLevelGoalReplacementRequest(BaseModel):
    rationale: str
    generation_project_description: str
    actor: Actor


class LowLevelGoalDecision(str, Enum):
    KEEP = "KEEP_LOW_LEVEL_GOALS"
    REGENERATE = "REGENERATE_LOW_LEVEL_GOALS"


class LowLevelGoalEvaluation(BaseModel):
    rationale: str
    decision: LowLevelGoalDecision
    regeneration_feedback: str | None = None
    unsupported_or_misleading_llg_ids: list[str] = Field(default_factory=list)
    missing_essential_capabilities: list[str] = Field(default_factory=list)
    voter_count: int = 1
    valid_vote_count: int = 1
    winning_vote_count: int = 1
    vote_distribution: dict[str, int] = Field(default_factory=dict)
    evaluator_votes: list[EvaluatorVoteRecord] = Field(default_factory=list)

    @field_validator("unsupported_or_misleading_llg_ids", mode="before")
    @classmethod
    def coerce_llg_ids_to_strings(cls, value):
        """Accept numeric IDs occasionally emitted by the evaluator JSON."""
        if value is None:
            return []
        if isinstance(value, (str, int, float)):
            value = [value]
        return [str(item) for item in value]

    @model_validator(mode="after")
    def check_llg_decision_evidence(self):
        if self.decision == LowLevelGoalDecision.KEEP:
            if (
                self.regeneration_feedback
                or self.unsupported_or_misleading_llg_ids
                or self.missing_essential_capabilities
            ):
                raise ValueError(
                    "KEEP_LOW_LEVEL_GOALS cannot report unresolved LLG defects"
                )
        if self.decision == LowLevelGoalDecision.REGENERATE:
            if not self.regeneration_feedback:
                raise ValueError(
                    "REGENERATE_LOW_LEVEL_GOALS requires regeneration_feedback"
                )
            if not (
                self.unsupported_or_misleading_llg_ids
                or self.missing_essential_capabilities
            ):
                raise ValueError(
                    "REGENERATE_LOW_LEVEL_GOALS requires structured defect evidence"
                )
        return self


class MissingHighLevelGoalDecision(str, Enum):
    NO_MISSING = "NO_MISSING_HIGH_LEVEL_GOALS"
    FOUND = "MISSING_HIGH_LEVEL_GOALS_FOUND"
    INCONCLUSIVE = "MISSING_HIGH_LEVEL_GOAL_EVALUATION_INCONCLUSIVE"


class MissingHighLevelGoalRequest(BaseModel):
    actor: str
    generation_project_description: str

    @field_validator("generation_project_description")
    @classmethod
    def require_missing_outcome_description(cls, value):
        if not value.strip():
            raise ValueError("A missing HLG request must describe the missing outcome")
        return value.strip()


class MissingHighLevelGoalEvaluation(BaseModel):
    rationale: str
    decision: MissingHighLevelGoalDecision
    missing_goal_requests: list[MissingHighLevelGoalRequest] = Field(
        default_factory=list
    )
    voter_count: int = 1
    valid_vote_count: int = 1
    winning_vote_count: int = 1
    vote_distribution: dict[str, int] = Field(default_factory=dict)
    evaluator_votes: list[EvaluatorVoteRecord] = Field(default_factory=list)

    @field_validator("rationale")
    @classmethod
    def require_completeness_reason(cls, value):
        if not value.strip():
            raise ValueError("Completeness analysis requires a non-empty rationale")
        return value.strip()

    @model_validator(mode="after")
    def validate_requests(self):
        if self.decision == MissingHighLevelGoalDecision.FOUND and not self.missing_goal_requests:
            raise ValueError(
                "MISSING_HIGH_LEVEL_GOALS_FOUND requires at least one request."
            )
        if self.decision == MissingHighLevelGoalDecision.NO_MISSING and self.missing_goal_requests:
            raise ValueError(
                "NO_MISSING_HIGH_LEVEL_GOALS cannot contain requests."
            )
        if self.decision == MissingHighLevelGoalDecision.INCONCLUSIVE and self.missing_goal_requests:
            raise ValueError(
                "An inconclusive coverage vote cannot contain requests."
            )
        return self


class GlobalGoalEvaluationDecision(str, Enum):
    CONFIRM_BRANCH = "CONFIRM_BRANCH"
    EVALUATION_INCONCLUSIVE = "EVALUATION_INCONCLUSIVE"
    REGENERATE_LOW_LEVEL_GOALS = "REGENERATE_LOW_LEVEL_GOALS"
    LLG_REGENERATION_LIMIT_REACHED = "LLG_REGENERATION_LIMIT_REACHED"
    REWRITE_ORIGINAL_HIGH_LEVEL_GOAL = "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"
    REMOVE_ORIGINAL_HIGH_LEVEL_GOAL = "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL"
    ADD_NEW_HIGH_LEVEL_GOAL = "ADD_NEW_HIGH_LEVEL_GOAL"
    MISSING_HIGH_LEVEL_GOALS_FOUND = "MISSING_HIGH_LEVEL_GOALS_FOUND"
    NO_MISSING_HIGH_LEVEL_GOALS = "NO_MISSING_HIGH_LEVEL_GOALS"


class BottomUpRepairFeedback(BaseModel):
    """Evaluator-authored correction; previous goals are recovered from the branch."""
    model_config = ConfigDict(extra="forbid")
    critique: str = Field(min_length=1)

    @field_validator("critique")
    @classmethod
    def require_nonblank_critique(cls, value):
        if not value.strip():
            raise ValueError("Repair critique cannot be blank")
        return value


class GlobalGoalEvaluationResult(BaseModel):
    branch_id: str
    rationale: str
    final_decision: GlobalGoalEvaluationDecision
    high_level_evaluation: HighLevelGoalEvaluation
    replacement_request: HighLevelGoalReplacementRequest | None = None
    low_level_evaluation: LowLevelGoalEvaluation | None = None
    critic_votes: dict[str, list[EvaluatorVoteRecord]] = Field(default_factory=dict)
    evaluation_status: str | None = None
    similarity: dict = Field(default_factory=dict)
    regeneration_feedback: str | None = None
    feedback: BottomUpRepairFeedback | None = None
    new_goal_focus: str | None = None


class HighLevelGoalGenerationAction(str, Enum):
    ADD_NEW_HIGH_LEVEL_GOAL = "ADD_NEW_HIGH_LEVEL_GOAL"
    REPLACE_EXISTING_HIGH_LEVEL_GOAL = "REPLACE_EXISTING_HIGH_LEVEL_GOAL"


class HighLevelGoalGeneratorInput(BaseModel):
    project_description: str
    actors: Actors


class HighLevelGoalGenerationRequest(BaseModel):
    request_id: str
    action: HighLevelGoalGenerationAction
    generator_input: HighLevelGoalGeneratorInput
    rationale: str
    generator_guidance: str | None = None
    target_branch_id: str | None = None
    original_high_level_goal: HighLevelGoal | None = None
    consolidated_high_level_goals: HighLevelGoals = Field(default_factory=lambda: HighLevelGoals(goals=[]))
    existing_high_level_goals: HighLevelGoals = Field(default_factory=lambda: HighLevelGoals(goals=[]))
    existing_low_level_goals: LowLevelGoals = Field(default_factory=lambda: LowLevelGoals(low_level_goals=[]))

    @field_validator("rationale")
    @classmethod
    def require_generation_reason(cls, value):
        if not value.strip():
            raise ValueError("HLG generation requires a non-empty rationale")
        return value.strip()


class LowLevelGoalRegenerationRequest(BaseModel):
    high_level_goals: HighLevelGoals
    rationale: str
    feedback: BottomUpRepairFeedback | None = None
    project_description: str = ""
    sibling_high_level_goals: HighLevelGoals = Field(default_factory=lambda: HighLevelGoals(goals=[]))
    guidance_by_parent_name: dict[str, str] = Field(default_factory=dict)
    existing_low_level_goals: LowLevelGoals = Field(
        default_factory=lambda: LowLevelGoals(low_level_goals=[])
    )

    @field_validator("rationale")
    @classmethod
    def require_regeneration_reason(cls, value):
        if not value.strip():
            raise ValueError("LLG generation requires a non-empty rationale")
        return value.strip()


class GlobalGoalCycleStopReason(str, Enum):
    ALL_BRANCHES_CONFIRMED = "ALL_BRANCHES_CONFIRMED"
    NO_CHANGES_APPLIED = "NO_CHANGES_APPLIED"
    NO_ACTIONS_REMAIN_WITH_WARNINGS = "NO_ACTIONS_REMAIN_WITH_WARNINGS"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


class GlobalGoalCycleIteration(BaseModel):
    iteration: int
    reconstructions: dict[str, BottomUpHighLevelGoal]
    evaluations: dict[str, GlobalGoalEvaluationResult]
    critic1_global_votes: list[EvaluatorVoteRecord] = Field(default_factory=list)
    hlg_structure_summary: dict = Field(default_factory=dict)
    structural_action_outcomes: dict[str, str] = Field(default_factory=dict)
    high_level_goals: HighLevelGoals
    low_level_goals: LowLevelGoals
    input_hierarchy: dict | None = None
    output_hierarchy: dict | None = None
    hlg_consolidations: dict[str, list[str]] = Field(default_factory=dict)
    missing_hlg_evaluation: MissingHighLevelGoalEvaluation | None = None
    missing_hlg_decision: GlobalGoalEvaluationDecision | None = None
    llg_regeneration_counts: dict[str, int] = Field(default_factory=dict)
    pending_llg_audit_ids: list[str] = Field(default_factory=list)
    hlg_rewrite_counts: dict[str, int] = Field(default_factory=dict)
    confirmed_branch_ids: list[str] = Field(default_factory=list)
    repair_rejection_counts: dict[str, int] = Field(default_factory=dict)
    forced_confirmed_branch_ids: list[str] = Field(default_factory=list)
    completeness_rejection_counts: dict[str, int] = Field(
        default_factory=dict, validation_alias=AliasChoices("completeness_rejection_counts", "discovery_rejection_counts")
    )
    warnings: list[str] = Field(default_factory=list)
    checkpoint_complete: bool = True
    completed_branch_count: int = 0
    total_branch_count: int = 0
    similarity_analysis: dict = Field(default_factory=dict)
    stage_logs: dict[str, dict[str, str]] = Field(default_factory=dict)
    top_down_reflections: dict[str, dict] = Field(default_factory=dict)
    new_hlg_additions: dict[str, str] = Field(default_factory=dict)
    hlg_addition_counts: dict[str, int] = Field(default_factory=dict)


class GlobalGoalCycleResult(BaseModel):
    converged: bool
    stop_reason: GlobalGoalCycleStopReason
    completed_iterations: int
    final_high_level_goals: HighLevelGoals
    final_low_level_goals: LowLevelGoals
    iterations: list[GlobalGoalCycleIteration] = Field(default_factory=list)
    llg_regeneration_counts: dict[str, int] = Field(default_factory=dict)
    pending_llg_audit_ids: list[str] = Field(default_factory=list)
    hlg_rewrite_counts: dict[str, int] = Field(default_factory=dict)
    confirmed_branch_ids: list[str] = Field(default_factory=list)
    repair_rejection_counts: dict[str, int] = Field(default_factory=dict)
    forced_confirmed_branch_ids: list[str] = Field(default_factory=list)
    completeness_rejection_counts: dict[str, int] = Field(
        default_factory=dict, validation_alias=AliasChoices("completeness_rejection_counts", "discovery_rejection_counts")
    )
    warnings: list[str] = Field(default_factory=list)
    configuration: dict = Field(default_factory=dict)
    hlg_addition_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def kaos_analysis(self) -> None:
        """Compatibility for stale notebook cells; KAOS is no longer run."""
        return None
