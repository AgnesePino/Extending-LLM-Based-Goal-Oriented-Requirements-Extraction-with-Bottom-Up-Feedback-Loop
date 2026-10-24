"""Shared models for the original top-down and bottom-up pipelines."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


def indexed_high_level_goals(high_level_goals: HighLevelGoals) -> dict[str, HighLevelGoal]:
    """Number the existing top-down HLGs without changing or deduplicating them."""
    return {str(index): goal for index, goal in enumerate(high_level_goals.goals, 1)}


def low_level_goals_for_hlg(parent: HighLevelGoal, low_level_goals: LowLevelGoals) -> list[LowLevelGoal]:
    """Read exact embedded top-down references; leave source objects unchanged."""
    expected = parent.model_dump(mode="json")
    return [goal for goal in low_level_goals.low_level_goals
            if goal.high_level_associated.model_dump(mode="json") == expected]


# Only the current bottom-up wire format is supported.
RESULT_SCHEMA_VERSION = 3


class BottomUpHighLevelGoalLLMOutput(BaseModel):
    reconstructed_high_level_goal: str


class BottomUpHighLevelGoal(BottomUpHighLevelGoalLLMOutput):
    branch_id: str
    source_low_level_goal_ids: list[str]


class HighLevelGoalDecision(str, Enum):
    KEEP = "KEEP_ORIGINAL_HIGH_LEVEL_GOAL"
    REWRITE = "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"
    REMOVE = "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL"
    DUPLICATE_REMOVAL = "DUPLICATE_REMOVAL"


class BottomUpRepairFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    critique: str = Field(min_length=1)

    @field_validator("critique")
    @classmethod
    def require_nonblank_critique(cls, value):
        if not value.strip():
            raise ValueError("Repair critique cannot be blank")
        return value


class DocumentaryDecision(BaseModel):
    """Exact per-branch JSON contract returned by each evaluator."""
    model_config = ConfigDict(extra="forbid")
    branch_id: str = Field(min_length=1)
    decision: HighLevelGoalDecision
    feedback: BottomUpRepairFeedback | None = None
    surviving_branch_id: str | None = None

    @model_validator(mode="after")
    def check_parameters(self):
        for value in (self.branch_id, self.surviving_branch_id):
            if value is not None and not value.strip():
                raise ValueError("Text fields cannot be blank")
        if bool(self.feedback) != (self.decision == HighLevelGoalDecision.REWRITE):
            raise ValueError("Only REWRITE requires textual feedback")
        if bool(self.surviving_branch_id) != (self.decision == HighLevelGoalDecision.DUPLICATE_REMOVAL):
            raise ValueError("Only DUPLICATE_REMOVAL requires surviving_branch_id")
        return self


class EvaluatorVoteRecord(BaseModel):
    """One global response and one technical history per model and iteration."""
    model_config = ConfigDict(extra="forbid")
    model: str
    status: Literal["VALID", "INVALID"]
    decisions: list[DocumentaryDecision] = Field(default_factory=list)
    error: str | None = None
    attempts: list[dict] = Field(default_factory=list)


class GlobalGoalEvaluationResult(BaseModel):
    """An approved action, or no action when the global review is inconclusive."""
    model_config = ConfigDict(extra="forbid")
    branch_id: str
    decision: HighLevelGoalDecision | None
    diagnostic: str | None = None
    feedback: BottomUpRepairFeedback | None = None
    surviving_branch_id: str | None = None
    evaluation_status: str

    @model_validator(mode="after")
    def check_aggregate(self):
        if self.decision is None:
            if self.feedback is not None or self.surviving_branch_id is not None:
                raise ValueError("An inconclusive review cannot authorize repair or removal")
            if self.evaluation_status == "DECIDED":
                raise ValueError("DECIDED requires an approved action")
        else:
            DocumentaryDecision.model_validate(self.model_dump(exclude={"evaluation_status", "diagnostic"}))
            if self.evaluation_status != "DECIDED":
                raise ValueError("An approved action must be DECIDED")
        return self


class GlobalGoalCycleStopReason(str, Enum):
    ALL_BRANCHES_CONFIRMED = "ALL_BRANCHES_CONFIRMED"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


class GlobalGoalCycleIteration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iteration: int
    reconstructions: dict[str, BottomUpHighLevelGoal]
    evaluations: dict[str, GlobalGoalEvaluationResult]
    evaluator_responses: list[EvaluatorVoteRecord] = Field(default_factory=list)
    structural_action_outcomes: dict[str, str] = Field(default_factory=dict)
    high_level_goals: HighLevelGoals
    low_level_goals: LowLevelGoals
    input_hierarchy: dict
    warnings: list[str] = Field(default_factory=list)
    checkpoint_complete: bool = True
    completed_branch_count: int = 0
    total_branch_count: int = 0
    similarity_analysis: dict = Field(default_factory=dict)
    top_down_reflections: dict[str, dict] = Field(default_factory=dict)
    changes: dict = Field(default_factory=dict)


class GlobalGoalCycleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[3]
    converged: bool
    stop_reason: GlobalGoalCycleStopReason
    completed_iterations: int
    final_high_level_goals: HighLevelGoals
    final_low_level_goals: LowLevelGoals
    iterations: list[GlobalGoalCycleIteration] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    configuration: dict = Field(default_factory=dict)
