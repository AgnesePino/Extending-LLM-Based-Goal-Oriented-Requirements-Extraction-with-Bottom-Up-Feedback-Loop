"""Reconstruct one diagnostic HLG from the LLGs of a branch."""

from src.data_model import (
    BottomUpHighLevelGoal,
    BottomUpHighLevelGoalLLMOutput,
    GoalBranch,
)
from src.llm_clients import generate_response


SYSTEM_PROMPT = """You are an expert in Goal-Oriented Requirements Engineering.
Infer one High-Level Goal from the supplied Low-Level Goals. The reconstruction
is a lossless diagnostic summary, not a charitable abstraction: it must expose
every distinct functional capability present in the children, including an
outlier that does not belong with the others. Never silently discard, weaken,
or generalize an inconvenient child merely to make the set look coherent.
Explain exactly how every supplied LLG contributes to the reconstructed scope.
Do not invent functionality and do not mention the hidden parent goal. Return
only the requested JSON."""


def reconstruct_high_level_goal(branch: GoalBranch) -> BottomUpHighLevelGoal:
    if not branch.low_level_goals:
        return BottomUpHighLevelGoal(
            branch_id=branch.branch_id,
            reconstructed_high_level_goal=(
                "No High-Level Goal can be reconstructed because this branch "
                "has no Low-Level Goals."
            ),
            rationale=(
                "The empty decomposition is a material coverage gap. Evaluate "
                "the original HLG normally; if it is valid, generate its "
                "required Low-Level Goals instead of removing it merely because "
                "the branch is empty."
            ),
            source_low_level_goal_ids=[],
        )

    lines = [
        f"- llg_{index:03d}: {goal.description}"
        for index, goal in enumerate(branch.low_level_goals, start=1)
    ]
    prompt = """Given these Low-Level Goals, reconstruct the single High-Level
Goal that describes the complete aggregate scope implemented by them.

Mandatory lossless reconstruction rules:
- account for every llg_NNN; no child may be ignored as noise;
- preserve materially different actions, objects, actors, lifecycle phases,
  and outcomes instead of hiding them behind a vague verb such as manage,
  handle, support, process, or use;
- if one child introduces an unrelated or broader capability, include that
  capability explicitly in the reconstructed HLG so the evaluator can detect
  the mismatch;
- do not infer missing operations that are not present in the supplied LLGs;
- in the rationale, reference every llg_NNN and state the capability it adds.

Low-Level Goals:
{goals}

Output only valid JSON:
{{
  "reconstructed_high_level_goal": "...",
  "rationale": "Brief explanation of the inferred intention."
}}""".format(goals="\n".join(lines))

    output = generate_response(prompt, SYSTEM_PROMPT, BottomUpHighLevelGoalLLMOutput)
    return BottomUpHighLevelGoal(
        branch_id=branch.branch_id,
        reconstructed_high_level_goal=output.reconstructed_high_level_goal,
        rationale=output.rationale,
        source_low_level_goal_ids=[goal.name for goal in branch.low_level_goals],
    )


def reconstruct_all_branches(branches: list[GoalBranch]):
    return {branch.branch_id: reconstruct_high_level_goal(branch) for branch in branches}
