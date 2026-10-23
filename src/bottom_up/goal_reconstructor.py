"""Reconstruct one diagnostic HLG from the LLGs of a branch."""

from src.data_model import (
    BottomUpHighLevelGoal,
    BottomUpHighLevelGoalLLMOutput,
    GoalBranch,
)
from src.llm_clients import generate_response, loaded_local_model, QWEN_GENERATOR_MODEL


SYSTEM_PROMPT = (
    "You are a helpful assistant expert in software engineering tasks. "
    "You are tasked to reconstruct one High-Level Goal from the supplied Low-Level Goals. "
    "Following the Goal-Oriented Requirements Engineering (GORE) framework, express the "
    "functional outcome represented by the concrete operations in the low-level goals. "
    "Write a concise, action-oriented goal phrase starting with a verb, followed by the "
    "functional object and any necessary scope qualification. "
    "Use only the supplied low-level goals. Return only the requested JSON, without explanations."
)


def reconstruct_high_level_goal(branch: GoalBranch) -> BottomUpHighLevelGoal:
    if not branch.low_level_goals:
        return BottomUpHighLevelGoal(
            branch_id=branch.branch_id,
            reconstructed_high_level_goal=(
                "No High-Level Goal can be reconstructed because this branch "
                "has no Low-Level Goals."
            ),
            source_low_level_goal_ids=[],
            source_low_level_goal_map={},
        )

    lines = [
        f"- llg_{index:03d}: {goal.name}\n  Description: {goal.description}"
        for index, goal in enumerate(branch.low_level_goals, start=1)
    ]
    prompt = """Your task: summarize the following low-level goals as one
high-level functional goal.

- Account for every supplied LLG.
- Combine related operations into their common functional outcome.
- Preserve distinct capabilities and scope constraints when omitting them
  would change the meaning.
- Use a concise verb-led phrase, rather than a list of operations.
- Omit introductory wording such as "The stakeholder aims to", "Enable users to",
  or "The system shall".
- Include actors or ownership restrictions only when needed to distinguish
  responsibilities or scope.
- Do not add functionality, explanations, assessments, or corrections.
- If the LLGs express distinct outcomes, retain them in one concise composite
  goal; do not invent a common purpose to make them appear coherent.

**Low-level goals:**
{goals}

**Output:**
Return only valid JSON:
{{
  "reconstructed_high_level_goal": "..."
}}""".format(goals="\n".join(lines))

    output = generate_response(prompt, SYSTEM_PROMPT, BottomUpHighLevelGoalLLMOutput)
    return BottomUpHighLevelGoal(
        branch_id=branch.branch_id,
        reconstructed_high_level_goal=output.reconstructed_high_level_goal,
        source_low_level_goal_ids=[goal.name for goal in branch.low_level_goals],
    )


def reconstruct_all_branches(branches: list[GoalBranch]):
    if not branches:
        return {}
    with loaded_local_model(QWEN_GENERATOR_MODEL):
        return {branch.branch_id: reconstruct_high_level_goal(branch) for branch in branches}
