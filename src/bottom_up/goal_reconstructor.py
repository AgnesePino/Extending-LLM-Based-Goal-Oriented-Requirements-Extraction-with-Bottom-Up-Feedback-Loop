"""Reconstruct one diagnostic HLG from the LLGs of a branch."""

from src.data_model import (
    BottomUpHighLevelGoal,
    BottomUpHighLevelGoalLLMOutput,
    LowLevelGoal, HighLevelGoals, LowLevelGoals,
    indexed_high_level_goals, low_level_goals_for_hlg,
)
from src.llm_clients import generate_response, loaded_local_model, QWEN_GENERATOR_MODEL


SYSTEM_PROMPT = (
    "You are an expert in software engineering and Goal-Oriented Requirements Engineering (GORE). "
    "Reconstruct one High-Level Goal from only the supplied Low-Level Goal descriptions. "
    "Apply bottom-up goal abstraction by asking WHY these lower-level goals should be achieved: "
    "identify the higher-level functional purpose that justifies them together. "
    "Infer only a purpose supported by the supplied children; do not invent stakeholder motivations "
    "or additional capabilities. Return a concise verb-led goal description in the requested JSON, "
    "without a separate name, explanations or assessments."
)


def reconstruct_high_level_goal(low_level_goals: list[LowLevelGoal]) -> BottomUpHighLevelGoalLLMOutput:
    """Read original structured LLGs; copy only their descriptions into the prompt.

    Do not rewrite descriptions or mutate the source goals/parent references.
    Every request, including JSON-validation retries, excludes prior responses.
    """
    if not low_level_goals:
        return BottomUpHighLevelGoalLLMOutput(
            reconstructed_high_level_goal=(
                "No High-Level Goal can be reconstructed because this branch "
                "has no Low-Level Goals."
            ),
        )

    # Local labels organize the input; each description is copied verbatim.
    lines = [
        f"- llg_{index:03d}: {goal.description}"
        for index, goal in enumerate(low_level_goals, start=1)
    ]
    prompt = """Apply GORE goal abstraction to the following low-level goal descriptions.
Ask WHY these goals should be achieved together: what common functional purpose
do their operations serve? Express that purpose as one higher-level goal.

- Use only the supplied descriptions and account for every LLG.
- Move from concrete operations to their supported functional intention;
  do not merely list, concatenate or paraphrase the operations.
- Preserve distinct capabilities, actor responsibilities and ownership/scope
  restrictions whenever omitting them would change the meaning.
- Do not infer unsupported functionality, business benefits or motivations.
- Do not correct, assess or omit an LLG to make the branch appear coherent.
- If no single common purpose is supported, retain the distinct outcomes in one
  concise composite goal rather than inventing a unifying purpose.
- Start with a verb and its functional object, adding only necessary scope.
  Omit introductions such as "The stakeholder aims to", "Enable users to",
  or "The system shall".
- Return only the goal description in JSON, without a separate name or reasoning.

**Low-level goals:**
{goals}

**Output:**
Return only valid JSON:
{{
  "reconstructed_high_level_goal": "..."
}}""".format(goals="\n".join(lines))

    return generate_response(prompt, SYSTEM_PROMPT, BottomUpHighLevelGoalLLMOutput,
                             stateless_retries=True)


def reconstruct_all_branches(
    goal_lists: dict[str, list[LowLevelGoal]] | HighLevelGoals,
    low_level_goals: LowLevelGoals | None = None,
):
    """Return reconstructions from existing top-down models or grouped LLGs."""
    if isinstance(goal_lists, HighLevelGoals):
        if low_level_goals is None:
            raise ValueError("The top-down LLG collection is required")
        goal_lists = {
            bid: children for bid, parent in indexed_high_level_goals(goal_lists).items()
            if (children := low_level_goals_for_hlg(parent, low_level_goals))
        }
    if not goal_lists:
        return {}
    with loaded_local_model(QWEN_GENERATOR_MODEL):
        results = {}
        for branch_id, low_level_goals in goal_lists.items():
            output = reconstruct_high_level_goal(low_level_goals)
            results[branch_id] = BottomUpHighLevelGoal(
                branch_id=branch_id,
                reconstructed_high_level_goal=output.reconstructed_high_level_goal,
                source_low_level_goal_ids=[goal.name for goal in low_level_goals])
        return results
