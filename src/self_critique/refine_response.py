import os
import re
import sys
from enum import Enum


from src import examples
import src.examples.shot_learning
from src.data_model import Critique
from  src.examples.shot_learning import ShotPromptingMode
from src.llm_clients import (
    generate_evaluator_response,
    generate_response_llama,
    generate_response,
)


MAX_ATTEMPTS = 3
QUALITY_THRESHOLD = 8.5
TOP_DOWN_EVALUATOR_MAX_INPUT_CHARS = int(
    os.getenv("TOP_DOWN_EVALUATOR_MAX_INPUT_CHARS", "24000")
)
TOP_DOWN_EVALUATOR_DESCRIPTION_MAX_CHARS = int(
    os.getenv("TOP_DOWN_EVALUATOR_DESCRIPTION_MAX_CHARS", "1800")
)
TOP_DOWN_EVALUATOR_ACTOR_DESCRIPTION_MAX_CHARS = 70
TOP_DOWN_EVALUATOR_HLG_DESCRIPTION_MAX_CHARS = 90
TOP_DOWN_EVALUATOR_LLG_DESCRIPTION_MAX_CHARS = 60

class EvalMode(Enum):
    ACTORS = "Actors"
    HIGH_LEVEL = "High Level Goals"
    LOW_LEVEL = "Low Level Goals"

class Feedback():
    def __init__(self, previous_output, critique):
        self.previous_output = previous_output
        self.critique = critique 

# Original baseline prompt builder.  The public ``get_evaluation`` below keeps
# these instructions and all four examples unchanged, compacting only target
# data before delegating here.
def _original_get_evaluation(eval_mode: EvalMode, description, actors, high_level_goals=None, low_level_goals=None):
    if not isinstance(eval_mode, EvalMode):
        raise TypeError(f"Expected an instance of EvalMode, but got {type(eval_mode).__name__}")
    sys_prompt = (
        "You're an helpful assistant, expert in the field of software engineering and specialised in the Goal-Oriented Requirements Engineering (GORE) framework.\n\n."
        "Following the Goal-Oriented Requirements Engineering (GORE) framework: "
        "-  an actor is active entity that has the capability to perform actions to achieve goals. Unlike goals, which are 'what' or 'why,'"
        " actors are the 'who.' \n"
        "-  high-level goals are strategic objectives that define the 'why' behind a system. \n"
        "   They are usually abstract, business-oriented, and independent of technical implementation. "
        "They represent the needs of stakeholders or the organization. \n"
        "   Focus: Vision and justification. \n"
        "-  low-level goals are technical objectives that describe 'how' the high-level goals will be achieved. \n"
        "   They are more concrete and are eventually refined into specific requirements or software specifications. \n"
        "   Focus: Implementation and constraints.\n"
        "You can propose new goals taking into account the already present ones. Consider that high-level "
        "goals often answer the WHY question, while low-level goals often address the HOW."
        "\n\n ### Examples:\n\n"
        )

    if eval_mode == EvalMode.ACTORS:
        sys_prompt += f"""
            {examples.shot_learning.example1_actors_withFeedback1}
            
            ---
            
            {examples.shot_learning.example1_actors_withFeedback2}
            
            ---
            
            {examples.shot_learning.example1_actors_withFeedback3}
            
            ---
            
            {examples.shot_learning.example1_actors_withFeedback4}
            
            ---
        """

    elif eval_mode == EvalMode.HIGH_LEVEL:
        sys_prompt += f"""
            {examples.shot_learning.example1_hl_withFeedback1}

            ---

            {examples.shot_learning.example1_hl_withFeedback2}

            ---

            {examples.shot_learning.example1_hl_withFeedback3}

            ---

            {examples.shot_learning.example1_hl_withFeedback4}

            ---
        """

    if eval_mode == EvalMode.LOW_LEVEL:
        sys_prompt += f"""
            {examples.shot_learning.example1_ll_withFeedback1}

            ---

            {examples.shot_learning.example1_ll_withFeedback2}

            ---

            {examples.shot_learning.example1_ll_withFeedback3}

            ---

            {examples.shot_learning.example1_ll_withFeedback4}

            ---
        """


    assume_this_is_ok = ""
    additional_prompt = ""
    if eval_mode == EvalMode.ACTORS:
        if high_level_goals != None or low_level_goals != None:
            raise ValueError("EvalMode.ACTORS can only be used when high_level_goals and low_level_goals are both None.")
        provided_with = "a software description and the actors (end user roles) for said software"
        assume_this_is_ok = "Considering only the actors' names and not their descriptions,"
        critique_this = "defining actors"
    elif eval_mode == EvalMode.HIGH_LEVEL:
        if low_level_goals != None or high_level_goals == None:
            raise ValueError("EvalMode.HIGH_LEVEL can only be used when low_level_goals is None and high_level_goals is not None.")
        provided_with = "a software description, actors and high-level goals for said software. "
        assume_this_is_ok = "Assuming the work done on actors is ok,"
        critique_this = "defining high-level end users goals. Multiple actors can have the same goals (i.e., overlapping goals). You must ensure that ONLY functional goals are present."
        additional_prompt = f"""
        **High-level goals:**\n\n
        {high_level_goals}

        """
    elif eval_mode == EvalMode.LOW_LEVEL:
        if low_level_goals == None or high_level_goals == None:
            raise ValueError("EvalMode.LOW_LEVEL can only be used when both low_level_goals and high_level_goals are not None.")
        provided_with = "a software description, actors, high-level goals and low-level goals for said software"
        assume_this_is_ok = "Assuming the work done on actors and high-level goals is ok,"
        critique_this = "defining low-level end users goals.  Each low-level goal should theoretically correspond to a single action of the actor with the software. Multiple actors can have the same goals (i.e., overlapping goals). You must ensure that ONLY functional goals are present."
        additional_prompt =  f"""
        **High-level goals:**\n\n
        {high_level_goals}

        **Low-level goals:**\n\n
        {low_level_goals}

        """

    prompt = f"""
        You are provided with {provided_with}.\n
        These informations were extracted by another assistant from the software description.\n
        {assume_this_is_ok} your job is to critique the work done by the assistant on {critique_this}. 
        Give a score from 0 to 10 based on the critique you produced. 
        Assign a score of 0 if you see any contradiction or important omissions.
        Assign the maximum score if you don't see any error.
        The {provided_with}
        needs to be COMPLETELY representative of the software system described.
        As in the previous examples, decrease the score if
        you think that something may be missing, or if there is something in contrast with the system's description. 
        In that case, suggest what to do to improve the {provided_with}.
        Otherwise, just produce the score. 
        Just respond with a score and a feedback, like in this example:\n
        
        Feedback: [Feedback here]\n
        Score: [0.0-10.0]\n

        Do not add any other comments, just the above mentioned lines.\n

        ### Target Analysis
        **Description:** 
        {description}

        **Actors:**
        {actors}

        {additional_prompt}
        
        ---
        ### Final Evaluation
        Generate the feedback and the score now.
        
        Feedback: 
    """

    critique = generate_response_llama(prompt, sys_prompt)
    return critique


def _normalize_top_down_text(value) -> str:
    return " ".join(str(value).split())


def _shorten_top_down_field(value, limit: int) -> str:
    normalized = _normalize_top_down_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _bounded_top_down_description(description) -> str:
    """Bound only the critic input; generation still uses the full source."""
    normalized = _normalize_top_down_text(description)
    limit = TOP_DOWN_EVALUATOR_DESCRIPTION_MAX_CHARS
    if len(normalized) <= limit:
        return normalized

    marker = " ... [middle omitted for evaluator token budget] ... "
    head_size = int(limit * 0.7)
    tail_size = limit - head_size - len(marker)
    return normalized[:head_size] + marker + normalized[-tail_size:]


def _top_down_collection_items(value, attribute: str) -> list:
    items = getattr(value, attribute, None)
    if items is not None:
        return list(items)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _format_top_down_actors(actors) -> str:
    items = _top_down_collection_items(actors, "actors")
    if not items:
        return _normalize_top_down_text(actors)
    return "\n".join(
        f"- {actor.name}: "
        f"{_shorten_top_down_field(actor.description, TOP_DOWN_EVALUATOR_ACTOR_DESCRIPTION_MAX_CHARS)}"
        for actor in items
    )


def _format_top_down_hlgs(high_level_goals) -> str:
    items = _top_down_collection_items(high_level_goals, "goals")
    if not items:
        return _normalize_top_down_text(high_level_goals)
    return "\n".join(
        f"- [{goal.actor.name}] {goal.name}: "
        f"{_shorten_top_down_field(goal.description, TOP_DOWN_EVALUATOR_HLG_DESCRIPTION_MAX_CHARS)}"
        for goal in items
    )


def _top_down_llg_branch_blocks(low_level_goals) -> list[str]:
    """Serialize every parent once instead of repeating nested model reprs."""
    items = _top_down_collection_items(low_level_goals, "low_level_goals")
    if not items:
        return [_normalize_top_down_text(low_level_goals)]

    branches: dict[tuple[str, str], dict] = {}
    for goal in items:
        parent = goal.high_level_associated
        key = (parent.actor.name, parent.name)
        branch = branches.setdefault(
            key,
            {
                "actor": parent.actor.name,
                "name": parent.name,
                "goals": [],
            },
        )
        branch["goals"].append(
            f"  - {goal.name}: "
            f"{_shorten_top_down_field(goal.description, TOP_DOWN_EVALUATOR_LLG_DESCRIPTION_MAX_CHARS)}"
        )

    return [
        "\n".join(
            [
                f"Branch [{branch['actor']}] {branch['name']}",
                *branch["goals"],
            ]
        )
        for branch in branches.values()
    ]


def _top_down_examples(eval_mode: EvalMode) -> list[str]:
    if eval_mode == EvalMode.ACTORS:
        return [
            examples.shot_learning.example1_actors_withFeedback1,
            examples.shot_learning.example1_actors_withFeedback2,
            examples.shot_learning.example1_actors_withFeedback3,
            examples.shot_learning.example1_actors_withFeedback4,
        ]
    if eval_mode == EvalMode.HIGH_LEVEL:
        return [
            examples.shot_learning.example1_hl_withFeedback1,
            examples.shot_learning.example1_hl_withFeedback2,
            examples.shot_learning.example1_hl_withFeedback3,
            examples.shot_learning.example1_hl_withFeedback4,
        ]
    return [
        examples.shot_learning.example1_ll_withFeedback1,
        examples.shot_learning.example1_ll_withFeedback2,
        examples.shot_learning.example1_ll_withFeedback3,
        examples.shot_learning.example1_ll_withFeedback4,
    ]


def _build_top_down_system_prompt(
    eval_mode: EvalMode,
    example_count: int,
) -> str:
    prompt = (
        "You are a software-engineering evaluator specialized in "
        "Goal-Oriented Requirements Engineering (GORE). An actor is the "
        "'who'; a high-level goal is the strategic 'why'; a low-level goal "
        "is an atomic functional interaction describing 'how'. Evaluate only "
        "functional goals. Identify missing, contradictory or unsupported "
        "items and propose concrete corrections."
    )
    selected = _top_down_examples(eval_mode)[:example_count]
    if selected:
        prompt += "\n\n### Evaluation examples\n\n" + "\n\n---\n\n".join(
            selected
        )
    return prompt


def _top_down_evaluation_task(eval_mode: EvalMode) -> tuple[str, str, str]:
    if eval_mode == EvalMode.ACTORS:
        return (
            "a software description and extracted actors",
            "Considering actor names rather than wording details, evaluate "
            "whether the end-user roles are correct and complete.",
            "",
        )
    if eval_mode == EvalMode.HIGH_LEVEL:
        return (
            "a software description, actors and high-level goals",
            "Assume the actors are correct and evaluate the high-level goals.",
            "Multiple actors may share goals. Reject non-functional goals.",
        )
    return (
        "a software description, actors, high-level goals and low-level goals",
        "Assume actors and high-level goals are correct and evaluate the "
        "low-level goals.",
        "Each low-level goal should be an atomic actor-system interaction "
        "that could map to one API capability. Reject non-functional goals.",
    )


def _build_top_down_user_prompt(
    eval_mode: EvalMode,
    description: str,
    actors_text: str,
    high_level_text: str = "",
    low_level_text: str = "",
    batch_note: str = "",
) -> str:
    provided_with, task, extra_rule = _top_down_evaluation_task(eval_mode)
    sections = [
        f"You are given {provided_with}.",
        f"{task} {extra_rule}",
        "Check correctness, contradictions, unsupported items and important "
        "omissions. Score 0 for a major contradiction or omission and 10 when "
        "there is no relevant error. Suggest concrete corrections when needed.",
    ]
    if batch_note:
        sections.append(batch_note)
    sections.extend(
        [
            "Respond with exactly these two fields and no other text:\n"
            "Feedback: [feedback]\nScore: [0.0-10.0]",
            f"### Description\n{description}",
            f"### Actors\n{actors_text}",
        ]
    )
    if high_level_text:
        sections.append(f"### High-level goals\n{high_level_text}")
    if low_level_text:
        sections.append(f"### Low-level goals\n{low_level_text}")
    sections.append("### Final evaluation\nFeedback:")
    return "\n\n".join(sections)


def _call_budgeted_top_down_evaluator(
    eval_mode: EvalMode,
    user_prompt: str,
) -> str:
    """Use the largest few-shot prefix that remains under the input budget."""
    for example_count in range(4, -1, -1):
        system_prompt = _build_top_down_system_prompt(eval_mode, example_count)
        input_chars = len(system_prompt) + len(user_prompt)
        if input_chars <= TOP_DOWN_EVALUATOR_MAX_INPUT_CHARS:
            if example_count < 4:
                print(
                    "[top-down evaluator] token budget: using "
                    f"{example_count}/4 few-shot examples "
                    f"({input_chars} input chars)."
                )
            return generate_response_llama(user_prompt, system_prompt)
    raise ValueError(
        "Top-down evaluator target is too large even without few-shot "
        "examples; semantic batching is required."
    )


def _pack_top_down_blocks(
    blocks: list[str],
    character_budget: int,
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for block in blocks:
        block_size = len(block) + 2
        if current and current_size + block_size > character_budget:
            batches.append(current)
            current = []
            current_size = 0
        current.append(block)
        current_size += block_size
    if current:
        batches.append(current)
    return batches


def _evaluate_top_down_llg_batches(
    description: str,
    actors_text: str,
    high_level_goals,
    branch_blocks: list[str],
) -> str:
    """Split only oversized LLG evaluations, aligned to complete HLG branches."""
    hlg_index = "\n".join(
        f"- [{goal.actor.name}] {goal.name}"
        for goal in _top_down_collection_items(high_level_goals, "goals")
    )
    system_prompt = _build_top_down_system_prompt(EvalMode.LOW_LEVEL, 0)
    common_note = (
        "This is one semantic batch of a larger result. Evaluate only the "
        "listed branches. Do not penalize goals belonging to omitted branches; "
        "check completeness within every listed branch."
    )
    combined_prompt = _build_top_down_user_prompt(
        EvalMode.LOW_LEVEL,
        description,
        actors_text,
        hlg_index,
        "\n\n".join(branch_blocks),
        "The detailed low-level goals below form the complete result. The "
        "high-level section is a compact index used to avoid repeated text.",
    )
    if (
        len(system_prompt) + len(combined_prompt)
        <= TOP_DOWN_EVALUATOR_MAX_INPUT_CHARS
    ):
        print(
            "[top-down evaluator] using compact zero-shot LLG evaluation "
            f"({len(system_prompt) + len(combined_prompt)} input chars)."
        )
        return generate_response_llama(combined_prompt, system_prompt)

    empty_prompt = _build_top_down_user_prompt(
        EvalMode.LOW_LEVEL,
        description,
        actors_text,
        hlg_index,
        "",
        common_note,
    )
    branch_budget = max(
        2500,
        TOP_DOWN_EVALUATOR_MAX_INPUT_CHARS
        - len(system_prompt)
        - len(empty_prompt)
        - 500,
    )
    batches = _pack_top_down_blocks(branch_blocks, branch_budget)
    print(
        "[top-down evaluator] oversized low-level evaluation split into "
        f"{len(batches)} semantic batch(es)."
    )

    scores: list[float] = []
    feedback_parts: list[str] = []
    for batch_index, batch in enumerate(batches, start=1):
        batch_prompt = _build_top_down_user_prompt(
            EvalMode.LOW_LEVEL,
            description,
            actors_text,
            hlg_index,
            "\n\n".join(batch),
            f"Semantic batch {batch_index}/{len(batches)}. {common_note}",
        )
        response = generate_response_llama(batch_prompt, system_prompt)
        score, feedback = parse_evaluation(response)
        scores.append(score)
        feedback_parts.append(f"Batch {batch_index}: {feedback}")

    aggregate_score = min(scores)
    aggregate_feedback = " | ".join(feedback_parts)
    return f"Feedback: {aggregate_feedback}\nScore: {aggregate_score}"


def get_evaluation(
    eval_mode: EvalMode,
    description,
    actors,
    high_level_goals=None,
    low_level_goals=None,
):
    """Run the original baseline prompt with token-safe target serialization."""
    description_text = _bounded_top_down_description(description)
    actors_text = _format_top_down_actors(actors)
    high_level_text = (
        _format_top_down_hlgs(high_level_goals)
        if high_level_goals is not None
        else ""
    )
    branch_blocks: list[str] = []
    low_level_text = ""
    if low_level_goals is not None:
        branch_blocks = _top_down_llg_branch_blocks(low_level_goals)
        low_level_text = "\n\n".join(branch_blocks)

    # The instructions, evaluator examples and output contract remain exactly
    # those of the original baseline.  Only repeated nested model data is
    # serialized compactly before interpolation into that prompt.
    if eval_mode == EvalMode.ACTORS:
        return _original_get_evaluation(eval_mode, description_text, actors_text)
    if eval_mode == EvalMode.HIGH_LEVEL:
        return _original_get_evaluation(
            eval_mode,
            description_text,
            actors_text,
            high_level_text,
        )
    return _original_get_evaluation(
        eval_mode,
        description_text,
        actors_text,
        high_level_text,
        low_level_text,
    )


def parse_evaluation(evaluation: str|Critique):
    if type(evaluation) == str:
        normalized = evaluation.strip()
        score_matches = list(
            re.finditer(
                r"(?im)^\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
                normalized,
            )
        )
        feedback_match = re.search(
            r"(?is)Feedback\s*:\s*(.*?)(?=\n\s*Score\s*:)",
            normalized,
        )
        if not score_matches or feedback_match is None:
            raise ValueError(
                "Input text must contain Feedback: ... and a final Score: line."
            )
        score = float(score_matches[-1].group(1))
        feedback = " ".join(feedback_match.group(1).split())
        if not 0 <= score <= 10:
            raise ValueError("Evaluation score must be between 0 and 10.")

    elif type(evaluation) == Critique:
        score = evaluation.score
        feedback = evaluation.comment

    else:
        raise ValueError("Input evaluation is not in the expected format.")

    return score, feedback





def generate_response_with_reflection(target_type, call_function, define_args, eval_mode, eval_args, shotPromptingMode=ShotPromptingMode.ZERO_SHOT, max_attempts=MAX_ATTEMPTS, llama_ablation = False):
    feedback = None
    for attempt in range(1, max_attempts + 1):
        print(f"{target_type} STARTING... (attempt {attempt})")
        result = call_function(*define_args, feedback=feedback, mode = shotPromptingMode )
        print(f"{target_type} DONE...")
        print(result)

        if llama_ablation:
            return result, 10, None

        print(f"Evaluation for {target_type} STARTING...")
        evaluation = get_evaluation(eval_mode, *eval_args, result)
        print(f"Evaluation for {target_type} DONE...")

        try:
            score, critique = parse_evaluation(evaluation)
            print(f"Score: {score}")
            print(f"Critique: {critique}")

            #log this to check output
            #with open("output.txt", "a") as file:  # Use "w" to overwrite or "a" to append
            #   file.write(f"Critique: {critique}\nScore: {score}\nHLG: 

            if score >= QUALITY_THRESHOLD:
                print("Satisfactory score achieved! Breaking out of the loop.")
                return result, score, critique
            else:
                print("Unsatisfactory score. Retrying...")
                feedback = Feedback(previous_output=result, critique=critique)
        except ValueError as e:
            print(f"Error while parsing evaluation: {e}")
            sys.exit(1)  # Exit the program if parsing fails

    #raise RuntimeError("Failed to achieve a satisfactory score within the maximum number of attempts.")
    print("Failed to achieve a satisfactory score within the maximum number of attempts.")
    return result, score, critique


# =============================================================================
# BOTTOM-UP EXTENSION
#
# Everything above this marker is the original top-down implementation.  The
# bottom-up pipeline uses the explicitly prefixed symbols below, so its focused
# prompts, thresholds and parsing rules cannot alter the baseline route.
# =============================================================================

BOTTOM_UP_MAX_ATTEMPTS = 5
BOTTOM_UP_QUALITY_THRESHOLD = 8.0
BOTTOM_UP_EVALUATOR_FEW_SHOT_EXAMPLE_COUNT = None


class BottomUpEvalMode(Enum):
    ACTORS = "Actors"
    HIGH_LEVEL = "High Level Goals"
    LOW_LEVEL = "Low Level Goals"


class BottomUpFeedback:
    def __init__(self, previous_output, critique):
        self.previous_output = previous_output
        self.critique = critique


def _bottom_up_few_shot_examples(eval_mode: BottomUpEvalMode) -> list[str]:
    examples_by_mode = {
        BottomUpEvalMode.ACTORS: [
            examples.shot_learning.example1_actors_withFeedback1,
            examples.shot_learning.example1_actors_withFeedback2,
            examples.shot_learning.example1_actors_withFeedback3,
            examples.shot_learning.example1_actors_withFeedback4,
        ],
        BottomUpEvalMode.HIGH_LEVEL: [
            examples.shot_learning.example1_hl_withFeedback1,
            examples.shot_learning.example1_hl_withFeedback2,
            examples.shot_learning.example1_hl_withFeedback3,
            examples.shot_learning.example1_hl_withFeedback4,
        ],
        BottomUpEvalMode.LOW_LEVEL: [
            examples.shot_learning.example1_ll_withFeedback1,
            examples.shot_learning.example1_ll_withFeedback2,
            examples.shot_learning.example1_ll_withFeedback3,
            examples.shot_learning.example1_ll_withFeedback4,
        ],
    }
    example_count = BOTTOM_UP_EVALUATOR_FEW_SHOT_EXAMPLE_COUNT
    if example_count is None:
        example_count = int(os.getenv("EVALUATOR_FEW_SHOT_EXAMPLE_COUNT", "2"))
    if not 1 <= example_count <= 4:
        raise ValueError(
            "EVALUATOR_FEW_SHOT_EXAMPLE_COUNT must be between 1 and 4"
        )
    return examples_by_mode[eval_mode][:example_count]


def get_bottom_up_evaluation(
    eval_mode: BottomUpEvalMode,
    description,
    actors,
    high_level_goals=None,
    low_level_goals=None,
    focused_scope: str | None = None,
):
    if not isinstance(eval_mode, BottomUpEvalMode):
        raise TypeError(
            "Expected an instance of BottomUpEvalMode, "
            f"but got {type(eval_mode).__name__}"
        )
    sys_prompt = (
        "You're an helpful assistant, expert in the field of software engineering and specialised in the Goal-Oriented Requirements Engineering (GORE) framework.\n\n."
        "Following the Goal-Oriented Requirements Engineering (GORE) framework: "
        "-  an actor is active entity that has the capability to perform actions to achieve goals. Unlike goals, which are 'what' or 'why,'"
        " actors are the 'who.' \n"
        "-  high-level goals are strategic objectives that define the 'why' behind a system. \n"
        "   They are usually abstract, business-oriented, and independent of technical implementation. "
        "They represent the needs of stakeholders or the organization. \n"
        "   Focus: Vision and justification. \n"
        "-  low-level goals are technical objectives that describe 'how' the high-level goals will be achieved. \n"
        "   They are more concrete and are eventually refined into specific requirements or software specifications. \n"
        "   Focus: Implementation and constraints.\n"
        "You can propose new goals taking into account the already present ones. Consider that high-level "
        "goals often answer the WHY question, while low-level goals often address the HOW."
        "\n\n ### Examples:\n\n"
    )
    sys_prompt += "\n\n---\n\n".join(
        _bottom_up_few_shot_examples(eval_mode)
    )
    sys_prompt += "\n\n---\n"

    assume_this_is_ok = ""
    additional_prompt = ""
    if eval_mode == BottomUpEvalMode.ACTORS:
        if high_level_goals != None or low_level_goals != None:
            raise ValueError(
                "BottomUpEvalMode.ACTORS can only be used when high_level_goals "
                "and low_level_goals are both None."
            )
        provided_with = "a software description and the actors (end user roles) for said software"
        assume_this_is_ok = "Considering only the actors' names and not their descriptions,"
        critique_this = "defining actors"
    elif eval_mode == BottomUpEvalMode.HIGH_LEVEL:
        if low_level_goals != None or high_level_goals == None:
            raise ValueError(
                "BottomUpEvalMode.HIGH_LEVEL can only be used when "
                "low_level_goals is None and high_level_goals is not None."
            )
        provided_with = "a software description, actors and high-level goals for said software. "
        assume_this_is_ok = "Assuming the work done on actors is ok,"
        critique_this = "defining high-level end users goals. Multiple actors can have the same goals (i.e., overlapping goals). You must ensure that ONLY functional goals are present."
        additional_prompt = f"""
        **High-level goals:**\n\n
        {high_level_goals}

        """
    elif eval_mode == BottomUpEvalMode.LOW_LEVEL:
        if low_level_goals == None or high_level_goals == None:
            raise ValueError(
                "BottomUpEvalMode.LOW_LEVEL can only be used when both "
                "low_level_goals and high_level_goals are not None."
            )
        provided_with = "a software description, actors, high-level goals and low-level goals for said software"
        assume_this_is_ok = "Assuming the work done on actors and high-level goals is ok,"
        critique_this = "defining low-level end users goals.  Each low-level goal should theoretically correspond to a single action of the actor with the software. Multiple actors can have the same goals (i.e., overlapping goals). You must ensure that ONLY functional goals are present."
        additional_prompt = f"""
        **High-level goals:**\n\n
        {high_level_goals}

        **Low-level goals:**\n\n
        {low_level_goals}

        """

    if focused_scope:
        representation_requirement = f"""
        This evaluation concerns a focused correction, not a complete extraction
        of the software system. Evaluate exclusively the supplied actor, parent
        goal and functional intention. Do not penalize the answer for omitting
        other actors, sibling HLGs, or unrelated responsibilities of the same
        actor.

        Focused scope:
        {focused_scope}
        """
    else:
        representation_requirement = f"""
        The {provided_with} needs to be COMPLETELY representative of the
        software system described.
        """

    prompt = f"""
            You are provided with {provided_with}.

            These informations were extracted by another assistant from the software description.

            {assume_this_is_ok} your job is to critique the work done by the assistant on {critique_this}.
            Give a score from 0 to 10 based on the critique you produced.
            Assign a score of 0 if you see any contradiction or important omissions.
            Assign the maximum score if you don't see any error.
            {representation_requirement}
            As in the previous examples, decrease the score if
            you think that something may be missing, or if there is something in contrast with the system's description.
            In that case, suggest what to do to improve the {provided_with}.
            Otherwise, just produce the score.
            Just respond with a score and a feedback, like in this example:

            Feedback: [Feedback here]

            Score: [0.0-10.0]

            Do not add any other comments, just the above mentioned lines.

            ### Target Analysis
            **Description:**
            {description}

            **Actors:**
            {actors}

            {additional_prompt}

            ---
            ### Final Evaluation
            Generate the feedback and the score now.

            Feedback:
        """
    return generate_evaluator_response(prompt, sys_prompt)


def parse_bottom_up_evaluation(evaluation: str | Critique):
    if type(evaluation) == str:
        text = evaluation.strip()
        score_match = re.search(
            r"(?:^|\n)\s*Score\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
            text,
            flags=re.IGNORECASE,
        )
        feedback_match = re.search(
            r"^\s*Feedback\s*:\s*(.*?)\s*(?=\n\s*Score\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not score_match or not feedback_match:
            raise ValueError(
                "Input text must contain Feedback: ... and a final Score: line."
            )
        score = float(score_match.group(1))
        if not 0 <= score <= 10:
            raise ValueError("Evaluation score must be between 0 and 10.")
        feedback = feedback_match.group(1).strip()
    elif type(evaluation) == Critique:
        score = evaluation.score
        feedback = evaluation.comment
    else:
        raise ValueError("Input evaluation is not in the expected format.")
    return score, feedback


def generate_bottom_up_response_with_reflection(
    target_type,
    call_function,
    define_args,
    eval_mode,
    eval_args,
    shotPromptingMode=ShotPromptingMode.ZERO_SHOT,
    max_attempts=BOTTOM_UP_MAX_ATTEMPTS,
    llama_ablation=False,
    focused_scope: str | None = None,
):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    feedback = None
    score = None
    critique = None
    parse_error = None
    for attempt in range(1, max_attempts + 1):
        print(f"{target_type} STARTING... (attempt {attempt})")
        result = call_function(
            *define_args,
            feedback=feedback,
            mode=shotPromptingMode,
        )
        print(f"{target_type} DONE...")
        print(result)
        if llama_ablation:
            return result, 10, None
        print(f"Evaluation for {target_type} STARTING...")
        evaluation = get_bottom_up_evaluation(
            eval_mode,
            *eval_args,
            result,
            focused_scope=focused_scope,
        )
        print(f"Evaluation for {target_type} DONE...")
        try:
            score, critique = parse_bottom_up_evaluation(evaluation)
            print(f"Score: {score}")
            print(f"Critique: {critique}")
            if score >= BOTTOM_UP_QUALITY_THRESHOLD:
                print("Satisfactory score achieved! Breaking out of the loop.")
                return result, score, critique
            print("Unsatisfactory score. Retrying...")
            feedback = BottomUpFeedback(
                previous_output=result,
                critique=critique,
            )
        except ValueError as exc:
            print(f"Error while parsing evaluation: {exc}")
            parse_error = exc
            feedback = BottomUpFeedback(
                previous_output=result,
                critique=(
                    "The evaluator response was malformed. Return exactly "
                    "'Feedback: ...' followed by 'Score: 0-10'."
                ),
            )
            critique = str(exc)
    if score is None and parse_error is not None:
        raise RuntimeError(
            "Evaluator returned malformed output after "
            f"{max_attempts} attempts: {parse_error}"
        ) from parse_error
    print("Failed to achieve a satisfactory score within the maximum number of attempts.")
    return result, score, critique
