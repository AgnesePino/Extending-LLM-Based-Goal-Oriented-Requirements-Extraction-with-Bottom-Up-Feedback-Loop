import os
import json
import re
from enum import Enum


from src import examples
import src.examples.shot_learning
from src.data_model import Critique
from  src.examples.shot_learning import ShotPromptingMode
from src.llm_clients import (
    configured_critic_model,
    generate_evaluator_response,
    generate_response_llama,
    generate_response,
    loaded_local_evaluator_model,
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
TOP_DOWN_EVALUATION_PARSE_ATTEMPTS = int(
    os.getenv("TOP_DOWN_EVALUATION_PARSE_ATTEMPTS", "3")
)

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
        normalized = re.sub(
            r"<think>.*?</think>", "", evaluation, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        normalized = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", normalized, flags=re.I)
        normalized = normalized.replace("**", "").strip()

        if normalized.startswith("{"):
            try:
                payload = json.loads(normalized)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                lowered = {str(key).lower(): value for key, value in payload.items()}
                if "score" in lowered and ("feedback" in lowered or "critique" in lowered):
                    score = float(lowered["score"])
                    feedback = str(
                        lowered.get("feedback", lowered.get("critique", ""))
                    ).strip()
                    if not 0 <= score <= 10:
                        raise ValueError("Evaluation score must be between 0 and 10.")
                    return score, feedback

        score_matches = list(
            re.finditer(
                r"(?i)\bScore\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*10)?",
                normalized,
            )
        )
        if not score_matches:
            raise ValueError(
                "Input text must contain Feedback: ... and a final Score: line."
            )
        final_score_match = score_matches[-1]
        feedback_matches = list(
            re.finditer(
                r"(?i)\bFeedback\s*:\s*",
                normalized[: final_score_match.start()],
            )
        )
        if not feedback_matches:
            raise ValueError(
                "Input text must contain Feedback: ... and a final Score: line."
            )
        score = float(final_score_match.group(1))
        feedback = " ".join(
            normalized[
                feedback_matches[-1].end() : final_score_match.start()
            ].split()
        ).strip("#>*- ")
        if not 0 <= score <= 10:
            raise ValueError("Evaluation score must be between 0 and 10.")

    elif type(evaluation) == Critique:
        score = evaluation.score
        feedback = evaluation.comment

    else:
        raise ValueError("Input evaluation is not in the expected format.")

    return score, feedback

def _compact_result_summary(result) -> str:
    """Keep notebook logs readable when a structured result is very large."""
    for field_name in ("actors", "goals", "low_level_goals"):
        items = getattr(result, field_name, None)
        if isinstance(items, list):
            return f"{type(result).__name__}: {len(items)} {field_name}"
    text = str(result)
    if len(text) > 500:
        text = text[:497] + "..."
    return text


def generate_response_with_reflection(target_type, call_function, define_args, eval_mode, eval_args, shotPromptingMode=ShotPromptingMode.ZERO_SHOT, max_attempts=MAX_ATTEMPTS, llama_ablation = False):
    feedback = None
    score = None
    critique = None
    for attempt in range(1, max_attempts + 1):
        print(f"{target_type} STARTING... (attempt {attempt})")
        result = call_function(*define_args, feedback=feedback, mode = shotPromptingMode )
        print(f"{target_type} DONE...")
        print(_compact_result_summary(result))

        if llama_ablation:
            return result, 10, None

        evaluation_parse_error = None
        for evaluation_attempt in range(1, TOP_DOWN_EVALUATION_PARSE_ATTEMPTS + 1):
            print(
                f"Evaluation for {target_type} STARTING... "
                f"(format attempt {evaluation_attempt})"
            )
            evaluation = get_evaluation(eval_mode, *eval_args, result)
            print(f"Evaluation for {target_type} DONE...")

            try:
                score, critique = parse_evaluation(evaluation)
                evaluation_parse_error = None
                break
            except (TypeError, ValueError) as exc:
                evaluation_parse_error = exc
                preview = " ".join(str(evaluation).split())
                if len(preview) > 500:
                    preview = preview[:497] + "..."
                print(f"Invalid evaluator format: {exc}")
                print(f"Evaluator output preview: {preview}")

        if evaluation_parse_error is not None:
            raise RuntimeError(
                f"The {target_type} critic returned malformed output after "
                f"{TOP_DOWN_EVALUATION_PARSE_ATTEMPTS} attempts."
            ) from evaluation_parse_error

        if score is not None:
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

# Paper-aligned generation reflection: initial candidate plus two refinements.
BOTTOM_UP_MAX_ATTEMPTS = 3
BOTTOM_UP_QUALITY_THRESHOLD = 8.5


class RepairCandidateRejected(RuntimeError):
    """The semantic generation budget was exhausted; the cycle may retry later."""
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
        example_count = int(os.getenv("EVALUATOR_FEW_SHOT_EXAMPLE_COUNT", "4"))
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
    format_feedback: str | None = None,
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
    sys_prompt += (
        "\nFor this bottom-up repair, the parent actor is the source-grounded "
        "stakeholder responsible for or benefiting from the outcome. Documented "
        "automatic notifications or system operations do not require a manual "
        "trigger by that actor. Judge documentary support and responsibility; "
        "do not invent UI actions or reject automation solely for lacking one. "
        "For a consolidation, assess the complete requested functional outcome "
        "and preserve supported capabilities of every supplied member.\n"
    )

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
    if format_feedback:
        prompt += "\nFORMAT CORRECTION FOR THIS SAME CANDIDATE:\n" + format_feedback
    critic_model = configured_critic_model()
    with loaded_local_evaluator_model(critic_model):
        return generate_evaluator_response(
            prompt,
            sys_prompt,
            model=critic_model,
        )


def parse_bottom_up_evaluation(evaluation: str | Critique):
    """Accept explicit Feedback/Score or JSON without inventing a missing score."""
    score, feedback = parse_evaluation(evaluation)
    if not 0 <= score <= 10:
        raise ValueError("Evaluation score must be between 0 and 10.")
    return score, feedback


def generate_bottom_up_response_with_reflection(
    target_type, call_function, define_args, eval_mode, eval_args,
    shotPromptingMode=ShotPromptingMode.ZERO_SHOT,
    max_attempts=BOTTOM_UP_MAX_ATTEMPTS, llama_ablation=False,
    focused_scope: str | None = None,
):
    if max_attempts < 1 or TOP_DOWN_EVALUATION_PARSE_ATTEMPTS < 1:
        raise ValueError("Generation and format attempt budgets must be positive")
    feedback = None
    score = critique = None
    for attempt in range(1, max_attempts + 1):
        print(f"{target_type} STARTING... (attempt {attempt})")
        result = call_function(*define_args, feedback=feedback, mode=shotPromptingMode)
        print(f"{target_type} DONE...")
        print(_compact_result_summary(result))
        if llama_ablation:
            return result, 10, None
        score = None
        format_feedback = None
        last_parse_error = None
        # A formatting problem belongs to the critic, not to Qwen. Retry the
        # identical candidate while Ministral stays loaded for the format loop.
        with loaded_local_evaluator_model(configured_critic_model()):
            for format_attempt in range(1, TOP_DOWN_EVALUATION_PARSE_ATTEMPTS + 1):
                print(f"Evaluation for {target_type} STARTING... (format attempt {format_attempt})")
                evaluation = get_bottom_up_evaluation(eval_mode, *eval_args, result,
                    focused_scope=focused_scope, format_feedback=format_feedback)
                try:
                    score, critique = parse_bottom_up_evaluation(evaluation)
                    break
                except ValueError as exc:
                    last_parse_error = exc
                    format_feedback = (f"Your previous response could not be parsed: {exc}. "
                        "Evaluate the SAME candidate. Return Feedback: ... and Score: 0-10. "
                        "Do not change a score merely to pass parsing.\nPrevious response:\n" + str(evaluation))
        if score is None:
            raise RuntimeError(f"Evaluator returned malformed output after {TOP_DOWN_EVALUATION_PARSE_ATTEMPTS} format attempts for the same candidate: {last_parse_error}") from last_parse_error
        print(f"Score: {score}\nCritique: {critique}")
        if score >= BOTTOM_UP_QUALITY_THRESHOLD:
            return result, score, critique
        feedback = BottomUpFeedback(previous_output=result, critique=critique)
    raise RepairCandidateRejected(f"Top-down repair critic did not approve the candidate after {max_attempts} generation attempts (score={score}): {critique}")
