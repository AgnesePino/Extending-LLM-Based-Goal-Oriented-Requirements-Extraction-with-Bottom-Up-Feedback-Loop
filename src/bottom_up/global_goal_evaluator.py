"""One joint global review per critic of goals, numerical matches and project context."""

import json
import os
from collections import Counter
from contextlib import nullcontext


from src.bottom_up.goal_similarity import analyze_similarities, build_cosine_payload, build_review_payload
from src.data_model import (
    Actors, HighLevelGoals, indexed_high_level_goals,
    EvaluatorVoteRecord,
    GlobalGoalEvaluationResult,
    HighLevelGoalDecision,
    DocumentaryDecision,
)
from src.llm_clients import (
    configured_bottom_up_critic_models,
    generate_evaluator_response,
    loaded_local_evaluator_model,
    LM_STUDIO_EVALUATOR_CONTEXT_LENGTH,
)

# Configuration

MAX_EVALUATOR_ATTEMPTS = 3

PIPELINE_POLICY_VERSION = "cosine_global_documentary_review_v105_restored_v103_review_prompt_schema3"
BOTTOM_UP_EVALUATOR_MODELS = configured_bottom_up_critic_models()

if len(set(BOTTOM_UP_EVALUATOR_MODELS)) != 3 or not all(BOTTOM_UP_EVALUATOR_MODELS):
    raise ValueError("The ensemble requires three different non-empty models")
EVALUATOR_VOTING_QUORUM = 2
BOTTOM_UP_EVALUATOR_TEMPERATURE = float(
    os.getenv("BOTTOM_UP_EVALUATOR_TEMPERATURE", "0.2")
)
BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS = int(
    os.getenv("BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS", str(MAX_EVALUATOR_ATTEMPTS))
)

if not 1 <= BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS <= MAX_EVALUATOR_ATTEMPTS:
    raise ValueError("Evaluator attempts must be between 1 and 3")
BOTTOM_UP_EVALUATOR_PROMPTING_MODE = "FEW_SHOT"
CRITIC1_FEW_SHOT_EXAMPLE_COUNT = 4
CRITIQUE_CHARACTER_TARGET = 500

SYSTEM_PROMPT = (
    "You are an expert in software engineering and Goal-Oriented Requirements Engineering (GORE). "
    "Start from each reconstructed HLG and its software-supplied similarity labels. "
    "Accept those labels as the completed similarity assessment; do not reassess or relabel it. "
    "Then use project description and actors to interpret the reconstructed and original HLG texts "
    "and identify any concrete correction. The description states requirements; "
    "generated goals reveal scope, consistency and overlap, not additional requirements. "
    "Return only the requested decision JSON. Do not invent functionality."
)


# Four source/branch/response examples, shared by all three evaluators.
FEW_SHOT_EXAMPLES = [{'source': 'Members reserve tools and cancel their reservations.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': 'Reserve tools',
                'hlg_prime': 'Request tool reservations',
                'relationship': 'EQUIVALENT'},
               {'branch_id': 'b',
                'hlg': 'Cancel tool reservations',
                'hlg_prime': 'View the reservation selected for cancellation',
                'relationship': 'NOT_EQUIVALENT'}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL'},
                             {'branch_id': 'b',
                              'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL'}]}},
 {'source': 'Members edit only their own profile.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': 'Edit all member profiles',
                'hlg_prime': 'Edit my profile',
                'relationship': 'NOT_EQUIVALENT',
                'original_hlg_comparisons_complete': True}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              
                              'feedback': {'critique': "For the goal 'Edit all member profiles': Own-profile "
                                                       'editing corrects the existing scope. Limit editing '
                                                       "to the member's own profile."}}]}},
 {'source': 'Members reserve tools; auctions and tool sales are excluded.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': 'Auction tools',
                'hlg_prime': 'Run tool auctions',
                'relationship': 'EQUIVALENT'},
               {'branch_id': 'b',
                'hlg': 'Reserve and auction tools',
                'hlg_prime': 'Manage tool reservations and auctions',
                'relationship': 'EQUIVALENT',
                'original_hlg_comparisons_complete': True},
               {'branch_id': 'c',
                'hlg': 'Auction tools',
                'hlg_prime': 'Sell tools',
                'relationship': 'NOT_EQUIVALENT'}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'REMOVE_ORIGINAL_HIGH_LEVEL_GOAL'},
                             {'branch_id': 'b',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              
                              'feedback': {'critique': "For the goal 'Reserve and auction tools': Despite "
                                                       'numerical equivalence, only reservations are '
                                                       'documented; auctions are excluded from both goals. '
                                                       'Preserve tool reservations and remove the '
                                                       'unsupported auction capability.'}},
                             {'branch_id': 'c',
                              'decision': 'REMOVE_ORIGINAL_HIGH_LEVEL_GOAL'}]}},
 {'source': 'Members reserve tools.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': 'Reserve tools',
                'hlg_prime': 'Request tool reservations',
                'relationship': 'EQUIVALENT',
                'duplicate_candidates': ['b'], 'duplicate_label': 'POTENTIAL_DUPLICATE'},
               {'branch_id': 'b',
                'hlg': 'Request tool reservations',
                'hlg_prime': 'Reserve tools',
                'relationship': 'EQUIVALENT',
                'duplicate_candidates': ['a'], 'duplicate_label': 'POTENTIAL_DUPLICATE'}],
  'duplicate_candidates': [{'branch_id': 'a', 'other_branch_id': 'b'}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL'},
                             {'branch_id': 'b',
                              'decision': 'DUPLICATE_REMOVAL',
                              
                              'surviving_branch_id': 'a'}]}}]


def evaluator_examples(count=None):
    """Nested example sets, independent of evaluation ground truths."""
    count = CRITIC1_FEW_SHOT_EXAMPLE_COUNT if count is None else count
    if type(count) is not int or count not in (0, 1, 2, 4):
        raise ValueError("Evaluator example count must be 0, 1, 2 or 4 (maximum 4)")
    return FEW_SHOT_EXAMPLES[:count]


def _operation_key(vote):
    return f"{vote.decision.value}|survivor={vote.surviving_branch_id or ''}"


def _merge_agreeing_judgments(records, operation_key):
    """Join only the feedback prose of the agreeing evaluators for generation."""
    votes = [vote for vote in records if _operation_key(vote) == operation_key]
    if len(votes) < EVALUATOR_VOTING_QUORUM:
        raise ValueError("Merged feedback requires at least two agreeing valid votes")

    def join_text(parts):
        return "\n\n".join(dict.fromkeys(part.strip() for part in parts if part and part.strip()))

    updates = {}
    if votes[0].feedback:
        updates["feedback"] = {"critique": join_text(vote.feedback.critique for vote in votes)}
    return DocumentaryDecision.model_validate({**votes[0].model_dump(), **updates})


def _documentary_response_schema(branches, analysis):
    """Constrain each duplicate operation to its own direct candidates."""
    variants = []
    actions = ["KEEP_ORIGINAL_HIGH_LEVEL_GOAL", "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
               "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL", "DUPLICATE_REMOVAL"]
    eligible_ids = [b for b in branches
                    if analysis["branches"][b]["status"] == "VALID"]
    for action in actions:
        groups = {}
        for branch in branches:
            record = analysis["branches"][branch]
            candidates = tuple(record.get("duplicate_candidates", []))
            probe = dict(branch_id=branch, decision=action)
            if action == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
                probe["feedback"] = {"critique": "Correct the documented scope."}
            if action == "DUPLICATE_REMOVAL":
                if not candidates:
                    continue
                probe["surviving_branch_id"] = candidates[0]
            try:
                _validate_decision(probe, branch, analysis)
            except ValueError:
                continue
            key = candidates if action == "DUPLICATE_REMOVAL" else ()
            groups.setdefault(key, []).append(branch)
        for candidates, ids in groups.items():
            properties = {
                "branch_id": {"type": "string", "enum": ids},
                "decision": {"type": "string", "enum": [action]},
            }
            if action == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
                properties["feedback"] = {
                    "type": "object", "properties": {"critique": {"type": "string", "minLength": 1}},
                    "required": ["critique"], "additionalProperties": False}
            if action == "DUPLICATE_REMOVAL":
                properties["surviving_branch_id"] = {"type": "string", "enum": list(candidates)}
            variants.append({"type": "object", "properties": properties,
                             "required": list(properties), "additionalProperties": False})
    return {
        "type": "object", "properties": {"decisions": {
            "type": "array",
            "items": {"anyOf": variants} if variants else {"type": "object", "additionalProperties": False},
            "minItems": len(eligible_ids), "maxItems": len(eligible_ids)}},
        "required": ["decisions"], "additionalProperties": False,
    }


def _validate_decision(item, branch, analysis):
    vote = DocumentaryDecision.model_validate(item)
    if vote.branch_id != branch:
        raise ValueError("Wrong branch_id")
    record = analysis["branches"][branch]
    if record["status"] != "VALID":
        raise ValueError("Similarity unavailable; branch must remain inconclusive")
    if vote.decision == "DUPLICATE_REMOVAL":
        if vote.surviving_branch_id not in record["duplicate_candidates"]:
            raise ValueError("Survivor must be a direct same-actor text-review candidate")
    elif vote.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
        if not record["reconstructable"] or record["relationship"] not in {"EQUIVALENT", "NOT_EQUIVALENT"}:
            raise ValueError("REWRITE requires a reconstructable HLG prime and a valid numerical relationship")
        if record.get("original_hlg_comparisons_complete") is False:
            raise ValueError("Cannot choose REWRITE before completing available reconstructed-HLG comparisons")
    elif vote.decision == "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL":
        if record["relationship"] not in {"EQUIVALENT", "NOT_EQUIVALENT"}:
            raise ValueError("REMOVE requires a valid numerical relationship")
    return vote


# Joint global review prompt


GLOBAL_REVIEW_INSTRUCTIONS = """A. ROLE AND OBJECTIVE
Review the whole hierarchy, starting from each reconstructed HLG (prime), which
summarizes its LLGs. Prime is diagnostic, not an approved parent replacement.
Check original HLG scope, formulation, missing documented capabilities, actor
responsibility and redundancy against project documentation. Other generated
goals provide context, not requirements. Wording, abstraction or composite scope
alone is not a defect. LLG texts are absent: do not audit children or synthesis fidelity.

B. FIXED SEMANTIC LABELS
EQUIVALENT: classified semantic alignment, not proof of identical coverage,
correctness or documentary support.
NOT_EQUIVALENT: classified non-alignment, not proof of a parent defect.
POTENTIAL_DUPLICATE: a screened cross-branch candidate, not confirmed redundancy.
NO_DUPLICATE_MATCH: no screened candidate, not proof of independent functionality.
Accept these labels. Never recalculate cosine similarity, relabel, or invent
scores/thresholds. Coverage differences may be identified without changing labels.

C. LABEL-GUIDED REVIEW
Case 1: EQUIVALENT
Check the shared intention, parent's formulation/boundaries and actor against source.
KEEP a valid supported parent. REWRITE a documented defect, even if shared by prime.
REMOVE only if the shared outcome is unsupported, no supported parent outcome
remains, and removing its children loses no required capability. Partial support
requires correction. Equivalence alone never justifies KEEP.

Case 2: NOT_EQUIVALENT
Identify the difference: documented parent defect/missing capability, legitimate
abstraction/scope, or narrower/incomplete prime. KEEP a valid supported parent;
REWRITE only a concrete documented parent defect. Non-equivalence alone never
justifies REWRITE. REMOVE only if neither parent nor prime supplies a supported
outcome to preserve and no required child capability is lost. Preserve supported
scope; do not add unsupported capabilities, absorb independent goals or request
corrections already present in the parent's name or description.

Case 3: POTENTIAL_DUPLICATE (additional review)
Apply alongside either parent label. Compare BOTH parents, current and relevant
same-actor primes, source and responsibilities. DUPLICATE_REMOVAL requires a direct
same-actor candidate's parent to cover ALL supported capabilities of this branch.
Shared terms or partial overlap are insufficient; preserve complementary outcomes.
Choose the most complete valid parent; equivalent ties follow branch order.
Survivor must receive KEEP, never REWRITE/removal. No chains, cycles or child transfers.

D. GLOBAL DECISION CONSISTENCY
Return one globally consistent set: exactly one action per requested branch,
with duplicate survivors kept in the same response. Do not split contexts by label
or force duplicates without full coverage. Every correction requires source evidence.
REWRITE requires a valid reconstruction and complete required comparisons.

E. OUTPUT REQUIREMENTS
Return only {"decisions":[{"branch_id":"...","decision":"..."}]}.
Each requested branch_id once; schema fields only. Be concise and avoid repetition.
Aim for at most {critique_characters} characters per feedback.critique, including spaces and punctuation
in the decoded text value (not JSON syntax or escaping). This is a brevity target,
not an exact required length. Preserve actionable feedback.
Use these exact decision values (the short names above are explanatory only):
KEEP_ORIGINAL_HIGH_LEVEL_GOAL
REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
DUPLICATE_REMOVAL
Each entry: branch_id and decision. Do not return rationale or explanations.
No extra actions.
REWRITE alone adds feedback: {"critique":"..."}: name the goal, what to
preserve/change, boundaries and actor responsibility. Only critique prose reaches
the generator; agreeing critiques are joined. No IDs, scores or JSON in critique.
DUPLICATE_REMOVAL alone adds surviving_branch_id from direct candidates.
No abstention, intermediate analysis, replacement collections or separate LLG repair.
""".replace("{critique_characters}", str(CRITIQUE_CHARACTER_TARGET))


def _critic1_prompt(
    description,
    actors,
    branches,
    reconstructions,
    analysis,
    cosine_payload=None,
    few_shot_example_count=None,
):
    payload = (
        cosine_payload
        if cosine_payload is not None
        else build_cosine_payload(branches, reconstructions, analysis)
    )
    payload = build_review_payload(payload)
    selected_examples = evaluator_examples(few_shot_example_count)
    examples = "\n\n".join(
        f"Example {i}\nDescription: {e['source']}\n"
        f"Actor: {e['actor']}\nGoals: {json.dumps(e['branches'], separators=(',', ':'))}\n"
        f"Output: {json.dumps(e['response'], separators=(',', ':'))}"
        for i, e in enumerate(selected_examples, 1)
    )
    example_section = f"EXAMPLES (fictional requirements, not project evidence)\n{examples}\n\n" if examples else ""
    return f"""{GLOBAL_REVIEW_INSTRUCTIONS}

PAYLOAD FORMAT
Goal list order defines zero-based matrix indices. original_labels[i][j] compares
reconstructed HLG i with current original HLG j, across ALL actors.
Diagonal labels are EQUIVALENT or NOT_EQUIVALENT for the own-parent comparison.
Off-diagonal labels are POTENTIAL_DUPLICATE or NO_DUPLICATE_MATCH and incorporate
same-actor eligibility. NO_DUPLICATE_MATCH means no candidate identified by the
fixed screening rule, not proof that the texts are functionally different.
Null cells have no available/eligible judgement; consult comparison_statuses.
The matrix is directional, not necessarily symmetric. relationship repeats the
diagonal own-parent label.
duplicate_label is POTENTIAL_DUPLICATE or NO_DUPLICATE_MATCH for valid goals.
duplicate_candidates excludes self and restricts potential survivors to the same
actor and admissible comparisons. These labels do not establish full coverage.
No cosine values or numerical threshold are supplied. comparison_statuses contains
[row,column,status] for unavailable comparisons. Confirm complete supported
coverage from BOTH parents, primes and source before duplicate removal.
This request supplies {len(selected_examples)} fictional examples; use them only
as decision demonstrations, not evidence. Keep rewrite critiques actionable;
avoid unnecessary explanations or repeated details.

{example_section}PROJECT DESCRIPTION
{description}
ACTORS
{actors.model_dump_json()}
Similarity labels JSON:
{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}

"""



# Response parsing and retries for one evaluator


def _is_context_length_error(error):
    message = str(error).lower()
    return (
        "n_keep:" in message and "n_ctx:" in message
        or ("context length" in message or "context window" in message)
        and any(term in message for term in ("exceed", "greater than", "too long", "maximum"))
    )


def _json_object(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"decisions"}
        or not isinstance(payload["decisions"], list)
    ):
        raise ValueError('Return only {"decisions": [...]}')
    return payload


def _critic1_vote(prompt, model, branches, analysis, *, temperature=None, manage_model=False):
    """Share one per-iteration budget across load, context and output failures."""
    temperature = BOTTOM_UP_EVALUATOR_TEMPERATURE if temperature is None else temperature
    eligible = [b for b in branches if analysis["branches"][b]["status"] == "VALID"]
    expected = {b for b in eligible}
    schema = _documentary_response_schema(eligible, analysis)
    history, accepted, errors = [], {}, {}
    raw, payload, retry = None, None, prompt
    requested_context = LM_STUDIO_EVALUATOR_CONTEXT_LENGTH
    for attempt in range(1, BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS + 1):
        if not eligible:
            break
        print(f"[bottom-up] global review | model={model} | attempt={attempt}/"
              f"{BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS}", flush=True)
        current_errors, generation, candidate = {}, {}, {}
        raw, payload, context_error = None, None, False
        stage = "loading"
        try:
            options = ({"context_length": requested_context}
                       if requested_context != LM_STUDIO_EVALUATOR_CONTEXT_LENGTH else {})
            lease = loaded_local_evaluator_model(model, **options) if manage_model else nullcontext()
            with lease:
                stage = "generation"
                raw = generate_evaluator_response(retry, SYSTEM_PROMPT,
                    temperature=temperature, model=model, response_schema=schema)
                generation = dict(getattr(raw, "metadata", {}))
                stage = "release"
            stage = "validation"
            # Also reject truncation if a custom client returns it instead of raising.
            if generation.get("finish_reason", "stop") != "stop":
                raise ValueError("Incomplete global response; regenerate complete concise JSON")
            payload = _json_object(raw)
            items = payload["decisions"]
            if any(not isinstance(item, dict) for item in items):
                raise ValueError("Every decision must be an object")
            counts = Counter(item.get("branch_id") for item in items
                             if isinstance(item.get("branch_id"), str))
            if set(counts) != expected or any(n != 1 for n in counts.values()) or len(items) != len(expected):
                raise ValueError("Return every requested branch exactly once; no extra or missing IDs")
            by_id = {item["branch_id"]: item for item in items}
            for branch in eligible:
                vote = _validate_decision(by_id[branch], branch, analysis)
                candidate[branch] = vote
        except Exception as exc:
            raw = getattr(exc, "raw_response", raw)
            generation = dict(getattr(exc, "metadata", generation))
            current_errors["response"] = f"{type(exc).__name__}: {exc}"
            context_error = _is_context_length_error(exc)
            if context_error:
                generation.update(error_type="CONTEXT_LENGTH_EXCEEDED", recovery_required=True)
            elif stage in {"loading", "release"}:
                generation.update(error_type="MODEL_LOAD_FAILED")
        success = not current_errors and len(candidate) == len(expected)
        if raw is not None:
            raw = str(raw)  # Persist plain text, not the metadata-bearing str subclass.
        generation.update(accepted=success, global_retry_policy="discard_entire_invalid_response",
                          recovery_requested_context_length=requested_context,
                          attempt_budget_scope="per_model_per_iteration")
        history.append(dict(attempt=attempt, errors=current_errors,
                            valid_branch_ids=list(candidate) if success else [],
                            generation=generation, requested_branch_ids=[b for b in eligible]))
        errors = current_errors
        if success:
            accepted = candidate
            break
        print(f"[bottom-up] model={model} | global response rejected: {current_errors}", flush=True)
        if context_error:
            requested_context = max(requested_context, generation.get("loaded_context_length", 0) or 0) * 2
            retry = prompt
            continue
        if stage in {"loading", "release"}:
            continue
        reason = current_errors.get("response", "Invalid global response")[:350]
        retry = (prompt + "\nCORRECTION: The previous global response was discarded. "
                 "Regenerate ALL requested decisions as one complete concise JSON. "
                 "Keep the explanation focused and the critique actionable. Error: " + reason)
    global_record = EvaluatorVoteRecord(
        model=model, status="VALID" if accepted else "INVALID",
        decisions=list(accepted.values()), error=errors.get("response"), attempts=history)
    return global_record, accepted


# Ensemble aggregation


def _inconclusive(branch, message, *, status="INCONCLUSIVE"):
    return GlobalGoalEvaluationResult(
        branch_id=branch, decision=None, diagnostic=message, evaluation_status=status)


def evaluate_all_branches(
    project_description,
    branches,
    reconstructions,
    on_evaluation=None,
    *,
    actors=None,
    on_global_vote=None,
    similarity_analysis=None,
    cosine_payload=None,
    few_shot_example_count=None,
    evaluator_temperature=None,
):
    if isinstance(branches, HighLevelGoals):
        branches = indexed_high_level_goals(branches)
    if not branches:
        return {}
    analysis = similarity_analysis if similarity_analysis is not None else analyze_similarities(branches, reconstructions)
    if all(analysis["branches"][b]["status"] == "ERROR" for b in branches):
        results = {b: _inconclusive(
            b, "Similarity unavailable: " + analysis["branches"][b]["error"],
            status="SIMILARITY_UNAVAILABLE") for b in branches}
        if on_evaluation:
            on_evaluation(len(results), len(branches), results)
        return results
    actors = actors or Actors(actors=list({branches[b].actor.name: branches[b].actor for b in branches}.values()))
    prompt = _critic1_prompt(project_description, actors, branches, reconstructions, analysis, cosine_payload,
                            few_shot_example_count=few_shot_example_count)
    records = {b: [] for b in branches}
    global_records = []
    for model in BOTTOM_UP_EVALUATOR_MODELS:
        global_record, votes = _critic1_vote(
            prompt, model, branches, analysis,
            temperature=evaluator_temperature, manage_model=True)
        global_records.append(global_record)
        if on_global_vote:
            on_global_vote(global_record)
        for branch in branches:
            if branch in votes:
                records[branch].append(votes[branch])
    results = {}
    for branch in branches:
        local = records[branch]
        counts = Counter(_operation_key(vote) for vote in local)
        match = analysis["branches"][branch]
        if not counts or counts.most_common(1)[0][1] < EVALUATOR_VOTING_QUORUM:
            result = _inconclusive(branch, "No two valid documentary actions agree.")
            if any(attempt.get("generation", {}).get("error_type") == "CONTEXT_LENGTH_EXCEEDED"
                   for record in global_records for attempt in record.attempts):
                result = _inconclusive(branch,
                    "Evaluator context exceeded; insufficient agreeing valid votes. No automatic action.",
                    status="CONTEXT_LENGTH_EXCEEDED")
        else:
            key, _ = counts.most_common(1)[0]
            vote = _merge_agreeing_judgments(local, key)
            if match["status"] != "VALID":
                result = _inconclusive(branch, "Similarity comparison unavailable.")
            else:
                result = GlobalGoalEvaluationResult(
                    **vote.model_dump(), evaluation_status="DECIDED")
        results[branch] = result
        print(f"[bottom-up] ensemble | branch={branch} | cosine={match.get('score')} | "
              f"relationship={match.get('relationship')} | votes={dict(counts)} | "
              f"decision={(result.decision.value if result.decision else 'INCONCLUSIVE')}", flush=True)
    # Check all survivors against the original aggregate, not a mutated order-dependent graph.
    kept = {key for key, result in results.items() if result.decision == HighLevelGoalDecision.KEEP}
    for branch in branches:
        survivor = results[branch].surviving_branch_id
        if survivor and survivor not in kept:
            results[branch] = _inconclusive(
                branch, "Duplicate survivor is not conclusively kept in this snapshot.",
                )
            print(f"[bottom-up] ensemble | branch={branch} | INCONCLUSIVE: "
                  "duplicate survivor is not kept", flush=True)
    if on_evaluation:
        on_evaluation(len(results), len(branches), results)
    return results
