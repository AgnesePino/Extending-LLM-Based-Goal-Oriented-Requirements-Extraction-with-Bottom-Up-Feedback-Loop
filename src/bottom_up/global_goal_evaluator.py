"""One joint global review per critic of goals, numerical matches and project context."""

import json
import os
from collections import Counter
from contextlib import nullcontext


from src.bottom_up.goal_similarity import (
    analyze_similarities, build_cosine_payload, build_review_payload,
    build_duplicate_candidate_pairs,
)
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

PIPELINE_POLICY_VERSION = "cosine_global_documentary_review_v112_minimal_candidate_pairs"
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
    "You are an expert in Goal-Oriented Requirements Engineering. Review original HLGs "
    "bottom-up using reconstructed HLGs, fixed labels, documentation and actors. "
    "Apply exactly the four action conditions below. Read each parent's name and description. "
    "Preserve supported functionality, invent no requirements, and return only the requested JSON."
)


# Four source/branch/response examples, shared by all three evaluators.
FEW_SHOT_EXAMPLES = [{'source': 'Requesters create, modify and cancel their own service requests.',
  'actor': 'Requester',
  'branches': [{'branch_id': 'a',
                'hlg': {'name': 'Manage my service requests',
                        'description': 'Create, modify and cancel my service requests.'},
                'hlg_prime': 'Submit and cancel my service requests',
                'relationship': 'EQUIVALENT',
                'duplicate_label': 'NO_DUPLICATE_MATCH',
                'duplicate_candidates': [],
                'original_hlg_comparisons_complete': True}],
  'response': {'decisions': [{'branch_id': 'a', 'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL'}]}},
 {'source': 'Requesters create, modify and cancel their own service requests. Only a service '
            'officer may approve requests. Requesters may view approval outcomes.',
  'actor': 'Requester',
  'branches': [{'branch_id': 'a',
                'hlg': {'name': 'Create service requests only',
                        'description': 'Changes and cancellations are excluded.'},
                'hlg_prime': 'Create, modify and cancel my service requests',
                'relationship': 'NOT_EQUIVALENT',
                'duplicate_label': 'NO_DUPLICATE_MATCH',
                'duplicate_candidates': [],
                'original_hlg_comparisons_complete': True},
               {'branch_id': 'b',
                'hlg': {'name': 'Approve service requests',
                        'description': 'Approve or reject submitted service requests as a '
                                       'requester.'},
                'hlg_prime': 'Approve or reject service requests',
                'relationship': 'EQUIVALENT',
                'duplicate_label': 'NO_DUPLICATE_MATCH',
                'duplicate_candidates': [],
                'original_hlg_comparisons_complete': True},
               {'branch_id': 'c',
                'hlg': {'name': "Manage other requesters' service requests",
                        'description': 'Create, modify and cancel requests on behalf of any '
                                       'requester.'},
                'hlg_prime': 'Manage my own service requests',
                'relationship': 'NOT_EQUIVALENT',
                'duplicate_label': 'NO_DUPLICATE_MATCH',
                'duplicate_candidates': [],
                'original_hlg_comparisons_complete': True}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'feedback': {'critique': "For 'Create service requests only': "
                                                       'requirements allow creation, modification '
                                                       "and cancellation of the requester's own "
                                                       'requests. The parent explicitly excludes '
                                                       'two required operations. Preserve creation '
                                                       'and include modification and cancellation '
                                                       'within the same request scope.'}},
                             {'branch_id': 'b',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'feedback': {'critique': "For 'Approve service requests': the source "
                                                       'assigns approval and rejection to the '
                                                       'service officer, not the requester. '
                                                       'Correct the actor responsibility to the '
                                                       'service officer while preserving the '
                                                       'approval and rejection functionality. Keep '
                                                       "the requester's ability to view outcomes "
                                                       'distinct.'}},
                             {'branch_id': 'c',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'feedback': {'critique': "For 'Manage other requesters' service "
                                                       "requests': documentation allows requesters "
                                                       'to manage only their own requests. The '
                                                       'parent incorrectly extends responsibility '
                                                       'to other requesters. Restrict creation, '
                                                       'modification and cancellation to the '
                                                       "acting requester's own requests, "
                                                       'preserving all three operations.'}}]}},
 {'source': 'Members reserve tools. Auctions and sales are explicitly excluded.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': {'name': 'Auction tools', 'description': 'Run tool auctions.'},
                'hlg_prime': 'Organize tool auctions',
                'relationship': 'EQUIVALENT',
                'duplicate_label': 'NO_DUPLICATE_MATCH',
                'duplicate_candidates': [],
                'original_hlg_comparisons_complete': True},
               {'branch_id': 'b',
                'hlg': {'name': 'Reserve and sell tools',
                        'description': 'Reserve tools and offer them for sale.'},
                'hlg_prime': 'Manage tool reservations and sales',
                'relationship': 'EQUIVALENT',
                'duplicate_label': 'NO_DUPLICATE_MATCH',
                'duplicate_candidates': [],
                'original_hlg_comparisons_complete': True}],
  'response': {'decisions': [{'branch_id': 'a', 'decision': 'REMOVE_ORIGINAL_HIGH_LEVEL_GOAL'},
                             {'branch_id': 'b',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'feedback': {'critique': "For 'Reserve and sell tools': requirements "
                                                       'support reservations and explicitly '
                                                       'exclude sales. Preserve reservations and '
                                                       'remove sales. The branch is partly '
                                                       'supported, so removing it would lose '
                                                       'required functionality.'}}]}},
 {'source': 'Requesters create, modify and cancel their own service requests and download '
            'confirmation receipts.',
  'actor': 'Requester',
  'branches': [{'branch_id': 'a',
                'hlg': {'name': 'Manage my service requests',
                        'description': 'Create, modify and cancel my service requests.'},
                'hlg_prime': 'Create, modify and cancel my service requests',
                'relationship': 'EQUIVALENT',
                'duplicate_label': 'POTENTIAL_DUPLICATE',
                'duplicate_candidates': ['b', 'c'],
                'original_hlg_comparisons_complete': True},
               {'branch_id': 'b',
                'hlg': {'name': 'Create my service requests',
                        'description': 'Submit a service request for myself.'},
                'hlg_prime': 'Submit a service request for myself',
                'relationship': 'EQUIVALENT',
                'duplicate_label': 'POTENTIAL_DUPLICATE',
                'duplicate_candidates': ['a', 'c'],
                'original_hlg_comparisons_complete': True},
               {'branch_id': 'c',
                'hlg': {'name': 'Cancel requests and download receipts',
                        'description': 'Cancel my service requests and download confirmation '
                                       'receipts.'},
                'hlg_prime': 'Cancel my service requests and retrieve their receipts',
                'relationship': 'EQUIVALENT',
                'duplicate_label': 'POTENTIAL_DUPLICATE',
                'duplicate_candidates': ['a', 'b'],
                'original_hlg_comparisons_complete': True}],
  'response': {'decisions': [{'branch_id': 'a', 'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL'},
                             {'branch_id': 'b',
                              'decision': 'DUPLICATE_REMOVAL',
                              'surviving_branch_id': 'a'},
                             {'branch_id': 'c', 'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL'}]}}]


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
        if record["relationship"] != "EQUIVALENT":
            raise ValueError("REMOVE requires an EQUIVALENT parent relationship")
    return vote


# Joint global review prompt


GLOBAL_REVIEW_INSTRUCTIONS = """A. CONTEXT AND OBJECTIVE
Review original HLGs bottom-up: start from HLG', compare goals, then check the
project documentation. HLG' is reconstructed evidence, not an approved replacement.

B. INPUTS AND THEIR MEANING
PROJECT CONTEXT
- PROJECT DESCRIPTION: original requirements.
- ACTORS: roles and responsibilities.
- requested_branch_ids: branches requiring a decision.

BRANCHES (one entry per original HLG)
- branch_id / actor: branch identifier and responsible actor.
- hlg: original parent's name AND description; read both.
- hlg_prime: HLG' reconstructed from the branch's LLGs. LLG texts are not supplied.
- relationship: fixed parent comparison, EQUIVALENT or NOT_EQUIVALENT.
- duplicate_candidates: IDs of eligible same-actor parents.
- duplicate_label: POTENTIAL_DUPLICATE if candidates exist; otherwise NO_DUPLICATE_MATCH.
- status / reconstructable / original_hlg_comparisons_complete: availability checks.

POTENTIAL_DUPLICATES (only candidate pairs)
- branch_id / hlg_prime: source branch identifier and its reconstruction.
- candidate_branch_id / candidate_hlg: candidate parent's ID, name and description.
Each entry contains only HLG' and its candidate; parent and actor remain in BRANCHES.
Compare source HLG' -> candidate original HLG. The reverse pair is not implied.
An empty list means no eligible candidate, not proof that goals are independent.

Use documentation as authority; generated goals provide context, not requirements.
No matrix, cosine scores or threshold are supplied.
Never recalculate cosine similarity or modify the supplied classifications.

C. FOUR ACTIONS AND THEIR CONDITIONS
1. KEEP -> KEEP_ORIGINAL_HIGH_LEVEL_GOAL
Keep the HLG when comparison with HLG', considering EQUIVALENT, and verification
of documentation confirm a valid, supported functionality correctly assigned
to its actor. KEEP is not available with NOT_EQUIVALENT.

2. REWRITE -> REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
Propose rewriting when analysis of HLG', considering EQUIVALENT or NOT_EQUIVALENT,
reveals concrete problems in the original parent's formulation, functional scope
or assigned responsibility. In particular, NOT_EQUIVALENT may indicate that the
original HLG inadequately represents functionality expressed by the LLGs.
Propose REWRITE when documentation confirms this problem, preserving supported
functionality and respecting the functional boundaries of other HLGs.
Requires an available reconstruction and complete required comparisons.

3. REMOVE -> REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
Remove the HLG and its LLGs when HLG' is EQUIVALENT to the original HLG and
documentation confirms that the represented functionality is unsupported.
Removal must not lose required functionality. Partial support requires correction.
Never REMOVE with NOT_EQUIVALENT.

4. DUPLICATE REMOVAL -> DUPLICATE_REMOVAL
Remove a branch when comparison of HLG' with another same-actor HLG signals
POTENTIAL_DUPLICATE and documentary verification confirms full redundancy.
Use documentation to select the survivor, ensuring it covers ALL supported
functionality of the removed branch, including outcomes absent from HLG'.
Partial overlap is insufficient. Keep the most complete valid parent; equivalent
ties follow branch order. The survivor must meet KEEP conditions and receive KEEP.
No chains, cycles or child transfers.

D. DECISION DISCIPLINE
Choose one action per requested branch using the conditions above. All decisions
must be justified by original project documentation, never similarity labels alone.
NOT_EQUIVALENT flags a possible problem; it alone does not justify REWRITE.
Read the whole parent before claiming omissions: do not request changes already
satisfied by its name or description, invent requirements, or treat HLG' as a replacement.

E. OUTPUT REQUIREMENTS
Return only {"decisions":[...]} with each requested branch_id exactly once and
the exact decision values above. REWRITE alone adds feedback: {"critique":"..."},
identifying the documented parent problem, specific changes and functionality
to preserve. Aim for at most {critique_characters} characters per feedback.critique, including spaces and punctuation.
This is guidance, not a hard limit. Be concise. No replacement goals or reasoning trace.
DUPLICATE_REMOVAL alone adds surviving_branch_id from duplicate_candidates.
KEEP and REMOVE add no fields. Only approved rewrite critique reaches the generator;
agreeing critiques are joined. Examples are fictional demonstrations, not evidence.
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
        f"Potential duplicates: {json.dumps(build_duplicate_candidate_pairs([{**b, 'actor': e['actor']} for b in e['branches']]), separators=(',', ':'))}\n"
        f"Output: {json.dumps(e['response'], separators=(',', ':'))}"
        for i, e in enumerate(selected_examples, 1)
    )
    example_section = f"EXAMPLES (fictional requirements, not project evidence)\n{examples}\n\n" if examples else ""
    return f"""{GLOBAL_REVIEW_INSTRUCTIONS}

This request supplies {len(selected_examples)} fictional decision examples.

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
