"""One joint global review per critic of goals, numerical matches and project context."""

import json
import os
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.bottom_up.goal_similarity import analyze_similarities, build_cosine_payload, build_review_payload
from src.data_model import (
    Actors,
    BottomUpRepairFeedback,
    EvaluatorVoteRecord,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    HighLevelGoalDecision,
    HighLevelGoalEvaluation,
    HighLevelGoalReplacementRequest,
)
from src.llm_clients import (
    configured_bottom_up_critic_models,
    generate_evaluator_response,
    loaded_local_evaluator_model,
)

# Configuration

PIPELINE_POLICY_VERSION = "cosine_global_documentary_review_v69_textual_duplicate_review"
BOTTOM_UP_EVALUATOR_MODELS = configured_bottom_up_critic_models()

if len(set(BOTTOM_UP_EVALUATOR_MODELS)) != 3 or not all(BOTTOM_UP_EVALUATOR_MODELS):
    raise ValueError("The ensemble requires three different non-empty models")
EVALUATOR_VOTING_QUORUM = 2
BOTTOM_UP_EVALUATOR_TEMPERATURE = float(
    os.getenv("BOTTOM_UP_EVALUATOR_TEMPERATURE", "0.2")
)
BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS = int(
    os.getenv("BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS", "3")
)

if not 1 <= BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS <= 3:
    raise ValueError("Evaluator attempts must be between 1 and 3")
BOTTOM_UP_EVALUATOR_PROMPTING_MODE = "FEW_SHOT"
CRITIC1_FEW_SHOT_EXAMPLE_COUNT = 4

SYSTEM_PROMPT = (
    "You are an expert in software engineering and Goal-Oriented Requirements Engineering (GORE). "
    "Start from each reconstructed HLG and jointly review project description, actors, current HLGs, "
    "reconstructed HLGs and fixed similarity results. The description states requirements; "
    "generated goals reveal scope, consistency and overlap, not additional requirements. "
    "Return only the requested decision JSON. Do not invent functionality."
)


# Four source/branch/response examples, shared by all three evaluators.
FEW_SHOT_EXAMPLES = [{'source': 'Members reserve tools.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': 'Reserve tools',
                'hlg_prime': 'Request tool reservations',
                'relationship': 'EQUIVALENT'}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL',
                              'rationale': 'Tool reservations are documented.'}]}},
 {'source': 'Members edit only their own profile.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': 'Edit all member profiles',
                'hlg_prime': 'Edit my profile',
                'relationship': 'NOT_EQUIVALENT',
                'original_hlg_comparisons_complete': True}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'rationale': 'Own-profile editing corrects the existing scope.',
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
                              'decision': 'REMOVE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'rationale': 'Auctions are explicitly excluded.'},
                             {'branch_id': 'b',
                              'decision': 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'rationale': 'Despite numerical equivalence, only reservations are documented; '
                                           'auctions are excluded from both goals.',
                              'feedback': {'critique': "For the goal 'Reserve and auction tools': Despite "
                                                       'numerical equivalence, only reservations are '
                                                       'documented; auctions are excluded from both goals. '
                                                       'Preserve tool reservations and remove the '
                                                       'unsupported auction capability.'}},
                             {'branch_id': 'c',
                              'decision': 'REMOVE_ORIGINAL_HIGH_LEVEL_GOAL',
                              'rationale': 'The parent concerns auctions and the reconstruction concerns '
                                           'sales; neither outcome is documented. Removing this branch loses '
                                           'no supported capability.'}]}},
 {'source': 'Members reserve tools.',
  'actor': 'Member',
  'branches': [{'branch_id': 'a',
                'hlg': 'Reserve tools',
                'hlg_prime': 'Request tool reservations',
                'relationship': 'EQUIVALENT',
                'duplicate_candidates': ['b']},
               {'branch_id': 'b',
                'hlg': 'Request tool reservations',
                'hlg_prime': 'Reserve tools',
                'relationship': 'EQUIVALENT',
                'duplicate_candidates': ['a']}],
  'duplicate_candidates': [{'branch_id': 'a', 'other_branch_id': 'b'}],
  'response': {'decisions': [{'branch_id': 'a',
                              'decision': 'KEEP_ORIGINAL_HIGH_LEVEL_GOAL',
                              'rationale': 'This parent represents the documented outcome.'},
                             {'branch_id': 'b',
                              'decision': 'DUPLICATE_REMOVAL',
                              'rationale': 'The same member outcome is covered by a.',
                              'surviving_branch_id': 'a'}]}}]


def evaluator_examples(count=None):
    """Nested example sets, independent of evaluation ground truths."""
    count = CRITIC1_FEW_SHOT_EXAMPLE_COUNT if count is None else count
    if type(count) is not int or count not in (0, 1, 2, 4):
        raise ValueError("Evaluator example count must be 0, 1, 2 or 4 (maximum 4)")
    return FEW_SHOT_EXAMPLES[:count]


class DocumentaryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_id: str = Field(min_length=1)
    decision: Literal[
        "KEEP_ORIGINAL_HIGH_LEVEL_GOAL", "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
        "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL", "DUPLICATE_REMOVAL",
    ]
    rationale: str = Field(min_length=1)
    feedback: BottomUpRepairFeedback | None = None
    surviving_branch_id: str | None = None

    @model_validator(mode="after")
    def check_parameters(self):
        for value in (self.branch_id, self.rationale, self.surviving_branch_id):
            if value is not None and not value.strip():
                raise ValueError("Text fields cannot be blank")
        if bool(self.feedback) != (self.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"):
            raise ValueError("Only REWRITE requires textual feedback")
        if bool(self.surviving_branch_id) != (self.decision == "DUPLICATE_REMOVAL"):
            raise ValueError("Only DUPLICATE_REMOVAL requires surviving_branch_id")
        return self


def _operation_key(vote):
    return f"{vote.decision}|survivor={vote.surviving_branch_id or ''}"


def _merge_agreeing_judgments(records, operation_key):
    """Join only the feedback prose of the agreeing evaluators for generation."""
    votes = [DocumentaryDecision.model_validate(record.parsed_response)
             for record in records if record.status == "VALID" and record.decision == operation_key]
    if len(votes) < EVALUATOR_VOTING_QUORUM:
        raise ValueError("Merged feedback requires at least two agreeing valid votes")

    def join_text(parts):
        return "\n\n".join(dict.fromkeys(part.strip() for part in parts if part and part.strip()))

    updates = {"rationale": join_text(vote.rationale for vote in votes)}
    if votes[0].feedback:
        updates["feedback"] = {"critique": join_text(vote.feedback.critique for vote in votes)}
    return DocumentaryDecision.model_validate({**votes[0].model_dump(), **updates})


def _documentary_response_schema(branches, analysis):
    """Constrain each duplicate operation to its own direct candidates."""
    variants = []
    actions = ["KEEP_ORIGINAL_HIGH_LEVEL_GOAL", "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
               "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL", "DUPLICATE_REMOVAL"]
    eligible_ids = [b.branch_id for b in branches
                    if analysis["branches"][b.branch_id]["status"] == "VALID"]
    for action in actions:
        groups = {}
        for branch in branches:
            record = analysis["branches"][branch.branch_id]
            candidates = tuple(record.get("duplicate_candidates", []))
            probe = dict(branch_id=branch.branch_id, decision=action, rationale="Schema eligibility")
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
            groups.setdefault(key, []).append(branch.branch_id)
        for candidates, ids in groups.items():
            properties = {
                "branch_id": {"type": "string", "enum": ids},
                "decision": {"type": "string", "enum": [action]},
                "rationale": {"type": "string", "minLength": 1},
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
    if vote.branch_id != branch.branch_id:
        raise ValueError("Wrong branch_id")
    record = analysis["branches"][branch.branch_id]
    if record["status"] != "VALID":
        raise ValueError("Similarity unavailable; branch must remain inconclusive")
    if vote.decision == "DUPLICATE_REMOVAL":
        if vote.surviving_branch_id not in record["duplicate_candidates"]:
            raise ValueError("Survivor must be a direct same-actor text-review candidate")
    elif vote.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
        if not record["reconstructable"] or record["relationship"] not in {"EQUIVALENT", "NOT_EQUIVALENT"}:
            raise ValueError("REWRITE requires a reconstructable HLG prime and a valid numerical relationship")
        if record.get("original_hlg_comparisons_complete") is False:
            raise ValueError("Cannot choose REWRITE before completing comparisons with original HLGs")
    elif vote.decision == "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL":
        if record["relationship"] not in {"EQUIVALENT", "NOT_EQUIVALENT"}:
            raise ValueError("REMOVE requires a valid numerical relationship")
    return vote


# Joint global review prompt


GLOBAL_REVIEW_INSTRUCTIONS = """START FROM HLG PRIME: it summarizes the branch's child capabilities, not an
approved replacement. Read its scope, then compare it with its parent, ALL
current HLGs, same-actor primes, project description and actors. Use the fixed
scores/labels to locate matches; read goal texts to distinguish actual coverage
from proximity. Other goals provide context, not new requirements evidence.
Composite scope or different wording alone is not an error. LLG texts are absent:
do not audit individual children or reconstruction fidelity.

For every requested branch, apply the comparisons for the chosen action:

KEEP_ORIGINAL_HIGH_LEVEL_GOAL
Compare prime with parent (score, label, texts), then parent with source, actor
and other HLGs. Keep a correctly scoped supported parent with an independent
outcome, or the selected duplicate survivor. EQUIVALENT supports retention only
if the shared outcome is required. If NOT_EQUIVALENT, keep a valid parent when
prime's different scope does not establish a parent defect; explain that distinction.

REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
Compare prime with parent to identify scope differences; inspect prime's matches
with other HLGs and check source/actor to locate the correct responsibility.
Rewrite when the parent's supported outcome needs a concrete scope, ownership
or boundary correction. With EQUIVALENT, both texts may share unsupported scope;
with NOT_EQUIVALENT, prime may reveal a required correction, not an automatic
replacement. Preserve supported capabilities without absorbing another goal's
independent scope. Requires reconstructable=true and original_hlg_comparisons_complete=true.

REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
Compare BOTH parent and prime with source, actors and remaining HLGs. Remove
only if the parent has no supported outcome to preserve and removing its children
loses no required capability. EQUIVALENT: their shared outcome is unsupported.
NOT_EQUIVALENT: neither outcome supplies supported functionality to preserve.
Partial support requires correction; supported redundancy uses DUPLICATE_REMOVAL.

DUPLICATE_REMOVAL
Compare the texts of same-actor primes and BOTH original parents, then check
source and actor responsibilities. No prime/prime cosine is computed. Remove only if the
survivor's parent fully covers this branch's supported capabilities. Shared words
or partial overlap cannot replace this coverage check. Preserve
complementary outcomes. Keep the most complete valid parent; ties follow branch
order. Survivor must be KEEP, not rewritten/removed. No chains, cycles or child transfer.

OUTPUT: only {"decisions":[...]}; every requested ID once, schema fields only.
Each entry: branch_id, decision, rationale (<=25 words), naming the decisive
comparison and concrete scope finding. Do not output the intermediate analysis.
REWRITE alone adds feedback: {"critique":"..."} (<=60 words): name the goal,
state what to preserve/change and relevant boundaries/responsibilities. Only
this prose reaches the HLG generator; agreeing critiques are joined. No IDs,
scores or JSON in critique, replacement collections or separate LLG repair.
DUPLICATE_REMOVAL alone adds surviving_branch_id from its direct candidates.
No abstention or extra actions. Do not recalculate scores or replace fixed labels.
"""


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
Branch list order defines zero-based matrix indices. original_scores[i][j] is
cosine(prime of branch i, original HLG of branch j), across ALL actors; diagonal
is parent_score. Values are unrounded. Null means unavailable; original_errors
contains [row,column,status,error] for non-VALID comparisons.
threshold applies only to prime/original-HLG matches. duplicate_candidates lists
all other evaluable same-actor branches, NOT cosine-selected or confirmed duplicates.
Compare their parents and primes by text; preserve independent capabilities.
This request supplies {len(selected_examples)} fictional examples; use them only
as decision demonstrations, not evidence. Their prose illustrates policy;
apply the current 25/60-word limits to your response.

{example_section}PROJECT DESCRIPTION
{description}
ACTORS
{actors.model_dump_json()}
Cosine JSON:
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


def _critic1_vote(prompt, model, branches, analysis, *, temperature=None):
    """Accept one complete global response per model; never mix attempts."""
    temperature = BOTTOM_UP_EVALUATOR_TEMPERATURE if temperature is None else temperature
    eligible = [b for b in branches if analysis["branches"][b.branch_id]["status"] == "VALID"]
    expected = {b.branch_id for b in eligible}
    schema = _documentary_response_schema(eligible, analysis)
    history, accepted, errors = [], {}, {}
    raw, payload, retry = None, None, prompt
    for attempt in range(1, BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS + 1):
        if not eligible:
            break
        print(f"[bottom-up] global review | model={model} | attempt={attempt}/"
              f"{BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS}", flush=True)
        current_errors, generation, candidate = {}, {}, {}
        raw, payload, context_error = None, None, False
        try:
            raw = generate_evaluator_response(retry, SYSTEM_PROMPT, conversation=None,
                temperature=temperature, model=model, response_schema=schema)
            generation = dict(getattr(raw, "metadata", {}))
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
                vote = _validate_decision(by_id[branch.branch_id], branch, analysis)
                if len(vote.rationale.split()) > 25:
                    raise ValueError("Each rationale must use at most 25 words")
                if vote.feedback and len(vote.feedback.critique.split()) > 60:
                    raise ValueError("Each critique must use at most 60 words")
                candidate[branch.branch_id] = EvaluatorVoteRecord(
                    model=model, status="VALID", decision=_operation_key(vote), raw_response=raw,
                    parsed_response=vote.model_dump(exclude_none=True))
        except Exception as exc:
            raw = getattr(exc, "raw_response", raw)
            generation = dict(getattr(exc, "metadata", generation))
            current_errors["response"] = f"{type(exc).__name__}: {exc}"
            context_error = _is_context_length_error(exc)
            if context_error:
                generation.update(error_type="CONTEXT_LENGTH_EXCEEDED", retryable=False)
        success = not current_errors and len(candidate) == len(expected)
        if raw is not None:
            raw = str(raw)  # Persist plain text, not the metadata-bearing str subclass.
        generation.update(accepted=success, global_retry_policy="discard_entire_invalid_response")
        history.append(dict(attempt=attempt, raw_response=raw, errors=current_errors,
                            valid_branch_ids=list(candidate) if success else [],
                            generation=generation, requested_branch_ids=[b.branch_id for b in eligible]))
        errors = current_errors
        if success:
            accepted = candidate
            break
        print(f"[bottom-up] model={model} | global response rejected: {current_errors}", flush=True)
        if context_error:
            break  # Repeating the same oversized input cannot repair its context.
        reason = current_errors.get("response", "Invalid global response")[:350]
        retry = (prompt + "\nCORRECTION: The previous global response was discarded. "
                 "Regenerate ALL requested decisions as one complete concise JSON. "
                 "Rationale <=25 words; critique <=60 words. Error: " + reason)
    records = {}
    for branch in branches:
        record = accepted.get(branch.branch_id)
        if record is None:
            unavailable = analysis["branches"][branch.branch_id]["status"] != "VALID"
            record = EvaluatorVoteRecord(model=model, status="INVALID",
                error="Numerical comparison unavailable" if unavailable else errors.get("response", "Invalid global response"))
        record.attempt_history = history
        record.generation_attempts = [dict(attempt=h["attempt"], errors=h["errors"],
                                           **h["generation"]) for h in history]
        records[branch.branch_id] = record
    global_record = EvaluatorVoteRecord(model=model,
        status="VALID" if accepted else "INVALID",
        error=errors.get("response"), raw_response=raw,
        parsed_response=payload if accepted else None, attempt_history=history,
        generation_attempts=[dict(attempt=h["attempt"], errors=h["errors"],
                                  **h["generation"]) for h in history])
    return global_record, records


# Ensemble aggregation


def _inconclusive(branch, message, records, similarity):
    relation = similarity.get("relationship")
    return GlobalGoalEvaluationResult(
        branch_id=branch.branch_id, rationale=message,
        final_decision=GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
        high_level_evaluation=HighLevelGoalEvaluation(
            rationale=message, decision=HighLevelGoalDecision.KEEP,
            reconstruction_relationship=relation, requires_llg_evaluation=False),
        critic_votes={"documentary": records}, similarity=similarity, evaluation_status="INCONCLUSIVE")


def evaluation_status(evaluation):
    return evaluation.evaluation_status or (
        "INCONCLUSIVE" if evaluation.final_decision == GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
        else "DECIDED")


def evaluate_all_branches(
    project_description,
    branches,
    reconstructions,
    on_evaluation=None,
    *,
    actors=None,
    on_vote=None,
    on_global_vote=None,
    similarity_analysis=None,
    cosine_payload=None,
    few_shot_example_count=None,
    evaluator_temperature=None,
):
    if not branches:
        return {}
    analysis = similarity_analysis if similarity_analysis is not None else analyze_similarities(branches, reconstructions)
    if all(analysis["branches"][b.branch_id]["status"] == "ERROR" for b in branches):
        results = {b.branch_id: _inconclusive(
            b, "Similarity unavailable: " + analysis["branches"][b.branch_id]["error"],
            [], analysis["branches"][b.branch_id]) for b in branches}
        if on_evaluation:
            on_evaluation(len(results), len(branches), results)
        return results
    actors = actors or Actors(actors=list({b.high_level_goal.actor.name: b.high_level_goal.actor for b in branches}.values()))
    prompt = _critic1_prompt(project_description, actors, branches, reconstructions, analysis, cosine_payload,
                            few_shot_example_count=few_shot_example_count)
    records = {b.branch_id: [] for b in branches}
    for model in BOTTOM_UP_EVALUATOR_MODELS:
        try:
            with loaded_local_evaluator_model(model):
                global_record, votes = _critic1_vote(prompt, model, branches, analysis,
                                                   temperature=evaluator_temperature)
        except Exception as exc:
            # A failed model lease must not erase complete responses from peers.
            error = f"Model load/release failed: {type(exc).__name__}: {exc}"
            diagnostic = [dict(attempt=0, accepted=False, error_type="MODEL_LOAD_FAILED", error=error)]
            global_record = EvaluatorVoteRecord(model=model, status="INVALID", error=error,
                                               generation_attempts=diagnostic)
            votes = {b.branch_id: EvaluatorVoteRecord(model=model, status="INVALID", error=error,
                                                     generation_attempts=diagnostic) for b in branches}
        if on_global_vote:
            on_global_vote(global_record)
        for branch in branches:
            records[branch.branch_id].append(votes[branch.branch_id])
            if on_vote:
                on_vote(branch, "documentary", [votes[branch.branch_id]])
    results, by_id = {}, {b.branch_id: b for b in branches}
    for branch in branches:
        local = records[branch.branch_id]
        valid = [r for r in local if r.status == "VALID"]
        counts = Counter(r.decision for r in valid)
        match = analysis["branches"][branch.branch_id]
        if not counts or counts.most_common(1)[0][1] < EVALUATOR_VOTING_QUORUM:
            result = _inconclusive(branch, "No two valid documentary actions agree.", local, match)
            if any(attempt.get("generation", {}).get("error_type") == "CONTEXT_LENGTH_EXCEEDED"
                   for record in local for attempt in record.attempt_history):
                result = result.model_copy(update={
                    "evaluation_status": "CONTEXT_LENGTH_EXCEEDED",
                    "rationale": "Evaluator context exceeded; insufficient agreeing valid votes. No automatic action.",
                })
        else:
            key, count = counts.most_common(1)[0]
            vote = _merge_agreeing_judgments(valid, key)
            if match["status"] != "VALID":
                result = _inconclusive(branch, vote.rationale, local, match)
            else:
                survivor = by_id.get(vote.surviving_branch_id)
                hlg = HighLevelGoalEvaluation(
                    rationale=vote.rationale, decision=vote.decision,
                    reconstruction_relationship=match["relationship"], requires_llg_evaluation=False,
                    rewriting_focus=vote.feedback.critique if vote.feedback else None, covered_by_branch_id=vote.surviving_branch_id,
                    covered_by_high_level_goal_name=survivor.high_level_goal.name if survivor else None,
                    structural_decision=vote.model_dump(exclude_none=True), voter_count=3,
                    valid_vote_count=len(valid), winning_vote_count=count,
                    vote_distribution=dict(counts), evaluator_votes=local)
                if vote.decision in {"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL", "DUPLICATE_REMOVAL"}:
                    final = GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
                elif vote.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
                    final = GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
                else:
                    final = GlobalGoalEvaluationDecision.CONFIRM_BRANCH
                replacement = HighLevelGoalReplacementRequest(
                    rationale=vote.rationale, actor=branch.high_level_goal.actor,
                    generation_project_description=project_description) if vote.feedback else None
                result = GlobalGoalEvaluationResult(
                    branch_id=branch.branch_id, rationale=vote.rationale, final_decision=final,
                    high_level_evaluation=hlg, replacement_request=replacement,
                    feedback=vote.feedback, similarity=match,
                    critic_votes={"documentary": local}, evaluation_status="DECIDED")
        results[branch.branch_id] = result
        print(f"[bottom-up] ensemble | branch={branch.branch_id} | cosine={match.get('score')} | "
              f"relationship={match.get('relationship')} | votes={dict(counts)} | "
              f"decision={result.final_decision.value}\n  rationale: {result.rationale}", flush=True)
    # Check all survivors against the original aggregate, not a mutated order-dependent graph.
    kept = {key for key, result in results.items() if result.final_decision in {
        GlobalGoalEvaluationDecision.CONFIRM_BRANCH}}
    for branch in branches:
        survivor = results[branch.branch_id].high_level_evaluation.covered_by_branch_id
        if survivor and survivor not in kept:
            results[branch.branch_id] = _inconclusive(
                branch, "Duplicate survivor is not conclusively kept in this snapshot.",
                records[branch.branch_id], analysis["branches"][branch.branch_id])
            print(f"[bottom-up] ensemble | branch={branch.branch_id} | INCONCLUSIVE: "
                  "duplicate survivor is not kept", flush=True)
    if on_evaluation:
        on_evaluation(len(results), len(branches), results)
    return results
