"""One global documentary review per critic, using fixed numerical matches."""

import json
import os
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from src.bottom_up.goal_similarity import analyze_similarities, build_cosine_payload
from src.data_model import (
    Actors,
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

PIPELINE_POLICY_VERSION = "cosine_global_documentary_review_v58_three_limits"
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

SYSTEM_PROMPT = """Review software goals against the supplied project description.
Use only this source. Similarity scores and labels are fixed inputs, not questions.
Return one JSON object. Give short documentary reasons. Do not invent requirements."""

# Four source/branch/response examples, shared by all three evaluators.
FEW_SHOT_EXAMPLES = [
    {
        "source": "Members reserve tools.",
        "actor": "Member",
        "branches": [
            {
                "branch_id": "a",
                "hlg": "Reserve tools",
                "hlg_prime": "Request tool reservations",
                "relationship": "EQUIVALENT"
            }
        ],
        "response": {
            "decisions": [
                {
                    "branch_id": "a",
                    "decision": "KEEP_ORIGINAL_HIGH_LEVEL_GOAL",
                    "rationale": "Tool reservations are documented."
                }
            ]
        }
    },
    {
        "source": "Members reserve tools and edit only their own profile. Tool auctions are excluded.",
        "actor": "Member",
        "branches": [
            {
                "branch_id": "a",
                "hlg": "Edit all member profiles",
                "hlg_prime": "Edit my profile",
                "relationship": "NOT_EQUIVALENT",
                "original_hlg_comparisons_complete": True
            },
            {
                "branch_id": "c",
                "hlg": "Reserve tools",
                "hlg_prime": "Auction tools",
                "relationship": "NOT_EQUIVALENT"
            }
        ],
        "response": {
            "decisions": [
                {
                    "branch_id": "a",
                    "decision": "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
                    "rationale": "Own-profile editing corrects the existing scope.",
                    "rewriting_focus": "Limit editing to the member's own profile."
                },
                {
                    "branch_id": "c",
                    "decision": "KEEP_ORIGINAL_HIGH_LEVEL_GOAL",
                    "rationale": "Auctions are excluded; tool reservations remain valid.",
                    "regenerate_llgs": True,
                    "regeneration_feedback": "Generate tool-reservation capabilities."
                }
            ]
        }
    },
    {
        "source": "Members reserve tools; auctions are excluded.",
        "actor": "Member",
        "branches": [
            {
                "branch_id": "a",
                "hlg": "Auction tools",
                "hlg_prime": "Run tool auctions",
                "relationship": "EQUIVALENT"
            }
        ],
        "response": {
            "decisions": [
                {
                    "branch_id": "a",
                    "decision": "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL",
                    "rationale": "Auctions are explicitly excluded."
                }
            ]
        }
    },
    {
        "source": "Members reserve tools.",
        "actor": "Member",
        "branches": [
            {
                "branch_id": "a",
                "hlg": "Reserve tools",
                "hlg_prime": "Request tool reservations",
                "relationship": "EQUIVALENT",
                "duplicate_label": "DUPLICATE",
                "duplicate_candidates": [
                    "b"
                ]
            },
            {
                "branch_id": "b",
                "hlg": "Request tool reservations",
                "hlg_prime": "Reserve tools",
                "relationship": "EQUIVALENT",
                "duplicate_label": "DUPLICATE",
                "duplicate_candidates": [
                    "a"
                ]
            }
        ],
        "duplicate_candidates": [
            {
                "branch_id": "a",
                "other_branch_id": "b"
            }
        ],
        "response": {
            "decisions": [
                {
                    "branch_id": "a",
                    "decision": "KEEP_ORIGINAL_HIGH_LEVEL_GOAL",
                    "rationale": "This parent represents the documented outcome."
                },
                {
                    "branch_id": "b",
                    "decision": "DUPLICATE_REMOVAL",
                    "rationale": "The same member outcome is covered by a.",
                    "surviving_branch_id": "a"
                }
            ]
        }
    }
]


# Response contract

# Synthetic additions for the six-example experiment; the original four stay intact.
EXTRA_FEW_SHOT_EXAMPLES = [
    {
        "source": "Librarians record returned books. Fee payments are excluded.",
        "actor": "Librarian",
        "branches": [{"branch_id": "a", "hlg": "Record book returns",
                      "hlg_prime": "Collect late fees", "relationship": "NOT_EQUIVALENT"}],
        "response": {"decisions": [{"branch_id": "a",
            "decision": "KEEP_ORIGINAL_HIGH_LEVEL_GOAL",
            "rationale": "Returns are documented; payments are excluded.",
            "regenerate_llgs": True,
            "regeneration_feedback": "Generate capabilities for recording book returns only."}]},
    },
    {
        "source": "Editors publish only their own articles.",
        "actor": "Editor",
        "branches": [{"branch_id": "a", "hlg": "Publish all editors' articles",
                      "hlg_prime": "Publish my articles", "relationship": "NOT_EQUIVALENT",
                      "original_hlg_comparisons_complete": True}],
        "response": {"decisions": [{"branch_id": "a",
            "decision": "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
            "rationale": "Publication is restricted to the editor's own articles.",
            "rewriting_focus": "Restrict publication to articles owned by the acting editor."}]},
    },
]


def evaluator_examples(count=None):
    """Nested example sets, independent of evaluation ground truths."""
    count = CRITIC1_FEW_SHOT_EXAMPLE_COUNT if count is None else count
    if count not in (0, 2, 4, 6):
        raise ValueError("Evaluator example count must be 0, 2, 4 or 6")
    return (FEW_SHOT_EXAMPLES + EXTRA_FEW_SHOT_EXAMPLES)[:count]


class DocumentaryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_id: str = Field(min_length=1)
    decision: Literal[
        "KEEP_ORIGINAL_HIGH_LEVEL_GOAL",
        "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
        "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL",
        "DUPLICATE_REMOVAL",
    ]
    rationale: str = Field(min_length=1)
    rewriting_focus: str | None = None
    regenerate_llgs: StrictBool = False
    regeneration_feedback: str | None = None
    surviving_branch_id: str | None = None

    @model_validator(mode="after")
    def check_parameters(self):
        for value in (
            self.branch_id,
            self.rationale,
            self.rewriting_focus,
            self.regeneration_feedback,
            self.surviving_branch_id,
        ):
            if value is not None and not value.strip():
                raise ValueError("Text fields cannot be blank")
        if bool(self.rewriting_focus) != (self.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"):
            raise ValueError("Only REWRITE requires rewriting_focus")
        if bool(self.surviving_branch_id) != (self.decision == "DUPLICATE_REMOVAL"):
            raise ValueError("Only DUPLICATE_REMOVAL requires surviving_branch_id")
        if self.regenerate_llgs and self.decision != "KEEP_ORIGINAL_HIGH_LEVEL_GOAL":
            raise ValueError("Only KEEP may request regeneration")
        if bool(self.regeneration_feedback) != self.regenerate_llgs:
            raise ValueError("Only regeneration requires regeneration_feedback")
        return self


# Numerical validation and structured-output schema


def _operation_key(vote):
    return f"{vote.decision}|regenerate={vote.regenerate_llgs}|survivor={vote.surviving_branch_id or ''}"


def _documentary_response_schema(branches, analysis):
    """Constrain syntax and action-specific fields; validate semantics per branch."""
    variants = []
    actions = [
        ("KEEP_ORIGINAL_HIGH_LEVEL_GOAL", []),
        ("KEEP_ORIGINAL_HIGH_LEVEL_GOAL", ["regenerate_llgs", "regeneration_feedback"]),
        ("REWRITE_ORIGINAL_HIGH_LEVEL_GOAL", ["rewriting_focus"]),
        ("REMOVE_ORIGINAL_HIGH_LEVEL_GOAL", []),
        ("DUPLICATE_REMOVAL", ["surviving_branch_id"]),
    ]
    # A retry may request only unresolved branches; survivors can be accepted peers.
    ids = list(analysis["branches"])
    for action, fields in actions:
        eligible = []
        for branch in branches:
            # Reuse the live numerical/parameter validator instead of duplicating
            # its rules. Documentary support remains the model's responsibility.
            probe = {
                "branch_id": branch.branch_id,
                "decision": action,
                "rationale": "Schema eligibility",
            }
            for name in fields:
                probe[name] = True if name == "regenerate_llgs" else "Required feedback"
            if action == "DUPLICATE_REMOVAL":
                candidates = analysis["branches"][branch.branch_id].get("duplicate_candidates", [])
                if not candidates:
                    continue
                probe["surviving_branch_id"] = candidates[0]
            try:
                _validate_decision(probe, branch, analysis)
            except ValueError:
                continue
            eligible.append(branch.branch_id)
        if not eligible:
            continue
        properties = {
            "branch_id": {"type": "string", "enum": eligible},
            "decision": {"type": "string", "enum": [action]},
            "rationale": {"type": "string", "minLength": 1},
        }
        for name in fields:
            properties[name] = ({"type": "boolean", "enum": [True]} if name == "regenerate_llgs"
                                else {"type": "string", "enum": ids} if name == "surviving_branch_id"
                                else {"type": "string", "minLength": 1})
        variants.append({
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        })

    required_count = len({
        branch_id for variant in variants
        for branch_id in variant["properties"]["branch_id"]["enum"]
    })
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": ({"anyOf": variants} if variants else {"type": "object", "additionalProperties": False}),
                "minItems": required_count,
                "maxItems": required_count,
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
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
            raise ValueError("Survivor must be a direct same-actor cosine candidate")
    elif vote.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
        if not record["reconstructable"] or record["relationship"] != "NOT_EQUIVALENT":
            raise ValueError("REWRITE requires a non-equivalent, reconstructable HLG prime")
        if record.get("original_hlg_comparisons_complete") is False:
            raise ValueError("Cannot choose REWRITE before completing comparisons with original HLGs")
    elif vote.decision == "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL":
        if record["relationship"] != "EQUIVALENT":
            raise ValueError("REMOVE requires numerical equivalence")
    elif vote.regenerate_llgs != (record["relationship"] == "NOT_EQUIVALENT"):
        if record["relationship"] == "NOT_EQUIVALENT":
            raise ValueError("NOT_EQUIVALENT: KEEP requires regenerate_llgs=true and regeneration_feedback; plain KEEP is invalid")
        raise ValueError("EQUIVALENT: KEEP must omit regenerate_llgs and regeneration_feedback")
    return vote


# Global documentary prompt


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
    examples = "\n\n".join(
        f"Example {i}\nDescription: {e['source']}\n"
        f"Actor: {e['actor']}\nGoals: {json.dumps(e['branches'], separators=(',', ':'))}\n"
        f"Output: {json.dumps(e['response'], separators=(',', ':'))}"
        for i, e in enumerate(evaluator_examples(few_shot_example_count), 1)
    )
    example_section = f"EXAMPLES (fictional requirements, not project evidence)\n{examples}\n\n" if examples else ""
    return f"""TASK
Review each branch against the source for its actor. HLG prime comes from its LLGs.
Use the supplied cosine labels as fixed facts; do not reassess similarity.
Do not audit LLG quality or judge reconstruction fidelity; review the goals only.
original_branch_index refers to the branch list, starting at 0; absent comparison status means VALID.

DECISIONS
- EQUIVALENT: documented outcome -> KEEP_ORIGINAL_HIGH_LEVEL_GOAL;
  entirely unsupported outcome -> REMOVE_ORIGINAL_HIGH_LEVEL_GOAL.
- NOT_EQUIVALENT + documented prime: correct the SAME outcome ->
  REWRITE_ORIGINAL_HIGH_LEVEL_GOAL + rewriting_focus. Never create a new HLG.
- NOT_EQUIVALENT + unsupported prime + valid parent -> KEEP_ORIGINAL_HIGH_LEVEL_GOAL
  + regenerate_llgs=true + regeneration_feedback. Plain KEEP is forbidden here.
- DUPLICATE is only a candidate. Use DUPLICATE_REMOVAL + surviving_branch_id only
  for a direct duplicate_candidates member if the source justifies one parent
  without losing autonomous capabilities. Keep the most complete valid parent;
  ties: registry order. No child transfer, circular or transitive removals.
- Empty branch: KEEP with regeneration only if the parent is documented.
  Never REWRITE from empty children or incomplete original comparisons.
  Choose one of the four permitted actions for every requested branch.

{example_section}PROJECT DESCRIPTION
{description}
ACTORS
{actors.model_dump_json()}
Cosine JSON:
{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}

OUTPUT
Only {{"decisions":[...]}} matching the schema. Every requested branch exactly once.
No abstention, omitted branches, empty decisions or additional action labels.
Each entry: branch_id, decision, one short source-based rationale; add only the
fields required above. Do not repeat goals, scores, labels or unused fields.
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
    temperature = BOTTOM_UP_EVALUATOR_TEMPERATURE if temperature is None else temperature
    retained, history, errors = {}, [], {}
    raw, payload, retry = None, None, prompt
    expected = {b.branch_id for b in branches}
    for branch in branches:
        if analysis["branches"][branch.branch_id]["status"] != "VALID":
            retained[branch.branch_id] = EvaluatorVoteRecord(
                model=model, status="INVALID", error="Numerical comparison unavailable")
    for attempt in range(1, BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS + 1):
        print(
            f"[bottom-up] global review | model={model} | attempt={attempt}/"
            f"{BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS}",
            flush=True,
        )
        current_errors, raw, payload, generation = {}, None, None, {}
        context_error = False
        pending = [b for b in branches if b.branch_id not in retained]
        if not pending:
            break
        schema = _documentary_response_schema(pending, analysis)
        try:
            raw = generate_evaluator_response(
                retry,
                SYSTEM_PROMPT,
                conversation=None,
                temperature=temperature,
                model=model,
                response_schema=schema,
            )
            generation = getattr(raw, "metadata", {})
            payload = _json_object(raw)
            items = payload["decisions"]
            counts = Counter(x.get("branch_id") for x in items
                             if isinstance(x, dict) and isinstance(x.get("branch_id"), str))
            unknown = [i for i in counts if i not in expected]
            if unknown:
                current_errors["response"] = f"Unknown branch IDs: {unknown}"
            for branch in branches:
                if branch.branch_id in retained:
                    continue
                try:
                    if counts[branch.branch_id] != 1:
                        raise ValueError("Missing or repeated branch decision")
                    item = next(x for x in items if isinstance(x, dict) and x.get("branch_id") == branch.branch_id)
                    vote = _validate_decision(item, branch, analysis)
                    retained[branch.branch_id] = EvaluatorVoteRecord(
                        model=model, status="VALID", decision=_operation_key(vote), raw_response=raw,
                        parsed_response=vote.model_dump(exclude_none=True))
                    print(f"[bottom-up] model={model} | branch={branch.branch_id} | "
                          f"{_operation_key(vote)}\n  rationale: {vote.rationale}", flush=True)
                except (ValueError, TypeError) as exc:
                    current_errors[branch.branch_id] = str(exc)
        except Exception as exc:
            raw = getattr(exc, "raw_response", raw)
            generation = getattr(exc, "metadata", generation)
            current_errors["response"] = f"{type(exc).__name__}: {exc}"
            context_error = _is_context_length_error(exc)
            if context_error:
                generation = {**generation, "error_type": "CONTEXT_LENGTH_EXCEEDED", "retryable": False}
        history.append({
            "attempt": attempt,
            "raw_response": raw,
            "errors": current_errors,
            "valid_branch_ids": list(retained),
            "generation": generation,
            "requested_branch_ids": [b.branch_id for b in pending],
        })
        errors.update(current_errors)
        if len(retained) == len(branches):
            break
        print(f"[bottom-up] model={model} | invalid decisions: {current_errors}", flush=True)
        if context_error:
            print(f"[bottom-up] model={model} | context exceeded; identical retries skipped", flush=True)
            break
        pending_ids = [b.branch_id for b in branches if b.branch_id not in retained]
        retry = (prompt + "\nCORRECTION: This retry overrides the output branch list only.\n"
                 "Return decisions ONLY for these unresolved IDs: " + json.dumps(pending_ids)
                 + ". Do not repeat accepted decisions. Keep the full project context above. Fix: "
                 + json.dumps(current_errors, ensure_ascii=False))
    for branch in branches:
        if branch.branch_id not in retained:
            retained[branch.branch_id] = EvaluatorVoteRecord(
                model=model, status="INVALID", raw_response=raw,
                error=errors.get(branch.branch_id, errors.get("response", "Invalid decision")))
        retained[branch.branch_id].attempt_history = history
    global_record = EvaluatorVoteRecord(
        model=model, status="VALID" if all(v.status == "VALID" for v in retained.values()) else "PARTIAL",
        raw_response=raw, parsed_response=payload, attempt_history=history)
    return global_record, retained


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
        with loaded_local_evaluator_model(model):
            global_record, votes = _critic1_vote(prompt, model, branches, analysis,
                                               temperature=evaluator_temperature)
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
            vote = DocumentaryDecision.model_validate(next(r for r in valid if r.decision == key).parsed_response)
            if match["status"] != "VALID":
                result = _inconclusive(branch, vote.rationale, local, match)
            else:
                survivor = by_id.get(vote.surviving_branch_id)
                hlg = HighLevelGoalEvaluation(
                    rationale=vote.rationale, decision=vote.decision,
                    reconstruction_relationship=match["relationship"], requires_llg_evaluation=False,
                    rewriting_focus=vote.rewriting_focus, covered_by_branch_id=vote.surviving_branch_id,
                    covered_by_high_level_goal_name=survivor.high_level_goal.name if survivor else None,
                    structural_decision=vote.model_dump(exclude_none=True), voter_count=3,
                    valid_vote_count=len(valid), winning_vote_count=count,
                    vote_distribution=dict(counts), evaluator_votes=local)
                if vote.decision in {"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL", "DUPLICATE_REMOVAL"}:
                    final = GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
                elif vote.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
                    final = GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
                elif vote.regenerate_llgs:
                    final = GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
                else:
                    final = GlobalGoalEvaluationDecision.CONFIRM_BRANCH
                replacement = HighLevelGoalReplacementRequest(
                    rationale=vote.rationale, actor=branch.high_level_goal.actor,
                    generation_project_description=project_description) if vote.rewriting_focus else None
                result = GlobalGoalEvaluationResult(
                    branch_id=branch.branch_id, rationale=vote.rationale, final_decision=final,
                    high_level_evaluation=hlg, replacement_request=replacement,
                    regeneration_feedback=vote.regeneration_feedback, similarity=match,
                    critic_votes={"documentary": local}, evaluation_status="DECIDED")
        results[branch.branch_id] = result
        print(f"[bottom-up] ensemble | branch={branch.branch_id} | cosine={match.get('score')} | "
              f"relationship={match.get('relationship')} | votes={dict(counts)} | "
              f"decision={result.final_decision.value}\n  rationale: {result.rationale}", flush=True)
    # Check all survivors against the original aggregate, not a mutated order-dependent graph.
    kept = {key for key, result in results.items() if result.final_decision in {
        GlobalGoalEvaluationDecision.CONFIRM_BRANCH, GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS}}
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
