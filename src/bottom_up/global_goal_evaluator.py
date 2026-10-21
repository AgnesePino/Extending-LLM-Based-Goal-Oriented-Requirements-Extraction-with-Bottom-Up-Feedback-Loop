<<<<<<< HEAD
"""Sequential branch and global bottom-up evaluations with deterministic aggregation."""
=======
"""One global documentary review per critic, using fixed numerical matches."""
>>>>>>> 665ac6d9 (Update experimental pipeline and results)

import json
import os
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from src.bottom_up.goal_similarity import analyze_similarities, build_cosine_payload
from src.data_model import (
<<<<<<< HEAD
    BottomUpHighLevelGoal,
    EvaluatorVoteRecord,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    GoalBranch,
=======
    Actors,
    EvaluatorVoteRecord,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
>>>>>>> 665ac6d9 (Update experimental pipeline and results)
    HighLevelGoalDecision,
    HighLevelGoalEvaluation,
    HighLevelGoalRemovalBasis,
    HighLevelGoalReplacementRequest,
<<<<<<< HEAD
    HighLevelGoals,
    HlgStructuralDecision,
    GlobalHlgReviewResponse,
    HlgReconstructionDiagnosis,
    LowLevelGoalDecision,
    LowLevelGoalEvaluation,
    Actors,
    MissingHighLevelGoalDecision,
    MissingHighLevelGoalEvaluation,
    MissingHighLevelGoalRequest,
    ReconstructionRelationship,
=======
>>>>>>> 665ac6d9 (Update experimental pipeline and results)
)
from src.llm_clients import (
    EvaluatorConversation,
    configured_bottom_up_critic_models,
    generate_evaluator_response,
    loaded_local_evaluator_model,
)

# Configuration

<<<<<<< HEAD
SYSTEM_PROMPT = """You are a software engineering assistant specialized in Goal-Oriented
Requirements Engineering (GORE).
An actor is the stakeholder responsible for or benefiting from a capability.
A high-level goal (HLG) expresses WHAT outcome the stakeholder needs and WHY,
independently of implementation. A low-level goal (LLG) describes HOW that
outcome is realized through a concrete system interaction or effect.
A documented automatic effect does not require a manual trigger by the actor.

Your job is to critique artifacts extracted by another assistant from the
software description. Use that description as the source of truth and evaluate
only the assigned task. Examples illustrate criteria, not project requirements.
Do not invent features or responsibilities. Return one valid JSON object without
Markdown. In rationale give concise feedback (at most 200 words): source evidence,
comparison and justified decision. Follow the task-specific score rules.
"""


BOTTOM_UP_EVALUATOR_MODELS = configured_bottom_up_critic_models()
COMPLETENESS_EVALUATOR_MODELS = BOTTOM_UP_EVALUATOR_MODELS
if any(not model_id for model_id in BOTTOM_UP_EVALUATOR_MODELS):
    raise ValueError("Every bottom-up critic model id must be non-empty.")
if len(set(BOTTOM_UP_EVALUATOR_MODELS)) != 3:
    raise ValueError("The bottom-up ensemble requires three different models.")
BOTTOM_UP_EVALUATOR_VOTERS = len(BOTTOM_UP_EVALUATOR_MODELS)
=======
PIPELINE_POLICY_VERSION = "cosine_global_documentary_review_v58_three_limits"
BOTTOM_UP_EVALUATOR_MODELS = configured_bottom_up_critic_models()

if len(set(BOTTOM_UP_EVALUATOR_MODELS)) != 3 or not all(BOTTOM_UP_EVALUATOR_MODELS):
    raise ValueError("The ensemble requires three different non-empty models")
EVALUATOR_VOTING_QUORUM = 2
>>>>>>> 665ac6d9 (Update experimental pipeline and results)
BOTTOM_UP_EVALUATOR_TEMPERATURE = float(
    os.getenv("BOTTOM_UP_EVALUATOR_TEMPERATURE", "0.2")
)
BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS = int(
    os.getenv("BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS", "3")
)
<<<<<<< HEAD
if not 1 <= BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS <= 3:
    raise ValueError(
        "BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS must be between 1 and 3."
    )
EVALUATOR_VOTING_QUORUM = BOTTOM_UP_EVALUATOR_VOTERS // 2 + 1
BOTTOM_UP_QUALITY_THRESHOLD = 4
BOTTOM_UP_EVALUATOR_PROMPTING_MODE = "FEW_SHOT"
PIPELINE_POLICY_VERSION = "reconstruction_first_critic1_v31"
CRITIC1_GLOBAL_EXAMPLES = """Four illustrative complete-list reviews, not project requirements.
For every parent, diagnose HLG' first, then justify the action using the source
and registry. A diagnosis is evidence about decomposition, not the final action.

1. VALID PARENTS, INCOMPLETE OR MISASSOCIATED CHILDREN.
Source: Members create, change and cancel equipment reservations and separately
report damage. Registry: a=Manage equipment reservations, covering all three
reservation actions; b=Report equipment damage. Both belong to Member.
HLG' a describes creating reservations only; HLG' b describes reservations
instead of damage. All current IDs: [a,b].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"PARTIAL_OVERLAP","rationale":"Creation alone does not reconstruct changes and cancellation."},"decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The source supports the complete parent; inspect children rather than narrowing it to creation.","outcome_justification":"Member obtains and manages reservations; damage reporting serves a different outcome."},{"branch_id":"b","reconstruction_diagnosis":{"reconstruction_relationship":"MATCHES_SIBLING","matches_branch_id":"a","rationale":"The reconstruction expresses reservations, not damage reporting."},"decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"Damage reporting is independently documented; the mismatch concerns its children, not the original parent's legitimacy.","outcome_justification":"Member alerts staff to damage; reservation management does not cover that purpose."}]}

2. SOURCE-GROUNDED PARENT REWRITE.
Source: Applicants submit applications and track their status; only staff decide
approval. Registry: a=Submit, track and automatically approve applications,
whose description grants automatic approval to Applicant. HLG' a describes
submission and status tracking. All current IDs: [a].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"PARTIAL_OVERLAP","rationale":"The reconstruction omits the original parent's automatic approval capability."},"decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The source reserves approval to staff; correct the parent's responsibility, independently of the reconstruction mismatch.","outcome_justification":"Applicant submits an application and follows its status; staff retain approval responsibility.","rewriting_focus":"Express application submission and status tracking for Applicant; remove automatic approval, which the source assigns to staff."}]}

3. EQUIVALENT BUT OVER-SPECIFIC.
Source: Buyers prepare and submit orders, saving and resuming drafts within
order preparation. Registry: a=Save and resume order drafts, Buyer. HLG' a
correctly expresses draft persistence. No broader order parent exists. IDs: [a].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"EQUIVALENT","rationale":"The children reconstruct draft saving and resumption."},"decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"Draft persistence is internal to documented order preparation; equivalence establishes coherence but not parent autonomy.","outcome_justification":"The containing outcome is preparing and submitting an order. Later discovery assesses the missing broader parent.","removal_basis":"OVER_SPECIFIC"}]}

4. EQUIVALENT BUT FULLY COVERED.
Source: Residents create, change and cancel pickup bookings.
Registry: a=Manage pickup bookings, explicitly covering all three actions;
b=Book a pickup, covering creation only. Both belong to Resident.
HLG' a covers all three actions; HLG' b covers creation. All current IDs: [a,b].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"EQUIVALENT","rationale":"All booking capabilities match."},"decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The parent represents the complete documented booking outcome and is not covered by the narrower peer.","outcome_justification":"Resident obtains and manages a pickup booking; b is its creation step."},{"branch_id":"b","reconstruction_diagnosis":{"reconstruction_relationship":"EQUIVALENT","rationale":"Creation children correctly reconstruct the narrow parent."},"decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The same actor's parent a wholly covers creation; equivalence does not prevent redundant removal.","outcome_justification":"Creation adds no independent outcome beyond managing the booking.","removal_basis":"FULLY_REDUNDANT","covered_by_branch_id":"a","covered_by_high_level_goal_name":"Manage pickup bookings"}]}
"""
=======

if not 1 <= BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS <= 3:
    raise ValueError("Evaluator attempts must be between 1 and 3")
BOTTOM_UP_EVALUATOR_PROMPTING_MODE = "FEW_SHOT"
CRITIC1_FEW_SHOT_EXAMPLE_COUNT = 4
>>>>>>> 665ac6d9 (Update experimental pipeline and results)

SYSTEM_PROMPT = """Review software goals against the supplied project description.
Use only this source. Similarity scores and labels are fixed inputs, not questions.
Return one JSON object. Give short documentary reasons. Do not invent requirements."""

<<<<<<< HEAD
HLG_EVALUATOR_EXAMPLE_COUNT = 4
LLG_EVALUATOR_EXAMPLE_COUNT = 4
COMPLETENESS_EVALUATOR_EXAMPLE_COUNT = 4


class EvaluatorVotingError(ValueError):
    """Voting failure that preserves every local model response for tracing."""

    def __init__(self, message: str, vote_records: list[EvaluatorVoteRecord]):
        super().__init__(message)
        self.vote_records = vote_records


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def _other_goal_branches(
    branches: list[GoalBranch],
    branch: GoalBranch,
) -> list[dict]:
    """Expose sibling semantics without repeating every embedded parent.

    ``LowLevelGoal.model_dump`` includes the complete parent HLG.  Repeating it
    for every child made the HLG critic prompt exceed the model's input-token limit
    on realistic datasets.  The compact representation retains all evidence
    needed for the responsibility/redundancy audit.
    """
    siblings = []
    actor_key = _key(branch.high_level_goal.actor.name)
    for sibling in branches:
        if (
            sibling.branch_id == branch.branch_id
            or _key(sibling.high_level_goal.actor.name) != actor_key
        ):
            continue
        siblings.append({
            "branch_id": sibling.branch_id,
            "high_level_goal": _compact_hlg(sibling.high_level_goal),
            "low_level_goal_count": len(sibling.low_level_goals),
            # Only same-actor branches can justify FULLY_REDUNDANT or receive
            # misplaced children from this branch.
            "low_level_goal_names": [
                goal.name for goal in sibling.low_level_goals
            ],
        })
    return siblings


def _compact_hlg(goal) -> dict:
    """Return the semantic HLG fields used by critic prompts."""
    return {
        "name": goal.name,
        "description": goal.description,
        "actor": goal.actor.name,
    }


def _compact_llg(
    goal,
    *,
    identifier: str | None = None,
    include_description: bool = True,
) -> dict:
    """Return an LLG without its recursively embedded parent HLG."""
    result = {"name": goal.name}
    if include_description:
        result["description"] = goal.description
    if identifier is not None:
        result["id"] = identifier
    return result


HLG_EVALUATOR_COMPACT_EXAMPLES = """Illustrative functional examples (not dataset annotations):
1. REMOVE fully covered fragment. Source: Applicant enters data, accepts
   declarations and submits a registration. CURRENT and HLG' both express
   "Enter household data". Same-actor branch_002 "Complete registration"
   explicitly covers all three capabilities, including household data.
   Output: {"rationale":"PART 1: DUPLICATE AUDIT: branch_002 fully covers it and survives. SOURCE: household data is required. GRANULARITY: CURRENT is only a step of the covered registration outcome; remove the narrow parent without deleting its required capability from the hierarchy. PART 2: the redundant-removal relationship records the covering sibling; the removal is justified by Part 1.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","reconstruction_relationship":"MATCHES_SIBLING","rewriting_focus":null,"consolidate_with_branch_ids":[],"removal_basis":"FULLY_REDUNDANT","covered_by_branch_id":"branch_002","covered_by_high_level_goal_name":"Complete registration"}
2. KEEP despite incomplete reconstruction. Source: Visitor reads service
   information and simulates a fee. CURRENT covers both; HLG' only reads.
   No other same-actor HLG covers CURRENT.
   Output: {"rationale":"PART 1: DUPLICATE AUDIT: no covering sibling. SOURCE: both outcomes are documented. GRANULARITY: one independent service-assessment outcome. The original already covers simulation; inspect children instead of rewriting it. PART 2: HLG' omits fee simulation, so inspect the actual children of this kept parent.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","reconstruction_relationship":"PARTIAL_OVERLAP","rewriting_focus":null,"consolidate_with_branch_ids":[],"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
3. REMOVE a supporting workflow capability even with a high-level title.
   Source: Buyers prepare and submit purchase orders; unfinished orders can be
   saved and resumed. CURRENT is "Support flexible order preparation", whose
   description only covers saving and resuming drafts. HLG' is equivalent.
   No current parent covers the complete purchase-order outcome.
   Output: {"rationale":"PART 1: DUPLICATE AUDIT: no complete covering parent. SOURCE: saving and resuming are documented within preparing a purchase order. GRANULARITY: this supports the order workflow rather than a separate stakeholder result; a high-level title does not make it autonomous. Remove this parent and let completeness assess the missing order outcome. PART 2: HLG' is equivalent, which confirms internal coherence but does not justify keeping an operational fragment as an HLG.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","reconstruction_relationship":"EQUIVALENT","rewriting_focus":null,"consolidate_with_branch_ids":[],"removal_basis":"OVER_SPECIFIC","covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
4. CONSOLIDATE complementary fragments. Source: one registration requires
   data, declarations and submission. CURRENT and HLG' cover data entry;
   unconfirmed same-actor branch_002 covers declarations and submission.
   Neither parent alone covers the complete outcome.
   Output: {"rationale":"PART 1: DUPLICATE AUDIT: neither fragment fully covers the other. SOURCE: all three capabilities form one registration. GRANULARITY: consolidate complementary steps, preserving data, declarations and submission together. PART 2: reconstruction matches the data-entry fragment; equivalence does not prevent consolidating the original fragments in Part 1.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","reconstruction_relationship":"EQUIVALENT","rewriting_focus":"Express complete registration including data, declarations and submission.","consolidate_with_branch_ids":["branch_002"],"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
"""
LLG_EVALUATOR_EXAMPLES = """Few-shot examples:
1. KEEP complete documented actions. Source: Staff accept, reject or suspend
   applications pending documents. Parent: "Manage application decisions".
   llg_001 accepts, llg_002 rejects and llg_003 suspends pending documents.
   Output: {"rationale":"All documented parent actions are present, supported and distinct.","decision":"KEEP_LOW_LEVEL_GOALS","quality_score":5,"regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[]}
2. REGENERATE missing capability. Source and parent require monitoring
   repository popularity and active contributors. llg_001 retrieves stars;
   llg_002 displays popularity trends. No child identifies contributors.
   Output: {"rationale":"Contributor monitoring is required but absent.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":1,"regeneration_feedback":"Preserve popularity monitoring and add identification of active contributors; return the full decomposition.","unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":["Identify active repository contributors"]}
3. REGENERATE unsupported addition. Source: users submit support requests.
   Parent: "Obtain support". llg_001 submits a request; llg_002 configures
   automatic status notifications, which are not documented.
   Output: {"rationale":"Request submission is valid; the notification capability is undocumented, not invalid merely because it is automatic.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Preserve request submission and remove the unsupported notification capability; return the full decomposition.","unsupported_or_misleading_llg_ids":["llg_002"],"missing_essential_capabilities":[]}
4. REGENERATE redundant UI fragments. Source and parent require submitting a
   completed application. llg_001 opens the form, llg_002 clicks Next,
   llg_003 clicks Submit, and llg_004 sends the completed application.
   The first three add no independently documented capability.
   Output: {"rationale":"Only sending the completed application represents the documented capability; the other entries fragment it into UI mechanics.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Preserve application submission and remove or consolidate the UI-only fragments; return the full decomposition.","unsupported_or_misleading_llg_ids":["llg_001","llg_002","llg_003"],"missing_essential_capabilities":[]}
"""
COVERAGE_EVALUATOR_EXAMPLES = """Few-shot examples:
1. NO_MISSING complete outcome. Source: users search a catalog and reserve
   items. Current HLG "Find and reserve items" explicitly covers both.
   Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Search and reservation map to the current HLG; no independent outcome is missing.","quality_score":5,"missing_goal_requests":[]}
2. FOUND independent actor outcome. Source: Customer places orders and Courier
   updates assigned delivery status. Current HLGs cover only Customer orders;
   Courier is already a known actor.
   Output: {"decision":"MISSING_HIGH_LEVEL_GOALS_FOUND","rationale":"Customer orders are covered, but the documented Courier delivery-status responsibility has no HLG.","quality_score":1,"missing_goal_requests":[{"actor":"Courier","generation_project_description":"Represent the Courier responsibility to manage assigned deliveries and update their status, which is explicit in the source and absent from Customer order goals."}]}
3. NO_MISSING undocumented conventional feature. Source: a librarian registers
   loans and returns. Current HLG "Manage book circulation" covers both;
   audit logging is not documented.
   Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Loans and returns map to circulation; conventional audit logging is not a documented missing outcome.","quality_score":4,"missing_goal_requests":[]}
4. NO_MISSING child-level gap. Source: an Applicant saves drafts and submits
   applications. Current HLG name and description explicitly cover both.
   A proposed extra HLG would only express saving a draft separately.
   Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Draft saving and submission are already covered by the current parent. A missing implementation step belongs to decomposition repair, not another HLG.","quality_score":5,"missing_goal_requests":[]}
"""



def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator response does not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _critic1_json_object(text: str) -> dict:
    """Normalize a complete JSON list without repairing or dropping decisions."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    if cleaned.startswith("["):
        # Decode the whole list: truncated JSON and trailing data must still fail.
        return {"decisions": json.loads(cleaned)}
    return _json_object(text)


def evaluation_status(evaluation):
    """Distinguish scheduling from invalid votes and semantic disagreement."""
    if evaluation.final_decision != GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE:
        return "DECIDED"
    if "CHILD_REVIEW_DEFERRED_PENDING_STRUCTURAL_ACTIONS" in evaluation.rationale:
        return "DEFERRED_STRUCTURAL_ACTIONS"
    if "Insufficient valid reconstruction diagnoses" in evaluation.rationale:
        return "INSUFFICIENT_VALID_DIAGNOSES"
    votes = evaluation.critic_votes
    task = "llg" if evaluation.rationale.startswith("LLG evaluation failed:") else "hlg"
    records = votes.get(task, [])
    if any(record.status == "INVALID" for record in records):
        return "NO_QUORUM_WITH_INVALID_VOTES"
    return "NO_QUORUM_DISAGREEMENT"


def _score_matches_decision(evaluation: BaseModel) -> bool:
    """Accept only decisions that apply the bottom-up quality threshold."""
    passing = evaluation.quality_score >= BOTTOM_UP_QUALITY_THRESHOLD
    if evaluation.decision in {
        HighLevelGoalDecision.KEEP,
        LowLevelGoalDecision.KEEP,
        MissingHighLevelGoalDecision.NO_MISSING,
    }:
        return passing
    if evaluation.decision in {
        HighLevelGoalDecision.REWRITE,
        HighLevelGoalDecision.REMOVE,
        LowLevelGoalDecision.REGENERATE,
        MissingHighLevelGoalDecision.FOUND,
    }:
        return not passing
    return False


def _operation_key(vote):
    """A rewrite is authorized only for the same complete consolidation group."""
    return (vote.decision, tuple(sorted(getattr(vote, "consolidate_with_branch_ids", []))))


def _operation_distribution(votes):
    return {decision.value + (":" + ",".join(group) if group else ""): count
            for (decision, group), count in Counter(_operation_key(v) for v in votes).items()}


def _parse_vote_payload(model: type[BaseModel], payload: dict):
    """Derive HLG bookkeeping without asking the model for redundant fields.

    Legacy score/routing fields remain visible in raw_response, but are not
    authoritative on this live path. Archived votes are never rewritten.
    """
    if model is HighLevelGoalEvaluation:
        if payload.get("rewrite_limit_reached"):
            raise ValueError("rewrite_limit_reached is an orchestrator flag, not a critic decision")
        payload = dict(payload)
        keep = payload.get("decision") == HighLevelGoalDecision.KEEP
        equivalent = payload.get("reconstruction_relationship") == ReconstructionRelationship.EQUIVALENT
        payload.update(
            requires_llg_evaluation=keep and not equivalent,
            quality_score=5 if keep and equivalent else 4 if keep else 2,
            quality_score_origin="decision_policy",
        )
    return model.model_validate(payload)


def _evaluate(
    prompt: str,
    model: type[BaseModel],
    conversation: EvaluatorConversation | None = None,
    memory_label: str | None = None,
    evaluator_models: tuple[str, ...] | None = None,
    manage_model_lifecycle: bool = True,
    record_conversation: bool = True,
    vote_validator: Callable[[BaseModel], None] | None = None,
):
    conversation = None  # Deliberately ignore historical chat input.
    voter_models = evaluator_models or BOTTOM_UP_EVALUATOR_MODELS
    if not voter_models:
        raise ValueError("At least one evaluator model is required.")
    if len(set(voter_models)) != len(voter_models):
        raise ValueError("Each evaluator voter must use a different model.")
    voter_count = len(voter_models)
    voting_quorum = voter_count // 2 + 1

    def request_vote(evaluator_model: str):
        last_error = "unknown evaluator error"
        last_response = None
        for _ in range(BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS):
            try:
                response = generate_evaluator_response(
                    prompt,
                    SYSTEM_PROMPT,
                    conversation=conversation,
                    temperature=BOTTOM_UP_EVALUATOR_TEMPERATURE,
                    model=evaluator_model,
                )
                last_response = response
                evaluation = _parse_vote_payload(model, _json_object(response))
                if getattr(evaluation, "rewrite_limit_reached", False):
                    raise ValueError("rewrite_limit_reached is an orchestrator flag, not a critic decision")
                if not _score_matches_decision(evaluation):
                    raise ValueError(
                        "quality_score conflicts with the selected decision"
                    )
                if vote_validator is not None:
                    vote_validator(evaluation)
                return evaluation, None, evaluator_model, response
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        return None, last_error, evaluator_model, last_response

    # LM Studio evaluates one local model at a time. The completed response is
    # retained in ``vote_results`` before the model instance is unloaded, then
    # the next voter is loaded. This prevents the three GGUF models from
    # occupying GPU/RAM simultaneously.
    vote_results = []
    for evaluator_model in voter_models:
        try:
            if manage_model_lifecycle:
                with loaded_local_evaluator_model(evaluator_model):
                    vote_result = request_vote(evaluator_model)
            else:
                vote_result = request_vote(evaluator_model)
        except Exception as exc:
            raise RuntimeError(
                f"LM Studio lifecycle failed for evaluator {evaluator_model!r}. "
                "Refusing to load the next voter until the current model has "
                "been safely unloaded."
            ) from exc
        vote_results.append(vote_result)

    valid_votes = [vote for vote, _, _, _ in vote_results if vote is not None]
    rejected_reasons = Counter(
        error for vote, error, _, _ in vote_results if vote is None and error
    )
    rejection_summary = "; ".join(
        f"{count}x {reason}" for reason, count in rejected_reasons.items()
    )
    def decoded_response(raw):
        try:
            return _json_object(raw or "")
        except (ValueError, TypeError):
            return None

    vote_records = [
        EvaluatorVoteRecord(
            model=evaluator_model,
            status="VALID" if vote is not None else "INVALID",
            decision=(vote.decision.value if vote is not None else None),
            raw_response=raw_response,
            parsed_response=(
                vote.model_dump(mode="json") if vote is not None else decoded_response(raw_response)
            ),
            error=error,
        )
        for vote, error, evaluator_model, raw_response in vote_results
    ]
    decision_counts = Counter(vote.decision for vote in valid_votes)
    if not decision_counts:
        raise EvaluatorVotingError(
            f"No valid {model.__name__} votes were returned by the evaluator. "
            f"Rejected votes: {rejection_summary or 'none recorded'}.",
            vote_records,
        )
    # Quorum is calculated on the action. Requiring agreement on the complete
    # HLG tuple made a 2/3 KEEP majority inconclusive whenever the two voters
    # differed only between EQUIVALENT and PARTIAL_OVERLAP. The representative
    # vote remains internally coherent; for KEEP disagreements the stricter
    # valid route (Critic 2 enabled) wins.
    winning_operation, winning_count = Counter(_operation_key(v) for v in valid_votes).most_common(1)[0]
    winning_decision = winning_operation[0]
    aggregation_policy = "MAJORITY"
    if winning_count < voting_quorum and aggregation_policy == "MAJORITY":
        raise EvaluatorVotingError(
            f"No {model.__name__} decision quorum: "
            f"{_operation_distribution(valid_votes)}; "
            f"required {voting_quorum}/{voter_count}. "
            f"Rejected votes: {rejection_summary or 'none'}.",
            vote_records,
        )

    winning_votes = [
        vote for vote in valid_votes
        if _operation_key(vote) == winning_operation
    ]

    relationship_strictness = {
        ReconstructionRelationship.EQUIVALENT: 0,
        ReconstructionRelationship.PARTIAL_OVERLAP: 1,
        ReconstructionRelationship.DIFFERENT: 2,
        ReconstructionRelationship.MATCHES_SIBLING: 3,
        ReconstructionRelationship.NOT_RECONSTRUCTABLE: 3,
    }

    def representative_rank(vote: BaseModel):
        evidence_count = (
            len(getattr(vote, "unsupported_or_misleading_llg_ids", []))
            + len(getattr(vote, "missing_essential_capabilities", []))
            + len(getattr(vote, "missing_goal_requests", []))
        )
        if (
            isinstance(vote, HighLevelGoalEvaluation)
            and vote.decision == HighLevelGoalDecision.KEEP
        ):
            return (
                int(vote.requires_llg_evaluation),
                relationship_strictness[vote.reconstruction_relationship],
                -vote.quality_score,
                evidence_count,
            )
        return (0, 0, 0, evidence_count)

    representative = max(
        winning_votes,
        key=representative_rank,
    )
    model_votes = ", ".join(
        f"{evaluator_model}="
        f"{vote.decision.value if vote is not None else 'INVALID'}"
        for vote, _, evaluator_model, _ in vote_results
    )
    result = representative.model_copy(update={
        "rationale": (
            f"Voting result {winning_count}/{voter_count} "
            f"for {winning_decision.value}. Model votes: {model_votes}. "
            f"Aggregation policy: {aggregation_policy}. "
            f"{representative.rationale}"
        ),
        "voter_count": voter_count,
        "valid_vote_count": len(valid_votes),
        "winning_vote_count": winning_count,
        "vote_distribution": {
            decision.value: count for decision, count in decision_counts.items()
        },
        "evaluator_votes": vote_records,
    })
    if isinstance(result, HighLevelGoalEvaluation):
        result.operation_vote_distribution = _operation_distribution(valid_votes)
    result = _merge_completeness_requests(result, winning_votes)
    if conversation is not None and record_conversation:
        conversation.record_exchange(
            memory_label or f"{model.__name__} evaluation",
            result.model_dump_json(),
        )
    return result


def _merge_completeness_requests(result, winning_votes):
    if (
        isinstance(result, MissingHighLevelGoalEvaluation)
        and result.decision == MissingHighLevelGoalDecision.FOUND
    ):
        # Quorum authorizes completeness, not just one representative's request.
        # Merge all positive voters' descriptions, keeping actors separate.
        by_actor = {}
        for vote in winning_votes:
            for request in vote.missing_goal_requests:
                actor_key = _key(request.actor)
                actor_name, descriptions = by_actor.setdefault(
                    actor_key, (request.actor, {})
                )
                description = request.generation_project_description.strip()
                if description:
                    descriptions.setdefault(_key(description), description)
        result = result.model_copy(update={
            "missing_goal_requests": [
                MissingHighLevelGoalRequest(
                    actor=actor_name,
                    generation_project_description="\n\n".join(
                        f"Completeness evidence {i}: {description}"
                        for i, description in enumerate(descriptions.values(), 1)
                    ),
                )
                for actor_name, descriptions in by_actor.values()
                if descriptions
            ],
            "rationale": result.rationale + "\nPositive critic rationales:\n"
                + "\n".join(dict.fromkeys(v.rationale for v in winning_votes)),
        })
    return result


def _aggregate_saved_votes(
    vote_records: list[EvaluatorVoteRecord],
    model: type[BaseModel],
    conversation: EvaluatorConversation | None,
    memory_label: str,
):
    """Aggregate model-major votes after all voters have released the model."""
    valid_votes = [
        model.model_validate(record.parsed_response)
        for record in vote_records
        if record.status == "VALID" and record.parsed_response is not None
    ]
    voter_count = len(vote_records)
    voting_quorum = voter_count // 2 + 1
    decision_counts = Counter(vote.decision for vote in valid_votes)
    if not decision_counts:
        raise EvaluatorVotingError(
            f"No valid {model.__name__} votes were returned by the evaluator.",
            vote_records,
        )
    winning_operation, winning_count = Counter(_operation_key(v) for v in valid_votes).most_common(1)[0]
    winning_decision = winning_operation[0]
    aggregation_policy = "MAJORITY"
    if winning_count < voting_quorum and aggregation_policy == "MAJORITY":
        raise EvaluatorVotingError(
            f"No {model.__name__} decision quorum: "
            f"{_operation_distribution(valid_votes)}; "
            f"required {voting_quorum}/{voter_count}.",
            vote_records,
        )

    winning_votes = [
        vote for vote in valid_votes if _operation_key(vote) == winning_operation
    ]
    relationship_strictness = {
        ReconstructionRelationship.EQUIVALENT: 0,
        ReconstructionRelationship.PARTIAL_OVERLAP: 1,
        ReconstructionRelationship.DIFFERENT: 2,
        ReconstructionRelationship.MATCHES_SIBLING: 3,
        ReconstructionRelationship.NOT_RECONSTRUCTABLE: 3,
    }

    def representative_rank(vote: BaseModel):
        evidence_count = (
            len(getattr(vote, "unsupported_or_misleading_llg_ids", []))
            + len(getattr(vote, "missing_essential_capabilities", []))
            + len(getattr(vote, "missing_goal_requests", []))
        )
        if (
            isinstance(vote, HighLevelGoalEvaluation)
            and vote.decision == HighLevelGoalDecision.KEEP
        ):
            return (
                int(vote.requires_llg_evaluation),
                relationship_strictness[vote.reconstruction_relationship],
                -vote.quality_score,
                evidence_count,
            )
        return (0, 0, 0, evidence_count)

    representative = max(winning_votes, key=representative_rank)
    model_votes = ", ".join(
        f"{record.model}={record.decision or 'INVALID'}"
        for record in vote_records
    )
    result = representative.model_copy(update={
        "rationale": (
            f"Voting result {winning_count}/{voter_count} for "
            f"{winning_decision.value}. Model votes: {model_votes}. "
            f"Aggregation policy: {aggregation_policy}. "
            f"{representative.rationale}"
        ),
        "voter_count": voter_count,
        "valid_vote_count": len(valid_votes),
        "winning_vote_count": winning_count,
        "vote_distribution": {
            decision.value: count for decision, count in decision_counts.items()
        },
        "evaluator_votes": vote_records,
    })
    if isinstance(result, HighLevelGoalEvaluation):
        result.operation_vote_distribution = _operation_distribution(valid_votes)
    result = _merge_completeness_requests(result, winning_votes)
    if conversation is not None:
        conversation.record_exchange(memory_label, result.model_dump_json())
    return result


def _validate_hlg_context(evaluation, branch, siblings, confirmed_branch_ids=None):
    """Check references before a model's vote may participate in a majority."""
    if (not branch.low_level_goals) != (
        evaluation.reconstruction_relationship == ReconstructionRelationship.NOT_RECONSTRUCTABLE
    ):
        raise ValueError("reconstruction_relationship conflicts with the presence of LLGs")
    eligible = {item["branch_id"] for item in siblings}
    for identifier in evaluation.consolidate_with_branch_ids:
        if identifier == branch.branch_id or identifier not in eligible:
            raise ValueError("Consolidation requires OTHER existing same-actor branches")
    if (evaluation.decision == HighLevelGoalDecision.REMOVE
            and evaluation.removal_basis == HighLevelGoalRemovalBasis.FULLY_REDUNDANT):
        if evaluation.covered_by_branch_id == branch.branch_id:
            raise ValueError("FULLY_REDUNDANT cannot use the current branch itself as covering target")
        if not any(
            item["branch_id"] == evaluation.covered_by_branch_id
            and _key(item["high_level_goal"]["name"])
            == _key(evaluation.covered_by_high_level_goal_name or "")
            for item in siblings
        ):
            raise ValueError("FULLY_REDUNDANT requires the exact ID and name of ANOTHER existing same-actor branch")


def evaluate_original_hlg(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals | None = None,
    conversation: EvaluatorConversation | None = None,
    current_branches: list[GoalBranch] | None = None,
    evaluator_models: tuple[str, ...] | None = None,
    manage_model_lifecycle: bool = True,
    record_conversation: bool = True,
    apply_destructive_gate: bool = True,
    confirmed_branch_ids: set[str] | None = None,
) -> HighLevelGoalEvaluation:
    """Compatibility single-branch API; the notebook uses evaluate_all_branches.

    The live cycle's Critic 1 uses _critic1_list_prompt, not this legacy prompt.
    """
    current_hlgs = current_hlgs or HighLevelGoals(goals=[branch.high_level_goal])
    if current_branches is None:
        current_branches = []
        current_assigned = False
        for index, goal in enumerate(current_hlgs.goals, start=1):
            if not current_assigned and goal == branch.high_level_goal:
                current_branches.append(branch)
                current_assigned = True
            else:
                current_branches.append(GoalBranch(
                    branch_id=f"legacy_branch_{index:03d}",
                    high_level_goal=goal,
                    low_level_goals=[],
                ))
    sibling_branches = _other_goal_branches(current_branches, branch)
    all_parents = [
        {"branch_id": item.branch_id,
         "is_current_branch": item.branch_id == branch.branch_id,
         "confirmed": item.branch_id in (confirmed_branch_ids or set()),
         "high_level_goal": item.high_level_goal.model_dump(mode="json")}
        for item in current_branches
    ]
    duplicate_audit_order = [
        item.branch_id
        for item in sorted(
            current_branches,
            key=lambda item: item.branch_id not in (confirmed_branch_ids or set()),
        )
        if _key(item.high_level_goal.actor.name) == _key(branch.high_level_goal.actor.name)
    ]
    prompt = f"""### High-Level Goal Evaluation
You are provided with a software description, an original high-level goal,
a goal reconstructed from its low-level decomposition (HLG'), and the complete
current HLG registry. Your job is to critique whether the ORIGINAL goal
expresses a distinct, documented functional outcome for the appropriate actor.
Complete two distinct parts in this order. Part 1 decides what to do with
the ORIGINAL HLG. Part 2 uses HLG' to diagnose the decomposition and classify
the relationship. Return one combined response, not two separate votes.

### PART 1 -- ORIGINAL HLG: source, duplicates and abstraction level
Use the software description and the original HLG registry. Read the current
goal's name, description and actor together. Decide its validity independently
of HLG'; an accurate reconstruction cannot validate a defective original.

Check these criteria in order; duplicate assessment comes FIRST:
1. DUPLICATE AUDIT: compare CURRENT with every OTHER registry entry, including
   confirmed goals. Two individually supported goals can duplicate the same
   outcome. Only another SAME-ACTOR branch may cover CURRENT.
   For equivalent duplicates preserve a confirmed equivalent first, otherwise
   the most source-faithful and complete goal; break ties by the supplied stable
   order. Apply the same rule to both members. If CURRENT survives, continue
   the checks; otherwise remove CURRENT and identify its surviving cover.
   A narrower goal is redundant only if the cover preserves its whole outcome.
   Prefer the broader documented stakeholder outcome over a fully covered
   detailed step, even when that detailed branch was previously confirmed.
   Similar topics, partial overlap or different responsibilities are insufficient.
   Use source evidence to verify the cover; do not select KEEP or REWRITE before
   completing this audit. State the duplicate finding first in the rationale.
2. SOURCE: identify the documented functional outcome and actor responsibility.
   A stakeholder may benefit from an automatic action without triggering it.
3. GRANULARITY: prefer outcome-level HLGs representing coherent stakeholder
   intentions rather than individual operations, fields, screens or workflow
   steps. Keep these operational details in the LLG decomposition. A detailed
   step already fully covered by a same-actor parent must be removed as
   FULLY_REDUNDANT. If no covering parent exists, REMOVE the over-detailed
   operational parent with OVER_SPECIFIC and null covering fields. The later
   completeness analysis must assess any resulting missing stakeholder outcome;
   it may request a new broader HLG. An immediate replacement is not required.
   REWRITE or CONSOLIDATE remain available for other recoverable scope defects
   and complementary outcomes. Excessive operational detail is a parent defect
   even when the step is documented. Prefer a broader coherent
   outcome, not a vague umbrella such as "Manage everything": retain clear
   actor, purpose and scope, without inventing capabilities or merging
   independent outcomes. Being documented, user-visible or phrased as
   "Enable/Support/Ensure" does not make a capability an independent HLG.
   Ask whether it is the stakeholder's end-to-end result or only a prerequisite,
   business rule, configuration choice or lifecycle phase of that result.
   Draft saving/resumption, consent acceptance, duplicate-data checks and
   payment-option selection normally belong inside their documented workflow,
   unless the source establishes an independently meaningful stakeholder outcome.
   To KEEP such a candidate, explain what independent result would be lost at
   parent level and why it cannot be represented in that workflow's LLGs.
   Otherwise REMOVE it as FULLY_REDUNDANT if a complete cover exists, or
   OVER_SPECIFIC without a cover; do not demand an exact duplicate title.
   A genuinely independent narrow outcome remains valid when the source gives
   it a separate purpose; do not delete outcomes merely to reach a target count.

PART 1 action:
- KEEP: the original is supported, independently meaningful and the selected
  survivor, if duplicates exist. A reconstruction mismatch alone is not a
  parent defect; different wording or optional API/UI details are not defects.
- REWRITE: cite a concrete source-grounded defect of the ORIGINAL and explain
  why its name AND description do not already cover the required outcome.
  For over-specific parents, explain why the current goal is only an operational
  fragment and identify the broader documented outcome that should replace it.
  Set rewriting_focus. To consolidate complementary fragments of ONE outcome,
  list ALL other eligible same-actor members in consolidate_with_branch_ids
  and explain every capability to preserve. Earlier confirmations are revisable;
  ordinary rewrite uses []. Do not merge independent outcomes. The same group
  must obtain two matching votes before application.
- REMOVE: use UNSUPPORTED for a wholly undocumented intention, NON_FUNCTIONAL
  for a pure quality/deployment constraint with no functional core, or
  FULLY_REDUNDANT for full coverage by an exact eligible same-actor branch,
  or OVER_SPECIFIC for an operational step incorrectly promoted to HLG, even
  without an existing cover. For OVER_SPECIFIC explain why it belongs at LLG
  level and leave missing-outcome recovery to the later completeness analysis.
  For mixed functional/non-functional scope defects, preserve the functional
  core through rewriting; this does not prohibit OVER_SPECIFIC removal.
  Redundancy requires the cover's exact ID and name. Never target CURRENT or
  delete the surviving cover; remove only CURRENT. Other removal bases have
  null covering fields. A duplicate need not be unsupported to be removed.

### PART 2 -- RECONSTRUCTED HLG': decomposition diagnosis
HLG' is inferred from the children without seeing the original parent or the
source. It is a fallible summary, not a requirement or a replacement parent.
Compare it with the ORIGINAL HLG and, when relevant, the other registry goals:
- EQUIVALENT: the same essential outcome and scope.
- PARTIAL_OVERLAP: only some essential capabilities match.
- MATCHES_SIBLING: HLG' expresses another same-actor branch's outcome.
- DIFFERENT: a different outcome with no matching same-actor sibling.
- NOT_RECONSTRUCTABLE: the branch has no children.
A match with a sibling diagnoses possible misplaced children; it does not
by itself prove the original HLG is duplicated. If only HLG' omits a capability
already stated in the original name or description, retain a valid original
and let Critic 2 verify actual children. You do not see those children here.
For a kept parent, a non-equivalent or absent reconstruction requires the
child audit; ordinary equivalent KEEP does not. Python derives that routing.
If Part 1 requires removal or rewriting, reconstruction agreement does not
cancel it. For FULLY_REDUNDANT removal, the response contract requires
MATCHES_SIBLING when children exist, or NOT_RECONSTRUCTABLE when absent;
the redundancy evidence must come from the original-goal comparison in Part 1.

### COMBINED RESPONSE
Give a concise rationale with PART 1: DUPLICATE AUDIT first, SOURCE and
GRANULARITY evidence, and the chosen action; then PART 2: the reconstruction
relationship and its diagnostic meaning. Report conclusions and evidence,
not an internal reasoning transcript. Return only the requested fields.
Do not return quality_score, requires_llg_evaluation or operational flags.
rewriting_focus is null except for REWRITE; removal fields are null except
for REMOVE. Registry IDs are references, not semantic evidence.

### CASE INPUTS
CURRENT branch ID: {branch.branch_id}
The is_current_branch=true registry row is CURRENT itself, not another duplicate.
Stable same-actor survivor order (confirmed first; otherwise registry order):
{json.dumps(duplicate_audit_order)}
Eligible redundant-removal target IDs (other branches, same actor only):
{json.dumps([item["branch_id"] for item in sibling_branches])}
Eligible consolidation member IDs (other same-actor branches):
{json.dumps([item["branch_id"] for item in sibling_branches])}

Examples (illustrative situations, not extra facts about this project):
{HLG_EVALUATOR_COMPACT_EXAMPLES}

Project description:
{project_description}

Original HLG:
branch_id: {branch.branch_id}
{branch.high_level_goal.model_dump_json()}

Reconstructed HLG':
{reconstruction.reconstructed_high_level_goal}

All current HLGs (IDs are references, not semantic evidence):
{json.dumps(all_parents, ensure_ascii=False)}

Return only JSON:
{{"rationale":"...","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL | REWRITE_ORIGINAL_HIGH_LEVEL_GOAL | REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","reconstruction_relationship":"EQUIVALENT | PARTIAL_OVERLAP | MATCHES_SIBLING | DIFFERENT | NOT_RECONSTRUCTABLE","rewriting_focus":null,"consolidate_with_branch_ids":[],"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}}"""
    evaluation = _evaluate(
        prompt,
        HighLevelGoalEvaluation,
        conversation,
        memory_label=(
            "HLG evaluation for actor "
            f"'{branch.high_level_goal.actor.name}', goal "
            f"'{branch.high_level_goal.name}'"
        ),
        evaluator_models=evaluator_models,
        manage_model_lifecycle=manage_model_lifecycle,
        record_conversation=record_conversation,
        vote_validator=lambda vote: _validate_hlg_context(vote, branch, sibling_branches, confirmed_branch_ids),
    )
    try:
        _validate_hlg_context(evaluation, branch, sibling_branches, confirmed_branch_ids)
    except ValueError as exc:
        raise _invalid_domain_vote(evaluation, str(exc)) from exc
    removal_is_supported = (
        evaluation.removal_basis in {HighLevelGoalRemovalBasis.UNSUPPORTED, HighLevelGoalRemovalBasis.NON_FUNCTIONAL, HighLevelGoalRemovalBasis.OVER_SPECIFIC}
        or (
            evaluation.removal_basis
            == HighLevelGoalRemovalBasis.FULLY_REDUNDANT
            and any(
                sibling["branch_id"] == evaluation.covered_by_branch_id
                and _key(sibling["high_level_goal"].get("name", ""))
                == _key(evaluation.covered_by_high_level_goal_name or "")
                and _key(
                    sibling["high_level_goal"].get("actor", "")
                ) == _key(branch.high_level_goal.actor.name)
                for sibling in sibling_branches
            )
        )
    )
    if apply_destructive_gate and evaluation.decision == HighLevelGoalDecision.REMOVE and (
        evaluation.winning_vote_count < EVALUATOR_VOTING_QUORUM
        or not removal_is_supported
    ):
        reasons = []
        if evaluation.winning_vote_count < EVALUATOR_VOTING_QUORUM:
            reasons.append("REMOVE did not reach the voting quorum")
        if not removal_is_supported:
            reasons.append(
                "no valid unsupported/redundant removal basis with an exact "
                "same-actor sibling branch was supplied"
            )
        return evaluation.model_copy(update={
            "rationale": (
                f"{evaluation.rationale} Destructive-action gate: "
                f"{'; '.join(reasons)}; the HLG is retained in this state."
            ),
            "decision": HighLevelGoalDecision.KEEP,
            "quality_score": BOTTOM_UP_QUALITY_THRESHOLD,
            "quality_score_origin": "removal_guard_policy",
            "requires_llg_evaluation": (
                evaluation.reconstruction_relationship
                != ReconstructionRelationship.EQUIVALENT
            ),
            "rewriting_focus": None,
            "removal_basis": None,
            "covered_by_branch_id": None,
            "covered_by_high_level_goal_name": None,
        })
    return evaluation


def build_replacement_request(
    project_description: str,
    branch: GoalBranch,
    evaluation: HighLevelGoalEvaluation,
) -> HighLevelGoalReplacementRequest:
    """Build the generator request from the HLG evaluator output."""
    return HighLevelGoalReplacementRequest(
        rationale=evaluation.rationale,
        generation_project_description=(
            f"{project_description}\n\n"
            f"Rewrite the functional intention for actor "
            f"'{branch.high_level_goal.actor.name}' using this focus: "
            f"{evaluation.rewriting_focus}"
        ),
        actor=branch.high_level_goal.actor,
    )


def _invalid_domain_vote(evaluation, message):
    """Keep the original model output when post-parsing validation fails."""
    error = ValueError(message)
    error.vote_records = [record.model_copy(update={
        "status": "INVALID", "error": f"ValueError: {message}",
    }) for record in evaluation.evaluator_votes]
    return error


def _normalize_llg_evaluation(
    branch: GoalBranch,
    evaluation: LowLevelGoalEvaluation,
) -> LowLevelGoalEvaluation:
    """Discard unusable repair evidence without turning failures into KEEP."""
    if not branch.low_level_goals and evaluation.decision == LowLevelGoalDecision.KEEP:
        raise _invalid_domain_vote(evaluation,
            "An empty decomposition cannot be confirmed; a critic must identify the required documented repair.")
    valid_ids = {
        f"llg_{index:03d}"
        for index, _ in enumerate(branch.low_level_goals, start=1)
    }
    supported_ids = [
        item
        for item in evaluation.unsupported_or_misleading_llg_ids
        if item in valid_ids
    ]
    missing = [
        item.strip()
        for item in evaluation.missing_essential_capabilities
        if item.strip()
    ]
    if (
        evaluation.decision == LowLevelGoalDecision.REGENERATE
        and not (supported_ids or missing)
    ):
        raise _invalid_domain_vote(evaluation,
            "REGENERATE_LOW_LEVEL_GOALS did not identify a valid current LLG "
            "or a non-empty missing essential capability; refusing to convert "
            "invalid repair evidence into KEEP_LOW_LEVEL_GOALS"
        )
    return evaluation.model_copy(update={
        "unsupported_or_misleading_llg_ids": supported_ids,
        "missing_essential_capabilities": missing,
    })


def evaluate_llg_decomposition(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals | None = None,
    conversation: EvaluatorConversation | None = None,
    evaluator_models: tuple[str, ...] | None = None,
    manage_model_lifecycle: bool = True,
    record_conversation: bool = True,
    upstream_hlg_evaluation: HighLevelGoalEvaluation | None = None,
) -> LowLevelGoalEvaluation:
    current_hlgs = current_hlgs or HighLevelGoals(goals=[branch.high_level_goal])
    llgs = [
        _compact_llg(goal, identifier=f"llg_{index:03d}")
        for index, goal in enumerate(branch.low_level_goals, start=1)
    ]
    prompt = f"""### Low-Level Goal Evaluation
You are provided with a software description, a high-level goal, its low-level
goals and the reconstructed HLG'. Your job is to critique whether these children
correctly and completely realize the parent through documented functional
interactions or effects. Keep the parent fixed and evaluate its decomposition.
Judge actual children against the SOURCE and original parent, independently of
all prior judgments. HLG' is a fallible diagnostic summary, not a requirement.
Verify any suspected mismatch in the actual children. For each child check
documentary support, correct scope and duplication.
Check that every essential documented parent capability has a child.
Do not invent conventional features or require stylistic changes. Differences
in wording or presentation alone do not justify regeneration. A single actual
unsupported or out-of-scope child is sufficient to request a full regeneration.
The current parent is fixed for this audit; do not rewrite it or compensate
for an unsupported parent clause by inventing children.
Use the same actor criterion as the parent audit: document-supported automatic
notifications and system operations do not require a manual trigger by the
parent actor. Check responsibility and scope, not an invented UI interaction.

KEEP_LOW_LEVEL_GOALS: complete, supported, non-redundant children, score 5,
null regeneration_feedback, empty defect lists.
REGENERATE_LOW_LEVEL_GOALS: score 0..3, list actual unsupported/misleading
llg_NNN IDs and/or specific essential missing capabilities. Give a short repair
request: PRESERVE valid actions; CORRECT/REMOVE identified defects; ADD missing
source-grounded actions; return the entire corrected decomposition.
For empty children request documented interactions, not an empty KEEP.
Rationale: source evidence, actual child IDs or missing action, required correction.

Examples (not requirements for this project):
{LLG_EVALUATOR_EXAMPLES}

Project description:
{project_description}

Current parent:
{branch.high_level_goal.model_dump_json()}

Current children:
{json.dumps(llgs, ensure_ascii=False)}

Reconstructed HLG' (diagnostic only):
{reconstruction.reconstructed_high_level_goal}

Return only JSON:
{{"rationale":"...","decision":"KEEP_LOW_LEVEL_GOALS | REGENERATE_LOW_LEVEL_GOALS","quality_score":0,"regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[]}}"""
    evaluation = _evaluate(
        prompt,
        LowLevelGoalEvaluation,
        conversation,
        memory_label=(
            "LLG decomposition evaluation for parent "
            f"'{branch.high_level_goal.name}'"
        ),
        evaluator_models=evaluator_models,
        manage_model_lifecycle=manage_model_lifecycle,
        record_conversation=record_conversation,
    )

    evaluation = _normalize_llg_evaluation(branch, evaluation)
    return evaluation


def evaluate_missing_high_level_goals(
    project_description: str,
    current_hlgs: HighLevelGoals,
    actors: Actors,
    conversation: EvaluatorConversation | None = None,
    *, evaluator_models: tuple[str, ...] | None = None,
    manage_model_lifecycle: bool = True, record_conversation: bool = True,
) -> MissingHighLevelGoalEvaluation:
    """Check HLG coverage after branch confirmation or audited repair exhaustion."""
    prompt = f"""### High-Level Goal Completeness Evaluation
You are provided with a software description, known actors and the complete
current set of high-level goals. Your job is to critique whether this set
represents every independent functional outcome required for those actors.
Identify documented outcomes not yet covered; do not produce final goal objects.
If an operational parent has been removed, recover a missing broader documented
stakeholder outcome when needed, rather than proposing a workflow step as an HLG.
For each documented actor outcome, check semantic coverage in current HLG names
AND descriptions. Report only independently missing high-level outcomes.
A missing step inside an existing parent belongs to its LLG audit, not a new HLG.
Do not add rewordings, implementation details, conventional features or new actors.
A source channel or exception alone does not automatically require a new HLG.
Before deciding, map the source's essential independent actor outcomes to
specific current HLG names in your rationale. Read their descriptions as well.
A generic title such as 'manage services' is not evidence of every related
capability: identify where each outcome is actually covered. An independently
missing outcome needs a new HLG; a missing step within a covered outcome needs
LLG repair, not an extra parent. Do not infer undocumented features.
This task checks missing outcomes only. NO_MISSING is not approval of every
current HLG or of their granularity. Do not create a new HLG merely to split
an existing covered outcome; redundancy and parent quality belong to Critic 1.

NO_MISSING_HIGH_LEVEL_GOALS: no supported independent gap, score 4..5, no requests.
MISSING_HIGH_LEVEL_GOALS_FOUND: score 0..3, one focused request for each distinct
missing outcome. Use an existing actor name, cite source support, and explain
why no current HLG covers it. Requests describe the gap, not final goal objects.
In rationale, explicitly describe WHAT is missing, WHICH actor needs it,
WHERE the source requires it, and WHY the existing HLGs do not cover it.
Keep each missing_goal_requests description self-contained so the positive
critics' descriptions can be merged into a top-down generation request.
No previous decisions or retired goals are evidence. Assess only current facts.

Examples:
{COVERAGE_EVALUATOR_EXAMPLES}

Project description:
{project_description}

Known actors:
{actors.model_dump_json()}

Current HLGs:
{current_hlgs.model_dump_json()}

Return only JSON:
{{"decision":"NO_MISSING_HIGH_LEVEL_GOALS | MISSING_HIGH_LEVEL_GOALS_FOUND","rationale":"...","quality_score":0,"missing_goal_requests":[{{"actor":"Existing actor","generation_project_description":"Missing outcome, source evidence and difference from existing HLGs"}}]}}"""
    try:
        evaluation = _evaluate(
            prompt,
            MissingHighLevelGoalEvaluation,
            conversation,
            memory_label=(
                "Project-wide HLG coverage evaluation with "
                f"{len(current_hlgs.goals)} current HLGs"
            ),
            evaluator_models=evaluator_models or COMPLETENESS_EVALUATOR_MODELS,
            manage_model_lifecycle=manage_model_lifecycle,
            record_conversation=record_conversation,
        )
    except (ValueError, TypeError) as exc:
        return MissingHighLevelGoalEvaluation(
            decision=MissingHighLevelGoalDecision.INCONCLUSIVE,
            rationale=f"Coverage voting failed: {exc}",
            quality_score=0,
            missing_goal_requests=[],
            voter_count=len(evaluator_models or COMPLETENESS_EVALUATOR_MODELS),
            valid_vote_count=0,
            winning_vote_count=0,
            vote_distribution={},
            evaluator_votes=getattr(exc, "vote_records", []),
        )
    return _restrict_missing_goals_to_existing_actors(evaluation, actors)


def aggregate_missing_high_level_goal_votes(vote_records, actors, conversation=None):
    """Aggregate already completed completeness votes without another GPU lease."""
    try:
        evaluation = _aggregate_saved_votes(vote_records, MissingHighLevelGoalEvaluation,
            conversation, "Project-wide HLG coverage evaluation")
    except (ValueError, TypeError) as exc:
        return MissingHighLevelGoalEvaluation(decision=MissingHighLevelGoalDecision.INCONCLUSIVE,
            rationale=f"Coverage voting failed: {exc}", quality_score=0, missing_goal_requests=[],
            voter_count=len(COMPLETENESS_EVALUATOR_MODELS), valid_vote_count=0, winning_vote_count=0,
            vote_distribution={}, evaluator_votes=getattr(exc, "vote_records", vote_records))
    return _restrict_missing_goals_to_existing_actors(evaluation, actors)


def _restrict_missing_goals_to_existing_actors(
    evaluation: MissingHighLevelGoalEvaluation,
    actors: Actors,
) -> MissingHighLevelGoalEvaluation:
    """Keep all distinct requests whose actors already exist."""
    if evaluation.decision == MissingHighLevelGoalDecision.INCONCLUSIVE:
        return evaluation
    actors_by_name = {_key(actor.name): actor.name for actor in actors.actors}
    valid_requests = []
    for request in evaluation.missing_goal_requests:
        actor_name = actors_by_name.get(_key(request.actor))
        if actor_name is None:
            continue
        normalized_request = request.model_copy(update={"actor": actor_name})
        request_key = (
            _key(normalized_request.actor),
            _key(normalized_request.generation_project_description),
        )
        if any(
            (_key(item.actor), _key(item.generation_project_description))
            == request_key
            for item in valid_requests
        ):
            continue
        valid_requests.append(normalized_request)

    if not valid_requests:
        if evaluation.decision == MissingHighLevelGoalDecision.NO_MISSING:
            return evaluation
        return evaluation.model_copy(update={
            "decision": MissingHighLevelGoalDecision.INCONCLUSIVE,
            "rationale": (
                f"{evaluation.rationale} Proposed requests were discarded "
                "because they did not reference an existing actor."
            ),
            "quality_score": 0,
            "missing_goal_requests": [],
        })
    return evaluation.model_copy(update={"missing_goal_requests": valid_requests})


def _requires_llg_audit(branch: GoalBranch, hlg: HighLevelGoalEvaluation) -> bool:
    """Route by semantic relationship, never by textual equality of goals."""
    return hlg.decision == HighLevelGoalDecision.KEEP and (
        not branch.low_level_goals
        or hlg.rewrite_limit_reached
        or hlg.post_repair_llg_audit_required
        or hlg.reconstruction_relationship != ReconstructionRelationship.EQUIVALENT
    )


def apply_hlg_rewrite_limit(hlg: HighLevelGoalEvaluation, rewrite_count: int,
                            max_rewrites: int = 3) -> HighLevelGoalEvaluation:
    """Retain raw votes while recording a budget-driven KEEP and mandatory LLG audit."""
    if hlg.decision != HighLevelGoalDecision.REWRITE or rewrite_count < max_rewrites:
        return hlg
    payload = hlg.model_dump(mode="python")
    payload.update(decision=HighLevelGoalDecision.KEEP, quality_score=4,
                   quality_score_origin="rewrite_limit_policy",
                   requires_llg_evaluation=True, rewriting_focus=None, consolidate_with_branch_ids=[],
                   rewrite_limit_reached=True,
                   rationale=f"HLG_REWRITE_LIMIT_REACHED ({rewrite_count}/{max_rewrites}). "
                             f"Operational KEEP; child audit required. Original critic: {hlg.rationale}")
    return HighLevelGoalEvaluation.model_validate(payload)


def evaluate_branch(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals,
    conversation: EvaluatorConversation | None = None,
    current_branches: list[GoalBranch] | None = None,
    *, hlg_rewrite_count: int = 0, max_hlg_rewrites: int = 3,
) -> GlobalGoalEvaluationResult:
    try:
        hlg = evaluate_original_hlg(
            project_description,
            branch,
            reconstruction,
            current_hlgs,
            conversation,
            current_branches,
        )
        hlg = apply_hlg_rewrite_limit(hlg, hlg_rewrite_count, max_hlg_rewrites)
    except (ValueError, TypeError) as exc:
        return _inconclusive_result(
            branch,
            f"HLG evaluation failed: {exc}",
            critic_votes={"hlg": getattr(exc, "vote_records", [])},
        )
    replacement = None
    llg = None

    if hlg.decision == HighLevelGoalDecision.REMOVE:
        decision = GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
    else:
        if hlg.decision == HighLevelGoalDecision.REWRITE:
            try:
                replacement = build_replacement_request(
                    project_description, branch, hlg
                )
            except (ValueError, TypeError) as exc:
                return _inconclusive_result(
                    branch,
                    f"HLG replacement request failed: {exc}",
                    hlg=hlg,
                    critic_votes={"hlg": hlg.evaluator_votes},
                )

        if hlg.decision == HighLevelGoalDecision.REWRITE:
            decision = GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        elif _requires_llg_audit(branch, hlg):
            try:
                llg = evaluate_llg_decomposition(
                    project_description,
                    branch,
                    reconstruction,
                    current_hlgs,
                    conversation,
                    upstream_hlg_evaluation=None,
                )
            except (ValueError, TypeError) as exc:
                return _inconclusive_result(
                    branch,
                    f"LLG evaluation failed: {exc}",
                    hlg=hlg,
                    critic_votes={
                        "hlg": hlg.evaluator_votes,
                        "llg": getattr(exc, "vote_records", []),
                    },
                )
            decision = (
                GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
                if llg.decision == LowLevelGoalDecision.REGENERATE
                else GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            )
        else:
            decision = GlobalGoalEvaluationDecision.CONFIRM_BRANCH
    rationale_parts = [hlg.rationale]
    if llg is not None:
        rationale_parts.append(llg.rationale)

    return GlobalGoalEvaluationResult(
        branch_id=branch.branch_id,
        rationale=" ".join(rationale_parts),
        final_decision=decision,
        high_level_evaluation=hlg,
        replacement_request=replacement,
        low_level_evaluation=llg,
        critic_votes={
            "hlg": hlg.evaluator_votes,
            **({"llg": llg.evaluator_votes} if llg is not None else {}),
        },
    )


def _inconclusive_result(
    branch: GoalBranch,
    rationale: str,
    *,
    hlg: HighLevelGoalEvaluation | None = None,
    llg: LowLevelGoalEvaluation | None = None,
    critic_votes: dict[str, list[EvaluatorVoteRecord]] | None = None,
) -> GlobalGoalEvaluationResult:
    hlg = hlg or HighLevelGoalEvaluation(
        rationale=rationale,
        decision=HighLevelGoalDecision.KEEP,
        quality_score=0,
        quality_score_origin="inconclusive_placeholder",
        reconstruction_relationship=ReconstructionRelationship.DIFFERENT,
        requires_llg_evaluation=True,
        voter_count=BOTTOM_UP_EVALUATOR_VOTERS,
        valid_vote_count=0,
        winning_vote_count=0,
        vote_distribution={},
        evaluator_votes=(critic_votes or {}).get("hlg", []),
    )
=======
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
        for i, e in enumerate(FEW_SHOT_EXAMPLES, 1)
    )
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

EXAMPLES (fictional requirements, not project evidence)
{examples}

PROJECT DESCRIPTION
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


def _critic1_vote(prompt, model, branches, analysis):
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
                temperature=BOTTOM_UP_EVALUATOR_TEMPERATURE,
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
>>>>>>> 665ac6d9 (Update experimental pipeline and results)
    return GlobalGoalEvaluationResult(
        branch_id=branch.branch_id,
        rationale=rationale,
        final_decision=GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
<<<<<<< HEAD
        high_level_evaluation=hlg,
        low_level_evaluation=llg,
        critic_votes=critic_votes or {
            "hlg": hlg.evaluator_votes,
            **({"llg": llg.evaluator_votes} if llg is not None else {}),
        },
    )


def _apply_batched_hlg_removal_gate(
    evaluation: HighLevelGoalEvaluation,
    branch: GoalBranch,
    all_branches: list[GoalBranch],
) -> HighLevelGoalEvaluation:
    if evaluation.decision != HighLevelGoalDecision.REMOVE:
        return evaluation
    sibling_branches = _other_goal_branches(all_branches, branch)
    removal_is_supported = (
        evaluation.removal_basis in {HighLevelGoalRemovalBasis.UNSUPPORTED, HighLevelGoalRemovalBasis.NON_FUNCTIONAL, HighLevelGoalRemovalBasis.OVER_SPECIFIC}
        or (
            evaluation.removal_basis == HighLevelGoalRemovalBasis.FULLY_REDUNDANT
            and any(
                sibling["branch_id"] == evaluation.covered_by_branch_id
                and _key(sibling["high_level_goal"].get("name", ""))
                == _key(evaluation.covered_by_high_level_goal_name or "")
                and _key(sibling["high_level_goal"].get("actor", ""))
                == _key(branch.high_level_goal.actor.name)
                for sibling in sibling_branches
            )
        )
    )
    if (
        evaluation.winning_vote_count >= EVALUATOR_VOTING_QUORUM
        and removal_is_supported
    ):
        return evaluation
    return evaluation.model_copy(update={
        "rationale": (
            f"{evaluation.rationale} Destructive-action gate: the REMOVE "
            "decision lacked quorum or an exact supported removal basis; "
            "the HLG is retained."
        ),
        "decision": HighLevelGoalDecision.KEEP,
        "quality_score": BOTTOM_UP_QUALITY_THRESHOLD,
        "quality_score_origin": "removal_guard_policy",
        "requires_llg_evaluation": (
            evaluation.reconstruction_relationship
            != ReconstructionRelationship.EQUIVALENT
        ),
        "rewriting_focus": None,
        "removal_basis": None,
        "covered_by_branch_id": None,
        "covered_by_high_level_goal_name": None,
    })


def _critic1_list_prompt(project_description, branches, reconstructions, confirmed_branch_ids=None):
    registry = [{"branch_id": b.branch_id,
                 "previously_confirmed": b.branch_id in (confirmed_branch_ids or set()),
                 "high_level_goal": b.high_level_goal.model_dump(mode="json")}
                for b in branches]
    reconstructed = [{"branch_id": b.branch_id, "has_children": bool(b.low_level_goals),
                      "reconstructed_hlg": reconstructions[b.branch_id].reconstructed_high_level_goal}
                     for b in branches]
    return f"""### Critic 1: complete-list High-Level Goal review
You are provided with a software description, ALL current original HLGs and
HLG' reconstructed from each branch's children. You do not see actual children.
Perform the following two parts inside this ONE review and return one entry
for EVERY original branch ID. These are two tasks, not two separate model calls.

### PART 1 -- Diagnose HLG' against the ORIGINAL HLG and SOURCE FIRST
For every branch compare the reconstructed intention with the original parent's
name AND description. Use the software description to distinguish a defect in
the parent from a possible defect in its decomposition. HLG' summarizes current
children; it is not a source of requirements and can itself omit a capability.
In reconstruction_diagnosis classify the relationship:
EQUIVALENT=same essential outcome; PARTIAL_OVERLAP=some capabilities differ;
MATCHES_SIBLING=another same-actor outcome (give matches_branch_id);
DIFFERENT=another outcome with no matching sibling;
NOT_RECONSTRUCTABLE=has_children is false (mandatory only in that case).
Use the registry to identify a matching sibling, not to assume the original
must be removed. A source-supported parent with an incomplete or different HLG'
should normally be kept so Critic 2 can inspect actual children. Do not narrow
that parent to fit the reconstruction or infer missing requirements from HLG'.
If the original has a recoverable source-grounded defect, explain that defect
for a rewrite; a reconstruction mismatch alone does not justify rewriting it.

### PART 2 -- Decide the parent action using the SOURCE and COMPLETE REGISTRY
After the diagnosis, assess the original parent's support, actor responsibility,
independent outcome and granularity. Compare it with all other original HLGs
for duplicates and same-actor full coverage. Read names AND descriptions.
Coherence with HLG' does not prove autonomy: a duplicate or workflow step can
have perfectly coherent children. Being documented, user-visible, or phrased as
Enable/Support/Ensure alone does not justify a standalone HLG.
For each original choose:
- KEEP_ORIGINAL_HIGH_LEVEL_GOAL: justify its independently useful outcome and
  its distinction from the other parents in outcome_justification. Preserve a
  supported independent parent when its reconstruction exposes a child problem.
- REMOVE_ORIGINAL_HIGH_LEVEL_GOAL: FULLY_REDUNDANT requires a complete covering
  SAME-ACTOR goal, exact surviving branch ID and name. Preserve the most complete
  documented outcome; for equivalent peers prefer prior confirmation, then registry
  order. CURRENT cannot cover itself. Different responsibilities and partial overlap
  are insufficient. OVER_SPECIFIC removes internal workflow capabilities even
  without a covering parent: drafts, consents, data checks and configuration may
  belong at LLG level. Explain the containing documented outcome; later discovery
  assesses any missing broader parent. UNSUPPORTED removes wholly undocumented
  intentions; NON_FUNCTIONAL removes pure constraints with no functional core.
  All non-redundant removal bases require null covering fields.
- REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: explain a recoverable defect supported by
  the source and provide rewriting_focus. Do not repair incorrect children by
  changing a valid parent. For complementary fragments of ONE outcome, nominate
  one anchor and list every OTHER same-actor member in consolidate_with_branch_ids.
  Retain source members until that transaction succeeds; avoid reciprocal or
  overlapping consolidation proposals. Preserve all documented capabilities.
A MATCHES_SIBLING diagnosis alone cannot delete a source-supported independent
parent. Conversely, structural REMOVE is valid even with diagnosis EQUIVALENT
when source and registry justify redundancy or over-specific granularity.
Keep diagnosis and action independently justified; do not alter one to fit the other.
Earlier confirmations are revisable. Do not invent broader requirements, combine
independent outcomes, or enforce a target HLG count. Several removals are allowed.
No other model votes or previous verdict texts are supplied.

### Response
Return one JSON object with decisions: a list containing every branch ID once.
Copy IDs exactly from the current registry; example IDs are not current IDs.
For each entry write branch_id and reconstruction_diagnosis first, then decision,
rationale and outcome_justification. The diagnosis requires reconstruction_relationship
and rationale. This order expresses the review sequence, not an extra model call.
Add matches_branch_id only for MATCHES_SIBLING. Add rewriting_focus only for
REWRITE; consolidate_with_branch_ids only for consolidation; removal_basis only
for REMOVE; covered_by_branch_id and covered_by_high_level_goal_name only for
FULLY_REDUNDANT. Omit unused optional fields. Do not return scores or routing flags.
Use concise evidence: at most 25 words each for rationale and outcome_justification,
and 15 for the diagnosis. Preserve semantic distinctions, not long reasoning transcripts.

### Four complete-list examples
{CRITIC1_GLOBAL_EXAMPLES}

### Software description
{project_description}

### All original HLGs (both parts; IDs are references, not semantic evidence)
{json.dumps(registry, ensure_ascii=False)}

### Reconstructions (Part 1 diagnosis; no child lists or reconstruction rationales)
{json.dumps(reconstructed, ensure_ascii=False)}

Return only the JSON object now."""


def _validate_structural_reference(vote, branch, branches):
    siblings = _other_goal_branches(branches, branch)
    eligible = {s["branch_id"] for s in siblings}
    if any(identifier not in eligible for identifier in vote.consolidate_with_branch_ids):
        raise ValueError("Consolidation requires other existing same-actor branches")
    if vote.removal_basis == HighLevelGoalRemovalBasis.FULLY_REDUNDANT:
        if not any(s["branch_id"] == vote.covered_by_branch_id
                   and _key(s["high_level_goal"]["name"]) == _key(vote.covered_by_high_level_goal_name)
                   for s in siblings):
            raise ValueError("Redundancy requires the exact ID/name of another same-actor branch")


def _critic1_list_vote(prompt, evaluator_model, branches):
    """One complete-list response; isolate local defects from envelope failures."""
    expected = {b.branch_id for b in branches}
    raw = parsed = None
    error = None
    for _ in range(BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS):
        parsed = None
        try:
            raw = generate_evaluator_response(prompt, SYSTEM_PROMPT, conversation=None,
                                             temperature=BOTTOM_UP_EVALUATOR_TEMPERATURE,
                                             model=evaluator_model)
            parsed = _critic1_json_object(raw)
            envelope = GlobalHlgReviewResponse.model_validate(parsed)
            identifiers = [item.get("branch_id") for item in envelope.decisions]
            if (any(not isinstance(i, str) or i not in expected for i in identifiers)
                    or len(set(identifiers)) != len(identifiers)):
                raise ValueError("Global response has duplicate, missing-identity or foreign branch IDs")
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        record = EvaluatorVoteRecord(model=evaluator_model, status="INVALID",
                                     raw_response=raw, parsed_response=parsed, error=error)
        invalid = {b.branch_id: record.model_copy() for b in branches}
        return record, invalid, {key: value.model_copy() for key, value in invalid.items()}
    global_record = EvaluatorVoteRecord(model=evaluator_model, status="VALID",
                                       raw_response=raw, parsed_response=parsed)
    entries = {item["branch_id"]: item for item in envelope.decisions}
    structure, diagnosis = {}, {}
    for branch in branches:
        item = entries.get(branch.branch_id)
        if item is None:
            missing = EvaluatorVoteRecord(model=evaluator_model, status="INVALID", raw_response=raw,
                                          error="Missing decision for this branch")
            structure[branch.branch_id] = missing
            diagnosis[branch.branch_id] = missing.model_copy()
            continue
        diagnostic_item = item.get("reconstruction_diagnosis")
        try:
            vote = HlgReconstructionDiagnosis.model_validate(diagnostic_item)
            if (not branch.low_level_goals) != (vote.reconstruction_relationship == ReconstructionRelationship.NOT_RECONSTRUCTABLE):
                raise ValueError("Diagnosis conflicts with presence of children")
            if vote.reconstruction_relationship == ReconstructionRelationship.MATCHES_SIBLING:
                if vote.matches_branch_id not in {s["branch_id"] for s in _other_goal_branches(branches, branch)}:
                    raise ValueError("MATCHES_SIBLING requires another existing same-actor branch")
            elif vote.matches_branch_id is not None:
                raise ValueError("Only MATCHES_SIBLING may identify a sibling")
            diagnosis[branch.branch_id] = EvaluatorVoteRecord(
                model=evaluator_model, status="VALID", decision=vote.reconstruction_relationship.value,
                raw_response=raw, parsed_response=vote.model_dump(mode="json"))
        except (ValueError, TypeError) as exc:
            diagnosis[branch.branch_id] = EvaluatorVoteRecord(
                model=evaluator_model, status="INVALID", raw_response=raw,
                parsed_response=diagnostic_item if isinstance(diagnostic_item, dict) else None,
                error=str(exc))
        try:
            vote = HlgStructuralDecision.model_validate(item)
            _validate_structural_reference(vote, branch, branches)
            structure[branch.branch_id] = EvaluatorVoteRecord(
                model=evaluator_model, status="VALID", decision=vote.decision.value,
                raw_response=raw, parsed_response=vote.model_dump(mode="json"))
        except (ValueError, TypeError) as exc:
            structure[branch.branch_id] = EvaluatorVoteRecord(
                model=evaluator_model, status="INVALID", raw_response=raw,
                parsed_response=item, error=str(exc))
    return global_record, structure, diagnosis


def _structural_to_hlg(vote):
    fields = {key: getattr(vote, key) for key in (
        "decision", "rewriting_focus", "consolidate_with_branch_ids", "removal_basis",
        "covered_by_branch_id", "covered_by_high_level_goal_name", "voter_count",
        "valid_vote_count", "winning_vote_count", "vote_distribution", "evaluator_votes")}
    return HighLevelGoalEvaluation(
        **fields, rationale=vote.rationale + " Outcome: " + vote.outcome_justification,
        quality_score=4 if vote.decision == HighLevelGoalDecision.KEEP else 2,
        quality_score_origin="structural_decision_policy",
        requires_llg_evaluation=vote.decision == HighLevelGoalDecision.KEEP,
        structural_decision=vote.model_dump(mode="json"))


def _attach_reconstruction_diagnosis(hlg, records):
    valid = [HlgReconstructionDiagnosis.model_validate(r.parsed_response)
             for r in records if r.status == "VALID" and r.parsed_response is not None]
    if len(valid) < EVALUATOR_VOTING_QUORUM:
        raise EvaluatorVotingError("Insufficient valid reconstruction diagnoses", records)
    # Every non-equivalent valid diagnosis is a reason to inspect actual children.
    representative = next((v for v in valid if v.reconstruction_relationship != ReconstructionRelationship.EQUIVALENT), valid[0])
    relation = representative.reconstruction_relationship
    return hlg.model_copy(update={
        "reconstruction_relationship": relation,
        "requires_llg_evaluation": hlg.rewrite_limit_reached or relation != ReconstructionRelationship.EQUIVALENT,
        "quality_score": 4 if hlg.rewrite_limit_reached or relation != ReconstructionRelationship.EQUIVALENT else 5,
        "rationale": hlg.rationale + " Reconstruction: " + representative.rationale,
    })


def evaluate_all_branches(
    project_description: str,
    branches: list[GoalBranch],
    reconstructions: dict[str, BottomUpHighLevelGoal],
    current_hlgs: HighLevelGoals,
    conversation: EvaluatorConversation | None = None,
    all_branches: list[GoalBranch] | None = None,
    on_evaluation: Callable[
        [int, int, dict[str, GlobalGoalEvaluationResult]], None
    ] | None = None,
    *, hlg_rewrite_counts: dict[str, int] | None = None,
    max_hlg_rewrites: int = 3,
    confirmed_branch_ids: set[str] | None = None,
    on_vote: Callable | None = None,
    on_global_vote: Callable | None = None,
    blocked_hlg_repair_ids: set[str] | None = None,
    required_llg_audit_ids: set[str] | None = None,
):
    all_branches = all_branches or branches
    total = len(branches)
    hlg_vote_records = {branch.branch_id: [] for branch in branches}

    llg_vote_records = {branch.branch_id: [] for branch in branches}
    diagnosis_records = {branch.branch_id: [] for branch in branches}
    if not branches:
        return {}

    def retain_failure(records, evaluator_model, exc):
        records.extend(getattr(exc, "vote_records", None) or [EvaluatorVoteRecord(
            model=evaluator_model, status="INVALID", error=f"{type(exc).__name__}: {exc}")])

    # The existing first critic reviews the complete list after reconstruction.
    # It requests reconstruction diagnoses first, then parent actions in ONE response.
    prompt = _critic1_list_prompt(project_description, all_branches, reconstructions, confirmed_branch_ids)
    for evaluator_model in BOTTOM_UP_EVALUATOR_MODELS:
        print(f"[bottom-up] loading complete critic pass: {evaluator_model}", flush=True)
        with loaded_local_evaluator_model(evaluator_model):
            print(f"[bottom-up] {evaluator_model}: Critic 1 global HLG list ({total} branches)", flush=True)
            global_record, structure, diagnoses = _critic1_list_vote(prompt, evaluator_model, all_branches)
            if on_global_vote is not None:
                on_global_vote(global_record)
            for branch in branches:
                hlg_vote_records[branch.branch_id].append(structure[branch.branch_id])
                diagnosis_records[branch.branch_id].append(diagnoses[branch.branch_id])
                if on_vote is not None:
                    on_vote(branch, "hlg", [structure[branch.branch_id]])
                    on_vote(branch, "reconstruction", [diagnoses[branch.branch_id]])
        print(f"[bottom-up] unloaded complete critic pass: {evaluator_model}", flush=True)

    hlg_evaluations: dict[str, HighLevelGoalEvaluation] = {}
    evaluations: dict[str, GlobalGoalEvaluationResult] = {}
    llg_branches: list[GoalBranch] = []
    for branch in branches:
        try:
            structural = _aggregate_saved_votes(
                hlg_vote_records[branch.branch_id],
                HlgStructuralDecision,
                conversation,
                f"HLG evaluation for actor '{branch.high_level_goal.actor.name}', "
                f"goal '{branch.high_level_goal.name}'",
            )
            hlg = _structural_to_hlg(structural)
            hlg = _apply_batched_hlg_removal_gate(hlg, branch, all_branches)
            hlg = apply_hlg_rewrite_limit(
                hlg, max_hlg_rewrites if branch.branch_id in (blocked_hlg_repair_ids or set())
                else (hlg_rewrite_counts or {}).get(branch.branch_id, 0), max_hlg_rewrites)
            hlg_evaluations[branch.branch_id] = hlg
            if hlg.decision == HighLevelGoalDecision.KEEP:
                hlg = _attach_reconstruction_diagnosis(hlg, diagnosis_records[branch.branch_id])
                if branch.branch_id in (required_llg_audit_ids or set()):
                    hlg = HighLevelGoalEvaluation.model_validate({
                        **hlg.model_dump(),
                        "post_repair_llg_audit_required": True,
                        "requires_llg_evaluation": True,
                        "quality_score": 4,
                        "quality_score_origin": "post_repair_llg_audit_policy",
                        "rationale": hlg.rationale + " POST_REPAIR_LLG_AUDIT_REQUIRED",
                    })
                hlg_evaluations[branch.branch_id] = hlg
            if _requires_llg_audit(branch, hlg):
                llg_branches.append(branch)
        except (ValueError, TypeError) as exc:
            evaluations[branch.branch_id] = _inconclusive_result(
                branch,
                f"Critic 1 evaluation failed: {exc}",
                hlg=hlg_evaluations.get(branch.branch_id),
                critic_votes={
                    "hlg": hlg_vote_records[branch.branch_id],
                    "reconstruction": diagnosis_records[branch.branch_id],
                    "llg": llg_vote_records[branch.branch_id],
                },
            )

    structural_changes_pending = any(
        hlg.decision in {HighLevelGoalDecision.REMOVE, HighLevelGoalDecision.REWRITE}
        for hlg in hlg_evaluations.values()
    )
    if structural_changes_pending:
        # Structural work takes precedence; do not repair children of the old hierarchy.
        llg_branches = []

    # Reload voters only for routed branches. No previous verdict or rationale
    # is included in their LLG prompts; routing is a Python responsibility.
    for evaluator_model in BOTTOM_UP_EVALUATOR_MODELS if llg_branches else ():
        print(f"[bottom-up] loading LLG critic pass: {evaluator_model}", flush=True)
        with loaded_local_evaluator_model(evaluator_model):
            for index, branch in enumerate(llg_branches, 1):
                print(f"[bottom-up] {evaluator_model}: LLG {index}/{len(llg_branches)}", flush=True)
                try:
                    single = evaluate_llg_decomposition(
                        project_description, branch, reconstructions[branch.branch_id],
                        current_hlgs, conversation, evaluator_models=(evaluator_model,),
                        manage_model_lifecycle=False, record_conversation=False,
                        upstream_hlg_evaluation=None)
                    llg_vote_records[branch.branch_id].extend(single.evaluator_votes)
                except (ValueError, TypeError) as exc:
                    retain_failure(llg_vote_records[branch.branch_id], evaluator_model, exc)
                if on_vote is not None:
                    on_vote(branch, "llg", [v for v in llg_vote_records[branch.branch_id] if v.model == evaluator_model])
        print(f"[bottom-up] unloaded LLG critic pass: {evaluator_model}", flush=True)

    # Aggregate only the decomposition votes actually requested.
    for branch in branches:
        if branch.branch_id in evaluations:
            continue
        hlg = hlg_evaluations[branch.branch_id]
        llg = None
        if branch in llg_branches:
            try:
                llg = _aggregate_saved_votes(
                    llg_vote_records[branch.branch_id],
                    LowLevelGoalEvaluation,
                    conversation,
                    f"LLG decomposition evaluation for parent "
                    f"'{branch.high_level_goal.name}'",
                )
                llg = _normalize_llg_evaluation(branch, llg)
            except (ValueError, TypeError) as exc:
                evaluations[branch.branch_id] = _inconclusive_result(
                    branch,
                    f"LLG evaluation failed: {exc}",
                    hlg=hlg,
                    critic_votes={
                        "hlg": hlg.evaluator_votes,
                        "reconstruction": diagnosis_records[branch.branch_id],
                        "llg": getattr(
                            exc,
                            "vote_records",
                            llg_vote_records[branch.branch_id],
                        ),
                    },
                )
                continue

        replacement = None
        if hlg.decision == HighLevelGoalDecision.REMOVE:
            decision = GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
        elif hlg.decision == HighLevelGoalDecision.REWRITE:
            replacement = build_replacement_request(
                project_description, branch, hlg
            )
            decision = GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        elif structural_changes_pending:
            decision = GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
        elif llg is None:
            decision = GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        else:
            decision = (
                GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
                if llg.decision == LowLevelGoalDecision.REGENERATE
                else GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            )
        rationale_parts = [hlg.rationale]
        if structural_changes_pending and hlg.decision == HighLevelGoalDecision.KEEP:
            rationale_parts.append("CHILD_REVIEW_DEFERRED_PENDING_STRUCTURAL_ACTIONS")
        if llg is not None:
            rationale_parts.append(llg.rationale)
        evaluations[branch.branch_id] = GlobalGoalEvaluationResult(
            branch_id=branch.branch_id,
            rationale=" ".join(rationale_parts),
            final_decision=decision,
            high_level_evaluation=hlg,
            replacement_request=replacement,
            low_level_evaluation=llg,
            critic_votes={
                "hlg": hlg.evaluator_votes,
                "reconstruction": diagnosis_records[branch.branch_id],
                "llg": llg.evaluator_votes if llg is not None else llg_vote_records[branch.branch_id],
            },
        )

    ordered_evaluations = {}
    for index, branch in enumerate(branches, start=1):
        evaluation = evaluations[branch.branch_id]
        evaluation.evaluation_status = evaluation_status(evaluation)
        ordered_evaluations[branch.branch_id] = evaluation
        if on_evaluation is not None:
            on_evaluation(index, total, dict(ordered_evaluations))
        llg_route = (
            evaluation.low_level_evaluation.decision.value
            if evaluation.low_level_evaluation is not None
            else "SKIPPED"
        )
        print(
            f"[bottom-up] branch {index}/{total} decision: "
            f"{evaluation.final_decision.value}; status={evaluation.evaluation_status}; "
            f"LLG critic={llg_route}",
            flush=True,
        )
    return ordered_evaluations
=======
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
    prompt = _critic1_prompt(project_description, actors, branches, reconstructions, analysis, cosine_payload)
    records = {b.branch_id: [] for b in branches}
    for model in BOTTOM_UP_EVALUATOR_MODELS:
        with loaded_local_evaluator_model(model):
            global_record, votes = _critic1_vote(prompt, model, branches, analysis)
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
>>>>>>> 665ac6d9 (Update experimental pipeline and results)
