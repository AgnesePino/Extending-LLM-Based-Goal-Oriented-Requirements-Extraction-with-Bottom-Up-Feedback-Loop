"""Sequential branch and global bottom-up evaluations with deterministic aggregation."""

from collections import Counter
import json
import os
import re
from typing import Callable

from pydantic import BaseModel

from src.data_model import (
    BottomUpHighLevelGoal,
    EvaluatorVoteRecord,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    GoalBranch,
    HighLevelGoalDecision,
    HighLevelGoalEvaluation,
    HighLevelGoalRemovalBasis,
    HighLevelGoalReplacementRequest,
    HighLevelGoals,
    LowLevelGoalDecision,
    LowLevelGoalEvaluation,
    Actors,
    MissingHighLevelGoalDecision,
    MissingHighLevelGoalEvaluation,
    ReconstructionRelationship,
)
from src.llm_clients import (
    EvaluatorConversation,
    configured_bottom_up_critic_models,
    generate_evaluator_response,
    loaded_local_evaluator_model,
)


SYSTEM_PROMPT = (
    "You are an expert in Software Engineering, Requirements Engineering, "
    "and Goal-Oriented Requirements Engineering. Return exactly one valid "
    "JSON object without markdown or surrounding prose. Keep rationale under "
    "200 words while preserving the required evidence. KEEP/NO_MISSING is not "
    "a default: use it only after every required audit passes. When concrete "
    "evidence identifies a material defect, choose the applicable corrective "
    "decision even if most of the artifact remains usable."
)

BOTTOM_UP_EVALUATOR_MODELS = configured_bottom_up_critic_models()
DISCOVERY_EVALUATOR_MODELS = BOTTOM_UP_EVALUATOR_MODELS
if any(not model_id for model_id in BOTTOM_UP_EVALUATOR_MODELS):
    raise ValueError("Every bottom-up critic model id must be non-empty.")
if len(set(BOTTOM_UP_EVALUATOR_MODELS)) != 3:
    raise ValueError("The bottom-up ensemble requires three different models.")
BOTTOM_UP_EVALUATOR_VOTERS = len(BOTTOM_UP_EVALUATOR_MODELS)
BOTTOM_UP_EVALUATOR_TEMPERATURE = float(
    os.getenv("BOTTOM_UP_EVALUATOR_TEMPERATURE", "0.2")
)
BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS = int(
    os.getenv("BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS", "3")
)
if not 1 <= BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS <= 3:
    raise ValueError(
        "BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS must be between 1 and 3."
    )
EVALUATOR_VOTING_QUORUM = BOTTOM_UP_EVALUATOR_VOTERS // 2 + 1
BOTTOM_UP_QUALITY_THRESHOLD = 4
BOTTOM_UP_EVALUATOR_PROMPTING_MODE = "FEW_SHOT"
HLG_EVALUATOR_EXAMPLE_COUNT = 9
LLG_EVALUATOR_EXAMPLE_COUNT = 7
DISCOVERY_EVALUATOR_EXAMPLE_COUNT = 3


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


def _other_high_level_goals(
    current_hlgs: HighLevelGoals,
    branch: GoalBranch,
) -> list[dict]:
    """Return compact sibling HLGs for the decomposition critic."""
    skipped_current = False
    siblings = []
    for goal in current_hlgs.goals:
        if not skipped_current and goal == branch.high_level_goal:
            skipped_current = True
            continue
        siblings.append(_compact_hlg(goal))
    return siblings


HLG_EVALUATOR_EXAMPLES = """Few-shot examples. The examples use the same
ground-truth style as the evaluated artifacts: a concise functional name, a
precise description, and one explicit actor. Judge the complete structured HLG,
not merely its label or shared domain vocabulary.

- Genuine equivalence.
  Documentation: a Resident can create a permit application, save it, resume
  it, update its data, and submit it.
  Original HLG: {"name":"Submit and manage a permit application","description":"Create, save, resume, update, and submit a permit application.","actor":{"name":"Resident","description":"Person requesting a municipal permit."}}
  HLG': "Manage the complete permit application lifecycle".
  Output: {"rationale":"For the same actor, both formulations cover the complete documented lifecycle: creation, preservation, modification, resumption, and submission. Neither adds or omits an independently meaningful outcome.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":5,"reconstruction_relationship":"EQUIVALENT","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- Same topic but narrower reconstruction.
  Documentation: a Visitor can read service information and simulate the fee
  they would pay.
  Original HLG: {"name":"Access service information and decision support","description":"Read service information and simulate the expected fee.","actor":{"name":"Visitor","description":"Unauthenticated service visitor."}}
  HLG': "Read service information".
  Output: {"rationale":"The reconstruction covers information access but omits the separately documented simulation outcome. Topic overlap is not bidirectional entailment; the parent itself is valid.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":4,"reconstruction_relationship":"PARTIAL_OVERLAP","requires_llg_evaluation":true,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- Reconstruction broader than the original parent.
  Documentation: Clinic Staff manage appointments; Billing Staff manage
  invoices and payments as a separate responsibility.
  Original HLG: {"name":"Manage patient appointments","description":"Create, reschedule, cancel, and consult patient appointments.","actor":{"name":"Clinic Staff","description":"Staff responsible for appointments."}}
  HLG': "Manage appointments and patient billing".
  Output: {"rationale":"Appointment management is documented and the original parent is valid, but the reconstruction adds billing, an independent outcome assigned to another actor. The reverse entailment fails.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":4,"reconstruction_relationship":"PARTIAL_OVERLAP","requires_llg_evaluation":true,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- Parent assigns an outcome to the wrong actor.
  Documentation: a Customer submits refund requests; only a Support Agent
  evaluates and approves or rejects them.
  Original HLG: {"name":"Request and approve refunds","description":"Submit a refund request and decide whether it is approved.","actor":{"name":"Customer","description":"Purchaser requesting assistance."}}
  HLG': "Request a refund".
  Output: {"rationale":"Requesting belongs to Customer, but approval belongs to Support Agent. The original parent contains a material actor-authority error and must be corrected rather than preserved.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":2,"reconstruction_relationship":"PARTIAL_OVERLAP","requires_llg_evaluation":false,"rewriting_focus":"Express only the Customer outcome of submitting and consulting a refund request; exclude approval decisions.","removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- HLG and HLG' agree but both represent only a workflow fragment.
  Documentation: an Applicant completes one registration outcome by entering
  household data, accepting mandatory declarations, and submitting the form.
  Original HLG: {"name":"Enter household data","description":"Record household details in the registration form.","actor":{"name":"Applicant","description":"Person completing registration."}}
  HLG': "Record household information". No sibling covers declarations or
  submission.
  Output: {"rationale":"The two formulations are equivalent to each other, but both reduce the documented end-to-end registration outcome to one data-entry phase. The parent is materially incomplete and must be rewritten.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":2,"reconstruction_relationship":"EQUIVALENT","requires_llg_evaluation":false,"rewriting_focus":"Express completing and submitting the registration, including household data and mandatory declarations.","removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- HLG' matches a sibling but the original parent is distinct and valid.
  Documentation: a Buyer manages orders and separately manages payments.
  Original HLG: {"name":"Manage customer payments","description":"Consult balance, authorize payment, and review payment transactions.","actor":{"name":"Buyer","description":"Customer purchasing products."}}
  HLG': "Manage customer orders". Same-actor branch_002 is HLG "Manage
  customer orders", covering creation, modification, and consultation of orders.
  Output: {"rationale":"The reconstruction matches branch_002, but payments are a separate documented Buyer outcome. Preserve the original parent and inspect the misplaced decomposition.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":4,"reconstruction_relationship":"MATCHES_SIBLING","requires_llg_evaluation":true,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- HLG' matches a sibling and the original parent is wholly redundant.
  Documentation: a Buyer schedules, changes, and cancels order pickup as one
  pickup-management outcome.
  Original HLG: {"name":"Book order pickup","description":"Select a pickup time for an order.","actor":{"name":"Buyer","description":"Customer collecting an order."}}
  HLG': "Manage order pickup". Same-actor branch_002 is HLG "Manage order
  pickup", whose description explicitly includes booking, rescheduling, and
  cancellation; it preserves the complete original capability.
  Output: {"rationale":"Branch_002 explicitly entails the original booking capability and the original parent contributes no independently scoped outcome.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":1,"reconstruction_relationship":"MATCHES_SIBLING","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":"FULLY_REDUNDANT","covered_by_branch_id":"branch_002","covered_by_high_level_goal_name":"Manage order pickup"}

- Different reconstruction, valid original parent.
  Documentation: an Athlete tracks physical activities and separately receives
  personalized workout recommendations.
  Original HLG: {"name":"Track physical activities","description":"Record and consult duration, distance, and type of completed activities.","actor":{"name":"Athlete","description":"User monitoring exercise."}}
  HLG': "Receive personalized workout recommendations". No sibling matches the
  reconstruction.
  Output: {"rationale":"Activity tracking is documented, correctly assigned, and independently scoped. Recommendations are a different outcome, so the error belongs to the decomposition rather than the parent.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":4,"reconstruction_relationship":"DIFFERENT","requires_llg_evaluation":true,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- Equivalent intentions but unsupported by documentation.
  Documentation: a User performs arithmetic calculations and interacts with
  calculator controls; sharing results is never described.
  Original HLG: {"name":"Share calculation results","description":"Publish calculation results to social networks.","actor":{"name":"User","description":"Calculator user."}}
  HLG': "Publish calculator results on social media".
  Output: {"rationale":"The two intentions are equivalent to each other, but the complete sharing capability is absent from the documentation. Agreement with an invented reconstruction cannot validate the parent.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":0,"reconstruction_relationship":"EQUIVALENT","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":"UNSUPPORTED","covered_by_branch_id":null,"covered_by_high_level_goal_name":null}

- Vague parent, recoverable documented intention.
  Documentation: a Moderator creates a planning session, invites estimators,
  adds items, and starts estimation rounds.
  Original HLG: {"name":"Use the platform","description":"Interact with the planning system.","actor":{"name":"Moderator","description":"Person coordinating estimation."}}
  HLG': "Set up and run a planning session".
  Output: {"rationale":"The original parent is too generic to delimit the Moderator's documented functional outcome. The reconstruction exposes a supported, recoverable intention, but the two scopes are not equivalent.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":2,"reconstruction_relationship":"PARTIAL_OVERLAP","requires_llg_evaluation":false,"rewriting_focus":"Express the Moderator outcome of setting up and running a planning session, including participants and estimation items.","removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
"""
HLG_EVALUATOR_COMPACT_EXAMPLES = """Compact ground-truth-shaped examples:
1. Documentation: Resident creates, saves, resumes and submits a permit request.
   HLG: {"name":"Manage permit request","description":"Create, save, resume and submit it","actor":{"name":"Resident"}}.
   HLG': "Create, save, resume and submit a permit request".
   Output: {"rationale":"Both outcome sets are {create, save, resume, submit}; actor and scope match.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":5,"reconstruction_relationship":"EQUIVALENT","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
2. Documentation: Visitor reads service information and simulates a fee.
   HLG covers both; HLG' only reads information.
   Output: {"rationale":"HLG outcomes are {read, simulate}; HLG' has only {read}.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":4,"reconstruction_relationship":"PARTIAL_OVERLAP","requires_llg_evaluation":true,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
3. Documentation: Customer requests refunds; Staff approve them. Customer HLG
   wrongly says "request and approve"; HLG' says "request refund".
   Output: {"rationale":"Approval is assigned to the wrong actor; repair the parent.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":2,"reconstruction_relationship":"PARTIAL_OVERLAP","requires_llg_evaluation":false,"rewriting_focus":"Keep Customer refund request; exclude Staff approval.","removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
4. Documentation: Buyer manages pickup booking, changes and cancellation.
   HLG "Book pickup" is fully covered by same-actor branch_002 "Manage pickup";
   HLG' also matches branch_002 and no distinct outcome remains.
   Output: {"rationale":"branch_002 entails the complete booking capability.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":1,"reconstruction_relationship":"MATCHES_SIBLING","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":"FULLY_REDUNDANT","covered_by_branch_id":"branch_002","covered_by_high_level_goal_name":"Manage pickup"}
5. Documentation: Buyer separately manages orders and payments. Original HLG is
   payments, while HLG' and branch_002 are orders.
   Output: {"rationale":"The payment parent is valid but its children implement the sibling outcome.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":4,"reconstruction_relationship":"MATCHES_SIBLING","requires_llg_evaluation":true,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
6. Documentation separately supports activity tracking and recommendations.
   HLG is tracking; HLG' is recommendations; no sibling matches HLG'.
   Output: {"rationale":"The valid parent and reconstruction express different documented outcomes.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":4,"reconstruction_relationship":"DIFFERENT","requires_llg_evaluation":true,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
7. Documentation only supports calculations. HLG and HLG' both publish results
   on social media, which is wholly unsupported.
   Output: {"rationale":"Agreement cannot validate an undocumented intention.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":0,"reconstruction_relationship":"EQUIVALENT","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":"UNSUPPORTED","covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
8. Documentation defines one end-to-end registration: enter data, accept required
   declarations and submit. HLG and HLG' both cover only data entry; no sibling
   covers the rest.
   Output: {"rationale":"Equivalent fragments remain materially incomplete parents.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":2,"reconstruction_relationship":"EQUIVALENT","requires_llg_evaluation":false,"rewriting_focus":"Express complete registration including data, declarations and submission.","removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}
9. Documentation defines one end-to-end registration. HLG "Enter household
   data" and HLG' both describe only data entry; same-actor branch_002
   "Complete registration" explicitly covers data, declarations and submission.
   Output: {"rationale":"branch_002 preserves the complete data-entry fragment and its end-to-end outcome; the narrow parent adds no independent responsibility.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":1,"reconstruction_relationship":"MATCHES_SIBLING","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":"FULLY_REDUNDANT","covered_by_branch_id":"branch_002","covered_by_high_level_goal_name":"Complete registration"}
"""
LLG_EVALUATOR_EXAMPLES = """Few-shot examples:
- Correct decomposition. Parent: "Monitor repository popularity". LLGs retrieve
  stars and forks and display their trends.
  Output: {"rationale":"The LLGs cover the documented parent capability.","decision":"KEEP_LOW_LEVEL_GOALS","quality_score":5,"regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[]}
- Major omission. Parent: "Monitor popularity and active contributors". LLGs:
  llg_001 retrieves stars and llg_002 displays popularity trends.
  Output: {"rationale":"Contributor monitoring is explicit in the parent but absent from every LLG.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":1,"regeneration_feedback":"Keep the popularity LLGs and add contributor monitoring.","unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":["Identify active repository contributors"]}
- Erroneous addition. Parent: "Analyze repository activity". llg_001 retrieves
  commits; llg_002 publishes a marketing post, which is undocumented.
  Output: {"rationale":"The marketing action is unsupported and outside the parent goal.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":1,"regeneration_feedback":"Keep repository analysis and remove the marketing action.","unsupported_or_misleading_llg_ids":["llg_002"],"missing_essential_capabilities":[]}
- Conventional but undocumented addition. Description: users submit support
  requests. Parent: "Obtain support". llg_001 submits a request and llg_002
  configures automatic status notifications; notifications are not documented.
  Output: {"rationale":"Automatic notifications are plausible but not entailed by the supplied documentation.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Keep request submission and remove the unsupported notification capability.","unsupported_or_misleading_llg_ids":["llg_002"],"missing_essential_capabilities":[]}
- UI fragments mistaken for stakeholder subgoals. Description: an Applicant
  submits a completed application. Parent: "Submit application". llg_001 opens
  the form, llg_002 clicks Next, llg_003 clicks Submit, and llg_004 sends the
  completed application. The first three are interface mechanics or duplicate
  fragments, not independently documented goals.
  Output: {"rationale":"Only sending the completed application is a documented stakeholder capability; opening the form and clicking navigation controls fragment that same action into undocumented UI mechanics.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Preserve the application-submission capability and remove or consolidate the UI-only fragments.","unsupported_or_misleading_llg_ids":["llg_001","llg_002","llg_003"],"missing_essential_capabilities":[]}
- Broad parent does not authorize arbitrary lifecycle variants. Description: a
  Customer can view an order and cancel it before dispatch. Parent: "Manage an
  order". llg_001 views the order, llg_002 cancels it, and llg_003 reassigns it
  to another customer. Reassignment is neither stated nor necessary.
  Output: {"rationale":"Viewing and cancellation are documented, while reassignment is an invented lifecycle operation that is not entailed by the broad verb manage.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":1,"regeneration_feedback":"Keep viewing and cancellation and remove customer reassignment.","unsupported_or_misleading_llg_ids":["llg_003"],"missing_essential_capabilities":[]}
- Upstream mismatch must be resolved. Description: staff evaluate applications
  and may accept, reject, or suspend them. Parent: "Evaluate applications".
  The aggregated HLG critic reports PARTIAL_OVERLAP because the reconstruction
  omits suspension. llg_001 accepts and llg_002 rejects applications.
  Output: {"rationale":"The upstream material mismatch is present in the decomposition: neither LLG covers the documented suspension outcome.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Preserve acceptance and rejection, and add the documented suspension outcome.","unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":["Suspend an application during evaluation"]}
"""
COVERAGE_EVALUATOR_EXAMPLES = """Few-shot examples:
- Correct coverage. Description: customers browse menus and place orders.
  Current HLG: "Find and order food" for Customer.
  Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Browsing and ordering are steps of the existing end-to-end intention.","quality_score":5,"missing_goal_requests":[]}
- Major omission. Description: customers place orders and couriers update
  delivery status. Current HLGs cover only Customer; Courier is an existing actor.
  Output: {"decision":"MISSING_HIGH_LEVEL_GOALS_FOUND","rationale":"The Courier has an explicit responsibility not covered by any current HLG.","quality_score":1,"missing_goal_requests":[{"actor":"Courier","generation_project_description":"Generate one HLG for managing assigned deliveries and their status."}]}
- Erroneous addition. Description: a librarian registers loans and returns.
  Current HLG: "Manage book circulation". Audit logging is not documented.
  Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"The current HLG covers loans and returns; conventional audit features are not evidence of a missing goal.","quality_score":4,"missing_goal_requests":[]}
"""

QUALITY_SCORE_RUBRIC = """Quality-score rubric for the current artifact:
- 0: unsupported or fundamentally contradictory;
- 1: severe omission or unsupported content requiring correction;
- 2: material defect, but the main intention remains recoverable;
- 3: acceptable and supported, with only minor non-material defects;
- 4: good coverage and alignment, with negligible issues;
- 5: complete, precise, supported, and internally consistent.
Only scores 4 and 5 pass. Scores 0, 1, 2, and 3 require a corrective
decision: a merely acceptable artifact with a known minor defect is not final.
The score evaluates the artifact, not confidence in your own answer."""


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator response does not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


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


def _evaluate(
    prompt: str,
    model: type[BaseModel],
    conversation: EvaluatorConversation | None = None,
    memory_label: str | None = None,
    evaluator_models: tuple[str, ...] | None = None,
    manage_model_lifecycle: bool = True,
    record_conversation: bool = True,
):
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
                evaluation = model.model_validate(_json_object(response))
                if not _score_matches_decision(evaluation):
                    raise ValueError(
                        "quality_score conflicts with the selected decision"
                    )
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
    vote_records = [
        EvaluatorVoteRecord(
            model=evaluator_model,
            status="VALID" if vote is not None else "INVALID",
            decision=(vote.decision.value if vote is not None else None),
            raw_response=raw_response,
            parsed_response=(
                vote.model_dump(mode="json") if vote is not None else None
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
    winning_decision, winning_count = decision_counts.most_common(1)[0]
    aggregation_policy = "MAJORITY"
    # A complete 5/5 LLG decomposition is required for KEEP. Consequently, a
    # single valid REGENERATE vote with structured defect evidence must not be
    # erased by two generic KEEP votes. Apply this repair veto only when all
    # three configured critics returned valid votes; partial/failed panels stay
    # inconclusive unless an ordinary quorum exists.
    if (
        model is LowLevelGoalEvaluation
        and voter_count == BOTTOM_UP_EVALUATOR_VOTERS
        and len(valid_votes) == voter_count
        and decision_counts.get(LowLevelGoalDecision.REGENERATE, 0) > 0
    ):
        winning_decision = LowLevelGoalDecision.REGENERATE
        winning_count = decision_counts[LowLevelGoalDecision.REGENERATE]
        aggregation_policy = "EVIDENCE_BACKED_LLG_REPAIR_VETO"
    if winning_count < voting_quorum and aggregation_policy == "MAJORITY":
        raise EvaluatorVotingError(
            f"No {model.__name__} decision quorum: "
            f"{dict((decision.value, count) for decision, count in decision_counts.items())}; "
            f"required {voting_quorum}/{voter_count}. "
            f"Rejected votes: {rejection_summary or 'none'}.",
            vote_records,
        )

    winning_votes = [
        vote for vote in valid_votes
        if vote.decision == winning_decision
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
    if conversation is not None and record_conversation:
        conversation.record_exchange(
            memory_label or f"{model.__name__} evaluation",
            result.model_dump_json(),
        )
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
    winning_decision, winning_count = decision_counts.most_common(1)[0]
    aggregation_policy = "MAJORITY"
    if (
        model is LowLevelGoalEvaluation
        and voter_count == BOTTOM_UP_EVALUATOR_VOTERS
        and len(valid_votes) == voter_count
        and decision_counts.get(LowLevelGoalDecision.REGENERATE, 0) > 0
    ):
        winning_decision = LowLevelGoalDecision.REGENERATE
        winning_count = decision_counts[LowLevelGoalDecision.REGENERATE]
        aggregation_policy = "EVIDENCE_BACKED_LLG_REPAIR_VETO"
    if winning_count < voting_quorum and aggregation_policy == "MAJORITY":
        raise EvaluatorVotingError(
            f"No {model.__name__} decision quorum: "
            f"{dict((decision.value, count) for decision, count in decision_counts.items())}; "
            f"required {voting_quorum}/{voter_count}.",
            vote_records,
        )

    winning_votes = [
        vote for vote in valid_votes if vote.decision == winning_decision
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
    if conversation is not None:
        conversation.record_exchange(memory_label, result.model_dump_json())
    return result


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
) -> HighLevelGoalEvaluation:
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
    prompt = f"""Evaluator calibration:
{HLG_EVALUATOR_COMPACT_EXAMPLES}

{QUALITY_SCORE_RUBRIC}

Decide whether the original High-Level Goal should be kept, rewritten, or
removed and explicitly diagnose whether any inconsistency belongs to the parent
or to its LLG decomposition. The project description is always the source of
truth. Use semantic functional intentions and entailment; never use name
equality, string similarity, token overlap, or an arbitrary similarity score.
KEEP is permitted only after the actor, documentary support, functional scope,
independent stakeholder outcome, and sibling-overlap checks all pass. Do not
use KEEP merely because the goal is plausible or partly correct. A score of 3
means a real defect remains and therefore requires REWRITE or REMOVE at this
level, or KEEP with score 4 followed by LLG repair when the parent itself is
valid and the defect belongs exclusively to its decomposition.

Before choosing a decision, independently validate the original parent's full
description against the documentation: actor authority, essential outcomes,
scope, formulation, and abstraction level. This validation applies to EVERY
relationship, including EQUIVALENT. A partly documented parent is not fully
valid merely because its main topic is documented. Rewrite only when the same
actor-owned intention remains identifiable and can be corrected without
turning it into a materially different goal. Remove a wholly unsupported
intention, or a proven fully redundant parent, under the existing gates. If
removing the erroneous clauses leaves no independently meaningful documented
outcome, prefer UNSUPPORTED removal over inventing a replacement intention.

A partially erroneous branch can require repair of BOTH levels. Choose REWRITE
when the parent itself omits an essential part of its documented stakeholder
outcome, is merely a workflow fragment with no covering sibling, or is
materially ambiguous/misformulated. Agreement between an incomplete HLG and an
equally incomplete HLG' does not make the parent valid: the relationship may
still be EQUIVALENT while the decision is REWRITE. Give rewriting_focus that
specifies the corrected outcome and its essential boundaries; the orchestrator
will rewrite the HLG and regenerate its LLGs before the next validation pass.
Do not rewrite a valid parent just because its children are incomplete, or for
cosmetic wording changes. Do not import an independent outcome owned by a
sibling or broaden a branch merely to cover the whole project.

EQUIVALENT requires bidirectional semantic entailment: HLG must entail HLG'
and HLG' must entail HLG, for the same actor, essential outcomes, and scope.
Test both directions explicitly. A shared domain object, common purpose,
compatible workflow, or matching 'core intent' is insufficient. Compare full
descriptions, not labels alone. Do not broaden a narrow reconstruction using
the project description or invent missing capabilities to make it equivalent.
If either side omits or adds an essential outcome while retaining meaningful
overlap, use PARTIAL_OVERLAP. If there is no shared functional outcome, use
DIFFERENT unless HLG' matches a sibling. Pure paraphrases remain EQUIVALENT;
do not manufacture defects or demand changes merely to avoid an all-KEEP pass.
For a valid parent, KEEP with any non-EQUIVALENT relationship records a
material decomposition defect that Critic 2 must repair. Do not use
PARTIAL_OVERLAP for a merely stylistic or non-material difference; classify
such a genuinely lossless variation as EQUIVALENT.

Apply the same granularity used by a curated ground truth: an HLG must state a
recognizable actor-owned functional outcome, with a concise name and a precise
description that delimit its responsibility. Read name, description, and actor
together. A generic umbrella such as "use the platform", "manage operations",
"ensure efficiency", or "handle information" is not automatically valid just
because the documentation contains related features. If the description does
not make the functional boundary precise, choose REWRITE when the intended
outcome is recoverable. Conversely, do not rewrite a precise description only
because its short name uses broad wording.

Use a strict burden of proof for EQUIVALENT. Every material clause of the
original name and description must be entailed by HLG', and every material
clause of HLG' must be entailed by the original structured HLG. If either
direction depends on interpreting vague verbs broadly, treating two workflow
phases as interchangeable, or assuming a conventional but unstated outcome,
the relationship is not EQUIVALENT. KEEP+EQUIVALENT is reserved for a clean,
fully supported match and therefore requires quality_score 5.

The reconstructor is required to produce a lossless aggregate of all children.
Do not accept a generic umbrella reconstruction as proof of equivalence. A
phrase such as "manage X" does not entail every operation on X, and a phrase
such as "complete the process" does not prove coverage of each material outcome
named by the original HLG. If the supplied LLG set contains a distinct action,
object, lifecycle phase, or outcome outside the original parent's boundary,
the aggregate reconstruction is broader or different and Critic 2 must run.
Likewise, if the reconstruction fails to state one material original outcome,
classify it as PARTIAL_OVERLAP even when its rationale claims that outcome.
Judge the reconstructed goal text, not charitable explanations in its
rationale.

Treat the reconstruction and its rationale as hypotheses, not an evaluator
verdict. Check the supplied LLGs to resolve ambiguous reconstruction wording;
do not assume a vague phrase such as 'process data' includes verification,
calculation, and approval unless its evidence actually supports those outcomes.
Previous critic decisions are context only, never proof for this branch.
Do not perform the atomic support/completeness audit of individual LLGs here:
that is Critic 2's responsibility. Critic 1 uses the children only to interpret
the aggregate meaning of HLG' and to decide whether Critic 2 must run.

Follow this decision sequence in order:
1. Compare the original HLG with the bottom-up reconstruction HLG'. If they
   express the same documented functional intention, set relationship to
   EQUIVALENT. Only if the parent passes the independent documentation and scope
   audit, choose KEEP and set requires_llg_evaluation=false. A recoverable but
   materially incomplete or malformed parent requires REWRITE even here.
   Linguistic
   differences alone do not matter. Equivalence does not preserve an intention
   that is unsupported by the project description: an unsupported parent must
   still be removed conservatively.
2. If HLG' covers only part of the original intention, or adds an independent
   outcome while overlapping it, set PARTIAL_OVERLAP and
   validate the original against the complete project description: support,
   formulation, actor, functional intention, and abstraction level. If the
   parent is valid, choose KEEP and requires_llg_evaluation=true. If the parent
   is materially wrong, choose REWRITE with a precise documentation-grounded
   rewriting_focus and requires_llg_evaluation=false. Never rewrite a valid HLG
   merely to match a partial reconstruction.
3. Compare HLG' with every other current HLG. If it is semantically equivalent
   to, or fully entailed by, a same-actor sibling, set MATCHES_SIBLING. Remove
   the original only when
   that original parent is fully redundant, identify the exact covering branch,
   and use removal_basis=FULLY_REDUNDANT. This includes a narrow workflow phase
   or CRUD fragment when one same-actor sibling explicitly preserves that whole
   fragment inside a broader documented outcome. In that case remove the narrow
   parent instead of rewriting it into a second copy of the broader sibling. If
   the original parent expresses a distinct documented intention, keep it and
   set requires_llg_evaluation=true, because its LLGs appear to implement the
   sibling's responsibility.
4. Otherwise set DIFFERENT. Validate the original against the project and the
   sibling responsibility map. A valid parent means the divergence is in the
   decomposition: KEEP with requires_llg_evaluation=true. A materially malformed
   parent requires REWRITE. A completely unsupported intention may be removed
   using the conservative UNSUPPORTED basis. Difference alone never justifies a
   destructive decision.

If this branch has no LLGs, set relationship to NOT_RECONSTRUCTABLE. Absence of
LLGs is not evidence against the parent. Evaluate the parent from documentation
and siblings; if it is valid, choose KEEP and requires_llg_evaluation=true so
the missing decomposition is generated. Never confirm an empty branch.

An HLG is independently scoped only when it represents a stakeholder outcome,
not merely a workflow phase, state transition, validation, exception path, or
CRUD operation. A same-actor sibling supports FULLY_REDUNDANT only if it entails
every documented capability and every distinctive LLG of the original. Partial
overlap, a shared object, or a broader-sounding label is insufficient. If the
hierarchy is fragmented and no sibling is a valid umbrella, REWRITE the most
suitable parent rather than deleting documented capabilities. Conversely, if
an exact same-actor umbrella is present and entails the entire fragment, REMOVE
the fragment rather than preserving two HLGs for one responsibility.

Complete project description:
{project_description}

Actor:
{branch.high_level_goal.actor.model_dump_json()}

Original High-Level Goal:
{branch.high_level_goal.model_dump_json()}

Original branch Low-Level Goals:
{json.dumps([_compact_llg(goal) for goal in branch.low_level_goals], ensure_ascii=False)}

Bottom-up reconstructed High-Level Goal:
{reconstruction.reconstructed_high_level_goal}

Reconstructor rationale (untrusted hypothesis to verify independently):
{reconstruction.rationale}

Other current HLG branches, including technical IDs and associated LLGs:
{json.dumps(sibling_branches, ensure_ascii=False)}

Do not rewrite a supported HLG merely because its LLGs use API-like wording.
For REWRITE or REMOVE always set requires_llg_evaluation=false: the old
decomposition must not be evaluated. For KEEP, this field records whether the
HLG comparison itself found a decomposition mismatch: set it false only for an
EQUIVALENT reconstruction and true for every other relationship. The pipeline
still performs an independent LLG quality audit for every kept branch, even
when this field is false.

For REMOVE, set removal_basis to UNSUPPORTED or FULLY_REDUNDANT. A
FULLY_REDUNDANT removal must identify exactly one surviving sibling using its
covered_by_branch_id and copy its HLG name into covered_by_high_level_goal_name.
The branch ID is only a technical reference, never evidence of semantic
identity. Otherwise use null for all three fields.

In rationale, cite concrete documentary support (or its absence), explicitly
enumerate the material outcome set expressed by HLG and the material outcome
set expressed by HLG', compare them in both directions, and identify any
missing/added outcome or actor mismatch. For EQUIVALENT, explain why neither
direction leaves an essential outcome unmatched and why the parent passes the
independent scope audit. If the two enumerated sets are not identical, an
EQUIVALENT answer is invalid. For redundancy, identify what the exact sibling
preserves. Do not use 'same core intent', 'same purpose', 'same domain', or
'distinct workflow stages' as the sole justification.

Output JSON:
{{"rationale":"...","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL | REWRITE_ORIGINAL_HIGH_LEVEL_GOAL | REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":0,"reconstruction_relationship":"EQUIVALENT | PARTIAL_OVERLAP | MATCHES_SIBLING | DIFFERENT | NOT_RECONSTRUCTABLE","requires_llg_evaluation":false,"rewriting_focus":null,"removal_basis":null,"covered_by_branch_id":null,"covered_by_high_level_goal_name":null}}"""
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
    )
    expected_empty_relationship = not branch.low_level_goals
    if expected_empty_relationship != (
        evaluation.reconstruction_relationship
        == ReconstructionRelationship.NOT_RECONSTRUCTABLE
    ):
        raise ValueError(
            "reconstruction_relationship conflicts with the presence of LLGs"
        )
    removal_is_supported = (
        evaluation.removal_basis == HighLevelGoalRemovalBasis.UNSUPPORTED
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


def _normalize_llg_evaluation(
    branch: GoalBranch,
    evaluation: LowLevelGoalEvaluation,
) -> LowLevelGoalEvaluation:
    """Discard unusable repair evidence without turning failures into KEEP."""
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
        raise ValueError(
            "REGENERATE_LOW_LEVEL_GOALS did not identify a valid current LLG "
            "or a non-empty missing essential capability; refusing to convert "
            "invalid repair evidence into KEEP_LOW_LEVEL_GOALS"
        )
    return evaluation.model_copy(update={
        "unsupported_or_misleading_llg_ids": supported_ids,
        "missing_essential_capabilities": missing,
    })


def _enforce_cross_level_consistency(
    branch: GoalBranch,
    hlg_evaluation: HighLevelGoalEvaluation,
    llg_evaluation: LowLevelGoalEvaluation | None,
) -> LowLevelGoalEvaluation | None:
    """Prevent Critic 2 from silently cancelling a material Critic 1 finding."""
    if (
        llg_evaluation is None
        or llg_evaluation.decision == LowLevelGoalDecision.REGENERATE
        or hlg_evaluation.reconstruction_relationship
        == ReconstructionRelationship.EQUIVALENT
    ):
        return llg_evaluation

    relationship = hlg_evaluation.reconstruction_relationship
    current_ids = [
        f"llg_{index:03d}"
        for index, _ in enumerate(branch.low_level_goals, start=1)
    ]
    unsupported_ids = (
        current_ids
        if relationship in {
            ReconstructionRelationship.DIFFERENT,
            ReconstructionRelationship.MATCHES_SIBLING,
        }
        else []
    )
    missing_capabilities = (
        []
        if unsupported_ids
        else [
            branch.high_level_goal.description.strip()
            or branch.high_level_goal.name.strip()
            or "Complete the documented parent outcome"
        ]
    )
    gate_rationale = (
        "Cross-level consistency gate: the aggregated HLG critic classified "
        f"the reconstruction as {relationship.value}, so the unchanged LLG "
        "set cannot be a complete 5/5 decomposition of this parent. "
        f"Upstream finding: {hlg_evaluation.rationale}"
    )
    return llg_evaluation.model_copy(update={
        "rationale": f"{llg_evaluation.rationale} {gate_rationale}",
        "decision": LowLevelGoalDecision.REGENERATE,
        "quality_score": min(llg_evaluation.quality_score, 2),
        "regeneration_feedback": (
            "Preserve individually supported LLGs, but regenerate the branch "
            "so its aggregate meaning exactly matches the documented parent "
            f"and resolves the {relationship.value} mismatch identified by "
            "Critic 1."
        ),
        "unsupported_or_misleading_llg_ids": unsupported_ids,
        "missing_essential_capabilities": missing_capabilities,
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
    sibling_hlgs = _other_high_level_goals(current_hlgs, branch)
    upstream_hlg = (
        {
            "decision": upstream_hlg_evaluation.decision.value,
            "quality_score": upstream_hlg_evaluation.quality_score,
            "reconstruction_relationship": (
                upstream_hlg_evaluation.reconstruction_relationship.value
            ),
            "rationale": upstream_hlg_evaluation.rationale,
        }
        if upstream_hlg_evaluation is not None
        else None
    )
    prompt = f"""Few-shot calibration:
{LLG_EVALUATOR_EXAMPLES}

{QUALITY_SCORE_RUBRIC}

Perform a strict evidence-based audit of the valid parent HLG decomposition.
KEEP_LOW_LEVEL_GOALS is allowed only for a complete, precise, non-redundant,
fully documented 5/5 decomposition. Regenerate when at least one essential
capability explicitly required by the parent HLG is absent, or a current LLG
is contradicted by, unsupported in, redundant with, or materially outside the
documentation and parent boundary. Do not regenerate merely because an
optional operation is absent, but do regenerate when an optional operation is
already present and lacks documentary support. API/CRUD wording alone is not a
defect when it expresses a documented stakeholder capability; pure UI actions,
transport details, technical bookkeeping, and multiple fragments of one action
are defects when the documentation does not establish them as distinct goals.
Do not change the HLG.

When supplied, Critic 1's aggregated HLG evaluation is an upstream constraint,
not optional context. If its relationship is PARTIAL_OVERLAP, DIFFERENT, MATCHES_SIBLING,
or NOT_RECONSTRUCTABLE, the quorum has identified a material mismatch between
the parent and the aggregate meaning of its children. In that case
KEEP_LOW_LEVEL_GOALS is invalid: locate how the current LLG set causes that
mismatch and choose REGENERATE_LOW_LEVEL_GOALS. Preserve supported LLGs in the
feedback while removing out-of-parent responsibilities or adding the essential
parent capability that is absent. A sibling's coverage never excuses this
decomposition from matching its own parent. With a supplied upstream verdict,
only EQUIVALENT permits KEEP, and it still requires the complete independent
5/5 audit below. If the upstream value is null, perform the independent audit
without inventing an upstream mismatch.

Audit every current LLG individually. Its capability must be explicitly stated
or necessarily entailed by the project description and the parent HLG. Being
common, useful, conventional, standard for the domain, or compatible with the
workflow is not evidence. Do not infer notifications, downloads, reassignment,
internal notes, audit operations, confirmations, status transitions, or CRUD
variants unless the supplied description entails them. Broad verbs such as
"manage", "monitor", or "process" do not authorize arbitrary lifecycle
operations. Put every unsupported current LLG in
unsupported_or_misleading_llg_ids and choose REGENERATE.

Necessary entailment is narrow: without that capability, the explicitly
documented parent outcome could not be achieved. A convenient interface step,
typical implementation choice, possible follow-up action, or one of several
alternative workflows is not necessary entailment. If an LLG mixes a supported
capability with an unsupported clause, mark the whole LLG as misleading. If two
or more LLGs duplicate the same stakeholder outcome or split it into
non-independent click/state/validation fragments, identify the redundant or
fragment-only IDs for consolidation.

Each missing capability must be essential to this parent, explicitly grounded
in its wording and documentation, and absent from every current LLG. Do not
import capabilities owned by another HLG. If the evidence is debatable, choose
REGENERATE_LOW_LEVEL_GOALS and identify the unsupported current LLG or the
explicitly missing essential capability. Uncertainty must not preserve invented
functionality. A KEEP rationale must map every llg_NNN to a concrete phrase in
the documentation or demonstrate why it is strictly necessary, then account
for every essential parent outcome. Generic statements such as "the LLGs cover
the workflow" are insufficient.

Complete project description:
{project_description}

Parent HLG:
{branch.high_level_goal.model_dump_json()}

Current LLGs:
{json.dumps(llgs, ensure_ascii=False)}

Sibling HLGs whose responsibilities must not be duplicated:
{json.dumps(sibling_hlgs, ensure_ascii=False)}

Reconstructed HLG:
{reconstruction.reconstructed_high_level_goal}

Reconstructor rationale (untrusted hypothesis, not calibration):
{reconstruction.rationale}

Aggregated upstream HLG evaluation:
{json.dumps(upstream_hlg, ensure_ascii=False)}

Use only the supplied llg_NNN identifiers in
unsupported_or_misleading_llg_ids. List concise, explicitly documented parent
capabilities in missing_essential_capabilities. A REGENERATE decision must
contain at least one valid unsupported ID or one missing essential capability
and actionable regeneration_feedback. A KEEP decision must use quality_score 5,
empty defect lists, and null regeneration_feedback.

Output JSON:
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
    if upstream_hlg_evaluation is not None:
        evaluation = _enforce_cross_level_consistency(
            branch,
            upstream_hlg_evaluation,
            evaluation,
        )
    return evaluation


def evaluate_missing_high_level_goals(
    project_description: str,
    current_hlgs: HighLevelGoals,
    actors: Actors,
    conversation: EvaluatorConversation | None = None,
    retired_hlgs: HighLevelGoals | None = None,
) -> MissingHighLevelGoalEvaluation:
    """Check project-wide HLG coverage once per bottom-up iteration."""
    retired_hlgs = retired_hlgs or HighLevelGoals(goals=[])
    prompt = f"""Evaluator rationale and calibration:
{COVERAGE_EVALUATOR_EXAMPLES}

{QUALITY_SCORE_RUBRIC}

Discovery condition is enabled for this evaluation. Unlike branch validation,
this task must inspect the complete project description and determine whether
at least one documented actor-level functional intention is absent from the
current HLG set. Apply this condition in addition to the normal evidence,
actor-authority, functional-scope, and non-hallucination requirements.

Perform a fresh requirement-by-requirement coverage audit. Check whether the
project description contains functional, actor-level High-Level Goals that are
missing from the current HLG list. Previous conversation conclusions are
context only, not evidence; reassess coverage from the supplied description and
current HLGs on every call.

NO_MISSING_HIGH_LEVEL_GOALS is permitted only after every extracted actor
capability has been mapped to a current HLG that actually entails it. Topic
similarity, a broad label, or partial workflow overlap is not coverage. A score
of 3 indicates an unresolved coverage defect and cannot pass.

Do not generate final HLG objects. Return only focused requests for the normal
top-down HLG generator. Do not propose goals already covered by current HLGs,
technical/API-level operations, or intentions unsupported by the description.

Use this audit procedure internally before deciding:
1. Extract every explicit actor capability, desired outcome, mandatory
   interaction, and externally visible responsibility from the description.
2. Assign each requirement to one supplied actor and one current HLG only when
   that HLG's wording semantically entails the same outcome.
3. List requirements left uncovered, then group only those that serve one
   coherent stakeholder WHY.
4. Report every distinct documented actor-level gap. Do not truncate the list
   to an arbitrary number and do not merge independent outcomes merely to keep
   the response short.

A current HLG may cover several workflow steps, CRUD operations, or narrower
variants that jointly realize the same outcome. However, thematic similarity is
not coverage. Do not treat a requirement as covered merely because it concerns
the same domain object, actor, workflow, or broad topic. Separate it when it has
an independently satisfiable outcome, a distinct interaction channel or
external-system integration, a different artifact being managed, or a separate
actor responsibility that would require its own coherent LLG decomposition.

One clear, explicit functional requirement is sufficient evidence for a gap; it
does not need to be repeated in two passages. An implicit gap still requires at
least two coherent workflow passages or responsibilities. Never infer a feature
from conventional domain knowledge. Return NO_MISSING_HIGH_LEVEL_GOALS only
when every explicit functional requirement is entailed by a current HLG or the
remaining evidence is non-functional, implementation-only, ambiguous, or
merely conventional.

For every proposed gap, use the actor name exactly as it appears in the supplied
actor list. The generation_project_description must identify the uncovered
outcome and cite its concrete support from the project description; it must not
request unrelated improvements or multiple intentions. Never introduce a new
actor.

Complete project description:
{project_description}

Already identified actors:
{actors.model_dump_json()}

Current High-Level Goals:
{json.dumps([goal.model_dump(mode="json") for goal in current_hlgs.goals], ensure_ascii=False)}

HLGs removed or replaced earlier in this cycle:
{json.dumps([goal.model_dump(mode="json") for goal in retired_hlgs.goals], ensure_ascii=False)}

Audit every responsibility represented by the retired HLGs. Do not restore a
retired goal when a current HLG already entails its documented outcome, but do
report a gap when deletion or rewriting left that outcome uncovered. The
retired list is an audit checklist, not evidence that a goal must be restored.

Output only valid JSON:
{{
  "decision": "NO_MISSING_HIGH_LEVEL_GOALS | MISSING_HIGH_LEVEL_GOALS_FOUND",
  "rationale": "...",
  "quality_score": 0,
  "missing_goal_requests": [
    {{"actor": "Actor name", "generation_project_description": "Focused stakeholder description"}}
  ]
}}"""
    try:
        evaluation = _evaluate(
            prompt,
            MissingHighLevelGoalEvaluation,
            conversation,
            memory_label=(
                "Project-wide HLG coverage evaluation with "
                f"{len(current_hlgs.goals)} current HLGs"
            ),
            evaluator_models=DISCOVERY_EVALUATOR_MODELS,
        )
    except (ValueError, TypeError) as exc:
        return MissingHighLevelGoalEvaluation(
            decision=MissingHighLevelGoalDecision.INCONCLUSIVE,
            rationale=f"Coverage voting failed: {exc}",
            quality_score=0,
            missing_goal_requests=[],
            voter_count=len(DISCOVERY_EVALUATOR_MODELS),
            valid_vote_count=0,
            winning_vote_count=0,
            vote_distribution={},
            evaluator_votes=getattr(exc, "vote_records", []),
        )
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


def evaluate_branch(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals,
    conversation: EvaluatorConversation | None = None,
    current_branches: list[GoalBranch] | None = None,
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
        else:
            # HLG equivalence validates the parent/aggregate relationship, not
            # every individual child. Always audit the LLG decomposition before
            # confirming a kept branch.
            try:
                llg = evaluate_llg_decomposition(
                    project_description,
                    branch,
                    reconstruction,
                    current_hlgs,
                    conversation,
                    upstream_hlg_evaluation=hlg,
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
            if not branch.low_level_goals and decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH:
                llg = llg.model_copy(update={
                    "rationale": (
                        f"{llg.rationale} Empty-branch invariant: a valid HLG "
                        "cannot be confirmed without an LLG decomposition."
                    ),
                    "decision": LowLevelGoalDecision.REGENERATE,
                    "quality_score": min(llg.quality_score, 2),
                    "regeneration_feedback": (
                        "Generate a complete, non-redundant decomposition of "
                        "the documented parent HLG into atomic functional "
                        "interactions."
                    ),
                    "missing_essential_capabilities": [
                        branch.high_level_goal.description
                    ],
                })
                decision = GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS

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
        reconstruction_relationship=ReconstructionRelationship.DIFFERENT,
        requires_llg_evaluation=True,
        voter_count=BOTTOM_UP_EVALUATOR_VOTERS,
        valid_vote_count=0,
        winning_vote_count=0,
        vote_distribution={},
        evaluator_votes=(critic_votes or {}).get("hlg", []),
    )
    return GlobalGoalEvaluationResult(
        branch_id=branch.branch_id,
        rationale=rationale,
        final_decision=GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
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
        evaluation.removal_basis == HighLevelGoalRemovalBasis.UNSUPPORTED
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
        "requires_llg_evaluation": (
            evaluation.reconstruction_relationship
            != ReconstructionRelationship.EQUIVALENT
        ),
        "rewriting_focus": None,
        "removal_basis": None,
        "covered_by_branch_id": None,
        "covered_by_high_level_goal_name": None,
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
):
    all_branches = all_branches or branches
    total = len(branches)
    hlg_vote_records = {branch.branch_id: [] for branch in branches}

    # Stage 1: one model load evaluates every HLG branch before switching to
    # the next voter. All calls use the same conversation snapshot; memory is
    # updated only after the three votes have been aggregated.
    for evaluator_model in BOTTOM_UP_EVALUATOR_MODELS:
        print(
            f"[bottom-up] loading HLG voter: {evaluator_model}",
            flush=True,
        )
        with loaded_local_evaluator_model(evaluator_model):
            for index, branch in enumerate(branches, start=1):
                print(
                    f"[bottom-up] HLG voter {evaluator_model} branch "
                    f"{index}/{total}: {branch.high_level_goal.actor.name}::"
                    f"{branch.high_level_goal.name}",
                    flush=True,
                )
                try:
                    single = evaluate_original_hlg(
                        project_description,
                        branch,
                        reconstructions[branch.branch_id],
                        current_hlgs,
                        conversation,
                        all_branches,
                        evaluator_models=(evaluator_model,),
                        manage_model_lifecycle=False,
                        record_conversation=False,
                        apply_destructive_gate=False,
                    )
                    hlg_vote_records[branch.branch_id].extend(
                        single.evaluator_votes
                    )
                except (ValueError, TypeError) as exc:
                    records = getattr(exc, "vote_records", None)
                    hlg_vote_records[branch.branch_id].extend(
                        records
                        or [EvaluatorVoteRecord(
                            model=evaluator_model,
                            status="INVALID",
                            error=f"{type(exc).__name__}: {exc}",
                        )]
                    )
        print(
            f"[bottom-up] unloaded HLG voter: {evaluator_model}",
            flush=True,
        )

    hlg_evaluations: dict[str, HighLevelGoalEvaluation] = {}
    evaluations: dict[str, GlobalGoalEvaluationResult] = {}
    llg_branches: list[GoalBranch] = []
    for branch in branches:
        try:
            hlg = _aggregate_saved_votes(
                hlg_vote_records[branch.branch_id],
                HighLevelGoalEvaluation,
                conversation,
                f"HLG evaluation for actor '{branch.high_level_goal.actor.name}', "
                f"goal '{branch.high_level_goal.name}'",
            )
            hlg = _apply_batched_hlg_removal_gate(hlg, branch, all_branches)
            hlg_evaluations[branch.branch_id] = hlg
            if hlg.decision == HighLevelGoalDecision.KEEP:
                llg_branches.append(branch)
        except (ValueError, TypeError) as exc:
            evaluations[branch.branch_id] = _inconclusive_result(
                branch,
                f"HLG evaluation failed: {exc}",
                critic_votes={
                    "hlg": getattr(
                        exc, "vote_records", hlg_vote_records[branch.branch_id]
                    )
                },
            )

    # Stage 2: every branch retained by the aggregated HLG decision is sent to
    # the LLG critic. HLG equivalence alone cannot certify each individual LLG.
    # Models are again loaded once for the complete stage.
    llg_vote_records = {branch.branch_id: [] for branch in llg_branches}
    for evaluator_model in BOTTOM_UP_EVALUATOR_MODELS:
        if not llg_branches:
            break
        print(
            f"[bottom-up] loading LLG voter: {evaluator_model}", flush=True
        )
        with loaded_local_evaluator_model(evaluator_model):
            for index, branch in enumerate(llg_branches, start=1):
                print(
                    f"[bottom-up] LLG voter {evaluator_model} branch "
                    f"{index}/{len(llg_branches)}: "
                    f"{branch.high_level_goal.actor.name}::"
                    f"{branch.high_level_goal.name}",
                    flush=True,
                )
                try:
                    single = evaluate_llg_decomposition(
                        project_description,
                        branch,
                        reconstructions[branch.branch_id],
                        current_hlgs,
                        conversation,
                        upstream_hlg_evaluation=(
                            hlg_evaluations[branch.branch_id]
                        ),
                        evaluator_models=(evaluator_model,),
                        manage_model_lifecycle=False,
                        record_conversation=False,
                    )
                    llg_vote_records[branch.branch_id].extend(
                        single.evaluator_votes
                    )
                except (ValueError, TypeError) as exc:
                    records = getattr(exc, "vote_records", None)
                    llg_vote_records[branch.branch_id].extend(
                        records
                        or [EvaluatorVoteRecord(
                            model=evaluator_model,
                            status="INVALID",
                            error=f"{type(exc).__name__}: {exc}",
                        )]
                    )
        print(
            f"[bottom-up] unloaded LLG voter: {evaluator_model}", flush=True
        )

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
                llg = _enforce_cross_level_consistency(branch, hlg, llg)
            except (ValueError, TypeError) as exc:
                evaluations[branch.branch_id] = _inconclusive_result(
                    branch,
                    f"LLG evaluation failed: {exc}",
                    hlg=hlg,
                    critic_votes={
                        "hlg": hlg.evaluator_votes,
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
        elif llg is None:
            decision = GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        else:
            decision = (
                GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
                if llg.decision == LowLevelGoalDecision.REGENERATE
                else GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            )
            if not branch.low_level_goals and decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH:
                llg = llg.model_copy(update={
                    "rationale": (
                        f"{llg.rationale} Empty-branch invariant: a valid HLG "
                        "cannot be confirmed without an LLG decomposition."
                    ),
                    "decision": LowLevelGoalDecision.REGENERATE,
                    "quality_score": min(llg.quality_score, 2),
                    "regeneration_feedback": (
                        "Generate a complete, non-redundant decomposition of "
                        "the documented parent HLG into atomic functional interactions."
                    ),
                    "missing_essential_capabilities": [
                        branch.high_level_goal.description
                    ],
                })
                decision = GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS

        rationale_parts = [hlg.rationale]
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
                **({"llg": llg.evaluator_votes} if llg is not None else {}),
            },
        )

    ordered_evaluations = {}
    for index, branch in enumerate(branches, start=1):
        evaluation = evaluations[branch.branch_id]
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
            f"{evaluation.final_decision.value}; LLG critic={llg_route}",
            flush=True,
        )
    return ordered_evaluations
