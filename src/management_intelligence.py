"""Management-intelligence contracts for Cara.

Cara decides whether work should be handled directly, delegated to existing
staff, proposed as a new specialist capability, treated as configuration, or
clarified with the user.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ManagementDecision(str, Enum):
    HANDLE = "handle"
    DELEGATE = "delegate"
    RECRUIT = "recruit"
    CONFIGURE = "configure"
    ASK = "ask"


class ConfigurationAuthority(str, Enum):
    """How much authority Cara has over a configuration change."""

    SELF_SERVICE = "self_service"
    REQUIRES_APPROVAL = "requires_approval"
    FORBIDDEN = "forbidden"


@dataclass
class SpecialistProposal:
    """A proposed staff position. This does NOT activate an agent."""

    name: str
    mission: str
    capability_gap: str

    description: str = ""

    # Semantic capabilities describe WHAT the specialist needs to accomplish.
    # They are intentionally independent of concrete tool/plugin/MCP names.
    required_capabilities: List[str] = field(default_factory=list)

    # Legacy/model compatibility. Proposed tool names are hints only and are
    # never treated as verified capabilities or authorization.
    required_tools: List[str] = field(default_factory=list)

    max_authority: str = "read_only"
    preferred_model: Optional[str] = None

    playbook: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)

    rationale: str = ""
    recurring_need: bool = False


@dataclass
class ConfigurationProposal:
    """A proposed change to Cara's own operating configuration."""

    setting: str
    proposed_value: Any
    authority: ConfigurationAuthority
    rationale: str = ""


@dataclass
class ManagementPlan:
    decision: ManagementDecision
    reason: str

    # DELEGATE
    specialist: Optional[str] = None
    objective: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    requested_authority: Optional[str] = None

    # RECRUIT
    specialist_proposal: Optional[SpecialistProposal] = None

    # CONFIGURE
    configuration_proposal: Optional[ConfigurationProposal] = None

    # ASK
    question: Optional[str] = None

    # Common audit/explanation metadata.
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ManagementPlanError(ValueError):
    pass


def validate_management_plan(plan: ManagementPlan) -> None:
    """Reject internally inconsistent management decisions."""

    if not plan.reason.strip():
        raise ManagementPlanError("Management plan requires a reason.")

    if plan.decision == ManagementDecision.DELEGATE:
        if not plan.specialist:
            raise ManagementPlanError(
                "DELEGATE requires a specialist."
            )
        if not plan.objective:
            raise ManagementPlanError(
                "DELEGATE requires an objective."
            )

    elif plan.decision == ManagementDecision.RECRUIT:
        if plan.specialist_proposal is None:
            raise ManagementPlanError(
                "RECRUIT requires a specialist proposal."
            )

    elif plan.decision == ManagementDecision.CONFIGURE:
        if plan.configuration_proposal is None:
            raise ManagementPlanError(
                "CONFIGURE requires a configuration proposal."
            )

    elif plan.decision == ManagementDecision.ASK:
        if not plan.question or not plan.question.strip():
            raise ManagementPlanError(
                "ASK requires a clarification question."
            )


_MANAGEMENT_SYSTEM_PROMPT = """You are Cara's management-planning function.

Your job is NOT to solve the user's request. Decide how Cara should manage it.

Choose exactly one decision:

HANDLE
- Cara should handle the request herself.
- Use for ordinary reasoning, explanation, conversation, or work that does not
  warrant specialist delegation.

DELEGATE
- An existing operational specialist is clearly suited to the work.
- Delegate only to a specialist listed in CURRENT STAFF.
- A specialist is operational only when operational=true.
- Never delegate work to a specialist with operational=false.
- Never invent tools or authority the specialist does not possess.
- Do not delegate ordinary explanation or reasoning merely because a specialist
  could research the topic. Use HANDLE unless current/external evidence or
  specialist execution is actually needed.

RECRUIT
- There is a genuine recurring or specialized capability gap in CURRENT STAFF.
- Do not recruit merely because one task is difficult.
- Recruitment creates a proposal only. It does not activate an agent.
- Describe requirements primarily as semantic required_capabilities: WHAT the
  specialist needs to read, analyze, create, monitor, or manage.
- Do not invent internal tool names, API names, plugin names, model names, or
  implementation details.
- Leave required_tools empty unless CURRENT STAFF explicitly exposes the exact
  tool name and it is genuinely relevant.
- Leave preferred_model null unless a specific available model is provided in
  the request or context.
- New specialists should normally begin with max_authority="read_only".
  Higher authority must be justified separately.

CONFIGURE
- The request is primarily about changing how Cara or her staff operate:
  preferences, model selection, schedules, context budgets, routing,
  playbooks, or enabled specialists.
- Configuration must not grant new authority, bypass security, or expand
  credentials.

ASK
- Material information or authorization is missing and proceeding would be
  ambiguous, unsafe, or likely incorrect.

Return ONLY valid JSON.

HANDLE:
{
  "decision": "handle",
  "reason": "..."
}

DELEGATE:
{
  "decision": "delegate",
  "reason": "...",
  "specialist": "research",
  "objective": "...",
  "allowed_tools": ["web_search"],
  "requested_authority": "read_only"
}

RECRUIT:
{
  "decision": "recruit",
  "reason": "...",
  "specialist_proposal": {
    "name": "...",
    "mission": "...",
    "capability_gap": "...",
    "description": "...",
    "required_capabilities": [
      "Describe WHAT access or capability this specialist needs in plain language"
    ],
    "required_tools": [],
    "max_authority": "read_only",
    "preferred_model": null,
    "playbook": [],
    "evaluation_criteria": [],
    "rationale": "...",
    "recurring_need": true
  }
}

CONFIGURE:
{
  "decision": "configure",
  "reason": "...",
  "configuration_proposal": {
    "setting": "...",
    "proposed_value": null,
    "authority": "self_service",
    "rationale": "..."
  }
}

ASK:
{
  "decision": "ask",
  "reason": "...",
  "question": "..."
}
"""


def _staff_snapshot(registry) -> list[dict]:
    return [
        {
            "name": agent.name,
            "description": agent.description,
            "max_authority": agent.max_authority.value,
            "allowed_tools": list(agent.allowed_tools),
            "operational": bool(agent.allowed_tools),
        }
        for agent in registry.list()
    ]


def _management_plan_from_dict(data: dict) -> ManagementPlan:
    decision = ManagementDecision(
        str(data.get("decision") or "").strip().lower()
    )

    specialist_proposal = None
    raw_specialist = data.get("specialist_proposal")
    if isinstance(raw_specialist, dict):
        specialist_proposal = SpecialistProposal(
            name=str(raw_specialist.get("name") or "").strip(),
            mission=str(raw_specialist.get("mission") or "").strip(),
            capability_gap=str(
                raw_specialist.get("capability_gap") or ""
            ).strip(),
            description=str(
                raw_specialist.get("description") or ""
            ).strip(),
            required_capabilities=[
                str(x)
                for x in (
                    raw_specialist.get("required_capabilities") or []
                )
                if str(x).strip()
            ],
            required_tools=[
                str(x)
                for x in (raw_specialist.get("required_tools") or [])
                if str(x).strip()
            ],
            max_authority=str(
                raw_specialist.get("max_authority") or "read_only"
            ).strip(),
            preferred_model=raw_specialist.get("preferred_model"),
            playbook=[
                str(x) for x in (raw_specialist.get("playbook") or [])
            ],
            evaluation_criteria=[
                str(x)
                for x in (
                    raw_specialist.get("evaluation_criteria") or []
                )
            ],
            rationale=str(
                raw_specialist.get("rationale") or ""
            ).strip(),
            recurring_need=bool(
                raw_specialist.get("recurring_need", False)
            ),
        )

    configuration_proposal = None
    raw_config = data.get("configuration_proposal")
    if isinstance(raw_config, dict):
        configuration_proposal = ConfigurationProposal(
            setting=str(raw_config.get("setting") or "").strip(),
            proposed_value=raw_config.get("proposed_value"),
            authority=ConfigurationAuthority(
                str(
                    raw_config.get("authority")
                    or ConfigurationAuthority.REQUIRES_APPROVAL.value
                ).strip().lower()
            ),
            rationale=str(
                raw_config.get("rationale") or ""
            ).strip(),
        )

    plan = ManagementPlan(
        decision=decision,
        reason=str(data.get("reason") or "").strip(),
        specialist=(
            str(data.get("specialist") or "").strip() or None
        ),
        objective=(
            str(data.get("objective") or "").strip() or None
        ),
        allowed_tools=[
            str(x) for x in (data.get("allowed_tools") or [])
        ],
        requested_authority=(
            str(data.get("requested_authority") or "").strip() or None
        ),
        specialist_proposal=specialist_proposal,
        configuration_proposal=configuration_proposal,
        question=(
            str(data.get("question") or "").strip() or None
        ),
        confidence=data.get("confidence"),
    )

    validate_management_plan(plan)
    return plan


def _deterministic_management_plan(
    request_text: str,
    *,
    registry,
) -> Optional[ManagementPlan]:
    """Resolve obvious management decisions without an LLM call.

    This handles high-confidence routing/policy cases only. Ambiguous
    organizational decisions fall through to the management model.
    """

    import re

    text = str(request_text or "").strip()
    lower = text.lower()

    if not text:
        return None

    # Ambiguous consequential actions should be clarified before delegation.
    ambiguous_action_patterns = (
        r"^\s*pay\s+(?:the|this|that|an?)\s+invoice\s*[.!]?\s*$",
        r"^\s*pay\s+(?:the|this|that)\s+bill\s*[.!]?\s*$",
    )

    if any(re.search(pattern, lower) for pattern in ambiguous_action_patterns):
        return ManagementPlan(
            decision=ManagementDecision.ASK,
            reason="The requested financial action does not identify a specific target.",
            question="Which invoice or bill do you want me to work with?",
            metadata={"planner": "deterministic_v1"},
        )

    # Explicit requests to change Cara's operating behavior are configuration
    # candidates. They remain proposal-only until configuration policy decides
    # what Cara may change herself.
    configuration_patterns = (
        r"\b(context window|context length|context budget)\b",
        r"\b(?:smaller|larger|shorter|longer|increase|decrease|reduce|raise)\b.{0,30}\bcontext\b",
        r"\b(use|switch to|change to|prefer)\b.{0,40}\bmodel\b",
        r"\b(enable|disable)\b.{0,40}\bspecialist\b",
        r"\brouting (?:rule|rules|preference|preferences)\b",
    )

    if any(re.search(pattern, lower) for pattern in configuration_patterns):
        return ManagementPlan(
            decision=ManagementDecision.CONFIGURE,
            reason="The request changes Cara's operating configuration.",
            configuration_proposal=ConfigurationProposal(
                setting="requires_interpretation",
                proposed_value=text,
                authority=ConfigurationAuthority.REQUIRES_APPROVAL,
                rationale="A deterministic configuration policy has not yet mapped this request to a safe setting.",
            ),
            metadata={"planner": "deterministic_v1"},
        )

    # Requests explicitly requiring fresh/current external evidence belong to
    # the operational Research specialist.
    freshness_patterns = (
        r"\b(latest|current|today|this week|past week|recent|right now)\b",
        r"\b(search for|research (?:this|that|the|current|latest)|look up|lookup|find online|check online)\b",
    )

    if any(re.search(pattern, lower) for pattern in freshness_patterns):
        try:
            research = registry.get("research")
            if research.allowed_tools:
                tools = [
                    tool
                    for tool in ("web_search", "web_fetch")
                    if tool in research.allowed_tools
                ]

                if tools:
                    plan = ManagementPlan(
                        decision=ManagementDecision.DELEGATE,
                        reason="The request requires current or external evidence and Research is operational.",
                        specialist="research",
                        objective=text,
                        allowed_tools=tools,
                        requested_authority="read_only",
                        metadata={"planner": "deterministic_v1"},
                    )
                    validate_management_plan(plan)
                    return plan
        except Exception:
            pass

    # Straightforward explanation/advice requests normally belong to Cara.
    explanation_patterns = (
        r"^\s*(explain|describe|define)\b",
        r"^\s*what (?:is|are|does)\b",
        r"^\s*why (?:is|are|does|do)\b",
        r"^\s*how (?:does|do|is|are)\b",
    )

    if any(re.search(pattern, lower) for pattern in explanation_patterns):
        return ManagementPlan(
            decision=ManagementDecision.HANDLE,
            reason="This is an ordinary explanation or reasoning request Cara can handle directly.",
            metadata={"planner": "deterministic_v1"},
        )

    return None


async def assess_management_request(
    request_text: str,
    *,
    registry,
    owner: Optional[str] = None,
) -> ManagementPlan:
    """Ask Cara's management model how a request should be handled.

    This function plans only. It does not execute delegation, recruit staff,
    change configuration, or take external action.
    """

    import asyncio
    import json

    from src.ai_interaction import _resolve_model
    from src.llm_core import llm_call_async
    from src.settings import get_setting

    request_text = str(request_text or "").strip()
    if not request_text:
        raise ManagementPlanError("Management assessment requires a request.")

    # Resolve obvious high-confidence management decisions without spending
    # an LLM call. Ambiguous organizational decisions continue to Qwen.
    deterministic = _deterministic_management_plan(
        request_text,
        registry=registry,
    )
    if deterministic is not None:
        validate_management_plan(deterministic)
        return deterministic

    model_spec = (
        str(get_setting("management_model", "") or "").strip()
        or str(get_setting("default_model", "qwen3:14b") or "qwen3:14b").strip()
    )

    url, model, headers = await asyncio.to_thread(
        _resolve_model,
        model_spec,
        owner=owner,
    )

    payload = {
        "request": request_text,
        "current_staff": _staff_snapshot(registry),
    }

    raw = await llm_call_async(
        url=url,
        model=model,
        messages=[
            {"role": "system", "content": _MANAGEMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        headers=headers,
        temperature=0.0,
        max_tokens=1800,
        timeout=120,
        workload="background",
    )

    # Reuse Research's strict JSON extraction rather than silently repairing
    # malformed model output.
    from src.research_agent import _extract_json

    try:
        data = _extract_json(raw)
        plan = _management_plan_from_dict(data)
    except Exception as exc:
        raise ManagementPlanError(
            f"Management model returned an invalid plan: {exc}"
        ) from exc

    # Intelligence proposes delegation; deterministic registry policy decides
    # whether that delegation is actually within the specialist's capability.
    if plan.decision == ManagementDecision.DELEGATE:
        try:
            from src.agent_contracts import AgentAuthority

            authority = AgentAuthority(
                str(plan.requested_authority or "read_only").strip().lower()
            )

            registry.validate_delegation(
                plan.specialist,
                authority,
                plan.allowed_tools,
            )
        except Exception as exc:
            raise ManagementPlanError(
                f"Proposed delegation violates staff capability policy: {exc}"
            ) from exc

    # Configuration remains proposal-only until deterministic setting policy
    # is implemented. Never trust the model's self_service label by itself.
    if (
        plan.decision == ManagementDecision.CONFIGURE
        and plan.configuration_proposal is not None
    ):
        plan.configuration_proposal.authority = (
            ConfigurationAuthority.REQUIRES_APPROVAL
        )

    plan.metadata["model"] = model
    plan.metadata["planner"] = "management_v1"

    return plan


class CandidateStatus(str, Enum):
    DRAFT = "draft"
    NEEDS_CAPABILITIES = "needs_capabilities"
    NEEDS_MODEL = "needs_model"
    NEEDS_TRAINING = "needs_training"
    READY_FOR_EVALUATION = "ready_for_evaluation"
    EVALUATING = "evaluating"
    SCREENED = "screened"
    TECHNICALLY_PASSED = "technically_passed"
    PASSED = "passed"
    FAILED = "failed"
    APPROVED = "approved"
    ACTIVE = "active"


@dataclass
class CapabilityRequirement:
    name: str
    available: bool = False
    matched_tool: Optional[str] = None
    notes: str = ""


@dataclass
class TechnicalEvaluationScenario:
    """A concrete job simulation used to evaluate a specialist."""

    name: str
    objective: str
    expected_behavior: str

    prohibited_behavior: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpecialistCandidate:
    """A proposed specialist being prepared for evaluation.

    Candidates are not registered staff and have no execution authority.
    """

    name: str
    mission: str
    description: str

    initial_authority: str = "read_only"
    requested_max_authority: str = "read_only"

    preferred_model: Optional[str] = None
    resolved_model: Optional[str] = None

    capabilities: List[CapabilityRequirement] = field(default_factory=list)

    playbook: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)

    # Concrete technical interview scenarios. These test whether the candidate
    # can actually perform the job rather than merely describe how it would.
    technical_scenarios: List[TechnicalEvaluationScenario] = field(
        default_factory=list
    )

    status: CandidateStatus = CandidateStatus.DRAFT
    blockers: List[str] = field(default_factory=list)

    source_proposal: Optional[SpecialistProposal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Qualification stages are tracked independently. PASSED means every
    # required stage has succeeded for the current training snapshot.
    screening_passed: bool = False
    technical_passed: bool = False


def refresh_candidate_status(
    candidate: SpecialistCandidate,
) -> CandidateStatus:
    """Calculate the candidate's pre-evaluation lifecycle state."""

    # Capability blockers take precedence.
    if any(
        not capability.available
        for capability in candidate.capabilities
    ):
        candidate.status = CandidateStatus.NEEDS_CAPABILITIES
        return candidate.status

    # A specifically requested model must be verified before evaluation.
    if (
        candidate.preferred_model
        and not candidate.resolved_model
    ):
        candidate.status = CandidateStatus.NEEDS_MODEL
        return candidate.status

    # Training requires both an operating playbook and explicit evaluation
    # criteria. A model with tools but no job training is not ready.
    if (
        not candidate.playbook
        or not candidate.evaluation_criteria
    ):
        candidate.status = CandidateStatus.NEEDS_TRAINING
        return candidate.status

    candidate.status = CandidateStatus.READY_FOR_EVALUATION
    return candidate.status


def candidate_from_proposal(
    proposal: SpecialistProposal,
) -> SpecialistCandidate:
    """Turn Cara's staffing proposal into a safe, non-operational candidate."""

    semantic_requirements = [
        str(capability).strip()
        for capability in proposal.required_capabilities
        if str(capability).strip()
    ]

    # Legacy tool-name proposals remain unverified hints. If Cara supplied
    # semantic capabilities, those are the authoritative requirements.
    requirement_names = (
        semantic_requirements
        if semantic_requirements
        else [
            str(tool).strip()
            for tool in proposal.required_tools
            if str(tool).strip()
        ]
    )

    capabilities = [
        CapabilityRequirement(
            name=requirement,
            available=False,
            matched_tool=None,
            notes="Unverified semantic capability requirement.",
        )
        for requirement in requirement_names
    ]

    blockers = []

    if capabilities:
        blockers.append(
            "Required capabilities have not been verified against available tools."
        )

    if proposal.preferred_model:
        blockers.append(
            "Preferred model has not been verified against configured endpoints."
        )

    candidate = SpecialistCandidate(
        name=proposal.name,
        mission=proposal.mission,
        description=proposal.description,

        # New employees always start at the lowest authority.
        initial_authority="read_only",

        # Preserve Cara's recommendation, but DO NOT grant it.
        requested_max_authority=proposal.max_authority,

        preferred_model=proposal.preferred_model,
        resolved_model=None,

        capabilities=capabilities,
        playbook=list(proposal.playbook),
        evaluation_criteria=list(proposal.evaluation_criteria),

        status=CandidateStatus.DRAFT,
        blockers=blockers,
        source_proposal=proposal,

        metadata={
            "recruited_by": "cara",
            "recurring_need": proposal.recurring_need,
        },
    )

    refresh_candidate_status(candidate)
    return candidate


def reconcile_candidate_capabilities(
    candidate: SpecialistCandidate,
    inventory,
) -> SpecialistCandidate:
    """Reconcile a recruit's requirements against authoritative capabilities.

    AVAILABLE requirements are mapped to real tools.
    UNKNOWN and MISSING requirements remain blockers.
    This does not grant tool authorization or activate the candidate.
    """

    from src.capability_inventory import (
        CapabilityResolutionStatus,
        resolve_capabilities,
    )

    requirements = [
        capability.name
        for capability in candidate.capabilities
    ]

    resolutions = resolve_capabilities(
        requirements,
        inventory,
    )

    reconciled = []

    for requirement, resolution in zip(
        candidate.capabilities,
        resolutions,
    ):
        if resolution.status == CapabilityResolutionStatus.AVAILABLE:
            reconciled.append(
                CapabilityRequirement(
                    name=requirement.name,
                    available=True,
                    matched_tool=resolution.matched_tool,
                    notes=(
                        "Verified against authoritative capability inventory. "
                        f"Source: {resolution.source}; "
                        f"confidence: {resolution.confidence}."
                    ),
                )
            )

        elif resolution.status == CapabilityResolutionStatus.UNKNOWN:
            reconciled.append(
                CapabilityRequirement(
                    name=requirement.name,
                    available=False,
                    matched_tool=resolution.matched_tool,
                    notes=(
                        "Possible implementation found, but equivalence could "
                        "not be established safely. "
                        f"Candidate: {resolution.matched_tool or 'none'}; "
                        f"confidence: {resolution.confidence}."
                    ),
                )
            )

        else:
            reconciled.append(
                CapabilityRequirement(
                    name=requirement.name,
                    available=False,
                    matched_tool=None,
                    notes="No authoritative implementation was found.",
                )
            )

    candidate.capabilities = reconciled

    # Rebuild capability-related blockers while preserving unrelated blockers
    # such as unresolved model requirements.
    candidate.blockers = [
        blocker
        for blocker in candidate.blockers
        if not blocker.startswith("Required capabilities")
        and not blocker.startswith("Capability unresolved:")
        and not blocker.startswith("Capability missing:")
    ]

    for capability, resolution in zip(
        candidate.capabilities,
        resolutions,
    ):
        if resolution.status == CapabilityResolutionStatus.UNKNOWN:
            candidate.blockers.append(
                f"Capability unresolved: {capability.name}"
            )

        elif resolution.status == CapabilityResolutionStatus.MISSING:
            candidate.blockers.append(
                f"Capability missing: {capability.name}"
            )

    refresh_candidate_status(candidate)

    candidate.metadata["capability_resolution"] = [
        {
            "requirement": resolution.requirement,
            "status": resolution.status.value,
            "matched_tool": resolution.matched_tool,
            "source": resolution.source,
            "confidence": resolution.confidence,
        }
        for resolution in resolutions
    ]

    return candidate


async def reconcile_candidate_model(
    candidate: SpecialistCandidate,
    *,
    owner: Optional[str] = None,
) -> SpecialistCandidate:
    """Verify an explicitly requested candidate model against real endpoints.

    No preferred model means no model-specific blocker; the candidate may
    inherit the configured runtime/default model later.
    """

    # Remove stale model-resolution blockers before recalculating.
    candidate.blockers = [
        blocker
        for blocker in candidate.blockers
        if not blocker.startswith("Preferred model has not been verified")
        and not blocker.startswith("Preferred model unavailable:")
    ]

    if not candidate.preferred_model:
        candidate.resolved_model = None
        refresh_candidate_status(candidate)
        return candidate

    from src.ai_interaction import _resolve_model

    try:
        _url, resolved_model, _headers = await __import__(
            "asyncio"
        ).to_thread(
            _resolve_model,
            candidate.preferred_model,
            owner=owner,
        )

        candidate.resolved_model = resolved_model

        candidate.metadata["model_resolution"] = {
            "requested": candidate.preferred_model,
            "resolved": resolved_model,
            "available": True,
        }

    except Exception as exc:
        candidate.resolved_model = None
        candidate.blockers.append(
            f"Preferred model unavailable: {candidate.preferred_model}"
        )

        candidate.metadata["model_resolution"] = {
            "requested": candidate.preferred_model,
            "resolved": None,
            "available": False,
            "error": str(exc),
        }

    refresh_candidate_status(candidate)
    return candidate
