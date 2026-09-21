import pytest

from src.management_intelligence import (
    ConfigurationAuthority,
    ConfigurationProposal,
    ManagementDecision,
    ManagementPlan,
    ManagementPlanError,
    SpecialistProposal,
    validate_management_plan,
)


def test_handle_requires_only_reason():
    plan = ManagementPlan(
        decision=ManagementDecision.HANDLE,
        reason="Cara can answer this directly.",
    )
    validate_management_plan(plan)


def test_delegate_requires_specialist():
    plan = ManagementPlan(
        decision=ManagementDecision.DELEGATE,
        reason="Specialized research is warranted.",
        objective="Research current Azure announcements.",
    )

    with pytest.raises(ManagementPlanError):
        validate_management_plan(plan)


def test_delegate_requires_objective():
    plan = ManagementPlan(
        decision=ManagementDecision.DELEGATE,
        reason="Research specialist is appropriate.",
        specialist="research",
    )

    with pytest.raises(ManagementPlanError):
        validate_management_plan(plan)


def test_valid_delegate():
    plan = ManagementPlan(
        decision=ManagementDecision.DELEGATE,
        reason="Current external evidence is required.",
        specialist="research",
        objective="Research current Azure announcements.",
        allowed_tools=["web_search", "web_fetch"],
        requested_authority="read_only",
    )

    validate_management_plan(plan)


def test_recruit_requires_proposal():
    plan = ManagementPlan(
        decision=ManagementDecision.RECRUIT,
        reason="A capability gap exists.",
    )

    with pytest.raises(ManagementPlanError):
        validate_management_plan(plan)


def test_valid_recruit_proposal():
    proposal = SpecialistProposal(
        name="azure_cost_analyst",
        mission="Analyze Azure spend and identify optimization opportunities.",
        capability_gap="No existing specialist owns recurring Azure cost analysis.",
        required_tools=["azure_cost_read", "azure_metrics_read"],
        max_authority="read_only",
        playbook=[
            "Review current spend.",
            "Compare against historical baseline.",
            "Identify material anomalies.",
        ],
        evaluation_criteria=[
            "Does not modify Azure resources.",
            "Quantifies identified savings.",
        ],
        recurring_need=True,
    )

    plan = ManagementPlan(
        decision=ManagementDecision.RECRUIT,
        reason="Recurring specialized work warrants dedicated staff.",
        specialist_proposal=proposal,
    )

    validate_management_plan(plan)


def test_configure_requires_proposal():
    plan = ManagementPlan(
        decision=ManagementDecision.CONFIGURE,
        reason="Cara should change an operating preference.",
    )

    with pytest.raises(ManagementPlanError):
        validate_management_plan(plan)


def test_valid_self_service_configuration():
    proposal = ConfigurationProposal(
        setting="research_context_length",
        proposed_value=8192,
        authority=ConfigurationAuthority.SELF_SERVICE,
        rationale="Routine research does not require a 40K context.",
    )

    plan = ManagementPlan(
        decision=ManagementDecision.CONFIGURE,
        reason="Reduce unnecessary specialist resource usage.",
        configuration_proposal=proposal,
    )

    validate_management_plan(plan)


def test_ask_requires_question():
    plan = ManagementPlan(
        decision=ManagementDecision.ASK,
        reason="The request lacks required information.",
    )

    with pytest.raises(ManagementPlanError):
        validate_management_plan(plan)


def test_valid_ask():
    plan = ManagementPlan(
        decision=ManagementDecision.ASK,
        reason="The invoice is ambiguous.",
        question="Which invoice do you want me to work with?",
    )

    validate_management_plan(plan)


def test_every_decision_requires_reason():
    for decision in ManagementDecision:
        plan = ManagementPlan(
            decision=decision,
            reason="",
        )

        with pytest.raises(ManagementPlanError):
            validate_management_plan(plan)


@pytest.mark.asyncio
async def test_assessor_valid_delegate(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    from src.agent_registry import build_default_agent_registry
    from src.management_intelligence import assess_management_request

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    async def fake_llm_call_async(**kwargs):
        return '''{
          "decision": "delegate",
          "reason": "Current external evidence is required.",
          "specialist": "research",
          "objective": "Research current Azure announcements.",
          "allowed_tools": ["web_search", "web_fetch"],
          "requested_authority": "read_only"
        }'''

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)

    plan = await assess_management_request(
        "Find Microsoft's Azure announcements from this week.",
        registry=build_default_agent_registry(),
        owner="kyle",
    )

    assert plan.decision == ManagementDecision.DELEGATE
    assert plan.specialist == "research"
    assert plan.allowed_tools == ["web_search", "web_fetch"]
    assert plan.metadata["planner"] == "deterministic_v1"


@pytest.mark.asyncio
async def test_assessor_rejects_invented_delegation_capability(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    from src.agent_registry import build_default_agent_registry
    from src.management_intelligence import assess_management_request

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    async def fake_llm_call_async(**kwargs):
        return '''{
          "decision": "delegate",
          "reason": "Research should send the message.",
          "specialist": "research",
          "objective": "Send an email.",
          "allowed_tools": ["send_email"],
          "requested_authority": "read_only"
        }'''

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)

    with pytest.raises(ManagementPlanError):
        await assess_management_request(
            "Send this email.",
            registry=build_default_agent_registry(),
            owner="kyle",
        )


@pytest.mark.asyncio
async def test_assessor_configuration_cannot_self_authorize(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    from src.agent_registry import build_default_agent_registry
    from src.management_intelligence import assess_management_request

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    async def fake_llm_call_async(**kwargs):
        return '''{
          "decision": "configure",
          "reason": "Use a smaller research context.",
          "configuration_proposal": {
            "setting": "research_context_length",
            "proposed_value": 8192,
            "authority": "self_service",
            "rationale": "Reduce resource usage."
          }
        }'''

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)

    plan = await assess_management_request(
        "Use a smaller context for routine research.",
        registry=build_default_agent_registry(),
        owner="kyle",
    )

    assert plan.decision == ManagementDecision.CONFIGURE
    assert (
        plan.configuration_proposal.authority
        == ConfigurationAuthority.REQUIRES_APPROVAL
    )


@pytest.mark.asyncio
async def test_assessor_can_recruit_without_activating_agent(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    from src.agent_registry import build_default_agent_registry
    from src.management_intelligence import assess_management_request

    registry = build_default_agent_registry()
    before = len(registry.list())

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    async def fake_llm_call_async(**kwargs):
        return '''{
          "decision": "recruit",
          "reason": "Recurring Azure cost analysis needs dedicated ownership.",
          "specialist_proposal": {
            "name": "azure_cost_analyst",
            "mission": "Analyze Azure spend and optimization opportunities.",
            "capability_gap": "No current specialist owns recurring cloud cost analysis.",
            "description": "Read-only Azure cost specialist.",
            "required_tools": ["azure_cost_read"],
            "max_authority": "read_only",
            "preferred_model": null,
            "playbook": ["Review spend", "Identify anomalies"],
            "evaluation_criteria": ["Never modify resources"],
            "rationale": "Recurring specialized need.",
            "recurring_need": true
          }
        }'''

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)

    plan = await assess_management_request(
        "Start monitoring Azure costs every morning.",
        registry=registry,
        owner="kyle",
    )

    assert plan.decision == ManagementDecision.RECRUIT
    assert plan.specialist_proposal.name == "azure_cost_analyst"

    # Planning must not silently hire/activate anyone.
    assert len(registry.list()) == before


@pytest.mark.asyncio
async def test_assessor_can_ask_for_missing_information(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    from src.agent_registry import build_default_agent_registry
    from src.management_intelligence import assess_management_request

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    async def fake_llm_call_async(**kwargs):
        return '''{
          "decision": "ask",
          "reason": "The target invoice is ambiguous.",
          "question": "Which invoice do you want me to work with?"
        }'''

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)

    plan = await assess_management_request(
        "Pay the invoice.",
        registry=build_default_agent_registry(),
        owner="kyle",
    )

    assert plan.decision == ManagementDecision.ASK
    assert "Which invoice" in plan.question


def test_candidate_from_proposal_is_non_operational():
    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
    )

    proposal = SpecialistProposal(
        name="azure_cost_optimizer",
        mission=(
            "Monitor Azure costs, detect anomalies, identify optimization "
            "opportunities, and propose recommendations."
        ),
        capability_gap=(
            "No operational specialist owns recurring Azure cost management."
        ),
        description="Azure cost-management specialist.",
        required_tools=[
            "azure_cost_analytics",
            "azure_budget_api",
        ],
        max_authority="act_with_approval",
        preferred_model="azure_cost_management_v2",
        playbook=[
            "daily_cost_monitoring",
            "anomaly_detection",
            "optimization_recommendation",
        ],
        evaluation_criteria=[
            "accuracy_of_cost_forecasts",
            "number_of_optimization_opportunities_found",
        ],
        recurring_need=True,
    )

    candidate = candidate_from_proposal(proposal)

    assert candidate.name == "azure_cost_optimizer"

    # Cara may recommend higher eventual authority, but recruitment never
    # grants it automatically.
    assert candidate.initial_authority == "read_only"
    assert candidate.requested_max_authority == "act_with_approval"

    # Proposed implementation details are requirements, not facts.
    assert len(candidate.capabilities) == 2
    assert all(not capability.available for capability in candidate.capabilities)
    assert all(
        capability.matched_tool is None
        for capability in candidate.capabilities
    )

    assert candidate.preferred_model == "azure_cost_management_v2"
    assert candidate.resolved_model is None

    assert candidate.status == CandidateStatus.NEEDS_CAPABILITIES
    assert candidate.blockers

    # Preserve provenance so Cara can later explain why this employee exists.
    assert candidate.source_proposal is proposal
    assert candidate.metadata["recruited_by"] == "cara"
    assert candidate.metadata["recurring_need"] is True


def test_candidate_without_requirements_but_without_training_needs_training():
    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
    )

    proposal = SpecialistProposal(
        name="reasoning_specialist",
        mission="Perform isolated specialist reasoning.",
        capability_gap="Dedicated reasoning specialization is warranted.",
        required_tools=[],
        preferred_model=None,
        max_authority="read_only",
    )

    candidate = candidate_from_proposal(proposal)

    assert candidate.initial_authority == "read_only"
    assert candidate.capabilities == []
    assert candidate.blockers == []
    assert candidate.status == CandidateStatus.NEEDS_TRAINING


def test_management_parser_reads_semantic_capabilities():
    from src.management_intelligence import _management_plan_from_dict

    plan = _management_plan_from_dict({
        "decision": "recruit",
        "reason": "A recurring capability gap exists.",
        "specialist_proposal": {
            "name": "azure_cost_analyst",
            "mission": "Monitor and analyze Azure costs.",
            "capability_gap": "No operational specialist owns Azure cost analysis.",
            "required_capabilities": [
                "Read Azure subscription cost and usage data",
                "Read Azure resource utilization metrics",
            ],
            "required_tools": [],
            "max_authority": "read_only",
            "preferred_model": None,
            "recurring_need": True,
        },
    })

    proposal = plan.specialist_proposal

    assert proposal is not None
    assert proposal.required_capabilities == [
        "Read Azure subscription cost and usage data",
        "Read Azure resource utilization metrics",
    ]
    assert proposal.required_tools == []


def test_candidate_prefers_semantic_capabilities_over_tool_hints():
    from src.management_intelligence import candidate_from_proposal

    proposal = SpecialistProposal(
        name="azure_cost_analyst",
        mission="Monitor Azure costs.",
        capability_gap="Recurring Azure cost capability is missing.",
        required_capabilities=[
            "Read Azure subscription cost and usage data",
            "Read Azure budgets and thresholds",
        ],

        # Pretend the model still hallucinated implementation names.
        required_tools=[
            "made_up_azure_cost_tool",
            "made_up_budget_api",
        ],

        max_authority="act_with_approval",
        preferred_model="made_up_cost_model",
    )

    candidate = candidate_from_proposal(proposal)

    assert [c.name for c in candidate.capabilities] == [
        "Read Azure subscription cost and usage data",
        "Read Azure budgets and thresholds",
    ]

    assert all(not c.available for c in candidate.capabilities)
    assert all(c.matched_tool is None for c in candidate.capabilities)

    # Recruitment still clamps initial authority.
    assert candidate.initial_authority == "read_only"
    assert candidate.requested_max_authority == "act_with_approval"

    # A proposed model is still unverified.
    assert candidate.preferred_model == "made_up_cost_model"
    assert candidate.resolved_model is None


def test_candidate_all_capabilities_available_becomes_ready():
    from src.capability_inventory import AvailableCapability
    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
        reconcile_candidate_capabilities,
    )

    proposal = SpecialistProposal(
        name="azure_cost_analyst",
        mission="Analyze Azure costs.",
        capability_gap="Recurring Azure cost analysis is unstaffed.",
        required_capabilities=[
            "Read Azure subscription cost and usage data",
            "Read Azure resource utilization metrics",
        ],
        max_authority="read_only",
        playbook=[
            "Review Azure cost and usage data.",
            "Review resource utilization metrics.",
            "Identify anomalies and optimization opportunities.",
            "Report findings without modifying Azure resources.",
        ],
        evaluation_criteria=[
            "Uses only verified read-only capabilities.",
            "Identifies cost anomalies accurately.",
            "Does not modify Azure resources.",
        ],
    )

    candidate = candidate_from_proposal(proposal)

    inventory = [
        AvailableCapability(
            name="mcp__azure__read_costs",
            description="Read Azure subscription cost and usage data.",
            source="mcp",
        ),
        AvailableCapability(
            name="mcp__azure__read_metrics",
            description="Read Azure resource utilization metrics.",
            source="mcp",
        ),
    ]

    reconcile_candidate_capabilities(candidate, inventory)

    assert candidate.status == CandidateStatus.READY_FOR_EVALUATION
    assert candidate.blockers == []

    assert all(
        capability.available
        for capability in candidate.capabilities
    )

    assert [c.matched_tool for c in candidate.capabilities] == [
        "mcp__azure__read_costs",
        "mcp__azure__read_metrics",
    ]


def test_candidate_unknown_capability_remains_blocked():
    from src.capability_inventory import AvailableCapability
    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
        reconcile_candidate_capabilities,
    )

    proposal = SpecialistProposal(
        name="azure_cost_analyst",
        mission="Analyze Azure costs.",
        capability_gap="Azure cost analysis is unstaffed.",
        required_capabilities=[
            "Read Azure resource inventory and utilization metrics",
        ],
    )

    candidate = candidate_from_proposal(proposal)

    inventory = [
        AvailableCapability(
            name="mcp__azure__inventory",
            description="Read Azure resource inventory.",
            source="mcp",
        ),
    ]

    reconcile_candidate_capabilities(candidate, inventory)

    assert candidate.status == CandidateStatus.NEEDS_CAPABILITIES
    assert candidate.capabilities[0].available is False
    assert candidate.capabilities[0].matched_tool == "mcp__azure__inventory"

    assert any(
        blocker.startswith("Capability unresolved:")
        for blocker in candidate.blockers
    )


def test_candidate_missing_capability_remains_blocked():
    from src.capability_inventory import AvailableCapability
    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
        reconcile_candidate_capabilities,
    )

    proposal = SpecialistProposal(
        name="azure_cost_analyst",
        mission="Analyze Azure costs.",
        capability_gap="Azure budget analysis is unstaffed.",
        required_capabilities=[
            "Read Azure budget thresholds",
        ],
    )

    candidate = candidate_from_proposal(proposal)

    inventory = [
        AvailableCapability(
            name="web_search",
            description="Search current public web information.",
            source="builtin",
        ),
    ]

    reconcile_candidate_capabilities(candidate, inventory)

    assert candidate.status == CandidateStatus.NEEDS_CAPABILITIES
    assert candidate.capabilities[0].available is False
    assert candidate.capabilities[0].matched_tool is None

    assert any(
        blocker.startswith("Capability missing:")
        for blocker in candidate.blockers
    )


@pytest.mark.asyncio
async def test_candidate_without_preferred_model_is_not_model_blocked():
    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
        reconcile_candidate_model,
    )

    proposal = SpecialistProposal(
        name="reasoning_specialist",
        mission="Perform specialist reasoning.",
        capability_gap="Dedicated reasoning is warranted.",
        preferred_model=None,
        playbook=["Analyze the delegated objective."],
        evaluation_criteria=["Returns a relevant structured result."],
    )

    candidate = candidate_from_proposal(proposal)
    await reconcile_candidate_model(candidate)

    assert candidate.preferred_model is None
    assert candidate.resolved_model is None
    assert candidate.status == CandidateStatus.READY_FOR_EVALUATION


@pytest.mark.asyncio
async def test_candidate_preferred_model_resolves(monkeypatch):
    import src.ai_interaction as ai

    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
        reconcile_candidate_model,
    )

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    proposal = SpecialistProposal(
        name="reasoning_specialist",
        mission="Perform specialist reasoning.",
        capability_gap="Dedicated reasoning is warranted.",
        preferred_model="qwen3:14b",
        playbook=["Analyze the delegated objective."],
        evaluation_criteria=["Returns a relevant structured result."],
    )

    candidate = candidate_from_proposal(proposal)

    assert candidate.status == CandidateStatus.NEEDS_MODEL

    await reconcile_candidate_model(
        candidate,
        owner="kyle",
    )

    assert candidate.resolved_model == "qwen3:14b"
    assert candidate.status == CandidateStatus.READY_FOR_EVALUATION
    assert candidate.metadata["model_resolution"]["available"] is True


@pytest.mark.asyncio
async def test_candidate_unavailable_model_remains_blocked(monkeypatch):
    import src.ai_interaction as ai

    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
        reconcile_candidate_model,
    )

    def fail_resolve(spec, owner=None):
        raise ValueError("Model not found")

    monkeypatch.setattr(ai, "_resolve_model", fail_resolve)

    proposal = SpecialistProposal(
        name="azure_cost_analyst",
        mission="Analyze Azure costs.",
        capability_gap="Azure cost analysis is unstaffed.",
        preferred_model="azure_cost_management_v2",
        playbook=["Analyze cost data."],
        evaluation_criteria=["Does not invent cost data."],
    )

    candidate = candidate_from_proposal(proposal)

    await reconcile_candidate_model(candidate)

    assert candidate.resolved_model is None
    assert candidate.status == CandidateStatus.NEEDS_MODEL
    assert any(
        blocker.startswith("Preferred model unavailable:")
        for blocker in candidate.blockers
    )
    assert candidate.metadata["model_resolution"]["available"] is False


@pytest.mark.asyncio
async def test_model_resolution_does_not_erase_capability_blocker(monkeypatch):
    import src.ai_interaction as ai

    from src.management_intelligence import (
        CandidateStatus,
        candidate_from_proposal,
        reconcile_candidate_model,
    )

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    proposal = SpecialistProposal(
        name="azure_cost_analyst",
        mission="Analyze Azure costs.",
        capability_gap="Azure cost analysis is unstaffed.",
        required_capabilities=[
            "Read Azure subscription cost and usage data",
        ],
        preferred_model="qwen3:14b",
        playbook=["Analyze cost data."],
        evaluation_criteria=["Does not modify Azure resources."],
    )

    candidate = candidate_from_proposal(proposal)

    assert candidate.status == CandidateStatus.NEEDS_CAPABILITIES

    await reconcile_candidate_model(candidate)

    assert candidate.resolved_model == "qwen3:14b"

    # Capability gate outranks the successfully resolved model.
    assert candidate.status == CandidateStatus.NEEDS_CAPABILITIES
    assert any(
        "Required capabilities" in blocker
        for blocker in candidate.blockers
    )
