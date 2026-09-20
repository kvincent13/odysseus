import pytest

from src.agent_contracts import AgentAuthority
from src.agent_registry import (
    AgentDefinition,
    AgentRegistry,
    AgentRegistryError,
    build_default_agent_registry,
)


def test_research_read_only_allowed():
    registry = build_default_agent_registry()

    definition = registry.validate_delegation(
        "research",
        AgentAuthority.READ_ONLY,
        ["web_search", "web_fetch"],
    )

    assert definition.name == "research"


def test_research_cannot_be_autonomous():
    registry = build_default_agent_registry()

    with pytest.raises(AgentRegistryError):
        registry.validate_delegation(
            "research",
            AgentAuthority.AUTONOMOUS,
            ["web_search"],
        )


def test_research_cannot_receive_email_tool():
    registry = build_default_agent_registry()

    with pytest.raises(AgentRegistryError):
        registry.validate_delegation(
            "research",
            AgentAuthority.READ_ONLY,
            ["send_email"],
        )


def test_azure_cannot_exceed_approval_authority():
    registry = build_default_agent_registry()

    with pytest.raises(AgentRegistryError):
        registry.validate_delegation(
            "azure",
            AgentAuthority.AUTONOMOUS,
            [],
        )


def test_finance_cannot_receive_execution_authority():
    registry = build_default_agent_registry()

    with pytest.raises(AgentRegistryError):
        registry.validate_delegation(
            "finance",
            AgentAuthority.ACT_WITH_APPROVAL,
            [],
        )


def test_unknown_agent_rejected():
    registry = build_default_agent_registry()

    with pytest.raises(AgentRegistryError):
        registry.validate_delegation(
            "does-not-exist",
            AgentAuthority.READ_ONLY,
            [],
        )


def test_duplicate_registration_rejected():
    registry = AgentRegistry()

    definition = AgentDefinition(
        name="research",
        description="Research specialist",
        max_authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    registry.register(definition)

    with pytest.raises(AgentRegistryError):
        registry.register(definition)


def test_agent_names_are_case_insensitive():
    registry = build_default_agent_registry()

    definition = registry.get("RESEARCH")

    assert definition.name == "research"


def test_declared_future_agents_have_no_tools_yet():
    registry = build_default_agent_registry()

    for name in ("azure", "email", "finance", "security"):
        assert registry.get(name).allowed_tools == []
