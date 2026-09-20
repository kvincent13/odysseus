"""Registry of specialist agents available to Cara."""

from dataclasses import dataclass
from typing import Dict, List

from src.agent_contracts import AgentAuthority


_AUTHORITY_RANK = {
    AgentAuthority.READ_ONLY: 0,
    AgentAuthority.PROPOSE: 1,
    AgentAuthority.ACT_WITH_APPROVAL: 2,
    AgentAuthority.AUTONOMOUS: 3,
}


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    max_authority: AgentAuthority
    allowed_tools: List[str]


class AgentRegistryError(ValueError):
    pass


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> None:
        name = definition.name.strip().lower()

        if not name:
            raise AgentRegistryError("Agent name cannot be empty.")

        if name in self._agents:
            raise AgentRegistryError(f"Agent '{name}' is already registered.")

        self._agents[name] = AgentDefinition(
            name=name,
            description=definition.description.strip(),
            max_authority=definition.max_authority,
            allowed_tools=list(definition.allowed_tools),
        )

    def get(self, name: str) -> AgentDefinition:
        key = (name or "").strip().lower()

        if key not in self._agents:
            raise AgentRegistryError(f"Unknown specialist agent '{name}'.")

        return self._agents[key]

    def list(self) -> List[AgentDefinition]:
        return list(self._agents.values())

    def validate_delegation(
        self,
        agent_name: str,
        authority: AgentAuthority,
        requested_tools: List[str],
    ) -> AgentDefinition:
        definition = self.get(agent_name)

        if _AUTHORITY_RANK[authority] > _AUTHORITY_RANK[definition.max_authority]:
            raise AgentRegistryError(
                f"Agent '{definition.name}' maximum authority is "
                f"{definition.max_authority.value}; requested {authority.value}."
            )

        allowed = set(definition.allowed_tools)
        outside_scope = sorted(set(requested_tools) - allowed)

        if outside_scope:
            raise AgentRegistryError(
                f"Agent '{definition.name}' cannot be delegated tools: "
                + ", ".join(outside_scope)
            )

        return definition


def build_default_agent_registry() -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(AgentDefinition(
        name="research",
        description="Research, investigate, compare, and gather evidence.",
        max_authority=AgentAuthority.READ_ONLY,
        allowed_tools=[
            "web_search",
            "web_fetch",
        ],
    ))

    # These are declared now so Cara has a stable staff model.
    # Their concrete tool mappings/executors will be added incrementally.

    registry.register(AgentDefinition(
        name="azure",
        description="Inspect and manage Azure infrastructure and cloud operations.",
        max_authority=AgentAuthority.ACT_WITH_APPROVAL,
        allowed_tools=[],
    ))

    registry.register(AgentDefinition(
        name="email",
        description="Monitor, analyze, draft, and eventually send email.",
        max_authority=AgentAuthority.ACT_WITH_APPROVAL,
        allowed_tools=[],
    ))

    registry.register(AgentDefinition(
        name="finance",
        description="Analyze bills, invoices, receivables, payments, and financial operations.",
        max_authority=AgentAuthority.PROPOSE,
        allowed_tools=[],
    ))

    registry.register(AgentDefinition(
        name="security",
        description="Investigate security posture, alerts, identity, and exposure.",
        max_authority=AgentAuthority.READ_ONLY,
        allowed_tools=[],
    ))

    return registry
