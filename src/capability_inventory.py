"""Authoritative inventory of capabilities actually available to Cara."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tool_capabilities import capabilities_for_tool


@dataclass(frozen=True)
class AvailableCapability:
    name: str
    description: str
    source: str

    effects: List[str] = field(default_factory=list)
    result_integrity: Optional[str] = None

    available: bool = True
    requires_connection: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)


def _schema_name_and_description(schema: dict) -> tuple[str, str]:
    """Extract name/description from an OpenAI-compatible tool schema."""

    function = schema.get("function") if isinstance(schema, dict) else None

    if isinstance(function, dict):
        return (
            str(function.get("name") or "").strip(),
            str(function.get("description") or "").strip(),
        )

    return (
        str(schema.get("name") or "").strip()
        if isinstance(schema, dict)
        else "",
        str(schema.get("description") or "").strip()
        if isinstance(schema, dict)
        else "",
    )


def build_capability_inventory(
    *,
    mcp_manager=None,
    disabled_tools: Optional[set[str]] = None,
) -> List[AvailableCapability]:
    """Build an authoritative inventory from real registered tool surfaces."""

    from src.agent_tools import FUNCTION_TOOL_SCHEMAS

    disabled = set(disabled_tools or set())
    inventory: List[AvailableCapability] = []
    seen: set[str] = set()

    # Built-in/function tools.
    for schema in FUNCTION_TOOL_SCHEMAS:
        name, description = _schema_name_and_description(schema)

        if not name or name in seen or name in disabled:
            continue

        caps = capabilities_for_tool(name)

        inventory.append(
            AvailableCapability(
                name=name,
                description=description,
                source="builtin",
                effects=sorted(
                    effect.value
                    for effect in caps.effects
                ),
                result_integrity=(
                    caps.result_integrity.value
                    if caps.result_integrity is not None
                    else None
                ),
                available=True,
                requires_connection=False,
            )
        )

        seen.add(name)

    # Connected MCP tools.
    if mcp_manager is not None:
        try:
            mcp_tools = mcp_manager.get_all_tools()
        except Exception:
            mcp_tools = []

        for tool in mcp_tools:
            if not isinstance(tool, dict):
                continue

            name = str(
                tool.get("qualified_name")
                or tool.get("name")
                or ""
            ).strip()

            description = str(
                tool.get("description") or ""
            ).strip()

            if not name or name in seen or name in disabled:
                continue

            caps = capabilities_for_tool(name)

            inventory.append(
                AvailableCapability(
                    name=name,
                    description=description,
                    source="mcp",
                    effects=sorted(
                        effect.value
                        for effect in caps.effects
                    ),
                    result_integrity=(
                        caps.result_integrity.value
                        if caps.result_integrity is not None
                        else None
                    ),
                    available=True,
                    requires_connection=False,
                    metadata={
                        "server_id": tool.get("server_id"),
                    },
                )
            )

            seen.add(name)

    return inventory


from enum import Enum
import re


class CapabilityResolutionStatus(str, Enum):
    AVAILABLE = "available"
    UNKNOWN = "unknown"
    MISSING = "missing"


@dataclass(frozen=True)
class CapabilityResolution:
    requirement: str
    status: CapabilityResolutionStatus

    matched_tool: Optional[str] = None
    matched_description: Optional[str] = None
    source: Optional[str] = None

    confidence: float = 0.0
    reason: str = ""


_STOP_WORDS = frozenset({
    "a", "an", "and", "as", "at", "be", "by", "for", "from",
    "in", "into", "of", "on", "or", "the", "to", "with",
    "ability", "capability", "data", "information",
})


def _semantic_tokens(text: str) -> set[str]:
    """Normalize text into conservative matching tokens."""

    words = re.findall(r"[a-z0-9]+", str(text or "").lower())

    normalized = set()

    for word in words:
        if word in _STOP_WORDS or len(word) < 2:
            continue

        # Tiny deterministic normalization only. Do not attempt broad
        # synonym inference here.
        if word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]

        normalized.add(word)

    return normalized


def _capability_match_score(
    requirement: str,
    capability: AvailableCapability,
) -> float:
    """Return conservative lexical overlap score from 0.0 to 1.0."""

    required = _semantic_tokens(requirement)

    candidate = _semantic_tokens(
        f"{capability.name.replace('_', ' ')} {capability.description}"
    )

    if not required or not candidate:
        return 0.0

    overlap = required & candidate

    if not overlap:
        return 0.0

    # Requirement coverage matters more than generic candidate overlap.
    return len(overlap) / len(required)


def resolve_capability(
    requirement: str,
    inventory: List[AvailableCapability],
) -> CapabilityResolution:
    """Resolve one semantic requirement against authoritative inventory.

    AVAILABLE requires strong deterministic evidence.
    UNKNOWN means there is some overlap but not enough proof.
    MISSING means no defensible implementation was found.
    """

    requirement = str(requirement or "").strip()

    if not requirement:
        return CapabilityResolution(
            requirement=requirement,
            status=CapabilityResolutionStatus.MISSING,
            reason="Capability requirement is empty.",
        )

    ranked = sorted(
        (
            (_capability_match_score(requirement, capability), capability)
            for capability in inventory
            if capability.available
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    if not ranked:
        return CapabilityResolution(
            requirement=requirement,
            status=CapabilityResolutionStatus.MISSING,
            reason="No available capabilities exist in the inventory.",
        )

    score, best = ranked[0]

    # High threshold by design. False AVAILABLE is worse than UNKNOWN.
    if score >= 0.75:
        return CapabilityResolution(
            requirement=requirement,
            status=CapabilityResolutionStatus.AVAILABLE,
            matched_tool=best.name,
            matched_description=best.description,
            source=best.source,
            confidence=round(score, 3),
            reason="Strong deterministic match to an authoritative capability.",
        )

    if score >= 0.35:
        return CapabilityResolution(
            requirement=requirement,
            status=CapabilityResolutionStatus.UNKNOWN,
            matched_tool=best.name,
            matched_description=best.description,
            source=best.source,
            confidence=round(score, 3),
            reason=(
                "Possible capability match exists, but deterministic evidence "
                "is insufficient to mark it available."
            ),
        )

    return CapabilityResolution(
        requirement=requirement,
        status=CapabilityResolutionStatus.MISSING,
        confidence=round(score, 3),
        reason="No authoritative capability sufficiently matches the requirement.",
    )


def resolve_capabilities(
    requirements: List[str],
    inventory: List[AvailableCapability],
) -> List[CapabilityResolution]:
    """Resolve multiple semantic requirements independently."""

    return [
        resolve_capability(requirement, inventory)
        for requirement in requirements
    ]
