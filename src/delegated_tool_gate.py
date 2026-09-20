"""Execution-time tool boundary for Cara's delegated specialist agents."""

from typing import Any, Optional

from src.agent_contracts import AgentAuthority, AgentTask
from src.agent_registry import AgentRegistry
from src.tool_capabilities import ToolEffect, ToolRunSecurityContext, capabilities_for_action
from src.tool_execution import execute_tool_block


class DelegatedToolError(PermissionError):
    pass


_READ_ONLY_EFFECTS = frozenset({
    ToolEffect.READ_PUBLIC,
    ToolEffect.BROKERED_NETWORK_READ,
    # Explicitly delegated read-only web tools may make outbound requests.
    # Tool scope is still constrained by AgentTask + AgentRegistry.
    ToolEffect.NETWORK_EGRESS,
})


def _validate_authority_for_tool(
    task: AgentTask,
    tool_name: str,
    content: Any,
) -> None:
    capabilities = capabilities_for_action(tool_name, content)

    if not capabilities.known:
        raise DelegatedToolError(
            f"Tool '{tool_name}' has unknown capabilities and cannot be delegated."
        )

    if task.authority == AgentAuthority.READ_ONLY:
        disallowed = capabilities.effects - _READ_ONLY_EFFECTS
        if disallowed:
            effects = ", ".join(sorted(effect.value for effect in disallowed))
            raise DelegatedToolError(
                f"READ_ONLY task cannot use tool '{tool_name}' "
                f"because it can cause: {effects}."
            )

    elif task.authority == AgentAuthority.PROPOSE:
        # Proposal agents may gather information but may not execute writes
        # or external side effects. Additional proposal-only tooling can be
        # introduced later without weakening this execution boundary.
        disallowed = capabilities.effects & {
            ToolEffect.WRITE_WORKSPACE,
            ToolEffect.WRITE_PRIVATE,
            ToolEffect.DESTRUCTIVE,
            ToolEffect.EXTERNAL_SIDE_EFFECT,
            ToolEffect.UI_SIDE_EFFECT,
        }
        if disallowed:
            effects = ", ".join(sorted(effect.value for effect in disallowed))
            raise DelegatedToolError(
                f"PROPOSE task cannot execute tool '{tool_name}' "
                f"because it can cause: {effects}."
            )


async def execute_delegated_tool(
    *,
    task: AgentTask,
    registry: AgentRegistry,
    block: Any,
    session_id: Optional[str] = None,
    workspace: Optional[str] = None,
    security_context: Optional[ToolRunSecurityContext] = None,
):
    """Execute one specialist tool call inside Cara's delegated authority."""

    tool_name = str(getattr(block, "tool_type", "") or "").strip()

    if not tool_name:
        raise DelegatedToolError("Delegated tool request has no tool_type.")

    # 1. Per-task capability scope.
    if tool_name not in set(task.allowed_tools):
        raise DelegatedToolError(
            f"Tool '{tool_name}' is not allowed for task '{task.task_id}'."
        )

    # 2. Specialist's permanent registry ceiling.
    registry.validate_delegation(
        task.agent,
        task.authority,
        task.allowed_tools,
    )
    definition = registry.get(task.agent)

    if tool_name not in set(definition.allowed_tools):
        raise DelegatedToolError(
            f"Tool '{tool_name}' is outside agent '{definition.name}' capability scope."
        )

    # 3. Translate authority into actual tool effects.
    _validate_authority_for_tool(
        task,
        tool_name,
        getattr(block, "content", None),
    )

    # 4. Reuse Odysseus's server-owned tool security state.
    # Delegated runs never inherit an interactive user's approval bypass.
    if security_context is None:
        security_context = ToolRunSecurityContext(
            delegated_credential=True,
        )
    elif not security_context.delegated_credential:
        raise DelegatedToolError(
            "Delegated specialist runs require delegated_credential=True."
        )

    # 5. Execute through the normal dispatcher/security path.
    # Reusing the same security_context across calls preserves taint/state
    # for the entire specialist run.
    return await execute_tool_block(
        block,
        session_id=session_id,
        owner=task.owner,
        workspace=workspace,
        security_context=security_context,
    )
