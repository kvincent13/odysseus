import pytest

from src.agent_contracts import AgentAuthority, AgentTask
from src.agent_registry import build_default_agent_registry
from src.delegated_tool_gate import (
    DelegatedToolError,
    _validate_authority_for_tool,
    execute_delegated_tool,
)


class Block:
    def __init__(self, tool_type, content=""):
        self.tool_type = tool_type
        self.content = content


def research_task(tools=None):
    return AgentTask(
        agent="research",
        objective="Research the issue",
        owner="kyle",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=tools or ["web_search", "web_fetch"],
    )


def test_read_only_allows_web_search_capability():
    task = research_task()
    _validate_authority_for_tool(
        task,
        "web_search",
        "Azure Container Apps",
    )


def test_read_only_allows_web_fetch_capability():
    task = research_task()
    _validate_authority_for_tool(
        task,
        "web_fetch",
        "https://example.com",
    )


def test_read_only_rejects_write_file():
    task = research_task(["web_search"])

    with pytest.raises(DelegatedToolError):
        _validate_authority_for_tool(
            task,
            "write_file",
            "/tmp/test.txt\nhello",
        )


def test_unknown_tool_rejected():
    task = research_task()

    with pytest.raises(DelegatedToolError):
        _validate_authority_for_tool(
            task,
            "totally_fake_tool",
            "whatever",
        )


@pytest.mark.asyncio
async def test_tool_not_in_task_scope_blocked_before_dispatch(monkeypatch):
    import src.delegated_tool_gate as gate

    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        return ("fake", {"ok": True})

    monkeypatch.setattr(gate, "execute_tool_block", fake_execute)

    task = research_task(["web_search"])
    registry = build_default_agent_registry()

    with pytest.raises(DelegatedToolError):
        await execute_delegated_tool(
            task=task,
            registry=registry,
            block=Block("web_fetch", "https://example.com"),
        )

    assert called is False


@pytest.mark.asyncio
async def test_tool_outside_agent_registry_blocked_before_dispatch(monkeypatch):
    import src.delegated_tool_gate as gate

    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        return ("fake", {"ok": True})

    monkeypatch.setattr(gate, "execute_tool_block", fake_execute)

    task = research_task(["send_email"])
    registry = build_default_agent_registry()

    with pytest.raises(Exception):
        await execute_delegated_tool(
            task=task,
            registry=registry,
            block=Block("send_email", "hello"),
        )

    assert called is False


@pytest.mark.asyncio
async def test_allowed_web_search_reaches_dispatch(monkeypatch):
    import src.delegated_tool_gate as gate

    captured = {}

    async def fake_execute(block, **kwargs):
        captured["tool"] = block.tool_type
        captured["content"] = block.content
        captured["security_context"] = kwargs["security_context"]
        captured["owner"] = kwargs["owner"]
        return (
            "web_search",
            {
                "results": [{"title": "Example"}],
                "untrusted_content": True,
            },
        )

    monkeypatch.setattr(gate, "execute_tool_block", fake_execute)

    task = research_task(["web_search"])
    registry = build_default_agent_registry()

    description, result = await execute_delegated_tool(
        task=task,
        registry=registry,
        block=Block("web_search", "Azure AI"),
    )

    assert description == "web_search"
    assert result["untrusted_content"] is True
    assert captured["tool"] == "web_search"
    assert captured["owner"] == "kyle"
    assert captured["security_context"].delegated_credential is True


@pytest.mark.asyncio
async def test_same_security_context_reused_across_calls(monkeypatch):
    import src.delegated_tool_gate as gate
    from src.tool_capabilities import ToolRunSecurityContext

    seen = []

    async def fake_execute(block, **kwargs):
        seen.append(kwargs["security_context"])
        return (
            block.tool_type,
            {"results": [], "untrusted_content": True},
        )

    monkeypatch.setattr(gate, "execute_tool_block", fake_execute)

    task = research_task(["web_search", "web_fetch"])
    registry = build_default_agent_registry()
    security_context = ToolRunSecurityContext(delegated_credential=True)

    await execute_delegated_tool(
        task=task,
        registry=registry,
        block=Block("web_search", "Azure AI"),
        security_context=security_context,
    )

    await execute_delegated_tool(
        task=task,
        registry=registry,
        block=Block("web_fetch", "https://example.com"),
        security_context=security_context,
    )

    assert len(seen) == 2
    assert seen[0] is security_context
    assert seen[1] is security_context
    assert seen[0] is seen[1]


@pytest.mark.asyncio
async def test_interactive_security_context_rejected(monkeypatch):
    import src.delegated_tool_gate as gate
    from src.tool_capabilities import ToolRunSecurityContext

    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        return ("fake", {"ok": True})

    monkeypatch.setattr(gate, "execute_tool_block", fake_execute)

    task = research_task(["web_search"])
    registry = build_default_agent_registry()

    interactive_context = ToolRunSecurityContext(
        delegated_credential=False,
    )

    with pytest.raises(DelegatedToolError):
        await execute_delegated_tool(
            task=task,
            registry=registry,
            block=Block("web_search", "Azure AI"),
            security_context=interactive_context,
        )

    assert called is False


@pytest.mark.asyncio
async def test_same_security_context_reused_across_calls(monkeypatch):
    import src.delegated_tool_gate as gate
    from src.tool_capabilities import ToolRunSecurityContext

    seen = []

    async def fake_execute(block, **kwargs):
        seen.append(kwargs["security_context"])
        return (
            block.tool_type,
            {"results": [], "untrusted_content": True},
        )

    monkeypatch.setattr(gate, "execute_tool_block", fake_execute)

    task = research_task(["web_search", "web_fetch"])
    registry = build_default_agent_registry()
    security_context = ToolRunSecurityContext(delegated_credential=True)

    await execute_delegated_tool(
        task=task,
        registry=registry,
        block=Block("web_search", "Azure AI"),
        security_context=security_context,
    )

    await execute_delegated_tool(
        task=task,
        registry=registry,
        block=Block("web_fetch", "https://example.com"),
        security_context=security_context,
    )

    assert len(seen) == 2
    assert seen[0] is security_context
    assert seen[1] is security_context
    assert seen[0] is seen[1]


@pytest.mark.asyncio
async def test_interactive_security_context_rejected(monkeypatch):
    import src.delegated_tool_gate as gate
    from src.tool_capabilities import ToolRunSecurityContext

    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        return ("fake", {"ok": True})

    monkeypatch.setattr(gate, "execute_tool_block", fake_execute)

    task = research_task(["web_search"])
    registry = build_default_agent_registry()

    interactive_context = ToolRunSecurityContext(
        delegated_credential=False,
    )

    with pytest.raises(DelegatedToolError):
        await execute_delegated_tool(
            task=task,
            registry=registry,
            block=Block("web_search", "Azure AI"),
            security_context=interactive_context,
        )

    assert called is False
