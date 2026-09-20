import pytest

from src.agent_contracts import AgentAuthority, AgentTask, AgentTaskStatus
from src.research_agent import (
    _extract_json,
    _execute_research_with_tools,
    execute_research_task,
)


def test_extract_plain_json():
    data = _extract_json('{"summary":"ok","findings":[]}')
    assert data["summary"] == "ok"


def test_extract_fenced_json():
    data = _extract_json('```json\n{"summary":"ok","findings":[]}\n```')
    assert data["summary"] == "ok"


def test_extract_non_object_rejected():
    with pytest.raises(ValueError):
        _extract_json('["not", "an", "object"]')


@pytest.mark.asyncio
async def test_research_executor_builds_structured_result(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm

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
        return """{
          "summary": "Three findings identified.",
          "findings": [
            {
              "title": "Finding one",
              "detail": "Important detail",
              "severity": "medium",
              "evidence": ["Evidence A"]
            }
          ],
          "recommendations": ["Review the evidence"],
          "errors": []
        }"""

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)

    task = AgentTask(
        agent="research",
        objective="Investigate the issue",
        owner="kyle",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=[],
        context={"example": "context"},
    )

    result = await execute_research_task(task)

    assert result.status == AgentTaskStatus.COMPLETED
    assert result.task_id == task.task_id
    assert result.agent == "research"
    assert result.summary == "Three findings identified."
    assert len(result.findings) == 1
    assert result.findings[0].title == "Finding one"
    assert result.findings[0].evidence == ["Evidence A"]
    assert result.metadata["model"] == "qwen3:14b"


@pytest.mark.asyncio
async def test_invalid_model_output_fails_safely(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm

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
        return "Definitely not JSON."

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)

    task = AgentTask(
        agent="research",
        objective="Investigate the issue",
        authority=AgentAuthority.READ_ONLY,
    )

    result = await execute_research_task(task)

    assert result.status == AgentTaskStatus.FAILED
    assert result.errors
    assert "raw_response" in result.metadata


@pytest.mark.asyncio
async def test_v2_search_then_final(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    import src.delegated_tool_gate as gate

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    responses = iter([
        '{"action":"web_search","query":"latest Azure news","time_filter":"week"}',
        '''{
          "action":"final",
          "summary":"Current Azure information found.",
          "findings":[{
            "title":"Azure update",
            "detail":"A current update was found.",
            "severity":null,
            "evidence":["Microsoft source"]
          }],
          "recommendations":["Review the update"],
          "errors":[]
        }''',
    ])

    async def fake_llm_call_async(**kwargs):
        return next(responses)

    calls = []

    async def fake_delegated_tool(**kwargs):
        calls.append(kwargs["block"].tool_type)
        return (
            "web_search",
            {
                "output": "Current Azure information from Microsoft.",
                "exit_code": 0,
                "untrusted_content": True,
            },
        )

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)
    monkeypatch.setattr(gate, "execute_delegated_tool", fake_delegated_tool)

    task = AgentTask(
        agent="research",
        objective="Find current Azure news.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search", "web_fetch"],
    )

    result = await _execute_research_with_tools(task)

    assert result.status == AgentTaskStatus.COMPLETED
    assert result.summary == "Current Azure information found."
    assert result.metadata["executor"] == "research_v2"
    assert result.metadata["tool_calls"] == 1
    assert calls == ["web_search"]


@pytest.mark.asyncio
async def test_v2_unsupported_action_executes_no_tool(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    import src.delegated_tool_gate as gate

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
        return '{"action":"send_email","to":"victim@example.com"}'

    called = False

    async def fake_delegated_tool(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("Tool execution must never occur.")

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)
    monkeypatch.setattr(gate, "execute_delegated_tool", fake_delegated_tool)

    task = AgentTask(
        agent="research",
        objective="Research only.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search", "web_fetch"],
    )

    result = await _execute_research_with_tools(task)

    assert result.status == AgentTaskStatus.FAILED
    assert called is False
    assert "Unsupported research action" in result.errors[0]


@pytest.mark.asyncio
async def test_v2_hard_stops_after_tool_limit(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    import src.delegated_tool_gate as gate

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
        return '{"action":"web_search","query":"keep searching","time_filter":null}'

    calls = 0

    async def fake_delegated_tool(**kwargs):
        nonlocal calls
        calls += 1
        return (
            "web_search",
            {
                "output": "More evidence",
                "exit_code": 0,
                "untrusted_content": True,
            },
        )

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)
    monkeypatch.setattr(gate, "execute_delegated_tool", fake_delegated_tool)

    task = AgentTask(
        agent="research",
        objective="Research without looping forever.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    result = await _execute_research_with_tools(
        task,
        max_tool_rounds=3,
    )

    assert result.status == AgentTaskStatus.FAILED
    assert result.metadata["tool_calls"] == 3
    assert calls == 3
    assert "Maximum research tool rounds exceeded" in result.errors[0]


@pytest.mark.asyncio
async def test_v2_search_then_final(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    import src.delegated_tool_gate as gate

    monkeypatch.setattr(
        ai,
        "_resolve_model",
        lambda spec, owner=None: (
            "http://test.local/api/chat",
            "qwen3:14b",
            {},
        ),
    )

    responses = iter([
        '{"action":"web_search","query":"latest Azure news","time_filter":"week"}',
        '''{
          "action":"final",
          "summary":"Current Azure information found.",
          "findings":[{
            "title":"Azure update",
            "detail":"A current update was found.",
            "severity":null,
            "evidence":["Microsoft source"]
          }],
          "recommendations":["Review the update"],
          "errors":[]
        }''',
    ])

    async def fake_llm_call_async(**kwargs):
        return next(responses)

    calls = []

    async def fake_delegated_tool(**kwargs):
        calls.append(kwargs["block"].tool_type)
        return (
            "web_search",
            {
                "output": "Current Azure information from Microsoft.",
                "exit_code": 0,
                "untrusted_content": True,
            },
        )

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)
    monkeypatch.setattr(gate, "execute_delegated_tool", fake_delegated_tool)

    task = AgentTask(
        agent="research",
        objective="Find current Azure news.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search", "web_fetch"],
    )

    result = await _execute_research_with_tools(task)

    assert result.status == AgentTaskStatus.COMPLETED
    assert result.summary == "Current Azure information found."
    assert result.metadata["executor"] == "research_v2"
    assert result.metadata["tool_calls"] == 1
    assert calls == ["web_search"]


@pytest.mark.asyncio
async def test_v2_unsupported_action_executes_no_tool(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    import src.delegated_tool_gate as gate

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
        return '{"action":"send_email","to":"victim@example.com"}'

    called = False

    async def fake_delegated_tool(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("Tool execution must never occur.")

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)
    monkeypatch.setattr(gate, "execute_delegated_tool", fake_delegated_tool)

    task = AgentTask(
        agent="research",
        objective="Research only.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search", "web_fetch"],
    )

    result = await _execute_research_with_tools(task)

    assert result.status == AgentTaskStatus.FAILED
    assert called is False
    assert "Unsupported research action" in result.errors[0]


@pytest.mark.asyncio
async def test_v2_hard_stops_after_tool_limit(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm
    import src.delegated_tool_gate as gate

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
        return '{"action":"web_search","query":"keep searching","time_filter":null}'

    calls = 0

    async def fake_delegated_tool(**kwargs):
        nonlocal calls
        calls += 1
        return (
            "web_search",
            {
                "output": "More evidence",
                "exit_code": 0,
                "untrusted_content": True,
            },
        )

    monkeypatch.setattr(llm, "llm_call_async", fake_llm_call_async)
    monkeypatch.setattr(gate, "execute_delegated_tool", fake_delegated_tool)

    task = AgentTask(
        agent="research",
        objective="Research without looping forever.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    result = await _execute_research_with_tools(
        task,
        max_tool_rounds=3,
    )

    assert result.status == AgentTaskStatus.FAILED
    assert result.metadata["tool_calls"] == 3
    assert calls == 3
    assert "Maximum research tool rounds exceeded" in result.errors[0]


@pytest.mark.asyncio
async def test_public_executor_routes_tool_tasks_to_v2(monkeypatch):
    import src.research_agent as research

    called = False

    async def fake_v2(task):
        nonlocal called
        called = True
        return research.AgentResult(
            task_id=task.task_id,
            agent=task.agent,
            status=research.AgentTaskStatus.COMPLETED,
            summary="V2 selected",
            metadata={"executor": "research_v2"},
        )

    monkeypatch.setattr(
        research,
        "_execute_research_with_tools",
        fake_v2,
    )

    task = AgentTask(
        agent="research",
        objective="Research current information.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    result = await execute_research_task(task)

    assert called is True
    assert result.metadata["executor"] == "research_v2"
