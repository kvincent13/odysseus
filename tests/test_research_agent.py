import pytest

from src.agent_contracts import AgentAuthority, AgentTask, AgentTaskStatus
from src.research_agent import _extract_json, execute_research_task


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
