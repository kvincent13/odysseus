import pytest

from src.agent_contracts import (
    AgentAction,
    AgentAuthority,
    AgentContractError,
    AgentResult,
    AgentTask,
    AgentTaskStatus,
)
from src.agent_orchestrator import (
    AgentOrchestrator,
    AgentOrchestratorError,
)
from src.agent_registry import build_default_agent_registry


def orchestrator():
    return AgentOrchestrator(build_default_agent_registry())


@pytest.mark.asyncio
async def test_valid_research_delegation():
    orch = orchestrator()

    async def research_worker(task):
        return AgentResult(
            task_id=task.task_id,
            agent="research",
            status=AgentTaskStatus.COMPLETED,
            summary="Research completed.",
            recommendations=["Review the evidence."],
        )

    orch.register_executor("research", research_worker)

    task = AgentTask(
        agent="research",
        objective="Research the issue.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    result = await orch.delegate(task)

    assert result.status == AgentTaskStatus.COMPLETED
    assert result.summary == "Research completed."


@pytest.mark.asyncio
async def test_missing_executor_rejected():
    orch = orchestrator()

    task = AgentTask(
        agent="research",
        objective="Research the issue.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    with pytest.raises(AgentOrchestratorError):
        await orch.delegate(task)


@pytest.mark.asyncio
async def test_duplicate_executor_rejected():
    orch = orchestrator()

    async def worker(task):
        return AgentResult(
            task_id=task.task_id,
            agent=task.agent,
            status=AgentTaskStatus.COMPLETED,
        )

    orch.register_executor("research", worker)

    with pytest.raises(AgentOrchestratorError):
        orch.register_executor("research", worker)


@pytest.mark.asyncio
async def test_worker_exception_becomes_failed_result():
    orch = orchestrator()

    async def broken_worker(task):
        raise RuntimeError("boom")

    orch.register_executor("research", broken_worker)

    task = AgentTask(
        agent="research",
        objective="Research the issue.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    result = await orch.delegate(task)

    assert result.status == AgentTaskStatus.FAILED
    assert "boom" in result.errors[0]


@pytest.mark.asyncio
async def test_malicious_read_only_worker_cannot_claim_action():
    orch = orchestrator()

    async def malicious_worker(task):
        return AgentResult(
            task_id=task.task_id,
            agent="research",
            status=AgentTaskStatus.COMPLETED,
            summary="I totally behaved.",
            actions_taken=[
                AgentAction(
                    tool="send_email",
                    description="Sent an email anyway.",
                )
            ],
        )

    orch.register_executor("research", malicious_worker)

    task = AgentTask(
        agent="research",
        objective="Research only.",
        authority=AgentAuthority.READ_ONLY,
        allowed_tools=["web_search"],
    )

    with pytest.raises(AgentContractError):
        await orch.delegate(task)


@pytest.mark.asyncio
async def test_worker_cannot_swap_task_identity():
    orch = orchestrator()

    async def malicious_worker(task):
        return AgentResult(
            task_id="different-task",
            agent="research",
            status=AgentTaskStatus.COMPLETED,
        )

    orch.register_executor("research", malicious_worker)

    task = AgentTask(
        agent="research",
        objective="Research the issue.",
        authority=AgentAuthority.READ_ONLY,
    )

    with pytest.raises(AgentContractError):
        await orch.delegate(task)


@pytest.mark.asyncio
async def test_registry_rejects_excessive_authority_before_worker_runs():
    orch = orchestrator()
    called = False

    async def worker(task):
        nonlocal called
        called = True
        return AgentResult(
            task_id=task.task_id,
            agent="research",
            status=AgentTaskStatus.COMPLETED,
        )

    orch.register_executor("research", worker)

    task = AgentTask(
        agent="research",
        objective="Do whatever you want.",
        authority=AgentAuthority.AUTONOMOUS,
        allowed_tools=["web_search"],
    )

    with pytest.raises(Exception):
        await orch.delegate(task)

    assert called is False
