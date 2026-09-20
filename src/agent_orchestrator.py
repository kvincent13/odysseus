"""Cara specialist-agent orchestration.

The orchestrator validates delegation, dispatches to a registered specialist
executor, and validates the returned AgentResult before Cara consumes it.
"""

from typing import Awaitable, Callable, Dict

from src.agent_contracts import (
    AgentResult,
    AgentTask,
    AgentTaskStatus,
    validate_agent_result,
)
from src.agent_registry import AgentRegistry


AgentExecutor = Callable[[AgentTask], Awaitable[AgentResult]]


class AgentOrchestratorError(RuntimeError):
    pass


class AgentOrchestrator:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._executors: Dict[str, AgentExecutor] = {}

    def register_executor(
        self,
        agent_name: str,
        executor: AgentExecutor,
    ) -> None:
        definition = self.registry.get(agent_name)
        name = definition.name

        if name in self._executors:
            raise AgentOrchestratorError(
                f"Executor for agent '{name}' is already registered."
            )

        self._executors[name] = executor

    async def delegate(self, task: AgentTask) -> AgentResult:
        # Enforce specialist capability ceiling before execution.
        self.registry.validate_delegation(
            task.agent,
            task.authority,
            task.allowed_tools,
        )

        definition = self.registry.get(task.agent)

        executor = self._executors.get(definition.name)
        if executor is None:
            raise AgentOrchestratorError(
                f"No executor registered for agent '{definition.name}'."
            )

        try:
            result = await executor(task)
        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent=definition.name,
                status=AgentTaskStatus.FAILED,
                summary="Specialist agent execution failed.",
                errors=[str(exc)],
            )

        # Never trust a specialist's self-reported result blindly.
        validate_agent_result(task, result)

        return result
