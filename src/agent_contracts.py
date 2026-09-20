"""Contracts shared by Cara and delegated specialist agents."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class AgentAuthority(str, Enum):
    READ_ONLY = "read_only"
    PROPOSE = "propose"
    ACT_WITH_APPROVAL = "act_with_approval"
    AUTONOMOUS = "autonomous"


class AgentTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    agent: str
    objective: str

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_task_id: Optional[str] = None

    requested_by: str = "cara"
    owner: Optional[str] = None

    authority: AgentAuthority = AgentAuthority.READ_ONLY
    allowed_tools: List[str] = field(default_factory=list)

    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentFinding:
    title: str
    detail: str
    severity: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class AgentAction:
    tool: str
    description: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False


@dataclass
class AgentResult:
    task_id: str
    agent: str
    status: AgentTaskStatus

    summary: str = ""

    findings: List[AgentFinding] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    proposed_actions: List[AgentAction] = field(default_factory=list)
    actions_taken: List[AgentAction] = field(default_factory=list)

    approval_required: bool = False
    approval_request: Optional[str] = None

    errors: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContractError(ValueError):
    """Raised when a delegated agent result violates its task contract."""


def validate_agent_result(task: AgentTask, result: AgentResult) -> None:
    """Validate that a specialist result stayed within Cara's delegation."""

    if result.task_id != task.task_id:
        raise AgentContractError("Result task_id does not match delegated task.")

    if result.agent != task.agent:
        raise AgentContractError("Result agent does not match delegated agent.")

    # Read-only and proposal tasks may never report external execution.
    if task.authority in {AgentAuthority.READ_ONLY, AgentAuthority.PROPOSE}:
        if result.actions_taken:
            raise AgentContractError(
                f"{task.authority.value} tasks cannot report actions_taken."
            )

    # Any claimed or proposed tool must be inside the delegated tool scope.
    if task.allowed_tools:
        allowed = set(task.allowed_tools)
        for action in [*result.proposed_actions, *result.actions_taken]:
            if action.tool not in allowed:
                raise AgentContractError(
                    f"Tool '{action.tool}' is outside delegated allowed_tools."
                )

    # Waiting for approval must be represented consistently.
    if result.status == AgentTaskStatus.WAITING_APPROVAL:
        if not result.approval_required:
            raise AgentContractError(
                "waiting_approval status requires approval_required=True."
            )
        if not result.approval_request:
            raise AgentContractError(
                "waiting_approval status requires an approval_request."
            )

    # READ_ONLY agents should report findings, not proposed mutations.
    if task.authority == AgentAuthority.READ_ONLY and result.proposed_actions:
        raise AgentContractError(
            "read_only tasks cannot propose executable actions."
        )
