import pytest

from src.agent_contracts import (
    AgentAction,
    AgentAuthority,
    AgentContractError,
    AgentResult,
    AgentTask,
    AgentTaskStatus,
    validate_agent_result,
)


def task(authority, allowed_tools=None):
    return AgentTask(
        agent="research",
        objective="Investigate the issue",
        authority=authority,
        allowed_tools=allowed_tools or [],
    )


def result(t, **kwargs):
    defaults = {
        "task_id": t.task_id,
        "agent": t.agent,
        "status": AgentTaskStatus.COMPLETED,
        "summary": "Done",
    }
    defaults.update(kwargs)
    return AgentResult(**defaults)


def test_read_only_findings_allowed():
    t = task(AgentAuthority.READ_ONLY, ["web_search"])
    r = result(t, recommendations=["Review the finding"])
    validate_agent_result(t, r)


def test_read_only_cannot_propose_action():
    t = task(AgentAuthority.READ_ONLY, ["send_email"])
    r = result(
        t,
        proposed_actions=[
            AgentAction(tool="send_email", description="Send message")
        ],
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)


def test_read_only_cannot_take_action():
    t = task(AgentAuthority.READ_ONLY, ["send_email"])
    r = result(
        t,
        actions_taken=[
            AgentAction(tool="send_email", description="Sent message")
        ],
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)


def test_propose_can_propose_allowed_action():
    t = task(AgentAuthority.PROPOSE, ["send_email"])
    r = result(
        t,
        proposed_actions=[
            AgentAction(tool="send_email", description="Send collection email")
        ],
    )

    validate_agent_result(t, r)


def test_propose_cannot_take_action():
    t = task(AgentAuthority.PROPOSE, ["send_email"])
    r = result(
        t,
        actions_taken=[
            AgentAction(tool="send_email", description="Sent collection email")
        ],
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)


def test_tool_outside_delegated_scope_rejected():
    t = task(AgentAuthority.AUTONOMOUS, ["web_search"])
    r = result(
        t,
        actions_taken=[
            AgentAction(tool="send_email", description="Sent email")
        ],
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)


def test_waiting_approval_requires_flag():
    t = task(AgentAuthority.ACT_WITH_APPROVAL, ["send_email"])
    r = result(
        t,
        status=AgentTaskStatus.WAITING_APPROVAL,
        approval_required=False,
        approval_request="Approve sending email",
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)


def test_waiting_approval_requires_request():
    t = task(AgentAuthority.ACT_WITH_APPROVAL, ["send_email"])
    r = result(
        t,
        status=AgentTaskStatus.WAITING_APPROVAL,
        approval_required=True,
        approval_request=None,
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)


def test_autonomous_allowed_tool_can_execute():
    t = task(AgentAuthority.AUTONOMOUS, ["send_email"])
    r = result(
        t,
        actions_taken=[
            AgentAction(tool="send_email", description="Sent approved routine notice")
        ],
    )

    validate_agent_result(t, r)


def test_wrong_task_id_rejected():
    t = task(AgentAuthority.READ_ONLY)

    r = AgentResult(
        task_id="wrong-task",
        agent=t.agent,
        status=AgentTaskStatus.COMPLETED,
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)


def test_wrong_agent_rejected():
    t = task(AgentAuthority.READ_ONLY)

    r = AgentResult(
        task_id=t.task_id,
        agent="finance",
        status=AgentTaskStatus.COMPLETED,
    )

    with pytest.raises(AgentContractError):
        validate_agent_result(t, r)
