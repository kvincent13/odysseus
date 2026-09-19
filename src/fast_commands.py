"""Deterministic Chief-of-Staff fast-command routing.

Fast commands:
    Remember ...       -> Memory
    Note ...           -> Note
    Add a todo ...     -> Checklist/todo
    Remind me ...      -> Reminder

Multiple explicit commands may be sent one-per-line.

If any nonblank line in a multi-line message is NOT an explicit fast command,
the entire message falls through to the normal LLM/agent pipeline. This avoids
partially executing mixed conversational requests.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FastCommandResult:
    response: str
    command_type: str


def _command_kind(text: str) -> Optional[str]:
    """Cheap classification only. Does not execute anything."""
    text = (text or "").strip()

    if not text:
        return None

    if re.match(r"^remember\b", text, re.IGNORECASE):
        return "memory"

    if re.match(r"^note\b", text, re.IGNORECASE):
        return "note"

    if re.match(
        r"^(?:add\s+(?:a\s+)?todo(?:\s+to)?|todo)\b",
        text,
        re.IGNORECASE,
    ):
        return "todo"

    if re.match(r"^remind\s+me\b", text, re.IGNORECASE):
        return "reminder"

    return None


async def _route_one(
    *,
    chat_handler,
    session,
    message: str,
    owner: Optional[str],
    tool_policy,
    incognito: bool,
    no_memory: bool,
) -> Optional[FastCommandResult]:

    kind = _command_kind(message)

    if kind is None:
        return None

    if kind in {"note", "todo", "reminder"}:
        if incognito or tool_policy.blocks("manage_notes"):
            return None

        response = await chat_handler.handle_note_command(
            session,
            message,
            owner=owner,
        )

        if not response:
            return None

        return FastCommandResult(
            response=response,
            command_type=kind,
        )

    if kind == "memory":
        if (
            incognito
            or no_memory
            or tool_policy.blocks("manage_memory")
        ):
            return None

        response = await chat_handler.handle_memory_command(
            session,
            message,
            owner=owner,
        )

        if not response:
            return None

        return FastCommandResult(
            response=response,
            command_type="memory",
        )

    return None


async def route_fast_command(
    *,
    chat_handler,
    session,
    message: str,
    owner: Optional[str],
    tool_policy,
    incognito: bool = False,
    no_memory: bool = False,
    approval_continuation: bool = False,
) -> Optional[FastCommandResult]:

    if not isinstance(message, str) or not message.strip():
        return None

    if approval_continuation:
        return None

    lines = [
        line.strip()
        for line in message.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    # --------------------------------------------------------------
    # MULTI-LINE SAFETY GATE
    #
    # Do not partially execute mixed conversation.
    #
    # Example:
    #   Remember Tanner owns licensing.
    #   What do you think about our Microsoft strategy?
    #
    # The second line is not deterministic, therefore execute NOTHING
    # here and let the full message go to Qwen.
    # --------------------------------------------------------------
    if len(lines) > 1:
        kinds = [_command_kind(line) for line in lines]

        if any(kind is None for kind in kinds):
            return None

        # Preflight permissions BEFORE executing the first command.
        # This prevents half-completed batches.
        if any(kind == "memory" for kind in kinds):
            if (
                incognito
                or no_memory
                or tool_policy.blocks("manage_memory")
            ):
                return None

        if any(kind in {"note", "todo", "reminder"} for kind in kinds):
            if incognito or tool_policy.blocks("manage_notes"):
                return None

        results = []

        for line in lines:
            result = await _route_one(
                chat_handler=chat_handler,
                session=session,
                message=line,
                owner=owner,
                tool_policy=tool_policy,
                incognito=incognito,
                no_memory=no_memory,
            )

            # Classification succeeded during preflight, so this normally
            # should not happen. Do not falsely claim full success.
            if result is None:
                return FastCommandResult(
                    response="I couldn't complete the entire batch.",
                    command_type="batch-partial",
                )

            results.append(result)

        counts = {
            "memory": 0,
            "note": 0,
            "todo": 0,
            "reminder": 0,
        }

        for result in results:
            if result.command_type in counts:
                counts[result.command_type] += 1

        parts = []

        if counts["memory"]:
            n = counts["memory"]
            parts.append(
                f"remembered {n}" if n > 1 else "remembered it"
            )

        if counts["note"]:
            n = counts["note"]
            parts.append(
                f"saved {n} notes" if n > 1 else "saved the note"
            )

        if counts["todo"]:
            n = counts["todo"]
            parts.append(
                f"added {n} todos" if n > 1 else "added the todo"
            )

        if counts["reminder"]:
            n = counts["reminder"]
            parts.append(
                f"set {n} reminders" if n > 1 else "set the reminder"
            )

        if not parts:
            summary = "Done."
        elif len(parts) == 1:
            summary = f"Done — {parts[0]}."
        elif len(parts) == 2:
            summary = f"Done — {parts[0]} and {parts[1]}."
        else:
            summary = (
                "Done — "
                + ", ".join(parts[:-1])
                + ", and "
                + parts[-1]
                + "."
            )

        return FastCommandResult(
            response=summary,
            command_type="batch",
        )

    # Single command.
    return await _route_one(
        chat_handler=chat_handler,
        session=session,
        message=lines[0],
        owner=owner,
        tool_policy=tool_policy,
        incognito=incognito,
        no_memory=no_memory,
    )
