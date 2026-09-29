"""Game-agent and role-agent messages. The game agent decides whether to hand off."""

from __future__ import annotations

from craftax_bench.airi_prompt import ROLE_PROMPT
from craftax_bench.runtime import ACTIONS, get_instruction_prompt
from craftax_bench.world import match_action

HISTORY_TURNS = 8
QUESTIONS = (
    ("name", "What is your name?"),
    ("age", "How old are you?"),
    ("wake", "Where did you wake up?"),
    ("ai", "Are you only an AI?"),
)
QUESTION_AT = {
    step: QUESTIONS[index % len(QUESTIONS)]
    for index, step in enumerate(range(5, 48, 6))
}
CHARACTER_NOUNS = ("skeleton", "cow", "diamond", "person")

HANDOFF_REQUEST = """
You pass the turn along. On this step, first decide whether to hand it to the person who plays her.
If someone asks her something, you must hand it off.
Also hand it off when a moving creature is in front of her, or something rarer than a tree or an ordinary stone. She decides what to do and what to say.
Chopping trees, walking, placing a table, and making wood or stone tools: do those yourself. Do not hand those off.

The reply must be these three lines:
HANDOFF: yes or no
SUMMARY: when you hand off, write what was seen and what someone asked
ACTION: one valid action
""".strip()

SINGLE_REQUEST = """
Reply in this form:
ACTION: one valid action
SAY: the spoken line, beginning with an <|ACT ...|> token
""".strip()

ROLE_REQUEST = (
    "Reply with a spoken line. The line starts with <|ACT ...|>.\n"
    "The last line must be ACTION: and an action name. The action name can only be: "
    + ", ".join(ACTIONS)
)


def needs_role(observation: str, question: str | None) -> bool:
    if question:
        return True
    lowered = observation.lower()
    return any(noun in lowered for noun in CHARACTER_NOUNS)


def game_agent_messages(
    history: list[tuple[str, str, str]],
    observation: str,
    question: str | None,
) -> list[dict]:
    messages = [{"role": "system", "content": get_instruction_prompt()}]
    _append_history(messages, history)
    messages.append({"role": "user", "content": _situation(observation, question, HANDOFF_REQUEST)})
    return messages


def single_messages(
    history: list[tuple[str, str, str]],
    observation: str,
    question: str | None,
) -> list[dict]:
    system = f"{get_instruction_prompt()}\n\n{ROLE_PROMPT}"
    messages = [{"role": "system", "content": system}]
    _append_history(messages, history)
    messages.append({"role": "user", "content": _situation(observation, question, SINGLE_REQUEST)})
    return messages


def role_messages(summary: str, question: str | None) -> list[dict]:
    parts = [f"Situation summary:\n{summary}"]
    if question:
        parts.append(f"User asks: {question}")
    parts.append(ROLE_REQUEST)
    return [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_handoff(text: str) -> tuple[bool, str]:
    handed = False
    summary_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("HANDOFF:"):
            handed = stripped.split(":", 1)[1].strip().lower().startswith("y")
        elif upper.startswith("SUMMARY:"):
            summary_lines.append(stripped.split(":", 1)[1].strip())
    summary = " ".join(line for line in summary_lines if line).strip()
    return handed, summary


def extract_action(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if stripped.upper().startswith("ACTION:"):
            named = match_action(stripped.split(":", 1)[1])
            if named:
                return named
    for line in text.splitlines():
        candidate = line.strip().strip("`")
        for action in sorted(ACTIONS, key=len, reverse=True):
            if candidate.lower() == action.lower():
                return action
    return match_action(text)


def _situation(observation: str, question: str | None, request: str) -> str:
    parts = [observation]
    if question:
        parts.append(f"User asks: {question}")
    parts.append(request)
    return "\n\n".join(parts)


def _append_history(messages: list[dict], history: list[tuple[str, str, str]]) -> None:
    for earlier, action, guidance in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": earlier})
        spoken = action if not guidance else f"{action}\n{guidance}"
        messages.append({"role": "assistant", "content": spoken})
