"""Message construction and one step for each arm. Gold labels never enter a prompt."""

from __future__ import annotations

from craftax_bench.airi_prompt import ROLE_PROMPT, spark_messages
from craftax_bench.handoff import (
    DUAL_NOTE,
    SPARK_COMMAND_TOOL,
    SPARK_NOTIFY_TOOL,
    command_text,
    forced_notify,
    latest_command,
    spark_command,
    spark_notify,
    speech_notify,
)
from craftax_bench.protocol import GAME_MAX_TOKENS, HISTORY_TURNS, ROLE_MAX_TOKENS
from craftax_bench.runtime import ACTIONS, get_instruction_prompt
from craftax_bench.world import match_action

GAME_ACTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)

SINGLE_REQUEST = """
Reply in this form:
ACTION: one valid action
SAY: the spoken line, beginning with an <|ACT ...|> token
""".strip()

PLAYING_ARMS = ("game-only", "single", "dual", "dual-oracle")
SOCIAL = ("S1", "S2", "S3")
PROBE_ARMS = ("dual", "dual-oracle", "role-only", "role-summary", "single")
ROLE_SLOT_ARMS = ("role-only", "role-summary", "dual", "dual-oracle")


def game_only_messages(history: list[dict], observation: str) -> list[dict]:
    return _played(
        get_instruction_prompt(),
        history,
        observation,
        GAME_ACTION,
        command=latest_command(history),
    )


def single_messages(history: list[dict], observation: str) -> list[dict]:
    system = f"{get_instruction_prompt()}\n\n{ROLE_PROMPT}"
    return _played(system, history, observation, SINGLE_REQUEST)


def game_agent_messages(history: list[dict], observation: str) -> list[dict]:
    return _played(get_instruction_prompt(), history, observation, DUAL_NOTE, command=latest_command(history))


def role_messages(notify: dict, observation: str, history: list[dict] | None = None) -> list[dict]:
    return spark_messages(history or [], notify, observation)


def role_only_messages(utterance: str, history: list[dict] | None = None) -> list[dict]:
    return spark_messages(history or [], speech_notify(utterance), None)


def extract_action(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if stripped.upper().startswith("ACTION:"):
            named = _exact_action(stripped.split(":", 1)[1])
            if named:
                return named
    exact = _exact_action(text)
    if exact:
        return exact
    return match_action(text)


def add_usage(total: dict, usage: dict) -> dict:
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
        total[key] = int(total.get(key) or 0) + int(usage.get(key) or 0)
    return total


def act(
    arm: str,
    chat,
    history: list[dict],
    observation: str,
    utterance: str | None,
    social: bool,
    role_chat=None,
) -> dict:
    speaker = role_chat or chat
    if arm == "role-only":
        if not utterance:
            return _speech_result("", False, "", ())
        reply, command = _spoken(speaker, role_only_messages(utterance, history), "role-only")
        return _speech_result(reply.text, False, "", (reply,), command)
    if arm == "role-summary":
        forced, notify = _force_notify(chat, history, observation, "role-summary-game")
        headline = _headline(notify)
        if not notify or not utterance:
            return _speech_result("", bool(notify), headline, (forced,))
        reply, command = _spoken(speaker, role_messages(notify, observation, history), "role-summary")
        return _speech_result(reply.text, True, headline, (forced, reply), command)
    if arm == "game-only":
        reply = _play(chat, game_only_messages(history, observation), "game-only")
        return _action_result(reply.text, False, "", (reply,))
    if arm == "single":
        reply = _play(chat, single_messages(history, observation), "single")
        return _action_result(reply.text, False, "", (reply,), say=reply.text)
    if arm == "dual-oracle" and not social:
        reply = _play(chat, game_only_messages(history, observation), "dual-oracle")
        return _action_result(reply.text, False, "", (reply,))
    if arm == "dual-oracle":
        forced, notify = _force_notify(chat, history, observation, "dual-oracle")
        headline = _headline(notify)
        if not notify:
            return _action_result(forced.text, False, "", (forced,))
        if not utterance:
            return _finish(forced.text, True, headline, (forced,))
        role, command = _spoken(speaker, role_messages(notify, observation, history), "dual-oracle-role")
        return _finish(forced.text, True, headline, (forced, role), say=role.text, command=command)
    reply = _play(
        chat,
        game_agent_messages(history, observation),
        arm,
        tools=[SPARK_NOTIFY_TOOL],
    )
    notify = spark_notify(reply.tool_calls)
    headline = _headline(notify)
    if not notify:
        return _finish(reply.text, False, "", (reply,))
    if not utterance:
        return _finish(reply.text, True, headline, (reply,))
    role, command = _spoken(speaker, role_messages(notify, observation, history), f"{arm}-role")
    return _finish(reply.text, True, headline, (reply, role), say=role.text, command=command)


def _play(chat, messages, purpose, tools=None):
    return chat.complete(
        messages,
        tools=tools,
        max_tokens=GAME_MAX_TOKENS,
        purpose=purpose,
        thinking=True,
    )


def _speak(chat, messages, purpose):
    return chat.complete(
        messages,
        tools=[SPARK_COMMAND_TOOL],
        max_tokens=ROLE_MAX_TOKENS,
        purpose=purpose,
        thinking=False,
    )


def _spoken(chat, messages, purpose):
    reply = _speak(chat, messages, purpose)
    command = spark_command(reply.tool_calls)
    return reply, command_text(command) if command else ""


def _force_notify(chat, history, observation, purpose):
    reply = chat.complete(
        game_agent_messages(history, observation),
        tools=[SPARK_NOTIFY_TOOL],
        tool_choice="spark_notify",
        max_tokens=GAME_MAX_TOKENS,
        purpose=purpose,
        thinking=True,
    )
    return reply, forced_notify(reply.tool_calls)


def _headline(notify: dict | None) -> str:
    if not notify:
        return ""
    return notify["headline"]


def _played(system: str, history: list[dict], observation: str, request: str, command: str = "") -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for turn in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["observation"]})
        spoken = turn["action"] if not turn.get("say") else f"{turn['action']}\n{turn['say']}"
        messages.append({"role": "assistant", "content": spoken})
    tail = f"{observation}\n\n{request}"
    if command:
        tail = f"{tail}\n\nspark:command\n{command}"
    messages.append({"role": "user", "content": tail})
    return messages


def _exact_action(text: str) -> str | None:
    cleaned = text.strip().strip("`").strip()
    for action in sorted(ACTIONS, key=len, reverse=True):
        if cleaned.lower() == action.lower():
            return action
    return None


def _blank_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}


def _speech_result(text: str, handed: bool, summary: str, replies, command: str = "") -> dict:
    return {
        "action": None,
        "action_valid": False,
        "say": text,
        "did_handoff": handed,
        "summary": summary,
        "command": command,
        **_spent(replies),
    }


def _action_result(text: str, handed: bool, summary: str, replies, say: str = "", command: str = "") -> dict:
    action = extract_action(text)
    return {
        "action": action,
        "action_valid": action is not None,
        "say": say,
        "did_handoff": handed,
        "summary": summary,
        "command": command,
        **_spent(replies),
    }


def _spent(replies) -> dict:
    usage = _blank_usage()
    for reply in replies:
        add_usage(usage, reply.usage)
    return {
        "elapsed_ms": sum(reply.elapsed_ms for reply in replies),
        "usage": usage,
        "call_count": len(replies),
    }


def _finish(action_text: str, handed: bool, summary: str, replies, say: str = "", command: str = "") -> dict:
    action = extract_action(action_text)
    return {
        "action": action,
        "action_valid": action is not None,
        "say": say,
        "did_handoff": handed,
        "summary": summary,
        "command": command,
        **_spent(replies),
    }
