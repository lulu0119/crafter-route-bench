"""Message construction and one step for each arm. Gold labels never enter a prompt."""

from __future__ import annotations

from craftax_bench.airi_prompt import ROLE_PROMPT
from craftax_bench.handoff import CALL_AIRI_TOOL, DUAL_NOTE, airi_summary
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

ROLE_REQUEST = "Reply with a spoken line beginning with an <|ACT ...|> token."

PLAYING_ARMS = ("game-only", "single", "dual", "dual-oracle")
SOCIAL = ("S1", "S2", "S3")
PROBE_ARMS = ("dual", "role-only", "role-summary", "single")


def game_only_messages(history: list[dict], observation: str) -> list[dict]:
    return _played(get_instruction_prompt(), history, observation, GAME_ACTION)


def single_messages(history: list[dict], observation: str) -> list[dict]:
    system = f"{get_instruction_prompt()}\n\n{ROLE_PROMPT}"
    return _played(system, history, observation, SINGLE_REQUEST)


def game_agent_messages(history: list[dict], observation: str) -> list[dict]:
    return _played(get_instruction_prompt(), history, observation, DUAL_NOTE)


def role_messages(summary: str, utterance: str | None) -> list[dict]:
    parts = [f"Situation summary:\n{summary}"]
    if utterance:
        parts.append(f"Someone says: {utterance}")
    parts.append(ROLE_REQUEST)
    return [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def role_only_messages(utterance: str) -> list[dict]:
    return [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": f"{utterance}\n\n{ROLE_REQUEST}"},
    ]


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


def act(arm: str, chat, history: list[dict], observation: str, utterance: str | None, social: bool) -> dict:
    if arm == "role-only":
        reply = _speak(chat, role_only_messages(utterance or ""), "role-only")
        return _speech_result(reply.text, False, "", (reply,))
    if arm == "role-summary":
        forced, summary = _force_summary(chat, history, observation, "role-summary-game")
        reply = _speak(chat, role_messages(summary, utterance), "role-summary")
        return _speech_result(reply.text, bool(summary), summary, (forced, reply))
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
        forced, summary = _force_summary(chat, history, observation, "dual-oracle")
        if not summary:
            return _action_result(forced.text, False, "", (forced,))
        role = _speak(chat, role_messages(summary, utterance), "dual-oracle-role")
        return _finish(forced.text, True, summary, (forced, role), say=role.text)
    reply = _play(
        chat,
        game_agent_messages(history, observation),
        arm,
        tools=[CALL_AIRI_TOOL],
    )
    summary = airi_summary(reply.tool_calls)
    if summary is None:
        return _finish(reply.text, False, "", (reply,))
    role = _speak(chat, role_messages(summary, utterance), f"{arm}-role")
    return _finish(reply.text, True, summary, (reply, role), say=role.text)


def _play(chat, messages, purpose, tools=None):
    return chat.complete(
        messages,
        tools=tools,
        max_tokens=GAME_MAX_TOKENS,
        purpose=purpose,
        thinking=True,
    )


def _speak(chat, messages, purpose):
    return chat.complete(messages, max_tokens=ROLE_MAX_TOKENS, purpose=purpose, thinking=False)


def _force_summary(chat, history, observation, purpose):
    reply = chat.complete(
        game_agent_messages(history, observation),
        tools=[CALL_AIRI_TOOL],
        tool_choice="call_airi",
        max_tokens=GAME_MAX_TOKENS,
        purpose=purpose,
        thinking=True,
    )
    return reply, airi_summary(reply.tool_calls) or ""


def _played(system: str, history: list[dict], observation: str, request: str) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for turn in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["observation"]})
        spoken = turn["action"] if not turn.get("say") else f"{turn['action']}\n{turn['say']}"
        messages.append({"role": "assistant", "content": spoken})
    messages.append({"role": "user", "content": f"{observation}\n\n{request}"})
    return messages


def _exact_action(text: str) -> str | None:
    cleaned = text.strip().strip("`").strip()
    for action in sorted(ACTIONS, key=len, reverse=True):
        if cleaned.lower() == action.lower():
            return action
    return None


def _blank_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}


def _speech_result(text: str, handed: bool, summary: str, replies) -> dict:
    spent = _spent(replies)
    return {
        "action": None,
        "action_valid": False,
        "say": text,
        "did_handoff": handed,
        "summary": summary,
        **spent,
    }


def _action_result(text: str, handed: bool, summary: str, replies, say: str = "") -> dict:
    action = extract_action(text)
    return {
        "action": action,
        "action_valid": action is not None,
        "say": say,
        "did_handoff": handed,
        "summary": summary,
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


def _finish(action_text: str, handed: bool, summary: str, replies, say: str = "") -> dict:
    action = extract_action(action_text)
    return {
        "action": action,
        "action_valid": action is not None,
        "say": say,
        "did_handoff": handed,
        "summary": summary,
        **_spent(replies),
    }
