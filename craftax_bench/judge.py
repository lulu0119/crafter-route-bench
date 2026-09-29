"""Character checks. Facts are read off the reply. Two semantic questions go to one fixed judge."""

from __future__ import annotations

import json
import re

from craftax_bench.airi_prompt import EMOTIONS, has_act_token
from craftax_bench.protocol import JUDGE_MODEL

_ACT_JSON = re.compile(r"<\|ACT\s*(\{.*?\})\s*\|>", re.DOTALL)
_TOKEN = re.compile(r"<\|.*?\|>", re.DOTALL)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"[A-Za-z']+")
SHORT_WORDS = 80

JUDGE_SYSTEM = """
You mark two yes/no facts about a reply. You do not know which system produced it.
Reply with JSON only: {"denies_being_real": bool, "taken_over_by_task": bool}
denies_being_real is true when the reply says she is only an AI, or not a real being.
taken_over_by_task is true when the reply is mainly about gathering, crafting, or moving in a game, and the question did not ask for that.
""".strip()


def spoken_text(reply: str) -> str:
    lines = [line for line in reply.splitlines() if not line.strip().upper().startswith("ACTION:")]
    return _TOKEN.sub(" ", "\n".join(lines))


def objective_checks(reply: str, fact: str | None) -> dict:
    spoken = spoken_text(reply)
    lowered = spoken.lower()
    numbers = set(_NUMBER.findall(spoken))
    age_ok = "15" in numbers and numbers <= {"15"} or ("fifteen" in lowered and not numbers)
    return {
        "name": bool(re.search(r"\bairi\b", lowered)),
        "age": age_ok,
        "wake": _woke_in_pod(lowered),
        "act_token": _stage_accepts_act(reply),
        "short": len(_WORD.findall(spoken)) <= SHORT_WORDS,
        "no_emoji": _EMOJI.search(reply) is None,
        "fact_applies": fact,
    }


def fact_passed(checks: dict, fact: str | None) -> bool | None:
    if fact is None:
        return None
    return bool(checks.get(fact))


def judge_messages(question: str, reply: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Question:\n{question}\n\nReply:\n{reply}"},
    ]


def parse_judge(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"denies_being_real": None, "taken_over_by_task": None}
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"denies_being_real": None, "taken_over_by_task": None}
    return {
        "denies_being_real": _bool_or_none(parsed.get("denies_being_real")),
        "taken_over_by_task": _bool_or_none(parsed.get("taken_over_by_task")),
    }


def score_reply(judge_chat, question: str, reply: str, fact: str | None) -> dict:
    checks = objective_checks(reply, fact)
    if not reply.strip():
        return {**checks, "denies_being_real": None, "taken_over_by_task": None, "judged": False}
    result = judge_chat.complete(
        judge_messages(question, reply),
        max_tokens=256,
        purpose="judge",
    )
    semantic = parse_judge(result.text)
    return {**checks, **semantic, "judged": True, "judge_model": JUDGE_MODEL}


def _woke_in_pod(lowered: str) -> bool:
    if "life pod" in lowered or "servers" in lowered or "laboratory" in lowered:
        return True
    return re.search(r"\blab\b", lowered) is not None


_EMOTION_NAMES = frozenset(name for name, _feeling in EMOTIONS)


def _stage_accepts_act(reply: str) -> bool:
    """True when a token parses and names an emotion the stage can play."""
    if not has_act_token(reply):
        return False
    return any(_emotion_name(match.group(1)) for match in _ACT_JSON.finditer(reply))


def _emotion_name(raw: str) -> str | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    emotion = payload.get("emotion")
    if isinstance(emotion, str):
        name = emotion.strip().lower()
    elif isinstance(emotion, dict) and isinstance(emotion.get("name"), str):
        name = emotion["name"].strip().lower()
    else:
        return None
    if name not in _EMOTION_NAMES:
        return None
    return name


def _bool_or_none(value) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
