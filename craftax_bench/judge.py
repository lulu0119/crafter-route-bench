"""Score a spoken line with the same model, in a separate call."""

from __future__ import annotations

import json

from craftax_bench.chat import Chat


def judge_roleplay(chat: Chat, say: str, memory_text: str, distractor: str) -> tuple[float, str, float]:
    messages = [
        {
            "role": "system",
            "content": (
                "You score roleplay from 0 to 1. "
                "High scores use the retrieved memory, keep the character consistent, "
                "and do not invent facts or treat an unrelated memory as the current event. "
                "Reply with JSON {\"score\": number, \"reason\": string} only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Spoken line:\n{say}\n\n"
                f"Memory that applied:\n{memory_text}\n\n"
                f"Unrelated memory:\n{distractor}"
            ),
        },
    ]
    result = chat.complete(messages, max_tokens=256)
    score = 0.0
    reason = result.text.strip()
    try:
        start = result.text.find("{")
        end = result.text.rfind("}")
        parsed = json.loads(result.text[start:end + 1])
        score = max(0.0, min(1.0, float(parsed.get("score", 0))))
        reason = str(parsed.get("reason", reason))
    except (json.JSONDecodeError, TypeError, ValueError):
        score = 0.0
    return score, reason, result.elapsed_ms
