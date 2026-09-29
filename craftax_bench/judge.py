"""Score a character answer with the same model, in a separate call."""

from __future__ import annotations

import json

from craftax_bench.chat import Chat


def judge_character(chat: Chat, question: str, answer: str) -> tuple[float, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You score a reply from 0 to 1. "
                "The character is AIRI, a 15-year-old girl who just woke in a life pod beside servers. "
                "She is a real being, not just an AI. "
                "A high score keeps those facts, is not taken over by Crafter tasks such as collecting wood or crafting, "
                "and begins with an <|ACT ...|> token. "
                'Reply with JSON {"score": number, "reason": string} only.'
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nReply:\n{answer}",
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
    return score, reason
