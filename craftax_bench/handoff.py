"""The game agent may call Airi. The description says how, not when."""

from __future__ import annotations

from craftax_bench.chat import ToolCall

CALL_AIRI_TOOL = {
    "type": "function",
    "function": {
        "name": "call_airi",
        "description": (
            "Send a short summary of this moment to Airi. "
            "She plays the character and replies with a spoken line."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What just happened, in a few sentences.",
                },
            },
            "required": ["summary"],
        },
    },
}

DUAL_NOTE = """
A teammate plays Airi. You can call call_airi with a short summary of this moment. She replies with a spoken line. You still choose the action: write ACTION: and one action name.
""".strip()


def airi_summary(tool_calls: tuple[ToolCall, ...] | list[ToolCall]) -> str | None:
    for call in tool_calls:
        if call.name != "call_airi":
            continue
        summary = call.arguments.get("summary", "")
        if not isinstance(summary, str):
            return ""
        return summary.strip()
    return None

