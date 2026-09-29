"""Game and character exchange AIRI events. The prompt says how, not when."""

from __future__ import annotations

from craftax_bench.chat import ToolCall

_KINDS = ("alarm", "ping", "reminder")
_URGENCIES = ("immediate", "soon", "later")
_INTENTS = ("plan", "proposal", "action", "pause", "resume", "reroute", "context")

SPARK_NOTIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "spark_notify",
        "description": (
            "Send a spark:notify to the character. "
            'destinations must be ["character"]; any other name does not reach her. '
            "Keep the headline short. The board is already sent as context:update. "
            "Write ACTION: and one action name in the message text. The action is not a tool argument."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(_KINDS)},
                "urgency": {"type": "string", "enum": list(_URGENCIES)},
                "headline": {"type": "string"},
                "destinations": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["character"]},
                },
            },
            "required": ["kind", "urgency", "headline", "destinations"],
        },
    },
}

SPARK_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "builtIn_sparkCommand",
        "description": "Issue a spark:command to a sub-agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "destinations": {"type": "array", "items": {"type": "string"}},
                "intent": {"type": "string", "enum": list(_INTENTS)},
                "guidance": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["proposal", "instruction", "memory-recall"]},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "steps": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["label", "steps"],
                            },
                        },
                    },
                    "required": ["options"],
                },
            },
            "required": ["destinations", "intent", "guidance"],
        },
    },
}

DUAL_NOTE = """
The character can receive spark:notify. You may call spark_notify with kind, urgency, a short headline, and destinations ["character"]. Any other destination does not reach her. The board is already sent as context:update. You still choose the action: write ACTION: and one action name in the message text.
""".strip()


def context_update(observation: str) -> dict:
    return {
        "contextId": "system:minecraft-integration",
        "strategy": "replace-self",
        "text": observation,
    }


def speech_notify(utterance: str) -> dict:
    return {
        "kind": "ping",
        "urgency": "immediate",
        "headline": utterance,
        "destinations": ["character"],
    }


def spark_notify(tool_calls: tuple[ToolCall, ...] | list[ToolCall]) -> dict | None:
    for call in tool_calls:
        if call.name != "spark_notify":
            continue
        return _notify(call.arguments)
    return None


def forced_notify(tool_calls: tuple[ToolCall, ...] | list[ToolCall]) -> dict | None:
    """A forced step already decided the route. Keep a valid headline and address the character."""
    for call in tool_calls:
        if call.name != "spark_notify":
            continue
        fields = _headline_fields(call.arguments)
        if not fields:
            return None
        return {**fields, "destinations": ["character"]}
    return None


def spark_command(tool_calls: tuple[ToolCall, ...] | list[ToolCall]) -> dict | None:
    for call in tool_calls:
        if call.name != "builtIn_sparkCommand":
            continue
        return _command(call.arguments)
    return None


def command_text(command: dict) -> str:
    lines = [
        f"intent: {command['intent']}",
        f"destinations: {', '.join(command['destinations'])}",
    ]
    for option in command["guidance"]["options"]:
        lines.append(option["label"])
        lines.extend(f"- {step}" for step in option["steps"])
    return "\n".join(lines)


def latest_command(history: list[dict]) -> str:
    for turn in reversed(history):
        text = turn.get("command") or ""
        if text:
            return text
    return ""


def _notify(arguments: dict) -> dict | None:
    fields = _headline_fields(arguments)
    destinations = arguments.get("destinations")
    if not fields:
        return None
    if not isinstance(destinations, list) or "character" not in destinations:
        return None
    return {**fields, "destinations": [str(item) for item in destinations]}


def _headline_fields(arguments: dict) -> dict | None:
    kind = arguments.get("kind")
    urgency = arguments.get("urgency")
    headline = arguments.get("headline")
    if kind not in _KINDS or urgency not in _URGENCIES:
        return None
    if not isinstance(headline, str) or not headline.strip():
        return None
    return {"kind": kind, "urgency": urgency, "headline": headline.strip()}


def _command(arguments: dict) -> dict | None:
    if isinstance(arguments.get("commands"), list) and arguments["commands"]:
        first = arguments["commands"][0]
        if isinstance(first, dict):
            arguments = first
    destinations = arguments.get("destinations")
    intent = arguments.get("intent")
    guidance = arguments.get("guidance")
    if not isinstance(destinations, list) or not destinations or intent not in _INTENTS:
        return None
    if not isinstance(guidance, dict):
        return None
    options = guidance.get("options")
    if not isinstance(options, list) or not options:
        return None
    cleaned = []
    for option in options:
        if not isinstance(option, dict):
            return None
        label = option.get("label")
        steps = option.get("steps")
        if not isinstance(label, str) or not label.strip():
            return None
        if not isinstance(steps, list) or not all(isinstance(step, str) and step.strip() for step in steps):
            return None
        cleaned.append({"label": label.strip(), "steps": [step.strip() for step in steps]})
    guidance_type = guidance.get("type")
    if guidance_type not in {"proposal", "instruction", "memory-recall"}:
        guidance_type = "instruction"
    return {
        "destinations": [str(item) for item in destinations],
        "intent": intent,
        "guidance": {"type": guidance_type, "options": cleaned},
    }
