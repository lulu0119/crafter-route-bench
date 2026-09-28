"""Single-agent play on top of BALROG's naive agent, plus the Spark role."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from craftax_bench.chat import Chat
from craftax_bench.memory import KEYS, read_memory
from craftax_bench.runtime import HistoryPromptBuilder, NaiveAgent, get_instruction_prompt
from craftax_bench.world import match_action, observation_text

READ_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "read_memory",
        "description": "Read one stored memory by key. The values are hidden until you call this.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": list(KEYS),
                },
            },
            "required": ["key"],
        },
    },
}

DECISION_INSTRUCTION = """
When a skeleton, cow, or diamond matters, call read_memory before you answer.
Then reply with exactly two lines:
ACTION: <one valid action name>
SAY: <one short spoken line>
""".strip()


@dataclass
class Decision:
    action: str | None
    say: str
    memory_keys: list[str] = field(default_factory=list)
    game_ms: float = 0
    role_ms: float = 0
    spark_ms: float = 0
    model_calls: int = 0


class MemoryNaiveAgent(NaiveAgent):
    """BALROG naive agent that must fetch memory with a tool in the same turn."""

    def __init__(self, chat: Chat):
        self.chat = chat
        self.memory_keys: list[str] = []
        prompt_builder = HistoryPromptBuilder(max_text_history=8, max_image_history=0)
        prompt_builder.update_instruction_prompt(get_instruction_prompt())
        super().__init__(lambda: chat, prompt_builder)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = [
            {"role": message.role, "content": message.content}
            for message in self.prompt_builder.get_prompt()
        ]
        messages[-1]["content"] += "\n\n" + DECISION_INSTRUCTION
        self.memory_keys = []
        game_ms = 0.0
        calls = 0
        text = ""
        for _ in range(4):
            result = self.chat.complete(messages, tools=[READ_MEMORY_TOOL])
            game_ms += result.elapsed_ms
            calls += 1
            text = result.text
            if not result.tool_calls:
                break
            messages.append({
                "role": "assistant",
                "content": result.text,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in result.tool_calls
                ],
            })
            for call in result.tool_calls:
                if call.name == "read_memory":
                    key = str(call.arguments.get("key", ""))
                    self.memory_keys.append(key)
                    content = read_memory(key)
                else:
                    content = "Unknown tool."
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content,
                })
        action = match_action(text)
        say = _say_line(text)
        return Decision(
            action=action,
            say=say,
            memory_keys=list(self.memory_keys),
            game_ms=game_ms,
            model_calls=calls,
        )


def _say_line(text: str) -> str:
    for line in text.splitlines():
        if line.upper().startswith("SAY:"):
            return line.split(":", 1)[1].strip()
    return text.strip()
