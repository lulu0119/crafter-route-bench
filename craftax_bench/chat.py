"""OpenAI-compatible chat against the single model in ``.env``."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from craftax_bench.runtime import ROOT

USER_AGENT = "crafter-route-bench/0.1"
SESSION = "crafter-route-bench"


def load_env(path: Path | None = None) -> dict[str, str]:
    env_path = path or ROOT / ".env"
    values: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ChatResult:
    text: str
    tool_calls: tuple[ToolCall, ...]
    elapsed_ms: float


class Chat:
    def __init__(self, env: dict[str, str] | None = None):
        loaded = env or load_env()
        self.model = loaded["MODEL"]
        self.url = loaded["MODEL_URL"]
        self.key = loaded["OPENCODE_API_KEY"]

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
                "x-opencode-session": SESSION,
            },
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    raw = response.read().decode()
                last_error = None
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")
                last_error = RuntimeError(f"chat HTTP {error.code}: {detail[:400]}")
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                    raise last_error from error
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                last_error = error
                if attempt == 3:
                    raise
            time.sleep(2 * (attempt + 1))
        if last_error is not None:
            raise last_error
        elapsed_ms = (time.perf_counter() - started) * 1000
        data = json.loads(raw)
        message = (data.get("choices") or [{}])[0].get("message") or {}
        text = message.get("content") or message.get("reasoning_content") or ""
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append(ToolCall(
                id=call.get("id") or function.get("name") or "tool",
                name=function.get("name") or "",
                arguments=arguments if isinstance(arguments, dict) else {},
            ))
        return ChatResult(text=text, tool_calls=tuple(calls), elapsed_ms=elapsed_ms)


def apply_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ.setdefault(key, value)
