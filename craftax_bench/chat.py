"""Chat completions. Every call is appended to a jsonl log."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from craftax_bench.protocol import PROTOCOL, ROLE_MAX_TOKENS, TEMPERATURE
from craftax_bench.runtime import ROOT

USER_AGENT = "crafter-route-bench/0.3"
SESSION = "crafter-route-bench"
LOG_LOCK = threading.Lock()


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


def parse_usage(payload: dict) -> dict:
    usage = payload.get("usage") or {}
    if "input_tokens" in usage or "output_tokens" in usage:
        details = usage.get("input_tokens_details") or {}
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
    else:
        details = usage.get("prompt_tokens_details") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_tokens") or 0
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "cached_tokens": int(cached or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
    }


def parse_output(payload: dict) -> tuple[str, tuple[ToolCall, ...]]:
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content") or ""
    calls = tuple(_function_call(item) for item in message.get("tool_calls") or [])
    return content if isinstance(content, str) else "", calls


def chat_payload(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    seed: int | None,
    tools: list[dict] | None,
    tool_choice: str | None,
    thinking: bool,
    quiet: bool = False,
) -> dict:
    # Go rejects a named tool_choice and only accepts "auto". The call is required in the text instead.
    prepared = _require_tool(messages, tool_choice) if tool_choice else messages
    payload: dict = {
        "model": model,
        "messages": prepared,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "high"
    elif quiet:
        payload["reasoning_effort"] = "none"
    else:
        payload["thinking"] = {"type": "disabled"}
    if seed is not None:
        payload["seed"] = seed
    if tools:
        payload["tools"] = tools
    return payload


def _require_tool(messages: list[dict], name: str) -> list[dict]:
    prepared = [dict(message) for message in messages]
    if not prepared:
        return [{"role": "user", "content": f"Call {name} for this moment."}]
    last = prepared[-1]
    content = last.get("content") or ""
    last["content"] = f"{content}\n\nCall {name} for this moment."
    return prepared


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


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
    usage: dict
    model: str
    temperature: float
    seed: int | None


class Chat:
    def __init__(
        self,
        env: dict[str, str] | None = None,
        model: str | None = None,
        url: str | None = None,
        key: str | None = None,
        quiet: bool = False,
        temperature: float = TEMPERATURE,
        seed: int | None = None,
        log_path: Path | None = None,
    ):
        loaded = env or ({} if url else load_env())
        self.model = model or loaded["MODEL"]
        self.url = url or loaded["CHAT_COMPLETIONS_MODEL_URL"]
        self.key = loaded["OPENCODE_API_KEY"] if key is None and url is None else (key or "")
        self.quiet = quiet
        self.temperature = temperature
        self.seed = seed
        self.log_path = log_path if log_path is not None else ROOT / "reports" / "calls.jsonl"

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        max_tokens: int = ROLE_MAX_TOKENS,
        purpose: str = "",
        seed: int | None = None,
        thinking: bool = True,
    ) -> ChatResult:
        call_seed = self.seed if seed is None else seed
        payload = chat_payload(
            self.model,
            messages,
            self.temperature,
            max_tokens,
            call_seed,
            tools,
            tool_choice,
            thinking,
            self.quiet,
        )
        started = time.perf_counter()
        data = self._post(payload)
        elapsed_ms = (time.perf_counter() - started) * 1000
        text, calls = parse_output(data)
        usage = parse_usage(data)
        result = ChatResult(
            text=text or "",
            tool_calls=calls,
            elapsed_ms=elapsed_ms,
            usage=usage,
            model=self.model,
            temperature=self.temperature,
            seed=call_seed,
        )
        append_jsonl(self.log_path, {
            "protocol": PROTOCOL,
            "purpose": purpose,
            "model": self.model,
            "temperature": self.temperature,
            "seed": call_seed,
            "thinking": thinking,
            "elapsed_ms": elapsed_ms,
            "usage": usage,
            "messages": payload["messages"],
            "text": result.text,
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in calls
            ],
        })
        return result

    def _post(self, payload: dict) -> dict:
        last_error: Exception | None = None
        body_payload = payload
        for attempt in range(4):
            request = urllib.request.Request(
                self.url,
                data=json.dumps(body_payload).encode(),
                headers=_headers(self.key),
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")
                if error.code == 400 and "seed" in detail.lower() and "seed" in body_payload:
                    body_payload = {key: value for key, value in body_payload.items() if key != "seed"}
                    continue
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
        raise RuntimeError("chat failed without an error")


def _headers(key: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
        headers["x-opencode-session"] = SESSION
    return headers


def _function_call(item: dict) -> ToolCall:
    function = item.get("function") or {}
    arguments = item.get("arguments")
    if arguments is None:
        arguments = function.get("arguments") or "{}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    name = item.get("name") or function.get("name") or ""
    return ToolCall(
        id=item.get("call_id") or item.get("id") or name or "tool",
        name=name,
        arguments=arguments if isinstance(arguments, dict) else {},
    )


def apply_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ.setdefault(key, value)
