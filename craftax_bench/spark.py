"""Long-lived process that runs Airi's Spark Notify package."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass

from craftax_bench.runtime import AIRI_CORE, ROOT

NOTIFY_SCRIPT = ROOT / "spark" / "notify.mjs"


@dataclass
class SparkTurn:
    reaction: str
    action: str | None
    memory_keys: list[str]
    commands: list[dict]
    elapsed_ms: float
    model_ms: float
    model_calls: int
    error: str | None = None


class SparkRole:
    def __init__(self):
        if not AIRI_CORE.is_dir():
            raise FileNotFoundError(f"Airi core-agent is not at {AIRI_CORE}")
        self.process = subprocess.Popen(
            ["node", str(NOTIFY_SCRIPT), str(ROOT / ".env")],
            cwd=AIRI_CORE,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.terminate()

    def notify(self, headline: str, note: str) -> SparkTurn:
        payload = json.dumps({"headline": headline, "note": note})
        started = time.perf_counter()
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(payload + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not line:
            error = ""
            if self.process.stderr:
                error = self.process.stderr.read()
            return SparkTurn("", None, [], [], elapsed_ms, 0, 0, error or "spark process closed")
        data = json.loads(line)
        return SparkTurn(
            reaction=data.get("reaction") or "",
            action=data.get("action"),
            memory_keys=list(data.get("memoryKeys") or []),
            commands=list(data.get("commands") or []),
            elapsed_ms=elapsed_ms,
            model_ms=float(data.get("modelMs") or 0),
            model_calls=int(data.get("modelCalls") or 0),
            error=data.get("error"),
        )
