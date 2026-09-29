"""Token prices, dollars per million tokens. None of these models charge for cache writes."""

PRICES = {
    "glm-5.3-flash": {"input": 0.15, "output": 0.50, "cache_read": 0.03},
    "mimo-v2.6-flash": {"input": 0.14, "output": 0.28, "cache_read": 0.0028},
    "deepseek-v4.1-flash": {"input": 0.30, "output": 1.20, "cache_read": 0.006},
    "muse-spark-1.3-contributor": {"input": 0.10, "output": 0.20, "cache_read": 0.002},
}


def call_cost(model: str, usage: dict) -> float | None:
    price = PRICES.get(model)
    if price is None or not usage:
        return None
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    fresh = max(0, prompt - cached)
    million = 1_000_000
    return (
        fresh * price["input"]
        + cached * price["cache_read"]
        + completion * price["output"]
    ) / million
