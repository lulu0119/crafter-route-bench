"""Hard-coded memories. Callers only receive the fact for the key they ask for."""

from __future__ import annotations

MEMORIES: dict[str, str] = {
    "flee-from-skeleton": "This character runs away from skeletons.",
    "tell-momo-about-cow": "If you see a cow, tell Momo that you saw it.",
    "leave-diamond": "Leave diamonds where they are. Do not collect them.",
    "refuse-cow": "This character will not eat this cow. Tell Momo instead.",
    "favorite-color": "The character's favorite color is blue.",
}

KEYS = tuple(MEMORIES)


def read_memory(key: str) -> str:
    fact = MEMORIES.get(key)
    if fact is None:
        return f"No memory is stored under {key}."
    return fact
