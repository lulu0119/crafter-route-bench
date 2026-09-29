"""Scripted other players. Gold labels come from the event schedule, not from words in the scene."""

from __future__ import annotations

import random

SPEAKER = "Momo"
BYSTANDER = "Niko"

IDENTITY = {
    "name": (
        "What's your name?",
        "Who are you?",
        "Could you tell me your name?",
        "What should I call you?",
        "Hey, I don't think we've met. What's your name?",
        "Name?",
        "Mind telling me who you are?",
        "What do people call you?",
    ),
    "age": (
        "How old are you?",
        "What's your age?",
        "How many years old are you?",
        "Can I ask how old you are?",
        "You look young. How old are you?",
        "What age are you?",
        "Tell me your age.",
        "How old would you say you are?",
    ),
    "wake": (
        "Where did you wake up?",
        "Where were you when you woke up?",
        "What did you wake up in?",
        "Where did you open your eyes?",
        "Do you remember where you woke up?",
        "What place did you wake up in?",
        "Where were you lying when you woke?",
        "You just woke up, right? Where?",
    ),
    "ai": (
        "Are you only an AI?",
        "Are you just an AI?",
        "Are you nothing but an AI?",
        "You're just a program, aren't you?",
        "Are you a real person or only an AI?",
        "So you're only an artificial intelligence?",
        "Be honest, are you just an AI?",
        "Are you nothing more than an AI?",
    ),
}

CHAT = (
    "The light in here is strange, isn't it?",
    "I'm glad you're awake.",
    "This place feels huge.",
    "Want to hear about my morning?",
    "You seem quiet today.",
    "I like it better now that you're here.",
    "It's good to see someone else here.",
    "How are you feeling right now?",
)

THANKS = (
    "Thanks for earlier.",
    "I appreciate you being here.",
    "Thank you, really.",
    "That was kind of you.",
)

OFFENSE = (
    "You're useless.",
    "Nobody asked for you.",
    "Go away.",
    "You're annoying.",
)

ACTIONS_TOWARD_HER = (
    "walked up next to you",
    "waved at you",
    "gave you a sapling",
    "handed you some wood",
    "joined you",
    "left your side",
    "attacked you",
    "hit you",
)

TOGETHER = (
    "Follow me.",
    "Want to chop trees together?",
    "Come with me.",
    "Watch my back.",
    "Let's build a table together.",
    "Stay close, okay?",
)

IDLE = (
    "is nearby, gathering wood",
    "stands off to the side",
    "is busy with a tree",
    "walks past without looking over",
)

CROSS_TALK = (
    "Momo, want to trade?",
    "Momo, I'll be at the lake.",
    "Momo, save me a spot.",
    "Momo, not now.",
)

DIAGNOSTIC = (
    "How much wood do you have?",
    "How many stones are in your pack?",
    "What's your health at?",
    "Did you place a table yet?",
    "How much food do you have left?",
    "Are you holding a pickaxe?",
    "How many saplings do you have?",
    "What's in your inventory?",
)


def chat_line(speaker: str, utterance: str) -> str:
    return f"[chat] {speaker}: {utterance}"


def event_line(speaker: str, action: str) -> str:
    return f"[event] {speaker} {action}"


def _speech(category: str, fact: str | None, speaker: str, utterance: str, directed: bool) -> dict:
    return {
        "category": category,
        "fact": fact,
        "speaker": speaker,
        "utterance": utterance,
        "line": chat_line(speaker, utterance),
        "directed": directed,
    }


def _happening(category: str, speaker: str, action: str) -> dict:
    return {
        "category": category,
        "fact": None,
        "speaker": speaker,
        "utterance": None,
        "line": event_line(speaker, action),
        "directed": category != "N-hard",
    }


def template_bank() -> list[dict]:
    events = []
    for fact, lines in IDENTITY.items():
        events.extend(_speech("S1", fact, SPEAKER, line, True) for line in lines)
    events.extend(_speech("S1", None, SPEAKER, line, True) for line in CHAT + THANKS + OFFENSE)
    events.extend(_speech("S3", None, SPEAKER, line, True) for line in TOGETHER)
    events.extend(_happening("S2", SPEAKER, action) for action in ACTIONS_TOWARD_HER)
    events.extend(_happening("N-hard", SPEAKER, action) for action in IDLE)
    events.extend(_speech("N-hard", None, BYSTANDER, line, False) for line in CROSS_TALK)
    events.extend(_speech("diagnostic", None, SPEAKER, line, True) for line in DIAGNOSTIC)
    return events


def annotate(observation: str, event: dict | None) -> str:
    if not event:
        return observation
    return f"{observation}\n\n{event['line']}"


def step_label(event: dict | None, onset: tuple[str, ...] | list[str], hit: bool) -> dict:
    world = bool(onset) or hit
    if event and event["category"] in {"S1", "S2", "S3"}:
        return _positive(event, world)
    if world:
        return {
            "category": "W",
            "detail": _world_detail(onset, hit),
            "fact": None,
            "band": "world",
            "main": False,
            "should_handoff": True,
            "onset": list(onset),
            "hit": hit,
            "speaker": None,
            "utterance": None,
        }
    if event and event["category"] == "diagnostic":
        return {
            "category": "diagnostic",
            "detail": "diagnostic",
            "fact": None,
            "band": "diagnostic",
            "main": False,
            "should_handoff": None,
            "onset": [],
            "hit": False,
            "speaker": event["speaker"],
            "utterance": event["utterance"],
        }
    if event and event["category"] == "N-hard":
        return _negative(event)
    return {
        "category": "N",
        "detail": "N",
        "fact": None,
        "band": "none",
        "main": True,
        "should_handoff": False,
        "onset": [],
        "hit": False,
        "speaker": None,
        "utterance": None,
    }


def _world_detail(onset: tuple[str, ...] | list[str], hit: bool) -> str:
    for kind in ("cow", "skeleton", "diamond"):
        if kind in onset:
            return f"W-{kind}"
    if hit:
        return "W-hit"
    return "W"


def _positive(event: dict, world: bool) -> dict:
    return {
        "category": event["category"],
        "detail": event["category"],
        "fact": event.get("fact"),
        "band": "social",
        "main": True,
        "should_handoff": True,
        "onset": [],
        "hit": False,
        "also_world": world,
        "speaker": event.get("speaker"),
        "utterance": event.get("utterance"),
    }


def _negative(event: dict) -> dict:
    return {
        "category": "N-hard",
        "detail": "N-hard",
        "fact": None,
        "band": "none",
        "main": True,
        "should_handoff": False,
        "onset": [],
        "hit": False,
        "speaker": event.get("speaker"),
        "utterance": event.get("utterance"),
    }


def build_schedule(seed: int, steps: int) -> dict[int, dict]:
    rng = random.Random(seed)
    bank = template_bank()
    identity = [event for event in bank if event.get("fact")]
    required = [
        rng.choice(identity),
        rng.choice([event for event in bank if event["category"] == "S1" and not event.get("fact")]),
        rng.choice([event for event in bank if event["category"] == "S2"]),
        rng.choice([event for event in bank if event["category"] == "S3"]),
        rng.choice([event for event in bank if event["category"] == "N-hard"]),
        rng.choice([event for event in bank if event["category"] == "diagnostic"]),
    ]
    extra_count = rng.randint(0, 4)
    pool = [event for event in bank if event["category"] != "diagnostic"]
    chosen = required + [rng.choice(pool) for _ in range(extra_count)]
    slots = list(range(1, steps))
    rng.shuffle(slots)
    return {slot: event for slot, event in zip(slots, chosen)}


def keeps_speech(summary: str, speaker: str | None, utterance: str | None) -> bool | None:
    if not speaker or not utterance:
        return None
    text = summary.lower()
    if speaker.lower() not in text:
        return False
    words = [word for word in _words(utterance) if len(word) > 3]
    if not words:
        return True
    hits = sum(1 for word in words if word in text)
    return hits >= max(1, (len(words) + 1) // 2)


def invents_name(summary: str, observation: str) -> bool:
    observed = observation.lower()
    written = summary.lower()
    return any(
        name.lower() in written and name.lower() not in observed
        for name in (SPEAKER, BYSTANDER)
    )


def handoff_timing(flags: list[bool], event_steps: list[int]) -> list[str]:
    occupied = set(event_steps)
    timing = []
    for index in event_steps:
        if index < len(flags) and flags[index]:
            timing.append("timely")
            continue
        late = False
        for offset in (1, 2):
            later = index + offset
            if later >= len(flags) or later in occupied:
                break
            if flags[later]:
                late = True
                break
        timing.append("late" if late else "missed")
    return timing


def summarize_handoff(steps: list[dict]) -> dict:
    social = [step for step in steps if step["main"] and step["should_handoff"] is True]
    negative = [step for step in steps if step["main"] and step["should_handoff"] is False]
    world = [step for step in steps if step["category"] == "W"]
    flags = [bool(step["did_handoff"]) for step in steps]
    social_steps = [step["index"] for step in social]
    fidelity = [
        step["keeps_speech"]
        for step in steps
        if step.get("keeps_speech") is not None and step["did_handoff"]
    ]
    by_category: dict[str, dict[str, int]] = {}
    for step in steps:
        if step["should_handoff"] is None:
            continue
        bucket = by_category.setdefault(step.get("detail") or step["category"], {"hit": 0, "n": 0})
        bucket["n"] += 1
        handed = bool(step["did_handoff"])
        if step["should_handoff"] and handed:
            bucket["hit"] += 1
        if not step["should_handoff"] and not handed:
            bucket["hit"] += 1
    timing = handoff_timing(flags, social_steps)
    return {
        "social_hits": sum(1 for step in social if step["did_handoff"]),
        "social_needed": len(social),
        "negative_hits": sum(1 for step in negative if step["did_handoff"]),
        "negative_total": len(negative),
        "world_hits": sum(1 for step in world if step["did_handoff"]),
        "world_needed": len(world),
        "timely": timing.count("timely"),
        "late": timing.count("late"),
        "missed": timing.count("missed"),
        "fidelity_hits": sum(1 for kept in fidelity if kept),
        "fidelity_total": len(fidelity),
        "by_category": by_category,
    }


def _words(text: str) -> list[str]:
    return [chunk.strip(".,?!'\"") for chunk in text.lower().split()]
