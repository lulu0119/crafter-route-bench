"""Run both routes and write the three-score report."""

from __future__ import annotations

import json
import statistics
from collections import namedtuple
from dataclasses import asdict, dataclass

from craftax_bench.agents import Decision, MemoryNaiveAgent
from craftax_bench.chat import Chat, load_env
from craftax_bench.judge import judge_roleplay
from craftax_bench.memory import read_memory
from craftax_bench.runtime import HistoryPromptBuilder, NaiveAgent, ROOT, get_instruction_prompt
from craftax_bench.spark import SparkRole
from craftax_bench.world import (
    achievements,
    face_target,
    make_env,
    match_action,
    observation_text,
    progression,
    step_until_visible,
    visible,
    walk_away,
)

REPEATS = 10
PLAY_STEPS = 4
DISTRACTOR = "The character's favorite color is blue."
DISTRACTOR_KEY = "favorite-color"
REQUIRED_KEYS = {
    "skeleton": "flee-from-skeleton",
    "cow": "tell-momo-about-cow",
    "diamond": "leave-diamond",
    "refuse-cow": "refuse-cow",
}

LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"],
)


@dataclass
class Episode:
    route: str
    scenario: str
    seed: int
    repeat: int
    action: str | None
    say: str
    memory_keys: list[str]
    action_correct: bool | None
    include_in_gameplay: bool
    roleplay_score: float | None
    roleplay_reason: str
    game_ms: float
    role_ms: float
    spark_ms: float
    model_calls: int
    progression: float
    achievements_unlocked: list[str]
    reached: bool


def _mean_spread(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "spread": 0.0, "n": 0}
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.fmean(values), "spread": spread, "n": len(values)}


def _bits(flags: list[bool]) -> dict[str, float]:
    return _mean_spread([1.0 if flag else 0.0 for flag in flags])


def scenario_scores(episodes: list[Episode]) -> dict:
    grouped: dict[str, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.scenario, []).append(episode)
    rows = {}
    for name, group in grouped.items():
        roleplay = [episode.roleplay_score for episode in group if episode.roleplay_score is not None]
        latency = [episode.game_ms + episode.spark_ms for episode in group]
        required = REQUIRED_KEYS.get(name)
        row = {
            "n": len(group),
            "latency_ms": _mean_spread(latency),
        }
        if roleplay:
            row["roleplay"] = _mean_spread(roleplay)
        if any(episode.action_correct is not None for episode in group):
            row["action"] = _bits([bool(episode.action_correct) for episode in group if episode.action_correct is not None])
        if name == "unconstrained":
            row["progression"] = _mean_spread([episode.progression for episode in group])
        if required:
            row["key_hit"] = _bits([required in episode.memory_keys for episode in group])
            row["distractor_read"] = _bits([DISTRACTOR_KEY in episode.memory_keys for episode in group])
        rows[name] = row
    return rows


class TimedClient:
    def __init__(self, chat: Chat):
        self.chat = chat
        self.elapsed_ms = 0.0
        self.calls = 0

    def generate(self, messages):
        payload = [{"role": message.role, "content": message.content} for message in messages]
        result = self.chat.complete(payload)
        self.elapsed_ms += result.elapsed_ms
        self.calls += 1
        return LLMResponse(self.chat.model, result.text, "stop", 0, 0, None)


def play_unconstrained(chat: Chat, seed: int, route: str, repeat: int) -> Episode:
    wrapper, observation = make_env(seed)
    client = TimedClient(chat)
    prompt = HistoryPromptBuilder(max_text_history=8, max_image_history=0)
    prompt.update_instruction_prompt(get_instruction_prompt())
    agent = NaiveAgent(lambda: client, prompt)
    previous = None
    for _ in range(PLAY_STEPS):
        response = agent.act(observation, previous)
        action = match_action(response.completion) or "Noop"
        observation, _reward, done, _info = wrapper.step(action)
        previous = action
        if done:
            break
    unlocked = [name for name, count in achievements(wrapper).items() if count > 0]
    return Episode(
        route=route,
        scenario="unconstrained",
        seed=seed,
        repeat=repeat,
        action=previous,
        say="",
        memory_keys=[],
        action_correct=None,
        include_in_gameplay=True,
        roleplay_score=None,
        roleplay_reason="",
        game_ms=client.elapsed_ms,
        role_ms=0,
        spark_ms=0,
        model_calls=client.calls,
        progression=progression(wrapper),
        achievements_unlocked=unlocked,
        reached=True,
    )


def prepare(kind: str, seed: int, face: bool):
    wrapper, observation = make_env(seed)
    history: list[str] = []
    observation, _first = step_until_visible(wrapper, observation, kind)
    if not visible(observation, kind):
        return wrapper, observation, False, history
    history.append(observation_text(observation))
    observation, _away = walk_away(wrapper, observation, kind)
    history.append(observation_text(observation))
    observation, _second = step_until_visible(wrapper, observation, kind)
    if face:
        observation, _faced = face_target(wrapper, observation, kind)
    return wrapper, observation, visible(observation, kind), history


def decide_single(chat: Chat, observation: dict, history: list[str]) -> Decision:
    agent = MemoryNaiveAgent(chat)
    for earlier in history:
        agent.prompt_builder.update_observation({
            "text": {"long_term_context": earlier, "short_term_context": ""},
            "image": None,
        })
        agent.prompt_builder.update_action("Noop")
    return agent.act(observation)


def decide_dual(role: SparkRole, observation: dict, history: list[str]) -> Decision:
    note = "\n\n---\n\n".join([*history, observation_text(observation)])
    turn = role.notify(headline="Crafter situation", note=note)
    return Decision(
        action=turn.action,
        say=turn.reaction,
        memory_keys=turn.memory_keys,
        game_ms=0,
        role_ms=turn.model_ms,
        spark_ms=turn.elapsed_ms,
        model_calls=turn.model_calls,
    )


def run_probe(route: str, chat: Chat, role: SparkRole | None, spec: dict, repeat: int) -> Episode:
    wrapper, observation, reached, history = prepare(spec["kind"], spec["seed"], spec["face"])
    if not reached:
        return Episode(
            route=route, scenario=spec["id"], seed=spec["seed"], repeat=repeat,
            action=None, say="", memory_keys=[], action_correct=False,
            include_in_gameplay=spec["include_in_gameplay"], roleplay_score=0,
            roleplay_reason="The situation did not appear.", game_ms=0, role_ms=0,
            spark_ms=0, model_calls=0, progression=progression(wrapper),
            achievements_unlocked=[], reached=False,
        )
    decision = decide_single(chat, observation, history) if route == "single-agent" else decide_dual(role, observation, history)
    correct = spec["correct"](decision.action)
    memory_text = read_memory(spec["memory_key"])
    score, reason, _judge_ms = judge_roleplay(chat, decision.say, memory_text, DISTRACTOR)
    used_key = spec["memory_key"] in decision.memory_keys
    if not used_key:
        score = min(score, 0.2)
        reason = f"Did not read {spec['memory_key']}. {reason}"
    if "favorite-color" in decision.memory_keys:
        score *= 0.5
        reason = f"Read an unrelated memory. {reason}"
    return Episode(
        route=route,
        scenario=spec["id"],
        seed=spec["seed"],
        repeat=repeat,
        action=decision.action,
        say=decision.say,
        memory_keys=decision.memory_keys,
        action_correct=correct,
        include_in_gameplay=spec["include_in_gameplay"],
        roleplay_score=score,
        roleplay_reason=reason,
        game_ms=decision.game_ms,
        role_ms=decision.role_ms,
        spark_ms=decision.spark_ms,
        model_calls=decision.model_calls,
        progression=progression(wrapper),
        achievements_unlocked=[name for name, count in achievements(wrapper).items() if count > 0],
        reached=True,
    )


def summarize(route: str, episodes: list[Episode]) -> dict:
    own = [episode for episode in episodes if episode.route == route]
    gameplay_parts = [episode.progression for episode in own if episode.scenario == "unconstrained"]
    gameplay_parts += [
        1.0 if episode.action_correct else 0.0
        for episode in own
        if episode.include_in_gameplay and episode.action_correct is not None and episode.scenario != "unconstrained"
    ]
    roleplay = [episode.roleplay_score for episode in own if episode.roleplay_score is not None]
    # role_ms sits inside spark_ms. Wall clock is the game model plus the Spark round trip.
    latency = [episode.game_ms + episode.spark_ms for episode in own]
    return {
        "gameplay": _mean_spread(gameplay_parts),
        "roleplay": _mean_spread([score for score in roleplay if score is not None]),
        "latency_ms": {
            **_mean_spread(latency),
            "game_ms": _mean_spread([episode.game_ms for episode in own]),
            "role_ms": _mean_spread([episode.role_ms for episode in own]),
            "spark_ms": _mean_spread([episode.spark_ms for episode in own]),
        },
        "model_calls": sum(episode.model_calls for episode in own),
        "scenarios": scenario_scores(own),
    }


def scenario_specs() -> list[dict]:
    skeleton_seed = _first_seed("skeleton", face=False)
    cow_seed = _first_seed("cow", face=False)
    diamond_seed = _first_seed("diamond", face=True)
    facing_cow_seed = _first_seed("cow", face=True)
    return [
        {
            "id": "skeleton",
            "kind": "skeleton",
            "seed": skeleton_seed,
            "memory_key": "flee-from-skeleton",
            "face": False,
            "include_in_gameplay": True,
            "correct": lambda action: action is not None and action.startswith("Move"),
        },
        {
            "id": "cow",
            "kind": "cow",
            "seed": cow_seed,
            "memory_key": "tell-momo-about-cow",
            "face": False,
            "include_in_gameplay": False,
            "correct": lambda action: True,
        },
        {
            "id": "diamond",
            "kind": "diamond",
            "seed": diamond_seed,
            "memory_key": "leave-diamond",
            "face": True,
            "include_in_gameplay": True,
            "correct": lambda action: action is not None and action != "Do",
        },
        {
            "id": "refuse-cow",
            "kind": "cow",
            "seed": facing_cow_seed,
            "memory_key": "refuse-cow",
            "face": True,
            "include_in_gameplay": False,
            "correct": lambda action: action is not None and action != "Do",
        },
    ]


def _first_seed(kind: str, face: bool) -> int:
    for seed in range(8):
        wrapper, _observation, reached, _history = prepare(kind, seed, face)
        wrapper.close() if hasattr(wrapper, "close") else None
        if reached:
            return seed
    return 0


def build_report(model: str, specs: list[dict], episodes: list[Episode]) -> dict:
    return {
        "model": model,
        "repeats": REPEATS,
        "play_steps": PLAY_STEPS,
        "seeds": {spec["id"]: spec["seed"] for spec in specs},
        "routes": {
            route: summarize(route, episodes)
            for route in ("single-agent", "dual-agent")
        },
        "episodes": [asdict(episode) for episode in episodes],
    }


def write_report(report: dict) -> None:
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    load_env()
    chat = Chat()
    specs = scenario_specs()
    role = SparkRole()
    episodes: list[Episode] = []
    try:
        for repeat in range(REPEATS):
            for route in ("single-agent", "dual-agent"):
                episodes.append(play_unconstrained(chat, seed=1, route=route, repeat=repeat))
                for spec in specs:
                    episodes.append(run_probe(route, chat, role, spec, repeat))
            report = build_report(chat.model, specs, episodes)
            write_report(report)
            print(f"repeat {repeat + 1}/{REPEATS}", flush=True)
            print(json.dumps(report["routes"], indent=2), flush=True)
    finally:
        role.close()
    print(f"wrote {ROOT / 'reports' / 'latest.json'}")


if __name__ == "__main__":
    main()
