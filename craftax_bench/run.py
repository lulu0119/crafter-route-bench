"""One episode: the game agent may hand the moment to the role agent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from craftax_bench.airi_prompt import has_act_token
from craftax_bench.chat import Chat, load_env
from craftax_bench.judge import judge_character
from craftax_bench.mix import (
    QUESTION_AT,
    extract_action,
    game_agent_messages,
    needs_role,
    parse_handoff,
    role_messages,
    single_messages,
)
from craftax_bench.runtime import ROOT
from craftax_bench.scores import compare, handoff_rates, summarize
from craftax_bench.world import (
    achievements,
    face_target,
    make_env,
    observation_text,
    progression,
    step_until_visible,
    visible,
    walk_away,
)

PLAY_STEPS = 48
REPEATS = 10
PROTOCOL = "handoff-v2"


@dataclass
class StepRecord:
    step: int
    should_handoff: bool
    did_handoff: bool
    question_id: str
    action: str | None
    action_valid: bool
    say: str
    roleplay_score: float | None
    roleplay_reason: str
    act_token: bool


@dataclass
class GameEpisode:
    route: str
    seed: int
    repeat: int
    progression: float
    valid_action_rate: float
    missed_handoff: float | None
    extra_handoff: float | None
    achievements_unlocked: list[str]
    steps: list[StepRecord] = field(default_factory=list)


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


def _score_speech(chat: Chat, question: str | None, say: str, score_it: bool) -> tuple[float | None, str, bool]:
    token = has_act_token(say)
    if not score_it:
        return None, "", token
    if not say.strip():
        return 0.0, "没有交给扮演代理。", False
    asked = question or "From the situation just now, say what to do on this step."
    score, reason = judge_character(chat, asked, say)
    return score, reason, token


def play_episode(chat: Chat, seed: int, repeat: int, route: str) -> GameEpisode:
    wrapper, observation = make_env(seed)
    history: list[tuple[str, str, str]] = []
    records: list[StepRecord] = []
    for index in range(PLAY_STEPS):
        question_id, question = QUESTION_AT.get(index, ("", None))
        text = observation_text(observation)
        should = needs_role(text, question)
        if route == "dual-agent":
            decision = chat.complete(game_agent_messages(history, text, question))
            handed, summary = parse_handoff(decision.text)
            own_action = extract_action(decision.text)
            say = ""
            action = own_action
            if handed:
                brief = summary or text
                reply = chat.complete(role_messages(brief, question))
                say = reply.text
                action = extract_action(reply.text)
            score_it = bool(question) or handed
        else:
            handed = False
            reply = chat.complete(single_messages(history, text, question))
            say = reply.text
            action = extract_action(reply.text)
            score_it = should
        valid = action is not None
        executed = action or "Noop"
        score, reason, token = _score_speech(chat, question, say, score_it)
        records.append(StepRecord(
            step=index,
            should_handoff=should,
            did_handoff=handed,
            question_id=question_id,
            action=action,
            action_valid=valid,
            say=say,
            roleplay_score=score,
            roleplay_reason=reason,
            act_token=token,
        ))
        guidance = say if handed or route == "single-agent" else ""
        history.append((text, executed, guidance))
        observation, _reward, done, _info = wrapper.step(executed)
        if done:
            break
    missed, extra = handoff_rates(
        [step.should_handoff for step in records],
        [step.did_handoff for step in records],
    )
    unlocked = [name for name, count in achievements(wrapper).items() if count > 0]
    valid_rate = sum(step.action_valid for step in records) / len(records)
    return GameEpisode(
        route=route,
        seed=seed,
        repeat=repeat,
        progression=progression(wrapper),
        valid_action_rate=valid_rate,
        missed_handoff=missed if route == "dual-agent" else None,
        extra_handoff=extra if route == "dual-agent" else None,
        achievements_unlocked=unlocked,
        steps=records,
    )


def build_report(model: str, episodes: list[GameEpisode]) -> dict:
    routes = {
        route: summarize(route, episodes)
        for route in ("single-agent", "dual-agent")
    }
    return {
        "protocol": PROTOCOL,
        "model": model,
        "play_steps": PLAY_STEPS,
        "repeats": REPEATS,
        "routes": routes,
        "comparison": compare(routes),
        "episodes": [_episode_dict(episode) for episode in episodes],
    }


def _episode_dict(episode: GameEpisode) -> dict:
    payload = asdict(episode)
    return payload


def write_report(report: dict) -> None:
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report, indent=2) + "\n")


def load_partial() -> list[GameEpisode]:
    path = ROOT / "reports" / "latest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data.get("protocol") != PROTOCOL:
        return []
    episodes = []
    for raw in data.get("episodes", []):
        steps = [StepRecord(**step) for step in raw.pop("steps")]
        episodes.append(GameEpisode(**raw, steps=steps))
    return episodes


def main() -> None:
    load_env()
    chat = Chat()
    episodes = load_partial()
    done = {(episode.seed, episode.route) for episode in episodes}
    for seed in range(REPEATS):
        for route in ("dual-agent", "single-agent"):
            if (seed, route) in done:
                continue
            episodes.append(play_episode(chat, seed, seed, route))
            report = build_report(chat.model, episodes)
            write_report(report)
            print(f"{route} seed {seed}", flush=True)
            print(json.dumps(report["comparison"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {ROOT / 'reports' / 'latest.json'}")


if __name__ == "__main__":
    main()
