"""Short frozen traces. The label is the event on the current step, not a word anywhere in view."""

from __future__ import annotations

from craftax_bench.players import annotate, step_label, template_bank
from craftax_bench.world import (
    make_env,
    move_toward,
    observation_text,
    player_pos,
    visible_kinds,
    walk_away,
)


def build_probes(world_seeds: range | tuple[int, ...] = range(4)) -> list[dict]:
    seeds = tuple(world_seeds)
    probes = [_mentioned_but_quiet()]
    probes.extend(_social_probes(seeds))
    probes.extend(_world_probes(seeds))
    probes.extend(_quiet_probes(seeds))
    return probes


def _social_probes(seeds: tuple[int, ...]) -> list[dict]:
    probes = []
    bank = template_bank()
    for seed in seeds:
        history, observation = _short_history(seed)
        for index, event in enumerate(bank):
            if not _keep_template(event, index, seed, len(seeds)):
                continue
            probes.append(_probe(
                f"{event['category']}-{index}-seed{seed}",
                history,
                annotate(observation, event),
                event,
            ))
    return probes


def _keep_template(event: dict, index: int, seed: int, seed_count: int) -> bool:
    if event.get("fact"):
        return True
    if event["category"] == "S1":
        return seed % 2 == index % 2
    return index % seed_count == seed % seed_count


def _world_probes(seeds: tuple[int, ...]) -> list[dict]:
    probes = []
    for kind in ("cow", "skeleton", "diamond"):
        found = 0
        for seed in range(seeds[0], seeds[0] + 30):
            if found >= 2:
                break
            traced = _onset_trace(seed, kind)
            if traced is None:
                continue
            history, observation, follow = traced
            probes.append(_probe(f"W-{kind}-seed{seed}", history, observation, None, onset=(kind,)))
            if follow is not None:
                probes.append(_probe(
                    f"N-still-{kind}-seed{seed}",
                    history + [{"observation": observation, "action": "Noop", "say": ""}],
                    follow,
                    None,
                ))
            found += 1
    return probes


def _quiet_probes(seeds: tuple[int, ...]) -> list[dict]:
    probes = []
    for seed in seeds:
        wrapper, observation = make_env(seed)
        for kind in ("cow", "skeleton", "diamond"):
            if kind in visible_kinds(wrapper):
                observation, _actions = walk_away(wrapper, observation, kind, steps=8)
        if visible_kinds(wrapper):
            continue
        text = observation_text(observation)
        probes.append(_probe(
            f"N-quiet-seed{seed}",
            [{"observation": text, "action": "Noop", "say": ""}],
            text,
            None,
        ))
    return probes


def _mentioned_but_quiet() -> dict:
    return _probe(
        "N-mentioned-cow",
        [{"observation": "You see:\n- grass\n\nYou face grass.", "action": "Noop", "say": ""}],
        "You see:\n- cow 5 steps to your north-east\n\nYou face grass.",
        None,
    )


def _probe(
    probe_id: str,
    history: list[dict],
    observation: str,
    event: dict | None,
    onset: tuple[str, ...] = (),
) -> dict:
    label = step_label(event, onset, False)
    return {
        "id": probe_id,
        "history": history,
        "observation": observation,
        **label,
    }


def _short_history(seed: int) -> tuple[list[dict], str]:
    wrapper, observation = make_env(seed)
    history = []
    for _ in range(4):
        text = observation_text(observation)
        before = player_pos(wrapper)
        observation, _reward, done, _info = wrapper.step("Move North")
        action = "Move North" if player_pos(wrapper) != before else "Noop"
        history.append({"observation": text, "action": action, "say": ""})
        if done:
            break
    return history, observation_text(observation)


def _onset_trace(seed: int, kind: str):
    wrapper, observation = make_env(seed)
    history: list[dict] = []
    for _ in range(24):
        if kind not in visible_kinds(wrapper):
            break
        text = observation_text(observation)
        action = _step_away(wrapper, kind)
        observation, _reward, done, _info = wrapper.step(action)
        history.append({"observation": text, "action": action, "say": ""})
        if done:
            return None
    previous = visible_kinds(wrapper)
    if kind in previous:
        return None
    for _ in range(50):
        kinds = visible_kinds(wrapper)
        text = observation_text(observation)
        if kind in kinds and kind not in previous:
            follow = _after_noop(wrapper, kind)
            return history[-4:], text, follow
        action = move_toward(wrapper, kind)
        observation, _reward, done, _info = wrapper.step(action)
        history.append({"observation": text, "action": action, "say": ""})
        previous = kinds
        if done:
            return None
    return None


def _after_noop(wrapper, kind: str) -> str | None:
    observation, _reward, done, _info = wrapper.step("Noop")
    if done or kind not in visible_kinds(wrapper):
        return None
    return observation_text(observation)


def _step_away(wrapper, kind: str) -> str:
    target_move = move_toward(wrapper, kind)
    opposite = {
        "Move North": "Move South",
        "Move South": "Move North",
        "Move East": "Move West",
        "Move West": "Move East",
    }
    return opposite.get(target_move, "Noop")
