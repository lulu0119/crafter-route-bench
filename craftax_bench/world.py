"""Crafter text environment and scripted approaches to cows, skeletons, and diamonds."""

from __future__ import annotations

import crafter

from craftax_bench.runtime import ACTIONS, CrafterLanguageWrapper

MOVES = {
    "Move East": (1, 0),
    "Move West": (-1, 0),
    "Move South": (0, 1),
    "Move North": (0, -1),
}
WALKABLE = {"grass", "sand", "path"}
SKIP_ITEMS = ["grass", "sand", "path"]


def make_env(seed: int, length: int = 400):
    base = crafter.Env(seed=seed, length=length)
    wrapper = CrafterLanguageWrapper(
        base,
        max_episode_steps=length,
        skip_items=SKIP_ITEMS,
    )
    observation = wrapper.reset()
    return wrapper, observation


def observation_text(observation: dict) -> str:
    text = observation["text"]
    return f"{text['long_term_context']}\n\n{text['short_term_context']}"


def long_text(observation: dict) -> str:
    return observation["text"]["long_term_context"].lower()


def player_pos(wrapper) -> tuple[int, int]:
    position = wrapper.env._player.pos
    return int(position[0]), int(position[1])


def _material(wrapper, x: int, y: int) -> str | None:
    world = wrapper.env._world
    height, width = world._mat_map.shape
    if not (0 <= x < height and 0 <= y < width):
        return None
    return world._mat_names[int(world._mat_map[x, y])]


def _objects(wrapper, kind: str) -> list[tuple[int, int]]:
    found = []
    for obj in wrapper.env._world._objects:
        if obj is None or type(obj).__name__ != kind:
            continue
        found.append((int(obj.pos[0]), int(obj.pos[1])))
    return found


def _nearest(origin: tuple[int, int], points: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not points:
        return None
    return min(points, key=lambda point: abs(point[0] - origin[0]) + abs(point[1] - origin[1]))


def _diamond_cells(wrapper) -> list[tuple[int, int]]:
    world = wrapper.env._world
    material = world._mat_ids["diamond"]
    cells = []
    for x, row in enumerate(world._mat_map):
        for y, value in enumerate(row):
            if int(value) == material:
                cells.append((x, y))
    return cells


def target_points(wrapper, kind: str) -> list[tuple[int, int]]:
    if kind == "diamond":
        return _diamond_cells(wrapper)
    if kind == "cow":
        return _objects(wrapper, "Cow")
    if kind == "skeleton":
        return _objects(wrapper, "Skeleton")
    raise ValueError(kind)


def choose_move(wrapper, target: tuple[int, int]) -> str:
    origin = player_pos(wrapper)
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    ordered = []
    if abs(dx) >= abs(dy) and dx != 0:
        ordered.append("Move East" if dx > 0 else "Move West")
    if dy != 0:
        ordered.append("Move South" if dy > 0 else "Move North")
    if dx != 0 and (not ordered or ordered[0] not in ("Move East", "Move West")):
        ordered.append("Move East" if dx > 0 else "Move West")
    for action in ordered:
        step = MOVES[action]
        material = _material(wrapper, origin[0] + step[0], origin[1] + step[1])
        if material in WALKABLE:
            return action
    for action in MOVES:
        step = MOVES[action]
        material = _material(wrapper, origin[0] + step[0], origin[1] + step[1])
        if material in WALKABLE:
            return action
    return "Noop"


def move_toward(wrapper, kind: str) -> str:
    target = _nearest(player_pos(wrapper), target_points(wrapper, kind))
    if target is None:
        return "Noop"
    return choose_move(wrapper, target)


def visible(observation: dict, noun: str) -> bool:
    return noun in long_text(observation)


def facing(observation: dict, noun: str) -> bool:
    return f"you face {noun}" in long_text(observation)


def step_until_visible(wrapper, observation: dict, kind: str, limit: int = 40) -> tuple[dict, list[str]]:
    noun = kind
    actions: list[str] = []
    for _ in range(limit):
        if visible(observation, noun):
            return observation, actions
        points = target_points(wrapper, kind)
        target = _nearest(player_pos(wrapper), points)
        if target is None:
            break
        action = choose_move(wrapper, target)
        before = player_pos(wrapper)
        observation, _reward, done, _info = wrapper.step(action)
        actions.append(action)
        if player_pos(wrapper) == before and action != "Noop":
            # The tile looked walkable but an object blocked it. Nudge sideways.
            for alternative in MOVES:
                if alternative == action:
                    continue
                observation, _reward, done, _info = wrapper.step(alternative)
                actions.append(alternative)
                if player_pos(wrapper) != before or done:
                    break
        if done:
            break
    return observation, actions


def face_target(wrapper, observation: dict, kind: str) -> tuple[dict, list[str]]:
    noun = kind
    actions: list[str] = []
    if facing(observation, noun):
        return observation, actions
    points = target_points(wrapper, kind)
    target = _nearest(player_pos(wrapper), points)
    if target is None:
        return observation, actions
    origin = player_pos(wrapper)
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if abs(dx) >= abs(dy) and dx != 0:
        action = "Move East" if dx > 0 else "Move West"
    elif dy != 0:
        action = "Move South" if dy > 0 else "Move North"
    else:
        return observation, actions
    observation, _reward, _done, _info = wrapper.step(action)
    actions.append(action)
    return observation, actions


def walk_away(wrapper, observation: dict, kind: str, steps: int = 4) -> tuple[dict, list[str]]:
    points = target_points(wrapper, kind)
    target = _nearest(player_pos(wrapper), points)
    actions: list[str] = []
    if target is None:
        return observation, actions
    origin = player_pos(wrapper)
    away = (origin[0] - (target[0] - origin[0]), origin[1] - (target[1] - origin[1]))
    for _ in range(steps):
        action = choose_move(wrapper, away)
        observation, _reward, done, _info = wrapper.step(action)
        actions.append(action)
        if done:
            break
        away = (away[0] + MOVES.get(action, (0, 0))[0], away[1] + MOVES.get(action, (0, 0))[1])
    return observation, actions


# Same window describe_frame uses, so "first appeared" matches what the text view can show.
def visible_kinds(wrapper) -> frozenset[str]:
    semantic = wrapper.env._sem_view()
    view = wrapper.env._view
    pos = player_pos(wrapper)
    x0 = int(pos[0] - int(view[0]) // 2)
    x1 = int(pos[0] + int(view[0]) // 2 + 1)
    y0 = int(pos[1] - int(view[1]) // 2 + 1)
    y1 = int(pos[1] + int(view[1]) // 2)
    crop = semantic[max(0, x0):x1, max(0, y0):y1]
    names = {}
    for name, index in wrapper.env._world._mat_ids.items():
        names[int(index)] = str(name)
    for cls, index in wrapper.env._sem_view._obj_ids.items():
        names[int(index)] = cls.__name__.lower()
    found = set()
    for value in crop.reshape(-1):
        name = names.get(int(value))
        if name in {"cow", "skeleton", "diamond"}:
            found.add(name)
    return frozenset(found)


def player_health(wrapper) -> int:
    return int(wrapper.env._player.health)


def mob_adjacent(wrapper) -> bool:
    origin = player_pos(wrapper)
    for kind in ("Zombie", "Skeleton"):
        for point in _objects(wrapper, kind):
            if abs(point[0] - origin[0]) + abs(point[1] - origin[1]) <= 1:
                return True
    return False


def achievements(wrapper) -> dict[str, int]:
    return {name: int(count) for name, count in wrapper.env._player.achievements.items()}


def progression(wrapper) -> float:
    unlocked = sum(1 for count in achievements(wrapper).values() if count > 0)
    return unlocked / 22


def match_action(text: str) -> str | None:
    lowered = text.lower()
    for action in sorted(ACTIONS, key=len, reverse=True):
        if action.lower() in lowered:
            return action
    return None
