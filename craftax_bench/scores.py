"""Per-task means. Handoff mistakes stay separate from gameplay and roleplay."""

from __future__ import annotations

import statistics


def mean_spread(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "spread": 0.0, "n": 0}
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.fmean(values), "spread": spread, "n": len(values)}


def mixture_effect(single_mean: float, dual_mean: float) -> str:
    if single_mean > dual_mean:
        return "提升"
    if single_mean < dual_mean:
        return "下降"
    return "持平"


def handoff_rates(should: list[bool], did: list[bool]) -> tuple[float, float]:
    missed_flags = [not happened for needed, happened in zip(should, did) if needed]
    extra_flags = [happened for needed, happened in zip(should, did) if not needed]
    missed = sum(missed_flags) / len(missed_flags) if missed_flags else 0.0
    extra = sum(extra_flags) / len(extra_flags) if extra_flags else 0.0
    return missed, extra


def summarize(route: str, episodes: list) -> dict:
    own = [episode for episode in episodes if episode.route == route]
    roleplay = [
        step.roleplay_score
        for episode in own
        for step in episode.steps
        if step.roleplay_score is not None
    ]
    summary = {
        "gameplay": mean_spread([episode.progression for episode in own]),
        "valid_action": mean_spread([episode.valid_action_rate for episode in own]),
        "roleplay": mean_spread(roleplay),
        "act_token": mean_spread([
            1.0 if step.act_token else 0.0
            for episode in own
            for step in episode.steps
            if step.roleplay_score is not None
        ]),
    }
    if route == "dual-agent":
        summary["missed_handoff"] = mean_spread([
            episode.missed_handoff for episode in own if episode.missed_handoff is not None
        ])
        summary["extra_handoff"] = mean_spread([
            episode.extra_handoff for episode in own if episode.extra_handoff is not None
        ])
    return summary


def compare(routes: dict) -> dict:
    single = routes["single-agent"]
    dual = routes["dual-agent"]
    compared = {}
    for name in ("gameplay", "valid_action", "roleplay", "act_token"):
        compared[name] = {
            "single": single[name]["mean"],
            "dual": dual[name]["mean"],
            "delta": single[name]["mean"] - dual[name]["mean"],
            "mixed": mixture_effect(single[name]["mean"], dual[name]["mean"]),
        }
    compared["missed_handoff"] = dual.get("missed_handoff")
    compared["extra_handoff"] = dual.get("extra_handoff")
    return compared
