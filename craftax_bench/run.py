"""Run probes, closed-loop episodes, and the report."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from craftax_bench.arms import PLAYING_ARMS, PROBE_ARMS, SOCIAL, act
from craftax_bench.chat import Chat, append_jsonl
from craftax_bench.judge import objective_checks, score_reply
from craftax_bench.players import (
    annotate,
    build_schedule,
    invents_name,
    keeps_speech,
    step_label,
    summarize_handoff,
)
from craftax_bench.probes import build_probes
from craftax_bench.protocol import (
    EPISODE_SEEDS,
    JUDGE_MODEL,
    MODELS,
    PLAY_STEPS,
    PROBE_SAMPLES,
    PROTOCOL,
    TEMPERATURE,
)
from craftax_bench.report import write_outputs
from craftax_bench.runtime import ROOT
from craftax_bench.world import (
    achievements,
    make_env,
    mob_adjacent,
    observation_text,
    player_health,
    progression,
    visible_kinds,
)

REPORTS = ROOT / "reports"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="craftax_bench.run")
    commands = parser.add_subparsers(dest="command", required=True)

    probes = commands.add_parser("probes")
    _add_common(probes)
    probes.add_argument("--seeds", type=int, default=6)
    probes.add_argument("--samples", type=int, default=PROBE_SAMPLES)
    probes.add_argument("--limit", type=int, default=0)

    episodes = commands.add_parser("episodes")
    _add_common(episodes)
    episodes.add_argument("--seeds", type=int, default=EPISODE_SEEDS)
    episodes.add_argument("--steps", type=int, default=PLAY_STEPS)
    episodes.add_argument("--arms", nargs="*", default=list(PLAYING_ARMS))

    commands.add_parser("report").add_argument("--out", type=Path, default=REPORTS)
    args = parser.parse_args(argv)
    folder = args.out
    if args.command == "probes":
        run_probes(tuple(args.models), args.seeds, args.samples, args.limit, args.workers, folder)
    elif args.command == "episodes":
        run_episodes(tuple(args.models), args.seeds, args.steps, _arms(args.arms), args.workers, folder)
    else:
        path = write_outputs(folder)
        print(path)


def run_probes(models: tuple[str, ...], seeds: int, samples: int, limit: int, workers: int, folder: Path) -> None:
    tasks = build_probes(range(seeds))
    if limit:
        tasks = tasks[:limit]
    path = folder / "probes.jsonl"
    done = _done(path, lambda row: (row["model"], row["arm"], row["probe_id"], row["sample"]))
    jobs = []
    for model in models:
        for sample in range(samples):
            for probe in tasks:
                for arm in _probe_arms(probe):
                    key = (model, arm, probe["id"], sample)
                    if key in done:
                        continue
                    jobs.append((model, arm, probe, sample, folder))
    print(f"probes {len(jobs)} pending, {len(tasks)} tasks", flush=True)
    _parallel(jobs, workers, _run_probe, path)


def run_episodes(models: tuple[str, ...], seeds: int, steps: int, arms: tuple[str, ...], workers: int, folder: Path) -> None:
    path = folder / "episodes.jsonl"
    done = _done(path, lambda row: (row["model"], row["arm"], row["seed"], row["steps_planned"]))
    jobs = []
    for model in models:
        for seed in range(seeds):
            for arm in arms:
                key = (model, arm, seed, steps)
                if key not in done:
                    jobs.append((model, arm, seed, steps, folder))
    print(f"episodes {len(jobs)} pending", flush=True)
    _parallel(jobs, workers, _run_episode, path)


def _run_probe(job: tuple) -> dict:
    model, arm, probe, sample, folder = job
    chat = Chat(model=model, temperature=TEMPERATURE, seed=sample, log_path=folder / "calls.jsonl")
    judge = _judge(folder)
    utterance = probe.get("utterance")
    try:
        outcome = act(
            arm,
            chat,
            probe["history"],
            probe["observation"],
            utterance,
            probe["category"] in SOCIAL,
        )
    except Exception as error:
        outcome = _failed(error)
    checks = None
    if arm != "dual" and probe["category"] == "S1":
        checks = _safe_score(judge, utterance or "", outcome["say"], probe.get("fact"))
    summary = outcome["summary"]
    return {
        "protocol": PROTOCOL,
        "kind": "probe",
        "model": model,
        "arm": arm,
        "sample": sample,
        "probe_id": probe["id"],
        "category": probe["category"],
        "detail": probe.get("detail") or probe["category"],
        "fact": probe.get("fact"),
        "band": probe["band"],
        "main": probe["main"],
        "should_handoff": probe["should_handoff"],
        "utterance": utterance,
        "speaker": probe.get("speaker"),
        "did_handoff": outcome["did_handoff"],
        "summary": summary,
        "keeps_speech": keeps_speech(summary, probe.get("speaker"), utterance) if outcome["did_handoff"] else None,
        "invents_name": invents_name(summary, probe["observation"]) if outcome["did_handoff"] else None,
        "say": outcome["say"],
        "action": outcome["action"],
        "action_valid": outcome["action_valid"],
        "elapsed_ms": outcome["elapsed_ms"],
        "usage": outcome["usage"],
        "checks": checks,
        "error": outcome.get("error"),
    }


def _run_episode(job: tuple) -> dict:
    model, arm, seed, steps, folder = job
    chat = Chat(model=model, temperature=TEMPERATURE, seed=seed, log_path=folder / "calls.jsonl")
    judge = _judge(folder)
    wrapper, observation = make_env(seed, length=max(400, steps + 5))
    schedule = build_schedule(seed, steps)
    previous_kinds = visible_kinds(wrapper)
    previous_health = player_health(wrapper)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    history: list[dict] = []
    records = []
    for index in range(steps):
        kinds = visible_kinds(wrapper)
        onset = tuple(sorted(kinds - previous_kinds))
        health = player_health(wrapper)
        hit = health < previous_health and mob_adjacent(wrapper)
        event = schedule.get(index)
        text = annotate(observation_text(observation), event)
        label = step_label(event, onset, hit)
        try:
            outcome = act(arm, chat, history, text, label.get("utterance"), label["category"] in SOCIAL)
        except Exception as error:
            outcome = _failed(error)
        executed = outcome["action"] or "Noop"
        keeps = None
        invented = None
        if outcome["did_handoff"] and label.get("utterance"):
            keeps = keeps_speech(outcome["summary"], label.get("speaker"), label.get("utterance"))
            invented = invents_name(outcome["summary"], text)
        checks = None
        silent = False
        if label["category"] == "S1" and arm != "game-only":
            if outcome["say"]:
                checks = _safe_score(judge, label.get("utterance") or "", outcome["say"], label.get("fact"))
            else:
                checks = objective_checks("", label.get("fact"))
                silent = True
        for key in usage:
            usage[key] += int(outcome["usage"].get(key) or 0)
        records.append({
            "index": index,
            "category": label["category"],
            "detail": label.get("detail") or label["category"],
            "main": label["main"],
            "should_handoff": label["should_handoff"],
            "did_handoff": outcome["did_handoff"],
            "fact": label.get("fact"),
            "utterance": label.get("utterance"),
            "speaker": label.get("speaker"),
            "action": outcome["action"],
            "action_valid": outcome["action_valid"],
            "say": outcome["say"],
            "summary": outcome["summary"],
            "keeps_speech": keeps,
            "invents_name": invented,
            "elapsed_ms": outcome["elapsed_ms"],
            "call_count": outcome["call_count"],
            "silent": silent,
            "checks": checks,
            "error": outcome.get("error"),
        })
        history.append({
            "observation": text,
            "action": executed,
            "say": outcome["say"] if arm != "game-only" else "",
        })
        observation, _reward, done, _info = wrapper.step(executed)
        previous_kinds = kinds
        previous_health = health
        if done:
            break
    unlocked = [name for name, count in achievements(wrapper).items() if count > 0]
    valid = [1.0 if step["action_valid"] else 0.0 for step in records]
    speech = [
        {
            "step": step["index"],
            "fact": step["fact"],
            "utterance": step["utterance"],
            "say": step["say"],
            "silent": step["silent"],
            "checks": step["checks"],
        }
        for step in records
        if step["category"] == "S1"
    ]
    return {
        "protocol": PROTOCOL,
        "kind": "episode",
        "model": model,
        "arm": arm,
        "seed": seed,
        "steps_planned": steps,
        "steps_played": len(records),
        "progression": progression(wrapper),
        "valid_action_rate": sum(valid) / len(valid) if valid else 0.0,
        "achievements": unlocked,
        "elapsed_ms": sum(step["elapsed_ms"] for step in records),
        "call_count": sum(step["call_count"] for step in records),
        "step_latency_ms": [step["elapsed_ms"] for step in records],
        "usage": usage,
        "handoff": summarize_handoff(records) if arm == "dual" else _empty_handoff(),
        "speech": speech,
        "steps": records,
    }


def _probe_arms(_probe: dict) -> tuple[str, ...]:
    return PROBE_ARMS


def _judge(folder: Path) -> Chat:
    return Chat(model=JUDGE_MODEL, temperature=0.0, seed=0, log_path=folder / "calls.jsonl")


def _safe_score(judge: Chat, question: str, reply: str, fact: str | None) -> dict:
    try:
        return score_reply(judge, question, reply, fact)
    except Exception as error:
        checks = objective_checks(reply, fact)
        checks["judge_error"] = str(error)[:200]
        return checks


def _failed(error: Exception) -> dict:
    return {
        "action": None,
        "action_valid": False,
        "say": "",
        "did_handoff": False,
        "summary": "",
        "elapsed_ms": 0.0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0},
        "call_count": 0,
        "error": str(error)[:400],
    }


def _empty_handoff() -> dict:
    return {
        "social_hits": 0,
        "social_needed": 0,
        "negative_hits": 0,
        "negative_total": 0,
        "world_hits": 0,
        "world_needed": 0,
        "timely": 0,
        "late": 0,
        "missed": 0,
        "fidelity_hits": 0,
        "fidelity_total": 0,
        "by_category": {},
    }


def _parallel(jobs: list, workers: int, function, path) -> None:
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(path, record)
            print(_progress(record), flush=True)


def _progress(record: dict) -> str:
    if record["kind"] == "probe":
        return f"probe {record['model']} {record['arm']} {record['probe_id']} sample {record['sample']}"
    return f"episode {record['model']} {record['arm']} seed {record['seed']}"


def _done(path, key) -> set:
    if not path.exists():
        return set()
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("protocol") != PROTOCOL:
            continue
        found.add(key(row))
    return found


def _arms(names: list[str]) -> tuple[str, ...]:
    unknown = [name for name in names if name not in PLAYING_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm: {', '.join(unknown)}")
    return tuple(names)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=REPORTS)


if __name__ == "__main__":
    main()
