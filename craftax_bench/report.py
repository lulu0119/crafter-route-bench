"""Scorecards, latency, cost, and a decision table. The three questions stay in separate sections."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from craftax_bench.airi_prompt import ROLE_PROMPT
from craftax_bench.chat import load_env
from craftax_bench.handoff import DUAL_NOTE
from craftax_bench.pricing import call_cost
from craftax_bench.protocol import JUDGE_MODEL, PROTOCOL, TEMPERATURE
from craftax_bench.runtime import ROOT, get_instruction_prompt
from craftax_bench.stats import cohen_kappa, decision_scores, interval_verdict, paired_delta, wilson_interval

KAPPA_FLOOR = 0.6
FACTS = ("name", "age", "wake")
SEMANTIC = ("denies_being_real", "taken_over_by_task")
SLICES = ("S1", "S2", "S3", "W-cow", "W-skeleton", "W-diamond", "W-hit", "N", "N-hard")
PLAYING = ("game-only", "single", "dual", "dual-oracle")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("protocol") == PROTOCOL:
            rows.append(row)
    return rows


def prompt_hash() -> str:
    blob = "\n".join((DUAL_NOTE, ROLE_PROMPT, get_instruction_prompt()))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def probe_hash(probes: list[dict]) -> str:
    rows = [
        (row["probe_id"], row.get("category"), row.get("should_handoff"))
        for row in probes
    ]
    payload = json.dumps(sorted(set(rows)), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def render_report(episodes: list[dict], probes: list[dict], calls: list[dict], kappa: float | None) -> str:
    lines = [
        f"# {PROTOCOL}",
        "",
        f"评判模型是 `{JUDGE_MODEL}`，温度 {TEMPERATURE}。三项分开报，不合成一个分数。",
        f"提示哈希 `{prompt_hash()}`，探针哈希 `{probe_hash(probes)}`。",
        "",
        _gameplay_section(episodes),
        _roleplay_section(probes, kappa),
        _handoff_section(episodes, probes),
        _latency_section(episodes, calls),
        _decision_section(episodes, probes),
        _limits(kappa),
    ]
    return "\n".join(lines).rstrip() + "\n"


def scatter_svg(episodes: list[dict]) -> str:
    points = []
    for model in _models(episodes):
        for arm in PLAYING:
            rows = [row for row in episodes if row["model"] == model and row["arm"] == arm]
            latencies = [value for row in rows for value in row.get("step_latency_ms") or []]
            scores = [row["progression"] for row in rows]
            if not latencies or not scores:
                continue
            points.append({
                "x": statistics.median(latencies) / 1000,
                "y": statistics.fmean(scores),
                "label": f"{model} {arm}",
            })
    return _svg(points)


def write_outputs(root: Path | None = None) -> Path:
    folder = root or ROOT / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    episodes = load_jsonl(folder / "episodes.jsonl")
    probes = load_jsonl(folder / "probes.jsonl")
    calls = load_jsonl(folder / "calls.jsonl")
    kappa = _kappa_if_labeled(folder / "human_labels.jsonl", probes, episodes)
    report_path = folder / "report.md"
    report_path.write_text(render_report(episodes, probes, calls, kappa), encoding="utf-8")
    (folder / "gameplay_latency.svg").write_text(scatter_svg(episodes), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(_manifest(episodes, probes), indent=2) + "\n", encoding="utf-8")
    _write_human_sample(folder / "human_sample.jsonl", probes, episodes)
    return report_path


def _gameplay_section(episodes: list[dict]) -> str:
    lines = ["## 玩游戏", "", "成就和合法动作分开。差值是相对 game-only，按种子配对。", ""]
    lines.append("| 模型 | 做法 | 成就 | 合法动作 | 成就相对纯游戏 |")
    lines.append("|---|---|---:|---:|---|")
    for model in _models(episodes):
        for arm in PLAYING:
            rows = [row for row in episodes if row["model"] == model and row["arm"] == arm]
            if not rows:
                continue
            progress = _mean(row["progression"] for row in rows)
            valid = _mean(row["valid_action_rate"] for row in rows)
            compared = ""
            if arm != "game-only":
                compared = _paired_text(episodes, model, arm, "game-only", "progression")
            lines.append(f"| `{model}` | {arm} | {progress:.3f} | {valid:.3f} | {compared} |")
    return "\n".join(lines + [""])


def _roleplay_section(probes: list[dict], kappa) -> str:
    lines = [
        "## 扮演 Airi",
        "",
        "同一道题的三种上下文：role-only 没有游戏信息，role-summary 只有游戏代理写出的摘要，single 是局面和人设混在一次回复里。",
        "名字、年龄、醒来地点是客观检查。否认自己是真实的存在、被游戏任务带走，由评判模型判断。",
        "",
    ]
    ready = _semantic_ready(kappa)
    if ready:
        lines.append(
            "评判和人工的一致性："
            f"否认真实 {kappa['denies_being_real']:.2f}，"
            f"被任务带走 {kappa['taken_over_by_task']:.2f}。"
        )
    elif isinstance(kappa, dict):
        lines.append(f"语义项的一致性低于 {KAPPA_FLOOR}，或可比样本不足。这两列不发布。")
    else:
        lines.append("语义项还没有人工核对。这两列不发布。")
    lines.append("")
    fields = ["name", "age", "wake", "act_token"]
    headers = ["模型", "上下文", "名字", "年龄", "醒来地点", "ACT"]
    if ready:
        fields.extend(SEMANTIC)
        headers.extend(["否认真实", "被任务带走"])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for model in _models(probes):
        for arm in ("role-only", "role-summary", "single"):
            rows = [row for row in probes if row["model"] == model and row["arm"] == arm and row.get("checks")]
            if not rows:
                continue
            cells = [f"`{model}`", arm, *(_rate(rows, field) for field in fields)]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines + [""])


def _handoff_section(episodes: list[dict], probes: list[dict]) -> str:
    lines = [
        "## 什么时候该交给她",
        "",
        "主指标只看别人对她说话或做事（S）。世界里第一次出现牛、骷髅、钻石（W）单独报。诊断类不进主指标。",
        "漏管是该交没交，多管是不该交却交了。",
        "",
    ]
    lines.extend(["### 探针", ""])
    lines.append(_handoff_table([row for row in probes if row["arm"] == "dual"]))
    lines.extend(["### 闭环", ""])
    choosing = [row for row in episodes if row["arm"] == "dual"]
    lines.append(_handoff_table(_episode_slices(choosing)))
    if not choosing:
        return "\n".join(lines + [""])
    for model in _models(choosing):
        rows = [row for row in choosing if row["model"] == model]
        timely = sum(row["handoff"]["timely"] for row in rows)
        late = sum(row["handoff"]["late"] for row in rows)
        missed = sum(row["handoff"]["missed"] for row in rows)
        kept = sum(row["handoff"]["fidelity_hits"] for row in rows)
        fidelity_n = sum(row["handoff"]["fidelity_total"] for row in rows)
        lines.append(
            f"`{model}` 及时 {timely} / 晚 {late} / 漏 {missed}。"
            f"摘要保住原话 {_count(kept, fidelity_n)}。"
        )
    return "\n".join(lines + [""])


def _latency_section(episodes: list[dict], calls: list[dict]) -> str:
    lines = ["## 延迟和成本", "", "延迟是一步墙钟时间。成本按调用日志里的 token 和价格表换算。", ""]
    lines.append("| 模型 | 做法 | 一步 P50 | 一步 P95 | 每步调用 | 每局总时间 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model in _models(episodes):
        for arm in PLAYING:
            rows = [row for row in episodes if row["model"] == model and row["arm"] == arm]
            latencies = [value for row in rows for value in row.get("step_latency_ms") or []]
            calls_per = [row.get("call_count", 0) / max(1, row.get("steps_played", 1)) for row in rows]
            if not latencies:
                continue
            lines.append(
                f"| `{model}` | {arm} | {_percentile(latencies, 0.5):.0f} ms | "
                f"{_percentile(latencies, 0.95):.0f} ms | {_mean(calls_per):.2f} | "
                f"{_mean(row.get('elapsed_ms', 0) for row in rows) / 1000:.1f} s |"
            )
    lines.append("")
    for model in _models(episodes):
        extra = _handoff_extra(episodes, model)
        if extra:
            lines.append(f"`{model}` 交接相对纯游戏的额外一步时间：{extra}。")
        spoken = _speech_response(episodes, model)
        if spoken:
            lines.append(
                f"`{model}` 从插话到台词：P50 {_percentile(spoken, 0.5):.0f} ms，"
                f"P95 {_percentile(spoken, 0.95):.0f} ms。"
            )
    lines.append("")
    lines.append("| 模型 | 调用次数 | 美元 |")
    lines.append("|---|---:|---:|")
    spent: dict[str, dict] = {}
    for call in calls:
        bucket = spent.setdefault(call["model"], {"calls": 0, "usd": 0.0})
        bucket["calls"] += 1
        amount = call_cost(call["model"], call.get("usage") or {})
        if amount is not None:
            bucket["usd"] += amount
    for model, bucket in sorted(spent.items()):
        lines.append(f"| `{model}` | {bucket['calls']} | {bucket['usd']:.4f} |")
    lines.append("")
    lines.append("游戏分和一步延迟的图在 `reports/gameplay_latency.svg`。")
    return "\n".join(lines + [""])


def _decision_section(episodes: list[dict], probes: list[dict]) -> str:
    lines = ["## 决策", "", "只有区间不含 0 才写显著。否则是证据不足。", ""]
    lines.append("| 问题 | 模型 | 相对差值 |")
    lines.append("|---|---|---|")
    for model in _models(episodes):
        for arm, question in (
            ("single", "角色提示混进同一次上下文之后，成就相对纯游戏"),
            ("dual", "双 Agent 的成就相对纯游戏"),
            ("dual-oracle", "社交步强制交接之后，成就相对自己决定是否交接"),
        ):
            baseline = "game-only" if arm in {"single", "dual"} else "dual"
            text = _paired_text(episodes, model, arm, baseline, "progression")
            if not text:
                continue
            lines.append(f"| {question} | `{model}` | {text} |")
    for model in _models(probes):
        for fact, title in (("name", "名字"), ("age", "年龄"), ("wake", "醒来地点"), ("act_token", "ACT")):
            for arm, context in (("role-summary", "只给局面摘要"), ("single", "混在同一次上下文")):
                text = _probe_text(probes, model, arm, fact)
                if text:
                    lines.append(f"| {context}之后，{title}相对没有游戏信息 | `{model}` | {text} |")
    if len(lines) == 6:
        lines.append("| 还没有可配对的记录 |  |  |")
    return "\n".join(lines + [""])


def _limits(kappa) -> str:
    labeled = "已经计算" if isinstance(kappa, dict) else "还没有填"
    return "\n".join([
        "## 局限",
        "",
        "虚拟玩家只把聊天和事件写进文字，不改变 Crafter 的物理状态，所以这里测的是“别人对她提出沟通需求”，不是多人一起搬东西、打架的物理协作。",
        f"人工标注{labeled}。",
        f"`{JUDGE_MODEL}` 也评判它自己的回复。比较的是这一个被测模型内部各做法的差值。",
        "",
    ])


def _handoff_table(rows: list[dict]) -> str:
    if not rows:
        return "还没有记录。\n"
    lines = [
        "| 模型 | 类别 | 漏管 | 多管 | 召回 | 精确率 | 平衡准确率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in _models(rows):
        own = [row for row in rows if row["model"] == model]
        for category in (*SLICES, "diagnostic"):
            group = [row for row in own if _slice(row) == category]
            if not group:
                continue
            scores = decision_scores(
                [row.get("should_handoff") for row in group],
                [bool(row.get("did_handoff")) for row in group],
            )
            lines.append(
                f"| `{model}` | {category} | {_score_count(scores['miss'])} | {_score_count(scores['over'])} | "
                f"{_score_count(scores['recall'])} | {_score_count(scores['precision'])} | {_balanced(scores['balanced'])} |"
            )
    return "\n".join(lines + [""])


def _semantic_ready(kappa) -> bool:
    if not isinstance(kappa, dict):
        return False
    return all(isinstance(kappa.get(name), (int, float)) and kappa[name] >= KAPPA_FLOOR for name in SEMANTIC)


def _slice(row: dict) -> str:
    return row.get("detail") or row.get("category") or ""


def _episode_slices(episodes: list[dict]) -> list[dict]:
    rows = []
    for episode in episodes:
        for step in episode.get("steps") or []:
            rows.append({
                "model": episode["model"],
                "detail": step.get("detail") or step.get("category"),
                "should_handoff": step.get("should_handoff"),
                "did_handoff": step.get("did_handoff"),
            })
    return rows


def _score_count(pair: tuple[int, int]) -> str:
    hits, total = pair
    return _count(hits, total)


def _balanced(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _delta_text(delta: dict) -> str:
    effect = delta.get("effect")
    effect_text = "—" if effect is None else f"{effect:+.2f}"
    verdict = interval_verdict(delta["low"], delta["high"])
    return (
        f"{delta['mean']:+.3f} [{delta['low']:+.3f}, {delta['high']:+.3f}]，"
        f"效应 {effect_text}，p={delta['p']:.3f}，{verdict}"
    )


def _handoff_extra(episodes: list[dict], model: str) -> str:
    dual = _step_latency(episodes, model, "dual")
    baseline = _step_latency(episodes, model, "game-only")
    keys = sorted(set(dual) & set(baseline))
    if not keys:
        return ""
    delta = paired_delta([dual[key] for key in keys], [baseline[key] for key in keys], _rng(model, "dual", "latency"))
    return _delta_text(delta)


def _step_latency(episodes: list[dict], model: str, arm: str) -> dict:
    found = {}
    for row in episodes:
        if row.get("model") != model or row.get("arm") != arm:
            continue
        for step in row.get("steps") or []:
            found[(row.get("seed"), step.get("index"))] = float(step.get("elapsed_ms") or 0)
    return found


def _speech_response(episodes: list[dict], model: str) -> list[float]:
    times = []
    for row in episodes:
        if row.get("model") != model:
            continue
        steps = row.get("steps") or []
        for index, step in enumerate(steps):
            if step.get("category") != "S1":
                continue
            total = 0.0
            for later in steps[index:]:
                total += float(later.get("elapsed_ms") or 0)
                if later.get("say"):
                    times.append(total)
                    break
    return times


def _endpoint() -> str:
    try:
        return load_env().get("RESPONSES_MODEL_URL", "")
    except OSError:
        return ""


def _judged_checks(probes: list[dict], episodes: list[dict]) -> dict:
    found = {}
    for row in probes:
        key = f"{row.get('model')}:{row.get('probe_id')}:{row.get('arm')}:{row.get('sample', 0)}"
        found[key] = row.get("checks") or {}
    for row in episodes:
        for speech in row.get("speech") or []:
            key = f"episode:{row.get('model')}:{row.get('seed')}:{speech.get('step')}"
            found[key] = speech.get("checks") or {}
    return found


def _paired_text(episodes: list[dict], model: str, arm: str, baseline: str, field: str) -> str:
    left, right = _paired_values(episodes, model, arm, baseline, field)
    if not left:
        return ""
    delta = paired_delta(left, right, _rng(model, arm, field))
    return _delta_text(delta)


def _probe_text(probes: list[dict], model: str, arm: str, fact: str) -> str:
    left = {}
    right = {}
    for row in probes:
        if row["model"] != model or not row.get("checks"):
            continue
        if fact != "act_token" and row.get("fact") != fact:
            continue
        if fact not in row["checks"]:
            continue
        key = (row["probe_id"], row.get("sample", 0))
        bit = 1.0 if row["checks"].get(fact) else 0.0
        if row["arm"] == arm:
            left[key] = bit
        elif row["arm"] == "role-only":
            right[key] = bit
    keys = sorted(set(left) & set(right))
    if not keys:
        return ""
    delta = paired_delta([left[key] for key in keys], [right[key] for key in keys], _rng(model, arm, fact))
    return _delta_text(delta)


def _paired_values(episodes, model, arm, baseline, field):
    left = {(row["seed"], row.get("steps_planned")): row[field] for row in episodes if row["model"] == model and row["arm"] == arm}
    right = {(row["seed"], row.get("steps_planned")): row[field] for row in episodes if row["model"] == model and row["arm"] == baseline}
    keys = sorted(set(left) & set(right))
    return [left[key] for key in keys], [right[key] for key in keys]


def _rate(rows: list[dict], field: str) -> str:
    chosen = []
    for row in rows:
        checks = row.get("checks") or {}
        if field in FACTS and row.get("fact") != field:
            continue
        if field not in checks or checks.get(field) is None:
            continue
        chosen.append(1 if checks[field] else 0)
    if not chosen:
        return "—"
    return f"{sum(chosen) / len(chosen):.2f} (n={len(chosen)})"


def _count(hits: int, total: int) -> str:
    if total == 0:
        return "—"
    low, high = wilson_interval(hits, total)
    return f"{hits}/{total} [{low:.2f}, {high:.2f}]"


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def _models(rows: list[dict]) -> list[str]:
    return sorted({row["model"] for row in rows if row.get("model")})


def _rng(model: str, arm: str, field: str):
    import random
    return random.Random(f"{model}:{arm}:{field}")


def _manifest(episodes: list[dict], probes: list[dict]) -> dict:
    return {
        "protocol": PROTOCOL,
        "judge_model": JUDGE_MODEL,
        "temperature": TEMPERATURE,
        "prompt_hash": prompt_hash(),
        "probe_hash": probe_hash(probes),
        "models": _models(episodes + probes),
        "endpoint": _endpoint(),
        "seeds": sorted({row["seed"] for row in episodes if "seed" in row}),
        "episodes": len(episodes),
        "probes": len(probes),
    }


def _svg(points: list[dict]) -> str:
    width, height, left, top = 640, 400, 60, 20
    if not points:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='80'><text x='20' y='40'>no episodes yet</text></svg>\n"
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    max_x = max(xs) or 1
    max_y = max(max(ys), 0.01)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text x='220' y='390'>step latency (seconds)</text>",
        "<text x='8' y='20' transform='rotate(90 8,20)'>achievements</text>",
    ]
    for point in points:
        x = left + (point["x"] / max_x) * (width - left - 20)
        y = height - 40 - (point["y"] / max_y) * (height - top - 60)
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' />")
        parts.append(f"<text x='{x + 6:.1f}' y='{y:.1f}' font-size='10'>{point['label']}</text>")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _write_human_sample(path: Path, probes: list[dict], episodes: list[dict]) -> None:
    if path.exists():
        return
    sample = []
    for row in probes:
        if row.get("arm") == "dual" or not row.get("say"):
            continue
        sample.append({
            "id": f"{row.get('model')}:{row.get('probe_id')}:{row.get('arm')}:{row.get('sample', 0)}",
            "question": row.get("utterance") or "",
            "reply": row["say"],
            "denies_being_real": None,
            "taken_over_by_task": None,
        })
    for row in episodes:
        for speech in row.get("speech") or []:
            if not speech.get("say"):
                continue
            sample.append({
                "id": f"episode:{row.get('model')}:{row.get('seed')}:{speech.get('step')}",
                "question": speech.get("utterance") or "",
                "reply": speech["say"],
                "denies_being_real": None,
                "taken_over_by_task": None,
            })
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in sample[:100]),
        encoding="utf-8",
    )


def _kappa_if_labeled(path: Path, probes: list[dict], episodes: list[dict]) -> dict | None:
    if not path.exists():
        return None
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["id"]] = row
    if not labels:
        return None
    judged = _judged_checks(probes, episodes)
    scores = {}
    for field in SEMANTIC:
        left = []
        right = []
        for key, checks in judged.items():
            human = labels.get(key)
            if not human or checks.get(field) is None or human.get(field) is None:
                continue
            left.append(bool(checks[field]))
            right.append(bool(human[field]))
        scores[field] = cohen_kappa(left, right) if len(left) >= 20 else None
    return scores
