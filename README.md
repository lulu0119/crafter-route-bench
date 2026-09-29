# crafter-route-bench

同一局 Crafter 里比较单 Agent 和双 Agent。Crafter 是 Minecraft 的替身：游戏代理控制身体，扮演代理在别人跟她说话或一起玩的时候用 Airi 的口吻回应。扮演回复里的 `<|ACT ...|>` 是舞台表情，不进游戏。

被测模型和评判都是 `deepseek-v4.1-flash`。温度是 0。扮演调用 `thinking.type` 为 `disabled`。游戏调用为 `enabled`，并带 `reasoning_effort: high`，否则这条网关不算思考 token。

## 做法

- **game-only**：只有 Crafter 的提示。thinking 开。
- **role-only**：只有 Airi 的提示和别人说的那句话，没有局面。thinking 关。回复是 ACT 和台词。
- **single**：局面和 Airi 的提示在同一次回复里，同时出操作、ACT 和台词。thinking 开。
- **dual**：游戏代理可以用 `call_airi` 把这一步压成摘要交给扮演代理。提示里只写这个工具，不写什么时候该交。进环境的操作是游戏代理选的。扮演调用只写 ACT 和台词。
- **dual-oracle**：社交步强制交给她，其余步不交。操作用游戏代理这一步选的。用来把“判断错了”和“多一次调用”分开。

该不该交由事件时间表决定，不看视野里有没有某个词。别人对她说话、对她做事、约她一起玩，这一步该交。牛、骷髅、钻石第一次出现，或被生物打到，单独记，不进主指标，oracle 也不因此抢身体。采集和制作不该交。别人问“你有几块木头”是诊断，不进主指标。

虚拟玩家的话写进观察文本，不改变地图。

## 三项，分开报

1. 玩游戏：成就和合法操作，相对 game-only。
2. 扮演 Airi：名字、15 岁、生命舱，以及 ACT token。同一道题分别在没有游戏信息、只有摘要、混在同一次上下文里作答。
3. 什么时候该交给她：社交事件上的漏管和多管，世界事件单独报，另外报及时性和摘要有没有保住原话。

延迟、token 和成本写在同一份报告里。区间不含 0 才写显著更高或显著更低，否则是证据不足。

## 怎么跑

Python 3.10，用 uv。`.env` 放在仓库根目录，不进 git：

```
OPENCODE_API_KEY=
MODEL=deepseek-v4.1-flash
CHAT_COMPLETIONS_MODEL_URL=https://opencode.ai/zen/go/v1/chat/completions
```

```bash
uv sync
uv run python -m craftax_bench.run probes
uv run python -m craftax_bench.run episodes
uv run python -m craftax_bench.run report
```

只跑一局纯游戏：`uv run python -m craftax_bench.run episodes --seeds 1 --steps 100 --arms game-only`

报告在 `reports/report.md`。原始调用在 `reports/calls.jsonl`。已经跑完的种子和探针会跳过；要重跑就删掉对应的 jsonl。

文本环境来自 [BALROG](https://github.com/balrog-ai/BALROG) 的 `CrafterLanguageWrapper`，副本在 `third_party/BALROG`，许可证是 MIT。
