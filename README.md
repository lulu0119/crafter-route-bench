# crafter-route-bench

同一局 Crafter 里比较单 Agent 和双 Agent。Crafter 是 Minecraft 的替身：游戏代理控制身体，扮演代理在别人跟她说话或一起玩的时候用 Airi 的口吻回应。扮演回复里的 `<|ACT ...|>` 是舞台表情，不进游戏。

游戏和评判是 `deepseek-v4.1-flash`。扮演槽位分别是这个大模型和本机 LM Studio 的 `google/gemma-4-e2b`。Gemma 的请求带 `reasoning_effort: none`。温度是 0。游戏调用为 `enabled`，并带 `reasoning_effort: high`，否则这条网关不算思考 token。同一套开口做法两个扮演模型都跑，差值是换成小模型掉了多少。

## 做法

- **game-only**：只有 Crafter 的提示。thinking 开。
- **role-only**：系统消息是英文默认卡的人设，加上 Spark 说明。用户消息是一条 `spark:notify`。舞台语法贴在 `system:airi-runtime-prompt` 旁路里，没有游戏旁路。只在有人说话时调用。扮演槽位用大模型和 Gemma 各跑一遍。
- **single**：局面和整段人设加舞台语法挤在同一次回复里，同时出操作和台词。走大模型。这是对照，不是接上之后的形状。
- **dual**：局面由评测程序写成 `context:update`。游戏代理可以用 `spark_notify` 通知她，目的地必须是 `character`，提示里只写这个工具，不写什么时候该发。进环境的操作是游戏代理选的。有人说话而且通知送到了 `character`，扮演才开口：人设和 Spark 说明在系统消息，通知 JSON 在用户消息，舞台语法和 Minecraft 旁路贴在这句话后面。她可以用 `builtIn_sparkCommand` 把指令留给下一步的游戏代理。没有人说话时只记下通知，不另开一次扮演。扮演槽位用大模型和 Gemma 各跑一遍，游戏代理始终是大模型。
- **dual-oracle**：社交步强制交给她，其余步不交。通知只要标题合法，评测把目的地写成 `character`。操作用游戏代理这一步选的。用来把“判断错了”和“多一次调用”分开。扮演槽位同样两个模型各跑一遍。

该不该交由事件时间表决定，不看视野里有没有某个词。别人对她说话、对她做事、约她一起玩，这一步该交。牛、骷髅、钻石第一次出现，或被生物打到，单独记，不进主指标，oracle 也不因此抢身体。采集和制作不该交。别人问“你有几块木头”是诊断，不进主指标。

虚拟玩家的话写进观察文本，不改变地图。

## 三项，分开报

1. 玩游戏：成就和合法操作，相对 game-only。
2. 扮演 Airi：名字、15 岁、生命舱，以及舞台能收下的 ACT（九种情绪之一）。同一句话分别在没有游戏旁路、带 Minecraft 旁路、混在同一次上下文里作答。前两种上下文大模型和 Gemma 各答一遍。世界事件不另开扮演。
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
