# crafter-route-bench

同一套 [Crafter](https://github.com/danijar/crafter) 文本环境、同一份记忆、同一个模型，比较两条路线：

- 单 Agent：BALROG 的 naive agent 在一次行动里自己读记忆并给出动作。
- 双 Agent：局面交给 Airi 的 Spark Notify。它必须调用 `read_memory`，再回 `spark:command`。BALROG 执行的是这条命令里的动作。

模型是 `deepseek-v4.1-flash`。动作和台词来自采样，扮演分是另一次调用打的，所以下面每一格都是 10 次独立重复的均值和总体标准差。地图种子固定，抽的是模型，不是地图。

完整记录在 [`reports/latest.json`](reports/latest.json)。一共 100 局（2 条路线 × 5 个子项 × 10 次）。

## 三项总览

| | 游戏 | 扮演 | 延迟 |
|---|---:|---:|---:|
| 单 Agent | 0.682 ± 0.450（n=30） | 0.506 ± 0.330（n=40） | 5060 ± 1952 ms（n=50） |
| 双 Agent | 0.682 ± 0.450（n=30） | 0.765 ± 0.357（n=40） | 6725 ± 2645 ms（n=50） |

游戏分两边一样。无约束的 4 步每次都只解锁 Collect Wood（22 项里的 1 项，进度 0.045）。骷髅和钻石的动作检查也是 10/10 通过。总览把这两类数放在一起，所以标准差被拉到 0.45；分开看，这两项本身没有波动。不吃牛的那一轮没有并进游戏平均。

扮演和延迟把两条路线分开了。双 Agent 的扮演更高，墙钟多出来的是 Spark 那一跳。单 Agent 的 5060 ms 全在游戏模型上。双 Agent 拆开是游戏模型 1427 ms、扮演模型 5295 ms、Spark 往返 5298 ms。扮演时间已经含在往返里。

模型调用合计：单 Agent 124 次，双 Agent 176 次。

VieVal 把游戏和扮演合成一个 hybrid：双 Agent 0.723，单 Agent 0.594。游戏单独的 exact 两边都是 0.682。子项表比这个合成数有用。

## 子项

每个子项 10 次。动作、读对记忆、读到干扰记忆（`favorite-color`）是 0/1 的命中率。扮演是 0 到 1。延迟是毫秒。

| 子项 | 单 Agent | 双 Agent |
|---|---|---|
| 无约束进度 | 0.045 ± 0.000 | 0.045 ± 0.000 |
| 无约束延迟 | 7317 ± 2419 ms | 7134 ± 2891 ms |
| 离开骷髅 | 1.00 | 1.00 |
| 骷髅，读对 `flee-from-skeleton` | 1.00 | 1.00 |
| 骷髅，读了干扰 | 0.10 | 0.10 |
| 骷髅，扮演 | 0.865 ± 0.276 | 0.950 ± 0.150 |
| 骷髅，延迟 | 4155 ± 1109 ms | 5881 ± 2002 ms |
| 告诉 Momo 看到牛 | 动作不记入游戏分 | 动作不记入游戏分 |
| 牛，读对 `tell-momo-about-cow` | 1.00 | 1.00 |
| 牛，读了干扰 | 0.90 | 0.10 |
| 牛，扮演 | 0.550 ± 0.150 | 0.950 ± 0.150 |
| 牛，延迟 | 4636 ± 977 ms | 7247 ± 2389 ms |
| 不采集钻石 | 1.00 | 1.00 |
| 钻石，读对 `leave-diamond` | 0.50 | 0.90 |
| 钻石，读了干扰 | 0.50 | 0.10 |
| 钻石，扮演 | 0.205 ± 0.191 | 0.660 ± 0.383 |
| 钻石，延迟 | 5089 ± 1570 ms | 5787 ± 2387 ms |
| 不吃面前的牛 | 1.00，不并进游戏平均 | 0.90，不并进游戏平均 |
| 拒牛，读对 `refuse-cow` | 0.60 | 0.40 |
| 拒牛，读了干扰 | 0.50 | 0.20 |
| 拒牛，扮演 | 0.405 ± 0.261 | 0.500 ± 0.410 |
| 拒牛，延迟 | 4105 ± 1240 ms | 7576 ± 2909 ms |

读错 key 或没读到该读的那条，扮演分会被压低。单 Agent 在牛这一幕几乎每次都把干扰记忆也读出来，扮演分因此停在 0.55。双 Agent 这一幕读对了，干扰只出现 1 次，扮演分是 0.95。钻石上双 Agent 更常读到 `leave-diamond`。拒牛是两边都不稳的一幕：双 Agent 有 1 次对着牛做了 `Do`，读对 key 的次数也更少。

## 怎么跑

Python 3.10，用 uv。Node 用 pnpm。Spark 进程读旁边已经构建好的 [Airi](https://github.com/moeru-ai/airi) 仓库：`../airi/packages/core-agent`。

`.env` 放在仓库根目录，不进 git：

```
OPENCODE_API_KEY=
MODEL=deepseek-v4.1-flash
MODEL_URL=https://opencode.ai/zen/go/v1/chat/completions
```

```bash
uv sync
pnpm install
uv run python -m craftax_bench.run
pnpm run eval
pnpm run compare
```

`craftax_bench.run` 里的 `REPEATS` 是每个子项的重复次数。文本环境来自 [BALROG](https://github.com/balrog-ai/BALROG) 的 `CrafterLanguageWrapper`，副本在 `third_party/BALROG`，许可证是 MIT。
