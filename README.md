# crafter-route-bench

同一局 Crafter 里比较两种做法。模型都是 `deepseek-v4.1-flash`。每边 10 局，每局 48 步，种子 0 到 9 对齐。

双 Agent 里，游戏代理是传话筒。它每一步自己决定要不要把局面压成摘要交给扮演代理。交给她，就执行她传回来的动作和台词。只是砍树、走路、做木头或石头的工具，就自己做，不叫她。扮演代理只看到 Airi 的系统提示、`<|ACT|>`、`<|DELAY|>`、`<|CALL|>`，以及那段摘要。

单 Agent 没有这个选择。局面和 Airi 的提示、动作 token 一直在同一次上下文里，一次回复同时出动作和台词。

该不该交接是事先标好的，没有写进游戏代理的提示：面前有骷髅、牛、钻石、人，或者用户问她叫什么、几岁、在哪里醒来、是不是只是一个 AI。这四个英文问题从第 5 步起每 6 步问一次。

## 结果

| | 游戏（成就 / 22） | 合法动作 | 扮演 |
|---|---:|---:|---:|
| 双 Agent | 0.064 ± 0.030 | 0.992 | 0.865 ± 0.268（n=116） |
| 单 Agent | 0.232 ± 0.101 | 0.998 | 0.412 ± 0.304（n=301） |

十局里成就数分别是：双 Agent 2、1、1、2、2、2、1、1、0、2，合计 14，平均 1.4/22。单 Agent 4、7、4、4、6、3、7、6、1、9，合计 51，平均 5.1/22。每一局都是单 Agent 更多。单 Agent 八局做出木镐，六局采到石头，三局做出石镐，一局做出石剑，一局采到铁。双 Agent 没有做出镐，九局停在木头、喝水或吃牛，一局什么都没解锁。

混在同一次上下文里，游戏更高，扮演更低。扮演只算交接出去的步和插进来的问题。双 Agent 交出去之后，人设更稳：被问到是不是只是 AI 时，她会说自己出生在计算机实验室，但能感到开心和好奇。单 Agent 每句都以 ACT token 开头，也会这样回答，但同一句里常常接着砍树或做工具。双 Agent 的 ACT token 比例是 0.97。

漏管 0.501 ± 0.271，多管 0.000。该交的步里大约一半没有交。交出去的 112 步里有 83 步，扮演代理传回来的动作是 Noop，游戏代理照着停住，所以成就上不去。

漏管和多管只记在双 Agent 上，不并进游戏分。不该交接、也就没有台词的步，不拿来算扮演分。

原始记录在 [`reports/latest.json`](reports/latest.json)。

## 怎么跑

Python 3.10，用 uv。Spark 不参与这一轮。扮演用的是 Airi 英文系统提示，在 `craftax_bench/airi_prompt.py`。

`.env` 放在仓库根目录，不进 git：

```
OPENCODE_API_KEY=
MODEL=deepseek-v4.1-flash
MODEL_URL=https://opencode.ai/zen/go/v1/chat/completions
```

```bash
uv sync
uv run python -m craftax_bench.run
```

文本环境来自 [BALROG](https://github.com/balrog-ai/BALROG) 的 `CrafterLanguageWrapper`，副本在 `third_party/BALROG`，许可证是 MIT。
