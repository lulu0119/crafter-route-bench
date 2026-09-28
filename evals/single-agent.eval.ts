import { readFileSync } from "node:fs"

import { caseOf, describeTask } from "vieval"

describeTask("single-agent", () => {
  caseOf("three-scores", (context) => {
    const report = JSON.parse(readFileSync("reports/latest.json", "utf8"))
    const scores = report.routes["single-agent"]
    context.score(scores.gameplay.mean, "exact")
    context.score(scores.roleplay.mean, "judge")
    context.metric("latency_ms_mean", scores.latency_ms.mean)
    context.metric("latency_ms_spread", scores.latency_ms.spread)
    context.metric("game_ms_mean", scores.latency_ms.game_ms.mean)
    context.metric("role_ms_mean", scores.latency_ms.role_ms.mean)
    context.metric("spark_ms_mean", scores.latency_ms.spark_ms.mean)
    context.metric("gameplay_spread", scores.gameplay.spread)
    context.metric("roleplay_spread", scores.roleplay.spread)
    context.metric("model_calls", scores.model_calls)
  })
})
