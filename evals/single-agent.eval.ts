import { readFileSync } from "node:fs"

import { caseOf, describeTask } from "vieval"

describeTask("single-agent", () => {
  caseOf("mixed-messages", (context) => {
    const report = JSON.parse(readFileSync("reports/latest.json", "utf8"))
    const scores = report.routes["single-agent"]
    context.score(scores.gameplay.mean, "exact")
    context.score(scores.roleplay.mean, "judge")
    context.metric("valid_action_mean", scores.valid_action.mean)
    context.metric("act_token_mean", scores.act_token.mean)
    context.metric("gameplay_spread", scores.gameplay.spread)
    context.metric("roleplay_spread", scores.roleplay.spread)
  })
})
