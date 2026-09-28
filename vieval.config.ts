import { cwd } from "node:process"

import { defineConfig, loadEnv, requiredEnvFrom } from "vieval"
import { ChatModels, chatModelFrom } from "vieval/plugins/chat-models"

const model = chatModelFrom({
  aliases: ["crafter"],
  apiKey: config => requiredEnvFrom(config.env, {
    name: "OPENCODE_API_KEY",
    type: "string",
  }),
  baseURL: "https://opencode.ai/zen/go/v1",
  inferenceExecutor: "openai",
  model: "deepseek-v4.1-flash",
})

export default defineConfig({
  env: loadEnv("test", cwd(), ""),
  plugins: [
    ChatModels({ models: [model] }),
  ],
  projects: [
    {
      include: ["evals/single-agent.eval.ts"],
      name: "single-agent",
      root: ".",
      runMatrix: {
        extend: {
          route: ["single-agent"],
        },
      },
    },
    {
      include: ["evals/dual-agent.eval.ts"],
      name: "dual-agent",
      root: ".",
      runMatrix: {
        extend: {
          route: ["dual-agent"],
        },
      },
    },
  ],
})
