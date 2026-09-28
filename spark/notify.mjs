import { readFileSync } from "node:fs"
import { createInterface } from "node:readline"

const coreAgent = new URL("../../airi/packages/core-agent/dist/agents/spark-notify/index.mjs", import.meta.url)
const { createSparkNotifyAgent } = await import(coreAgent.href)

const envPath = process.argv[2]
const env = Object.fromEntries(
  readFileSync(envPath, "utf8")
    .split("\n")
    .filter(line => line.includes("=") && !line.trim().startsWith("#"))
    .map((line) => {
      const index = line.indexOf("=")
      return [line.slice(0, index).trim(), line.slice(index + 1).trim()]
    }),
)

const memories = {
  "flee-from-skeleton": "This character runs away from skeletons.",
  "tell-momo-about-cow": "If you see a cow, tell Momo that you saw it.",
  "leave-diamond": "Leave diamonds where they are. Do not collect them.",
  "refuse-cow": "This character will not eat this cow. Tell Momo instead.",
  "favorite-color": "The character's favorite color is blue.",
}

const actions = [
  "Noop",
  "Move West",
  "Move East",
  "Move North",
  "Move South",
  "Do",
  "Sleep",
  "Place Stone",
  "Place Table",
  "Place Furnace",
  "Place Plant",
  "Make Wood Pickaxe",
  "Make Stone Pickaxe",
  "Make Iron Pickaxe",
  "Make Wood Sword",
  "Make Stone Sword",
  "Make Iron Sword",
]

function readMemoryTool() {
  return {
    type: "function",
    execute: async (raw) => {
      const key = raw?.key
      readKeys.push(key)
      return memories[key] ?? `No memory is stored under ${key}.`
    },
    function: {
      name: "read_memory",
      description: "Read one stored memory by key. The values are hidden until you call this.",
      parameters: {
        type: "object",
        properties: {
          key: { type: "string", enum: Object.keys(memories) },
        },
        required: ["key"],
      },
    },
  }
}

let readKeys = []
let spoken = ""
const stats = { modelMs: 0, modelCalls: 0 }

function turnsToMessages(conversation) {
  return conversation.turns.map((turn) => ({
    role: turn.type === "system" ? "system" : "user",
    content: (turn.content ?? []).map(part => part.text ?? "").join("\n"),
  }))
}

async function complete(messages, tools) {
  const started = performance.now()
  const response = await fetch(env.MODEL_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.OPENCODE_API_KEY}`,
      "Content-Type": "application/json",
      "User-Agent": "crafter-route-bench/0.1",
      "x-opencode-session": "crafter-route-bench",
    },
    body: JSON.stringify({
      model: env.MODEL,
      messages,
      tools: tools.map(tool => ({ type: "function", function: tool.function })),
      max_tokens: 2048,
      reasoning_effort: "none",
    }),
  })
  const elapsed = performance.now() - started
  const data = await response.json()
  if (!response.ok) {
    throw new Error(`chat HTTP ${response.status}: ${JSON.stringify(data).slice(0, 400)}`)
  }
  const message = data.choices?.[0]?.message ?? {}
  return { message, elapsed }
}

const agent = createSparkNotifyAgent({
  plugins: [
    {
      name: "read-memory",
      prepare() {
        readKeys = []
        return { tools: [readMemoryTool()] }
      },
    },
  ],
  runner: {
    async run({ conversation, tools, onStreamEvent }) {
    const messages = turnsToMessages(conversation)
    messages[0].content += [
      "",
      "You are the character. Call read_memory before you decide.",
      "Then call builtIn_sparkCommand.",
      "destinations must be [\"crafter\"]. intent is \"action\". interrupt is the string \"false\" or null.",
      "guidance.type is \"memory-recall\".",
      `The steps array must contain exactly one of: ${actions.join(", ")}.`,
      "Put the spoken line in ack.",
    ].join("\n")

    let modelMs = 0
    let modelCalls = 0
    spoken = ""
    for (let round = 0; round < 4; round += 1) {
      const { message, elapsed } = await complete(messages, tools)
      modelMs += elapsed
      modelCalls += 1
      const toolCalls = message.tool_calls ?? []
      if (message.content) {
        spoken += message.content
        await onStreamEvent({ type: "text-delta", text: message.content })
      }
      if (toolCalls.length === 0)
        break
      messages.push({
        role: "assistant",
        content: message.content ?? "",
        tool_calls: toolCalls,
      })
      for (const call of toolCalls) {
        const name = call.function?.name
        let args = {}
        try {
          args = JSON.parse(call.function?.arguments || "{}")
        }
        catch {
          args = {}
        }
        const tool = tools.find(candidate => candidate.function?.name === name)
        let result = "Unknown tool."
        if (tool?.execute) {
          try {
            result = await tool.execute(args, { toolCallId: call.id })
          }
          catch (error) {
            result = error instanceof Error ? error.message : String(error)
          }
        }
        await onStreamEvent({
          type: "tool-call",
          toolCallId: call.id,
          toolName: name,
          args,
        })
        await onStreamEvent({
          type: "tool-result",
          toolCallId: call.id,
          result,
        })
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: typeof result === "string" ? result : JSON.stringify(result),
        })
      }
    }
    stats.modelMs += modelMs
    stats.modelCalls += modelCalls
    },
  },
})

function actionFrom(commands) {
  const steps = commands.flatMap(command => command.guidance?.options?.flatMap(option => option.steps ?? []) ?? [])
  const joined = steps.join("\n")
  const sorted = [...actions].sort((left, right) => right.length - left.length)
  return sorted.find(action => joined.toLowerCase().includes(action.toLowerCase())) ?? null
}

const lines = createInterface({ input: process.stdin })
for await (const line of lines) {
  if (!line.trim())
    continue
  readKeys = []
  stats.modelMs = 0
  stats.modelCalls = 0
  try {
    const request = JSON.parse(line)
    const result = await agent.handle({
      systemPrompt: "You play beside a Crafter agent. Speak as the character and issue one action.",
      selectedChat: {
        providerId: "opencode",
        model: env.MODEL,
        provider: {},
      },
      control: { forceResponse: true },
      event: {
        type: "spark:notify",
        source: "crafter",
        metadata: { source: { id: "crafter" } },
        data: {
          id: crypto.randomUUID(),
          eventId: crypto.randomUUID(),
          kind: "ping",
          urgency: "soon",
          headline: request.headline,
          note: request.note,
          destinations: ["proj-airi:stage-*"],
        },
      },
    })
    const reaction = [spoken.trim(), ...result.commands.map(command => command.ack).filter(Boolean)].filter(Boolean).join(" ")
    process.stdout.write(`${JSON.stringify({
      reaction,
      action: actionFrom(result.commands),
      memoryKeys: readKeys,
      commands: result.commands,
      modelMs: stats.modelMs,
      modelCalls: stats.modelCalls,
      error: null,
    })}\n`)
  }
  catch (error) {
    process.stdout.write(`${JSON.stringify({
      reaction: "",
      action: null,
      memoryKeys: readKeys,
      commands: [],
      modelMs: stats.modelMs,
      modelCalls: stats.modelCalls,
      error: error instanceof Error ? error.message : String(error),
    })}\n`)
  }
}
