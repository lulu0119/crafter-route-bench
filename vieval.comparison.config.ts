import { defineConfig } from "vieval"

export default defineConfig({
  comparisons: [
    {
      id: "routes",
      benchmark: {
        id: "crafter-text",
        sharedCaseNamespace: "three-scores",
      },
      methods: [
        {
          id: "single-agent",
          project: "single-agent",
          workspace: ".",
        },
        {
          id: "dual-agent",
          project: "dual-agent",
          workspace: ".",
        },
      ],
    },
  ],
})
