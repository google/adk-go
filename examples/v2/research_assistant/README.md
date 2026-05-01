# research_assistant

Demonstrates **dynamic tool loading** — the agent keeps its LLM context
lean by listing available tools on demand and activating only what it
needs for the current task.

The toolregistry pattern:

1. The registry holds the catalog (lightweight `Info{Name, Description,
   Tags, Hints}` + a lazy builder for each tool).
2. The agent's `Tools` list contains only the always-on discovery tools
   `list_tools` and `load_tool` — the rest are dormant.
3. The LLM decides what's relevant, calls `list_tools` (optionally
   filtered by query / tags), then `load_tool(name)` for each one it
   wants.
4. Loaded names persist in session state, so on the next turn the
   `Toolset` surfaces them as real `FunctionDeclaration`s.

This example uses a deterministic stub LLM that scripts the
discovery → load → use sequence so you can see the loop end-to-end
without a real model. Replace the stub with your real LLM (Gemini /
Vertex / OpenAI) when adapting the pattern.
