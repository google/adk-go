# dynamic_flow

LLM-orchestrated multi-agent flow. Same writing team as `content_pipeline`,
but the **shape** of the flow is decided at runtime by an orchestrator
agent, not hard-coded into a `workflow.Graph`.

```
orchestrator ── tool: run_flow ──> [ recursive spec over the catalog ]

catalog: researcher, drafter, fact_checker, editor
```

The orchestrator emits a single `run_flow` tool call carrying a recursive
JSON spec. `flowtool` materialises and runs the flow, returning a
path-keyed outputs map plus the final output.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/dynamic_flow/             # console
go run ./examples/v2/dynamic_flow/ web         # adk-web
```

## Try it

> Write a short article about hummingbird migration. Run the researcher
> first, then run the drafter and the fact-checker in parallel on the
> brief, then have the editor merge their outputs.

What to watch for:

1. The orchestrator decides a flow shape — typically
   `seq[researcher → parallel(drafter, fact_checker) → editor]` — and
   calls `run_flow` once with the full spec.
2. Each leaf in the spec is one of the catalog agents; flowtool spawns
   them in isolated in-memory sessions and chains their outputs.
3. The `editor` node's `input` field uses templates like
   `{{nodes.seq[1].parallel[0].drafter.output}}` to pull specific
   upstream outputs into its prompt.
4. The tool returns `{outputs: {<path>: {output, error?}}, final_output}`;
   the orchestrator summarises `final_output` to the user.

## Spec shape

```json
{
  "type": "seq",
  "nodes": [
    {"type": "agent", "agent": "researcher"},
    {"type": "parallel", "nodes": [
      {"type": "agent", "agent": "drafter"},
      {"type": "agent", "agent": "fact_checker"}
    ]},
    {"type": "agent", "agent": "editor",
     "input": "Draft:\n{{nodes.seq[1].parallel[0].drafter.output}}\n\nVerdicts:\n{{nodes.seq[1].parallel[1].fact_checker.output}}"}
  ]
}
```

## Patterns shown

- `flowtool.New(catalog map[string]agent.Agent, opts...)` — registers a
  fixed set of agents; the LLM only chooses subset + shape.
- Recursive `seq` / `parallel` / `agent` spec; nesting is free.
- Per-call sub-sessions (no parent-state pollution by default; opt in
  with `flowtool.WithInheritState(true)`).
- Built-in safety limits: `WithMaxNodes`, `WithMaxDepth`,
  `WithMaxParallelWidth`, `WithTimeout`, `WithMaxRecursion`,
  `WithMaxConcurrency`.

## When to use this vs `content_pipeline`

- **Static graph (`content_pipeline`)**: the flow shape is known at
  build time, every run uses it. Cheaper, debuggable, no LLM judgment
  in the wiring.
- **Dynamic flow (this example)**: the flow shape varies by request.
  Same catalog of agents, different orchestrations. The LLM picks.
