# content_pipeline

Multi-agent **workflow graph**. Four Gemini sub-agents wired as nodes
on a workflow DAG; a `JoinNode` aggregates the parallel branch.

```
START → researcher ─┬─ drafter ─┐
                    └─ fact_chk ─┴─ join → editor
```

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/content_pipeline/             # console
go run ./examples/v2/content_pipeline/ web         # adk-web
```

## Try it

> Topic: how vector databases changed the design of search engines in
> the last two years.

Watch:

1. `researcher` produces a 5-bullet brief + 2 cited URLs + an argument
   sentence.
2. `drafter` and `fact_checker` BOTH receive the brief and run **in
   parallel** (workflow engine spawns one goroutine per node).
3. `join` aggregates `{"drafter": <draft>, "fact_checker":
   <verdicts>}` into a single map.
4. `editor` receives the map (rendered as labeled sections) and
   produces the final article. If any fact-check verdict starts with
   "VERIFY:", the editor prepends an editorial note.

## Patterns shown

- `workflow.New(...)` with a real DAG (fan-out + fan-in).
- `workflow.FromAgent(llmagent)` wrapping each Gemini agent as a node.
- `workflow.Join(...)` aggregating multiple predecessor outputs by
  predecessor name.
- `Workflow.AsAgent()` adapting the whole graph back into an
  `agent.Agent` so the standard `cmd/launcher` runs it.

This is the realistic pattern when one prompt would be too tangled to
do well: split responsibilities across small focused sub-agents and
let the graph orchestrate them.
