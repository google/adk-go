# research_planner

Real Gemini agent prompted with the **PlanReAct** template. The model
explicitly emits `/*PLANNING*/`, `/*REASONING*/`, `/*ACTION*/`,
`/*FINAL_ANSWER*/` sections, and the `planner.PlanReAct` post-processor
tags everything before the final answer as `Thought=true` so the UI
layer can hide chain-of-thought from end users by default.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/research_planner/             # console
go run ./examples/v2/research_planner/ web         # adk-web
```

## Try it

> Research the differences between gRPC and REST for high-throughput
> microservices. Cite at least two sources.

The model:

1. Outputs a `/*PLANNING*/` block listing sub-questions.
2. Outputs `/*REASONING*/` while it calls `search` for each
   sub-question.
3. Concludes with `/*FINAL_ANSWER*/` paragraph the user actually
   reads.

In `adk-web` the thought-tagged parts render in a collapsed pane.

The native planner-on-LlmAgent integration (so you can write
`Planner: &planner.PlanReAct{}` directly on `llmagent.Config`) is
landing in a follow-up; for now this example composes the planner
instruction manually onto the `Instruction` field.
