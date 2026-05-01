# research_pipeline

Multi-stage research workflow with explicit chain-of-thought planning:

```
START → plan → fan_out{search, fact_check} → join → synthesize
```

Demonstrates `workflow` + `planner.PlanReAct`. The planner injects a
Plan/Reasoning/Final-Answer template into the system instruction and
post-processes response parts so the workflow keeps planning text out
of the final answer.

In this offline demo each node is a typed Go function — the focus is
on the shape of the pipeline. Replace `synthesize` with a real LlmAgent
configured with `Planner: &planner.PlanReAct{}` to see the planning
parts emitted as `Thought=true` and the final answer flowing through.
