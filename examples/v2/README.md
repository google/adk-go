# adk-go v2 — real-world examples

Each example is a believable mini-application that exercises multiple
v2 features end-to-end. Run any example with:

```
go run ./examples/v2/<example>/
```

None of these require a network or API key — agents and tools that
would normally call a real LLM use a deterministic stub LLM defined
in the example, so the behavior under demonstration is observable
locally. Comments near each stub note where to wire a production
model (e.g. Gemini / Vertex / OpenAI) when you adapt the pattern.

## Domain examples

| Example | Story | v2 features |
|---|---|---|
| [`pr_triage_workflow`](pr_triage_workflow) | Auto-classify a GitHub PR: fetch metadata, parallel security/breaking-change/label scans, join, decide. | `workflow` (fan-out, JoinNode, retry on flaky API) |
| [`research_assistant`](research_assistant) | Research agent that lists / loads tools on demand instead of declaring them all upfront. Keeps the LLM context lean. | `toolregistry`, dynamic tool loading |
| [`incident_responder`](incident_responder) | On-call agent with a runbook library. Discovers + loads only the runbook(s) that match the incident type. | `skill`, `skillregistry`, `SkillsInstructionPlugin` |
| [`refund_approval`](refund_approval) | Refund workflow that pauses for a human manager's approval before processing the payout. | `workflow.RequestInput`, `runner.Resume` |
| [`travel_concierge`](travel_concierge) | Coordinator delegates flight + hotel research to specialist sub-agents and composes the itinerary. | Task API mesh runtime (`task.NewRequestTaskTool` / `FinishTaskTool`, `base_flow` auto-routing) |
| [`support_with_compaction`](support_with_compaction) | Long-running customer support session whose older turns auto-summarize so the model context doesn't blow up. | `app.EventsCompactionConfig`, `app.LlmEventSummarizer` |
| [`code_review_executor`](code_review_executor) | Code-review agent that runs the user's Python snippet in a sandboxed subprocess and reports the output. | `codeexec/unsafelocal`, `Executor` interface |
| [`prompt_optimizer`](prompt_optimizer) | Find the best prompt variant for a small QA eval set. | `optimize.GridSampler`, `eval.Runner`, `ExactMatchScorer` |
| [`production_llm_agent`](production_llm_agent) | Production-shape LLM agent: retried 5xx-tolerant model, structured logging, global safety instruction. | `model/retry`, `plugin/builtin/Logging`, `GlobalInstruction` |
| [`research_pipeline`](research_pipeline) | Multi-stage research workflow: plan → fan-out research → fact-check → synthesize, using PlanReAct planning. | `workflow`, `planner.PlanReAct` |

Each directory contains a single `main.go` plus a brief `README.md`
sketching the scenario.
