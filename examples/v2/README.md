# adk-go v2 — Gemini agent examples

Each subdirectory is a runnable Gemini-backed agent that exercises one
or more v2 features. Pattern mirrors `adk-python/contributing/samples`:
each example has a `main.go` that constructs a real `llmagent.New(...)`
agent (model = `gemini.NewModel(...)`) and starts it via the standard
launcher.

## Run

Set your Gemini API key once:

```
export GOOGLE_API_KEY=...
```

Then start any example:

```
go run ./examples/v2/<example>/                # interactive console
go run ./examples/v2/<example>/ web            # adk-web at http://localhost:8080
```

The launcher recognizes the same subcommands as `examples/quickstart/`.

## Examples

| Directory | What the agent does | v2 features |
|---|---|---|
| [`research_assistant`](research_assistant) | Researcher that lists / loads tools on demand to keep the LLM context lean | `toolregistry` dynamic tool loading, `list_tools` / `load_tool` |
| [`incident_responder`](incident_responder) | On-call agent that loads only the matching runbook for the alert | `skill`, `skillregistry`, `SkillsInstructionPlugin` |
| [`travel_concierge`](travel_concierge) | Coordinator delegates flight + hotel research to specialist sub-agents | Task API mesh: `RequestTaskTool` / `FinishTaskTool`, `base_flow` auto-routing |
| [`refund_approval`](refund_approval) | Refund agent with workflow that pauses for manager approval | `workflow`, `RequestInput`, `Runner.Resume`, `app.ResumabilityConfig` |
| [`pr_triage`](pr_triage) | GitHub PR triage as a workflow of parallel Gemini-backed scans | `workflow` + `JoinNode` + Gemini sub-agents inside a graph |
| [`support_with_compaction`](support_with_compaction) | Long support session that auto-summarizes older turns | `app.EventsCompactionConfig`, `app.LlmEventSummarizer` |
| [`code_reviewer`](code_reviewer) | Reviewer agent that runs reviewer-supplied Python via codeexec | `codeexec/unsafelocal` wired as a function tool |
| [`production_llm_agent`](production_llm_agent) | Production-shape Gemini agent: retry + structured logging + global safety instruction | `model/retry`, `plugin/builtin/Logging`, `GlobalInstruction` |
| [`research_planner`](research_planner) | Planner-driven research agent — model emits Plan/Reasoning/Final-Answer sections | `planner.PlanReAct` |
| [`qa_evaluator`](qa_evaluator) | Standalone evaluator that runs a Gemini agent against an eval set | `eval.Runner`, `eval/llmjudge` LLM-as-judge scorer |

## Conventions (matching adk-python samples)

- `main.go` is the entry point. The agent is constructed at the top of
  `main`, then handed to `launcher.NewLauncher().Execute`.
- Default model is `gemini-2.5-flash`. Override via `GOOGLE_GENAI_MODEL`
  env var if you want a different one.
- Sub-agents and toolset registrations live in the same file or a
  sibling `*.go` file in the same package — same shape as Python's
  per-sample `agent.py` + helper modules.
- Examples are ready to scale up: drop in production tools, swap the
  in-memory session for a database, add real plugins.
