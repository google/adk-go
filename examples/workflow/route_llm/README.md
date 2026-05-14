# LLM-driven routing sample

The smallest sample that uses an actual LLM as the routing brain
inside a workflow graph. An LLMAgent classifies the user's
message into one of three categories; a trivial Go function then
emits the corresponding `Event.Routes` value, dispatching to one
of three handlers.

Same shape as adk-python's
`contributing/workflow_samples/route/` sample (LLM classifier +
plain function emitting the routing event).

## Requirements

```sh
export GOOGLE_API_KEY=<your-key>
```

## Run it

```sh
go run ./examples/workflow/route_llm/ console
```

```text
User -> What time is it?
Agent -> question
answering question: What time is it?

User -> Hello world!
Agent -> exclamation
reacting to exclamation: Hello world!

User -> The sky is blue.
Agent -> statement
commenting on statement: The sky is blue.
```

The first line of agent output is the LLM's classification (it
prints because the console launcher streams every event with
text content). The second line is the handler's reply.

## Graph

```
START → classify (LLM) → route_by_classification ─┬─ "question"    → answer_question
                                                  ├─ "statement"   → comment_statement
                                                  └─ "exclamation" → react_exclamation
```

## What it shows

| Concept | Where |
|---|---|
| `workflow.NewAgentNode` wrapping an `LLMAgent` | `classifyNode := workflow.NewAgentNode(classifier, ...)` |
| LLM output flowing into the workflow's "output" magic key | `LLMAgent.OutputKey = "output"` — the LLMAgent code at `agent/llmagent/llmagent.go:425-429` writes the final reply text there automatically |
| Custom `BaseNode` translating the LLM's free-form text into a route value | `routeFromClassificationNode` reads the classifier's reply, normalises it, and emits `Event.Routes` |
| `StringRoute` matching one of three categories | three downstream edges, one per category |
| Handler reading the original user message from `ctx.UserContent` | each handler calls `userMessage(ctx)` rather than receiving it as graph input — the routing node only forwards the classification, not the original text |

## Why two nodes (classifier + router)?

Mirrors the canonical adk-python pattern: keep the LLM stateless
about routing, keep the routing logic in plain code. The
alternative — one custom node that calls the LLM and emits
`Routes` from the same Run body — is shorter but mixes
"LLM-driven decision" with "graph wiring", and reuses none of
the engine's normal LLMAgent machinery (output_key, telemetry,
etc.).

## Why register the classifier in `Config.SubAgents`?

`workflow.NewAgentNode` wraps an `agent.Agent` for graph
execution but does **not** make that agent visible in the
runner's agent tree (the structure `runner.findAgentToRun`
walks to resolve `event.Author` to an `agent.Agent`). Without
the explicit `SubAgents: []agent.Agent{classifier}`
registration, the runner logs

```
Event from an unknown agent: classify, event id: ...
```

on every turn. The warning is harmless — the runner falls back
to `rootAgent` (the workflow itself), and
`isTransferableAcrossAgentTree` blocks any actual re-routing to
the LLM agent because the chain to root contains a non-LLMAgent
(the workflow wrapper). But registering the classifier as a
sub-agent silences the warning and keeps the agent tree
consistent with the workflow graph. Treat it as required
boilerplate for any workflow that wraps a sub-agent.

## Tunable: pick a different model

Edit `gemini-flash-latest` in `main.go` to whatever model your
key has access to. The classifier prompt is very short and any
modern Gemini model handles it.
