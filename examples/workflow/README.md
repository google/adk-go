# Workflow examples

Runnable samples for the adk-go **graph workflow engine** (`google.golang.org/adk/v2/workflow` + `agent/workflowagent`). Each sample is a `main` package that builds a `workflowagent` from a graph of nodes and serves it through the console launcher.

Every example has its own README with a Mermaid diagram, a goal, run instructions, and an example session.

## Running any sample

```bash
go run ./examples/workflow/<name>/ console
```

Samples marked **LLM? Yes** call Gemini and need credentials — either a Gemini API key or Vertex AI Application Default Credentials:

```bash
# Gemini API key
export GOOGLE_API_KEY=...

# or Vertex AI via gcloud ADC
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=your-region
```

## Examples

| Example | Theme | What it demonstrates | LLM? |
|---|---|---|---|
| [`basic`](./basic) | Basics | Sequential chain of `FunctionNode`s with `workflow.Chain` | No |
| [`complex`](./complex) | Parallel | Fan-out → `JoinNode` fan-in → LLM synthesis (`AddFanOut` + `AddFanIn`) | Yes |
| [`routing/string`](./routing/string) | Routing | Branch on a string value with `StringRoute` (punctuation classifier) | No |
| [`routing/int`](./routing/int) | Routing | Branch on a number with `IntRoute` / `MultiRoute[int]` | No |
| [`routing/llm`](./routing/llm) | Routing | LLM classifies, the engine routes via `StringRoute` | Yes |
| [`hitl_simple`](./hitl_simple) | Human-in-the-loop | Minimal two-node pause/resume with `RequestInput` | No |
| [`hitl_rerun`](./hitl_rerun) | Human-in-the-loop | Single-node re-entry HITL with `ResumeOrRequestInput` | No |
| [`dynamic/basic`](./dynamic/basic) | Dynamic | Orchestrate children imperatively in Go via `NewDynamicNode` + `RunNode` | No |
| [`dynamic/hitl`](./dynamic/hitl) | Dynamic | Dynamic orchestrator that pauses for human input and resumes | No |
| [`dynamic/llm`](./dynamic/llm) | Dynamic | Dynamic orchestrator invoking an `LlmAgent` node via `RunNode` | Yes |

## Core concepts at a glance

- **Nodes** — units of work: `FunctionNode` (a Go function), `AgentNode` (wraps an `LlmAgent`), `ToolNode` (wraps a `tool.Tool`), `JoinNode` (fan-in barrier), and `DynamicNode` (imperative orchestrator).
- **Edges** — declared with `workflow.Chain`, `workflow.Concat`, and `EdgeBuilder` (`Add`, `AddFanOut`, `AddFanIn`).
- **Routing** — a node sets `Event.Routes`; matching edges use `StringRoute`, `IntRoute`, or `MultiRoute`.
- **Human-in-the-loop** — a node emits `RequestInput` to pause; the run resumes when the human replies.
