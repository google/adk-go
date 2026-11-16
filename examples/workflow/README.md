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
| [`dynamic/use_as_output`](./dynamic/use_as_output) | Dynamic | Promote a child's output to the orchestrator's own with `WithUseAsOutput` | Yes |
| [`schemas`](./schemas) | Schemas | Validate a root workflow's input and output with `InputSchema` and `OutputSchema` | No |

## Core concepts at a glance

- **Nodes** — units of work: `FunctionNode` (a Go function), `AgentNode` (wraps an `LlmAgent`), `ToolNode` (wraps a `tool.Tool`), `JoinNode` (fan-in barrier), and `DynamicNode` (imperative orchestrator).
- **Edges** — declared with `workflow.Chain`, `workflow.Concat`, and `EdgeBuilder` (`Add`, `AddFanOut`, `AddFanIn`).
- **Routing** — a node sets `Event.Routes`; matching edges use `StringRoute`, `IntRoute`, or `MultiRoute`.
- **Human-in-the-loop** — a node emits `RequestInput` to pause; the run resumes when the human replies.

## Workflow input and output schemas

A schema is a resolved JSON schema (`*jsonschema.Resolved` from `github.com/google/jsonschema-go/jsonschema`). Nodes, nested workflows and the root workflow can each have an input schema and an output schema. A `nil` schema means no validation.

### Where each schema applies

| Schema | How you set it | What it checks | Error on failure |
|---|---|---|---|
| Root workflow input | `workflowagent.Config.InputSchema`, or `workflow.WithInputSchema` on `workflow.New` | The user message, before any node runs. Resume turns check the message rebuilt from history. | Wraps `workflow.ErrInputValidation`; no node runs and no event is written |
| Root workflow output | `workflowagent.Config.OutputSchema`, or `workflow.WithOutputSchema` | The output of the single terminal node, when the run finishes with no node waiting | Wraps `workflow.ErrNodeFailed`; the terminal node's events are already written |
| Nested workflow | `workflow.NewWorkflowNodeWithSchemas(name, edges, in, out)` | The input the nested workflow receives and the output it hands to the next node | `ErrInputValidation` or `ErrNodeFailed`; the next node does not run |
| Node | `NewFunctionNodeWithSchema`, `NewBaseNodeWithSchemas` and the other `...WithSchema(s)` constructors | Each input the node receives and each output it emits | `ErrInputValidation` or `ErrNodeFailed` |

`workflow.Workflow.InputSchema` and `OutputSchema` return the root schemas, so code that builds workflows from configuration can read them back.

A paused workflow is not checked against its output schema. The check runs on the resume that finishes the run. When a workflow has more than one terminal output, it fails with `ErrMultipleTerminalOutputs` before the output schema is checked.

### What validation does to a value

- **Defaults.** A missing property that declares a default gets that default before the value is checked, so a required property with a default may be left out. Defaults are filled at any depth reached through `properties`, `patternProperties`, `additionalProperties`, `prefixItems`, `items`, `allOf` and local `$ref`s (`#/$defs/...`), but not through `anyOf` or `oneOf`. A default of `null` is not filled. The next node, or the workflow's caller, receives the filled value. The value you passed is not changed in place.
- **Nulls.** An optional property whose value is `null` is dropped after the check. A required property keeps its `null`.
- **Undeclared keys.** A key the schema does not declare is kept when the schema leaves `additionalProperties` open, and rejected when the schema sets `additionalProperties: false`. adk-python's pydantic models drop such keys instead.
- **Go structs.** A struct, pointer, typed map or typed slice is converted to its JSON form before validation. A struct output validated against a schema therefore arrives downstream as a `map[string]any`, whether or not a default was filled. A value already in JSON form (`map[string]any`, `[]any`, scalars) is returned unchanged when validation changes nothing.
- **Message content.** A user message or a `*genai.Content` output is validated as its text. The text is parsed as JSON when the schema expects something other than a string. Text that is not JSON is validated as a plain string, so `hello` satisfies `anyOf: [integer, string]`. A user message or a `*genai.Content` value is kept as it is when the schema is the empty schema `{}`, or when its text fails, the schema is the one generated from `genai.Content` (its root declares `parts`) and the content satisfies it. A user message kept by a root input schema reaches each successor of Start in the form it would without the schema. Against any other schema, text that fails is an error, even when the content's own JSON form would satisfy an open object schema.
- **Strings.** A string output of a function or tool node is validated as a string. Against a number or boolean schema, a string such as `"42"` or `"true"` becomes `float64(42)` or `true`; no other string is parsed. An LLM agent node's text output is parsed as JSON, because a model asked for structured output answers with JSON text.

adk-python converts more scalars than this, following pydantic's lax mode: for example `"yes"` to `true`, or `1` to `true`. adk-go does not port those rules.

### Example

[`schemas`](./schemas) builds a `workflowagent` with both root schemas and runs three messages through a runner. It needs no LLM and no `console` argument:

```bash
go run ./examples/workflow/schemas/
```

```text
message {"city": "Paris"}
  node input: map[city:Paris units:metric]
  run finished
message {"town": "Paris"}
  input rejected: workflow: input validation failed for node "weather": validating root: type: {"town": "Paris"} has type "string", want "object"
message {"city": "Atlantis", "units": "imperial"}
  node input: map[city:Atlantis units:imperial]
  output rejected: workflow: dynamic child failed: output validation failed for node "weather": validating root: validating /properties/summary: minLength: "" contains 0 Unicode code points, fewer than 1
```

The first message reaches the node with `units` filled from its default. The second fails the input schema, so the node never runs. The third reaches the node, but the node's output has an empty `summary`, so the run ends with an output validation error.
