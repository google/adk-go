# Human-in-the-Loop (single-node re-entry)

A Human-in-the-Loop workflow where **one** emitting `FunctionNode` both pauses for input and produces the final output. On resume the node is re-run from scratch (`NodeConfig.RerunOnResume = &true`).

- **Concept:** Single-node re-entry HITL with `workflow.ResumeOrRequestInput`.
- **Needs LLM?** No

For the two-node handoff variant, see [`../hitl_simple`](../hitl_simple).

## Goal

Contrast two ways to do HITL. The two-node *handoff* variant ([`../hitl_simple`](../hitl_simple)) has one node ask and a separate node consume the reply. This sample collapses both phases into a single re-run node:

- `workflow.ResumeOrRequestInput` emits a `RequestInput` and returns `ErrNodeInterrupted` on the **first** pass (pause, no output);
- after the human replies, the node is **re-run from the top**, and the same call now returns the reply, which the body turns into the terminal output.

## Workflow

```mermaid
graph LR
    User[User]
    subgraph "ADK Application Workflow"
        Start((Start)) --> G[Node: greet]
        G -.->|"re-run on resume"| G
        G --> End((End))
    end
    User -- "hello" --> Start
    G -- "1st pass: What's your name?" --> User
    User -- "Alice" --> G
    End -- "Hello, Alice!" --> User
```

The dotted self-loop is the re-entry: the same `greet` node executes twice — once to ask (and pause), once to answer. The `InterruptID` embeds the invocation ID so the reply still correlates across the re-run within a single run, yet a later run re-prompts.

## Running the sample

```bash
go run ./examples/workflow/hitl_rerun/ console
```

## Example session

```text
User -> hello
Agent -> What's your name?
User -> Alice
Agent -> Hello, Alice!
```
