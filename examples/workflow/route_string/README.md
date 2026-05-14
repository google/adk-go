# String routing sample — message classification → 3 branches

The smallest end-to-end demonstration of `workflow.StringRoute`
and the `Event.Routes` contract. No LLM, no HITL, no random — a
trivial classifier picks the route based on the message's
terminal punctuation.

## Run it

```sh
go run ./examples/workflow/route_string/ console
```

```text
User -> What time is it?
Agent -> answering question: What time is it?

User -> The sky is blue.
Agent -> commenting on statement: The sky is blue.

User -> Hello world!
Agent -> reacting to exclamation: Hello world!
```

## Graph

```
START → classify ─┬─ "question"    → answer_question
                  ├─ "statement"   → comment_statement
                  └─ "exclamation" → react_exclamation
```

## What it shows

| Concept | Where |
|---|---|
| Custom `BaseNode` emitting a routing event | `classifyNode` sets `Event.Routes = []string{category}` and `Event.Actions.StateDelta["output"] = msg` so downstream `FunctionNode`s get the original message as a typed `string` input |
| `StringRoute` matching a single value | three downstream edges, one per category |
| Direct port of adk-python's `route/` sample, minus the LLM | classifier is a plain Go function instead of an `Agent` with `output_schema` |
