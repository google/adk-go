# HITL workflow sample — expense approval

A runnable workflow agent that exercises the human-in-the-loop
patterns the workflow engine supports today, end-to-end through
the console launcher.

## What it shows

| Pattern | Where in this sample |
|---|---|
| Conditional pause (auto-approve vs ask human) | `evaluateRequestNode` routes `auto` vs `needs_review` |
| Workflow input request with typed payload | `reviewDecisionNode` yields `RequestInput{InterruptID: "manager_approval", Payload: report}` |
| Re-entry resume (`RerunOnResume = true`) | `reviewDecisionNode.Run` reads the reply via `ctx.ResumedInput("manager_approval")` |
| Routing on the resume reply | `reviewDecisionNode` emits `Event.Routes` keyed by `approved` / `rejected` / `revise` |
| Handoff resume (output flows to next edge) | the auto-approve path: `evaluateRequestNode` output → `adapter` → `file_disbursement` |
| Cross-process state rehydration tolerance | `expenseReportFromAny` normalises the typed-or-`map[string]any` shapes that arise after `Workflow.Resume` reloads `RunState` from session |
| Typed structures across edges | `expenseReport` (typed Go struct) flows directly through `evaluate_request → file_disbursement` (a ToolNode); the engine's `typeutil.ConvertToWithJSONSchema` coerces it into the tool's `disburseArgs` schema via matching JSON tags. No explicit adapter node needed. |

Tool confirmation (`functiontool.RequireConfirmation`) is a related
HITL kind exposed via the same console launcher dispatch, but it
lives inside the LLMAgent flow today and is not yet wired through
`workflow.ToolNode`. This sample therefore does not use it.

## Run it

```sh
go run ./examples/workflow/hitl/ console
```

## Try the four paths

```text
User -> 50 lunch with team
Agent -> ✓ Disbursement filed.
        # Auto-approved (≤ threshold). No prompt; the report flows
        # straight through evaluate → adapter → disburse → notify.
```

```text
User -> 250 client dinner
Agent ->
[HITL input] Approve $250 for "client dinner"? Reply 'yes', 'no', or 'revise: <feedback>'.
  Payload: {
    "amountUsd": 250,
    "description": "client dinner"
  }
[user]: yes
Agent -> ✓ Disbursement filed.
        # Re-entry: reviewDecisionNode re-runs, reads "yes" via
        # ResumedInput, routes to the "approved" branch which
        # files the disbursement.
```

```text
User -> 250 client dinner
[user]: no
Agent -> ✗ Rejected: {250 client dinner}
```

```text
User -> 300 birthday cake
[user]: revise: cake is too expensive
Agent -> ↺ Needs revision: cake is too expensive
```

## Tuning the auto-approve threshold

Default cutoff is $100. Override via env var:

```sh
HITL_AUTO_APPROVAL_THRESHOLD=500 go run ./examples/workflow/hitl/ console
```

## Wire-format notes

- The pause event carries a `FunctionCall` part with name
  `adk_request_workflow_input` and the `InterruptID` reused as
  the FunctionCall's `ID`. The console launcher detects it via
  `event.LongRunningToolIDs` (name-agnostic) and dispatches the
  prompt rendering by name.
- The console launcher attempts `json.Unmarshal` on the operator's
  reply first, so structured replies (JSON objects, arrays,
  numbers, booleans) round-trip as typed values. Free-form text
  passes through as a string.
- The reply is wrapped under `{"payload": <value>}` to match
  `agent/workflowagent.decodeWorkflowInputResponse`.
