# Tool error handling

This credential-free agent demonstrates how to recover from an expected function tool failure without hiding unexpected errors.

- **Concept:** Recover from selected tool failures with `OnToolErrorCallbacks`.
- **Needs LLM?** No. A deterministic local `model.LLM` drives the example.

## Goal

Show a function tool returning a normal Go error, an agent callback recognizing that error with `errors.Is`, and the agent continuing with a safe fallback result.

## How it works

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Tool as inventory_lookup
    participant Callback as OnToolErrorCallback
    User->>Agent: Is the sold-out item available?
    Agent->>Tool: item = sold-out
    Tool-->>Callback: inventory service unavailable
    Callback-->>Agent: status = temporarily unavailable
    Agent-->>User: Inventory is temporarily unavailable; please try again.
```

1. `lookupInventory` wraps the sentinel `errInventoryUnavailable` when its simulated dependency fails.
2. `recoverInventoryError` uses `errors.Is` to recover only from that known failure. It reports the original error for operators and returns a safe, retryable fallback to the agent. It returns unknown errors unchanged rather than masking them.
3. The agent receives the fallback as the tool's function response and completes its turn normally.

Without an `OnToolErrorCallback` recovery result, ADK supplies the model a function response containing the tool error. The model can then explain the failure, but application-specific details may be exposed and the response is model-dependent.

## Running the sample

From the repository root:

```bash
go run ./examples/tools/errorhandling
```

No credentials, environment variables, or network access are required.

## Example session

No external model runs here, so the output is deterministic:

```text
Tool inventory_lookup failed: inventory service unavailable for "sold-out"
Agent: Inventory is temporarily unavailable; please try again.
```

## What it shows

- Return ordinary wrapped Go errors from function tool handlers.
- Distinguish expected failures from programming or infrastructure errors.
- Give the model a stable fallback shape while retaining an operator-visible diagnostic.
- Test the complete tool-call, callback, and final-response path without credentials.

## Notes

`inventoryDemoModel` exists only to make this example self-contained. In a real application, configure the `llmagent` with your model provider; the tool and `OnToolErrorCallbacks` pattern remain the same. Avoid putting secrets or sensitive backend details in either the fallback map or user-facing response.
