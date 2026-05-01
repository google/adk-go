# travel_concierge

Coordinator agent delegates flight + hotel research to specialist
sub-agents and composes the itinerary. Demonstrates the **Task API
mesh runtime**:

- The coordinator's tools include `task.NewRequestTaskTool(flightAgent)`
  and `task.NewRequestTaskTool(hotelAgent)`.
- When the LLM calls `flight_agent(...)`, `RequestTaskTool.Run` writes
  a `session.TaskRequest` into `ctx.Actions().RequestTask` keyed by the
  function-call ID.
- `internal/llminternal/base_flow.runTaskRequests` (Phase 6D) sees the
  `RequestTask` entry, locates the named agent in the agent tree,
  invokes it with the rendered task input, and waits for the agent to
  call `task.NewFinishTaskTool` which writes a `session.TaskResult`.
- The runtime synthesizes a `FunctionResponse` for the coordinator
  carrying the task's output. The coordinator's next turn observes the
  result through standard contents-builder paths.

This example uses a deterministic stub LLM that scripts the
coordinator's three turns:

1. Call `flight_agent({"origin":"SFO","dest":"NRT"})`.
2. Call `hotel_agent({"city":"Tokyo","nights":3})`.
3. Compose an itinerary text from the two results.

The flight + hotel agents themselves use their own stub LLMs that
return canned `finish_task` payloads. Replace the stubs with a real
model when adapting the pattern.
