# travel_concierge

Real Gemini coordinator agent that delegates flight + hotel research to
two specialist sub-agents via the **Task API**.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/travel_concierge/             # console
go run ./examples/v2/travel_concierge/ web         # adk-web
```

## Try it

> Plan a 7-day trip from SFO to Tokyo for Sept 12-19, 2026, two
> travelers, budget $1500 for flights and $250/night for hotel.

What happens:

1. The coordinator calls the `flight_agent` tool with structured input.
   `RequestTaskTool` writes a `session.TaskRequest` into
   `Actions.RequestTask[<call_id>]`.
2. The Phase 6D mesh runtime (`internal/llminternal/runTaskRequests`)
   sees the entry, locates `flight_agent` in the agent tree, and runs
   it with the rendered task input.
3. `flight_agent` calls its `search_flights` tool, picks one, and calls
   its `finish_task` tool with the chosen flight.
4. The mesh runtime synthesizes a `FunctionResponse` keyed by the
   coordinator's original call id and feeds it back.
5. Coordinator does the same for `hotel_agent`, then composes the
   itinerary.

Replace `search_flights` / `search_hotels` with your real APIs to wire
this up for production.
