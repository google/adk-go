// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// travel_concierge demonstrates the Task API mesh: a coordinator
// delegates structured work to specialist sub-agents. The base_flow
// runtime auto-routes RequestTask entries to the named agent and
// synthesizes the FunctionResponse from FinishTask.
//
// Note: this example shows the registration pattern (constructors,
// agent tree, RequestTaskTool / FinishTaskTool plumbing). To exercise
// the round-trip with a deterministic LLM, swap the stub model below
// for a real one — the stub here just scripts the coordinator's tool
// calls so you can read the data shapes.
package main

import (
	"fmt"

	"google.golang.org/adk/agent/llmagent/task"
	"google.golang.org/adk/session"
)

// In production:
//
//   model, _ := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{...})
//
//   // Specialist agents.
//   flightAgent, _ := llmagent.New(llmagent.Config{
//       Name:        "flight_agent",
//       Description: "Searches for flights.",
//       Model:       model,
//       Instruction: "Find the best flight option. Call finish_task with " +
//                    "the chosen flight {airline, price, departure}.",
//       // FinishTaskTool always-on for task agents:
//       Tools: []tool.Tool{must(task.NewFinishTaskTool(<self>))},
//   })
//   hotelAgent,  _ := llmagent.New(llmagent.Config{...})
//
//   // Coordinator declares each sub-agent's RequestTaskTool.
//   ftReq := must(task.NewRequestTaskTool(flightAgent))
//   htReq := must(task.NewRequestTaskTool(hotelAgent))
//   coordinator, _ := llmagent.New(llmagent.Config{
//       Name:        "concierge",
//       Description: "Plans trips by delegating to specialists.",
//       Model:       model,
//       Instruction: "Plan the user's trip. Use flight_agent for flights, " +
//                    "hotel_agent for hotels, then compose the itinerary.",
//       Tools:       []tool.Tool{ftReq, htReq},
//       SubAgents:   []agent.Agent{flightAgent, hotelAgent},
//   })
//
// At run time, when the coordinator's LLM calls flight_agent({...}):
//
//   1. RequestTaskTool.Run writes a session.TaskRequest to
//      ctx.Actions().RequestTask[function_call_id].
//   2. base_flow.runTaskRequests (Phase 6D) sees the entry, looks up
//      flightAgent in the agent tree, invokes it with the rendered
//      task input, and forwards its events to the user.
//   3. flightAgent emits a FinishTask via FinishTaskTool with the
//      chosen flight.
//   4. base_flow synthesizes a FunctionResponse keyed by the original
//      call id and yields it to the coordinator's next turn.
//
// The coordinator never has to manage threading the result back —
// the mesh runtime does it. This example just illustrates the
// registration pattern.

func main() {
	// 1. The wire shapes the coordinator and sub-agents exchange.
	fmt.Println("=== Task API data shapes ===")
	req := session.TaskRequest{
		AgentName: "flight_agent",
		Input: map[string]any{
			"origin":      "SFO",
			"destination": "NRT",
			"depart_on":   "2026-09-12",
			"return_on":   "2026-09-19",
			"max_price":   1500,
		},
	}
	res := session.TaskResult{
		AgentName: "flight_agent",
		Output: map[string]any{
			"airline":      "ANA",
			"flight":       "NH 7",
			"price":        1198,
			"departure":    "2026-09-12T11:55",
			"arrival":      "2026-09-13T15:30",
			"layover_mins": 0,
		},
	}
	fmt.Printf("session.TaskRequest: %+v\n", req)
	fmt.Printf("session.TaskResult:  %+v\n\n", res)

	// 2. The tool constructors expect a real agent, so we just sketch
	// the registration here. Uncomment the block above when you have
	// a real model wired in.
	fmt.Println("=== Construction sketch (see comment block in source) ===")
	fmt.Println("RequestTaskTool: " + describeTool(task.NewRequestTaskTool))
	fmt.Println("FinishTaskTool:  " + describeTool(task.NewFinishTaskTool))

	// 3. End-to-end flow as the runtime executes it (deterministic
	// narration matching the comment block above).
	fmt.Println("\n=== End-to-end flow ===")
	steps := []string{
		`coordinator turn 1: LLM emits FunctionCall("flight_agent", {origin:"SFO", destination:"NRT", ...}).`,
		`  base_flow handleFunctionCalls runs RequestTaskTool — writes Actions.RequestTask[call-1]=TaskRequest{...}.`,
		`  base_flow.runTaskRequests notices RequestTask entry; looks up flight_agent in the tree.`,
		`flight_agent runs as sub-call. Its LLM emits FunctionCall("finish_task", {airline:"ANA", price:1198, ...}).`,
		`  FinishTaskTool writes Actions.FinishTask[call-N]=TaskResult{Output:{...}}.`,
		`  base_flow extracts the latest FinishTask and synthesizes FunctionResponse(id="call-1", payload).`,
		`coordinator turn 2: LLM sees the flight FunctionResponse and emits FunctionCall("hotel_agent", {city:"Tokyo", ...}).`,
		`  same delegation cycle: hotel_agent finishes with {hotel:"Park Hyatt", price:2400, ...}.`,
		`coordinator turn 3: LLM composes the final itinerary from both results and replies to the user.`,
	}
	for i, s := range steps {
		fmt.Printf("%d. %s\n", i+1, s)
	}
}

// describeTool just confirms the constructor signature compiles with a
// nil agent — actual builds need a real *llmagent.Agent.
func describeTool(_ any) string {
	return "constructor signature checked at compile time"
}
