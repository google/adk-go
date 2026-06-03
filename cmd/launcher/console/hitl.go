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

package console

import (
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// pendingInterrupt is one HITL prompt the agent emitted on the
// previous turn that still needs a user reply on the next. The
// id+name pair come from a FunctionCall part whose ID appears
// in Event.LongRunningToolIDs.
type pendingInterrupt struct {
	id   string
	name string
	args map[string]any
}

// collectPendingInterrupts scans events for FunctionCall parts
// referenced from LongRunningToolIDs on the same event and
// returns them in order of appearance.
func collectPendingInterrupts(events []*session.Event) []pendingInterrupt {
	var out []pendingInterrupt
	// seen dedups by call ID: in SSE streaming mode the same
	// function-call event can be emitted multiple times (partial
	// chunks plus the final aggregated event), each carrying the
	// same LongRunningToolIDs. Without dedup the console would
	// queue one prompt per duplicate and consume the user's reply
	// against a phantom interrupt instead of resuming the run.
	seen := map[string]struct{}{}
	for _, ev := range events {
		if ev == nil || len(ev.LongRunningToolIDs) == 0 {
			continue
		}
		// Skip partial streaming chunks; only the final aggregated
		// event represents a settled interrupt.
		if ev.LLMResponse.Partial {
			continue
		}
		lr := map[string]struct{}{}
		for _, id := range ev.LongRunningToolIDs {
			lr[id] = struct{}{}
		}
		c := ev.Content
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			fc := p.FunctionCall
			if fc == nil {
				continue
			}
			if _, isInterrupt := lr[fc.ID]; !isInterrupt {
				continue
			}
			if _, dup := seen[fc.ID]; dup {
				continue
			}
			seen[fc.ID] = struct{}{}
			out = append(out, pendingInterrupt{
				id:   fc.ID,
				name: fc.Name,
				args: fc.Args,
			})
		}
	}
	return out
}

// renderInterruptPrompt prints the pending interrupt and a
// "User -> " input prompt to stdout. The caller reads the user's
// reply and passes it to buildInterruptResponse.
func renderInterruptPrompt(p pendingInterrupt) {
	switch p.name {
	case workflow.WorkflowInputFunctionCallName:
		renderWorkflowInputPrompt(p.args)
	default:
		renderGenericInterruptPrompt(p.name, p.args)
	}
	fmt.Print("User -> ")
}

// buildInterruptResponse converts the operator's one-line input
// into a FunctionResponse part keyed by the interrupt's id and
// name. Same per-name dispatch as renderInterruptPrompt.
func buildInterruptResponse(p pendingInterrupt, userInput string) *genai.Part {
	line := strings.TrimRight(userInput, "\r\n")

	var response map[string]any
	switch p.name {
	case workflow.WorkflowInputFunctionCallName:
		response = workflowInputResponseFromUserInput(line)
	default:
		response = genericResponseFromUserInput(line)
	}
	return &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:       p.id,
			Name:     p.name,
			Response: response,
		},
	}
}

// renderWorkflowInputPrompt prints the workflow request prompt.
// Args carry the fields populated by workflow.NewRequestInputEvent:
// interruptId, message, payload, responseSchema.
func renderWorkflowInputPrompt(args map[string]any) {
	msg, _ := args["message"].(string)
	if msg == "" {
		msg = "Input requested"
	}
	fmt.Printf("Agent -> %s\n", msg)
	if payload, ok := args["payload"]; ok && payload != nil {
		// Strings (incl. those that survived a JSON persistence
		// roundtrip) print raw to avoid the noisy "\"escaped\""
		// form. Other values render as pretty JSON aligned under
		// the "  Payload: " label: prefix="  " puts continuation
		// lines (incl. the closing brace) at the label's column,
		// and indent="  " adds two spaces per nesting level — so
		// top-level keys sit two columns inside the opening brace.
		if s, ok := payload.(string); ok {
			fmt.Printf("  Payload: %s\n", s)
		} else if pretty, err := json.MarshalIndent(payload, "  ", "  "); err == nil {
			fmt.Printf("  Payload: %s\n", pretty)
		} else {
			fmt.Printf("  Payload: %v\n", payload)
		}
	}
	if schema, ok := args["responseSchema"]; ok && schema != nil {
		if pretty, err := json.Marshal(schema); err == nil {
			fmt.Printf("  Expected response schema: %s\n", pretty)
		}
	}
}

// workflowInputResponseFromUserInput shapes a one-line operator
// reply into a workflow-input FunctionResponse payload. Tries
// JSON first; a parsed object is returned verbatim so the
// operator can submit a fully-structured response, scalars and
// arrays are wrapped under "payload", and unparseable input is
// passed through under "payload" too.
func workflowInputResponseFromUserInput(line string) map[string]any {
	var parsed any
	if err := json.Unmarshal([]byte(line), &parsed); err == nil {
		if asMap, ok := parsed.(map[string]any); ok {
			return asMap
		}
		return map[string]any{"payload": parsed}
	}
	return map[string]any{"payload": line}
}

// renderGenericInterruptPrompt is the fallback for HITL kinds the
// launcher does not specifically recognise. Prints the kind name
// and the raw args so the operator can compose a sensible
// response by hand.
func renderGenericInterruptPrompt(name string, args map[string]any) {
	fmt.Printf("Agent -> waiting for response (kind: %s)\n", name)
	if len(args) > 0 {
		if pretty, err := json.Marshal(args); err == nil {
			fmt.Printf("  Args: %s\n", pretty)
		} else {
			fmt.Printf("  Args: %v\n", args)
		}
	}
}

// genericResponseFromUserInput shapes the operator's reply for
// HITL kinds without a dedicated parser. A parsed JSON object is
// returned verbatim so the operator can submit a fully-structured
// response; scalars and arrays are wrapped under "result"; raw
// text falls back to "result" too.
func genericResponseFromUserInput(line string) map[string]any {
	var parsed any
	if err := json.Unmarshal([]byte(line), &parsed); err == nil {
		if asMap, ok := parsed.(map[string]any); ok {
			return asMap
		}
		return map[string]any{"result": parsed}
	}
	return map[string]any{"result": line}
}
