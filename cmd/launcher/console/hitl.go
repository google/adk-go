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
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/adk/workflow"
)

// pendingInterrupt is one HITL prompt the agent emitted on the
// previous turn that still needs a user reply on the next.
//
// Detection mirrors the contract used by adk-python's
// run_interactively (cli.py:108-128): an interrupt is any
// FunctionCall part whose ID appears in
// Event.LongRunningToolIDs. The dispatch into "render this
// prompt" / "build that response" is then keyed on the call's
// Name (workflow input vs tool confirmation vs anything else).
type pendingInterrupt struct {
	id   string
	name string
	args map[string]any
}

// collectPendingInterrupts scans events for FunctionCall parts
// referenced from LongRunningToolIDs on the same event and returns
// them in order of appearance. A single event may carry multiple
// interrupts (rare but legal) — typically one per parallel pause
// in a workflow graph.
func collectPendingInterrupts(events []*session.Event) []pendingInterrupt {
	var out []pendingInterrupt
	for _, ev := range events {
		if ev == nil || len(ev.LongRunningToolIDs) == 0 {
			continue
		}
		lr := map[string]struct{}{}
		for _, id := range ev.LongRunningToolIDs {
			lr[id] = struct{}{}
		}
		// FunctionCall parts ride on Event.LLMResponse.Content
		// (the embedded model.LLMResponse field). Synthetic
		// HITL events — workflow's NewRequestInputEvent and
		// the LLM flow's generateRequestConfirmationEvent —
		// both populate this field with a single FunctionCall
		// part keyed in LongRunningToolIDs.
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
			out = append(out, pendingInterrupt{
				id:   fc.ID,
				name: fc.Name,
				args: fc.Args,
			})
		}
	}
	return out
}

// renderInterruptPrompt prints a one- or multi-line description
// of the pending interrupt to stdout and the "[user]: " input
// prompt below it. The caller is responsible for reading the
// user's reply from its existing stdin source (typically the
// shared inputChan) and feeding it into buildInterruptResponse.
//
// Mirrors adk-python cli.py:131-181 (_prompt_for_function_call):
// each long-running call name dispatches to a renderer/parser
// pair; the workflow-input and tool-confirmation paths are the
// two known kinds today.
func renderInterruptPrompt(p pendingInterrupt) {
	switch p.name {
	case workflow.WorkflowInputFunctionCallName:
		renderWorkflowInputPrompt(p.args)
	case toolconfirmation.FunctionCallName:
		renderToolConfirmationPrompt(p.args)
	default:
		fmt.Printf("[HITL] Waiting for input for %s(%v)\n", p.name, p.args)
	}
	fmt.Print("[user]: ")
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
	case toolconfirmation.FunctionCallName:
		response = toolConfirmationResponseFromUserInput(line)
	default:
		// Generic long-running call kind: pass the raw user
		// input through as {"result": <text>}, matching the
		// Python fallback. A future "adk_request_credential"
		// path would add its own renderer/parser pair above.
		response = map[string]any{"result": line}
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
	fmt.Printf("[HITL input] %s\n", msg)
	if payload, ok := args["payload"]; ok && payload != nil {
		if pretty, err := json.MarshalIndent(payload, "  ", "  "); err == nil {
			fmt.Printf("  Payload: %s\n", pretty)
		} else {
			fmt.Printf("  Payload: %v\n", payload)
		}
	}
	if schema, ok := args["responseSchema"]; ok && schema != nil {
		if pretty, err := json.Marshal(schema); err == nil {
			fmt.Printf("  Schema: %s\n", pretty)
		}
	}
}

// workflowInputResponseFromUserInput shapes a one-line operator
// reply into a workflow-input FunctionResponse payload. Tries
// JSON first (so the operator can submit objects, arrays,
// numbers, or booleans verbatim); falls back to passing the raw
// text through. The returned map is wrapped under "payload"
// because that is the key
// agent/workflowagent.decodeWorkflowInputResponse reads.
func workflowInputResponseFromUserInput(line string) map[string]any {
	var parsed any
	if err := json.Unmarshal([]byte(line), &parsed); err == nil {
		return map[string]any{"payload": parsed}
	}
	return map[string]any{"payload": line}
}

// renderToolConfirmationPrompt prints the tool-confirmation prompt.
// Args carry the fields populated by
// internal/llminternal/functions.go's generateRequestConfirmationEvent:
// "toolConfirmation" (with "hint") and "originalFunctionCall".
func renderToolConfirmationPrompt(args map[string]any) {
	hint := ""
	if tc, ok := args["toolConfirmation"].(map[string]any); ok {
		hint, _ = tc["hint"].(string)
	}
	if hint == "" {
		originalName := "unknown"
		if oc, ok := args["originalFunctionCall"].(map[string]any); ok {
			if name, _ := oc["name"].(string); name != "" {
				originalName = name
			}
		}
		hint = "Confirm " + originalName + "?"
	}
	fmt.Printf("[HITL confirm] %s\n", hint)
	fmt.Println("  Type 'yes' to confirm, anything else to reject.")
}

// toolConfirmationResponseFromUserInput maps yes-ish answers
// (y/yes/true/confirm, case-insensitive) to {"confirmed": true};
// everything else (including blank lines) maps to
// {"confirmed": false}. Mirrors _is_positive_response in
// adk-python cli.py:131-133.
func toolConfirmationResponseFromUserInput(line string) map[string]any {
	answer := strings.TrimSpace(strings.ToLower(line))
	switch answer {
	case "y", "yes", "true", "confirm":
		return map[string]any{"confirmed": true}
	default:
		return map[string]any{"confirmed": false}
	}
}
