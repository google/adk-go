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
	"reflect"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/adk/workflow"
)

// TestCollectPendingInterrupts_DetectionByLongRunningToolIDs verifies
// the Python-parity detection contract (cli.py:108-128): an event
// is a HITL prompt iff it has a non-empty LongRunningToolIDs and
// one of its content parts is a FunctionCall whose ID is in that
// set. The function name is not the discriminator — workflow input,
// tool confirmation, and any future kind all flow through the same
// detection path.
func TestCollectPendingInterrupts_DetectionByLongRunningToolIDs(t *testing.T) {
	tests := []struct {
		name   string
		events []*session.Event
		want   []pendingInterrupt
	}{
		{
			name:   "empty event list",
			events: nil,
			want:   nil,
		},
		{
			name: "event with FunctionCall but no LongRunningToolIDs is not an interrupt",
			events: []*session.Event{
				{
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{ID: "x", Name: "regular_tool"},
							}},
						},
					},
				},
			},
			want: nil,
		},
		{
			name: "event with LongRunningToolIDs but no matching FunctionCall is not an interrupt",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"abc"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{ID: "different_id", Name: "unmatched"},
							}},
						},
					},
				},
			},
			want: nil,
		},
		{
			name: "workflow input on Event.LLMResponse.Content is detected",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"int-1"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "int-1",
									Name: workflow.WorkflowInputFunctionCallName,
									Args: map[string]any{"message": "ok?"},
								},
							}},
						},
					},
				},
			},
			want: []pendingInterrupt{
				{id: "int-1", name: workflow.WorkflowInputFunctionCallName, args: map[string]any{"message": "ok?"}},
			},
		},
		{
			name: "tool confirmation on Event.LLMResponse.Content is detected",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"conf-1"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "conf-1",
									Name: toolconfirmation.FunctionCallName,
									Args: map[string]any{"toolConfirmation": map[string]any{"hint": "really delete?"}},
								},
							}},
						},
					},
				},
			},
			want: []pendingInterrupt{
				{id: "conf-1", name: toolconfirmation.FunctionCallName, args: map[string]any{"toolConfirmation": map[string]any{"hint": "really delete?"}}},
			},
		},
		{
			name: "multiple events, only ones with matching IDs surface",
			events: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "intro"}}}}},
				{
					LongRunningToolIDs: []string{"int-2"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "int-2", Name: "x"}}},
						},
					},
				},
				{LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "outro"}}}}},
			},
			want: []pendingInterrupt{
				{id: "int-2", name: "x", args: nil},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := collectPendingInterrupts(tc.events)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("collectPendingInterrupts() = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// TestBuildInterruptResponse_WorkflowInput verifies the
// workflow-input dispatch: the operator's line is wrapped under
// "payload" so the workflowagent's decodeWorkflowInputResponse
// reads it; JSON parsing is attempted first so structured replies
// round-trip as objects rather than strings.
func TestBuildInterruptResponse_WorkflowInput(t *testing.T) {
	tests := []struct {
		name      string
		userInput string
		wantValue any
	}{
		{
			name:      "raw text is passed through as a string",
			userInput: "approve\n",
			wantValue: "approve",
		},
		{
			name:      "JSON object is parsed and unwrapped",
			userInput: `{"approved": true, "days": 3}` + "\n",
			wantValue: map[string]any{"approved": true, "days": float64(3)},
		},
		{
			name:      "JSON scalar is parsed",
			userInput: "42\n",
			wantValue: float64(42),
		},
		{
			name:      "JSON array is parsed",
			userInput: `[1, 2, "three"]` + "\n",
			wantValue: []any{float64(1), float64(2), "three"},
		},
		{
			name:      "trailing CR is stripped",
			userInput: "approve\r\n",
			wantValue: "approve",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := pendingInterrupt{id: "x", name: workflow.WorkflowInputFunctionCallName}
			part := buildInterruptResponse(p, tc.userInput)
			if part.FunctionResponse == nil {
				t.Fatalf("expected FunctionResponse part, got %+v", part)
			}
			if got, want := part.FunctionResponse.ID, "x"; got != want {
				t.Errorf("ID = %q, want %q", got, want)
			}
			if got, want := part.FunctionResponse.Name, workflow.WorkflowInputFunctionCallName; got != want {
				t.Errorf("Name = %q, want %q", got, want)
			}
			payload, ok := part.FunctionResponse.Response["payload"]
			if !ok {
				t.Fatalf("Response missing 'payload' key; got %v", part.FunctionResponse.Response)
			}
			if !reflect.DeepEqual(payload, tc.wantValue) {
				t.Errorf("payload = %#v, want %#v", payload, tc.wantValue)
			}
		})
	}
}

// TestBuildInterruptResponse_ToolConfirmation verifies the
// tool-confirmation dispatch: positive answers map to
// {"confirmed": true}, everything else to {"confirmed": false},
// case-insensitive. Mirrors _is_positive_response in adk-python
// cli.py:131-133.
func TestBuildInterruptResponse_ToolConfirmation(t *testing.T) {
	tests := []struct {
		userInput string
		wantValue bool
	}{
		{"yes\n", true},
		{"YES\n", true},
		{" Yes \n", true},
		{"y\n", true},
		{"true\n", true},
		{"confirm\n", true},
		{"no\n", false},
		{"n\n", false},
		{"\n", false},
		{"anything else\n", false},
	}
	for _, tc := range tests {
		t.Run(tc.userInput, func(t *testing.T) {
			p := pendingInterrupt{id: "c", name: toolconfirmation.FunctionCallName}
			part := buildInterruptResponse(p, tc.userInput)
			confirmed, ok := part.FunctionResponse.Response["confirmed"]
			if !ok {
				t.Fatalf("Response missing 'confirmed'; got %v", part.FunctionResponse.Response)
			}
			if confirmed != tc.wantValue {
				t.Errorf("confirmed = %v, want %v", confirmed, tc.wantValue)
			}
		})
	}
}

// TestBuildInterruptResponse_GenericFallback verifies the catch-all
// path used for any long-running call name the launcher does not
// specifically know about.
func TestBuildInterruptResponse_GenericFallback(t *testing.T) {
	p := pendingInterrupt{id: "g", name: "adk_request_credential" /* hypothetical future kind */}
	part := buildInterruptResponse(p, "secret_value\n")
	got, ok := part.FunctionResponse.Response["result"]
	if !ok {
		t.Fatalf("Response missing 'result'; got %v", part.FunctionResponse.Response)
	}
	if got != "secret_value" {
		t.Errorf("result = %v, want %q", got, "secret_value")
	}
}
