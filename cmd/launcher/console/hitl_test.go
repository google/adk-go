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
	"io"
	"os"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// captureStdout runs f with os.Stdout redirected to a pipe and
// returns everything f wrote.
func captureStdout(t *testing.T, f func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	f()
	_ = w.Close()
	os.Stdout = orig
	return <-done
}

// TestCollectPendingInterrupts_DetectionByLongRunningToolIDs
// verifies the detection contract: an event is a HITL prompt
// iff it has a non-empty LongRunningToolIDs and one of its
// content parts is a FunctionCall whose ID is in that set. The
// function name is not the discriminator — workflow input and
// any future kind all flow through the same detection path.
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
		{
			// SSE streaming emits the same function-call event
			// repeatedly (partial chunks + a final aggregated
			// event), each carrying the same LongRunningToolIDs.
			// The interrupt must surface exactly once, from the
			// final event, so the console queues a single prompt.
			name: "duplicate call ID across partial and final events dedups to one",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"dup-1"},
					LLMResponse: model.LLMResponse{
						Partial: true,
						Content: &genai.Content{
							Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "dup-1", Name: "ask", Args: map[string]any{"a": float64(1)}}}},
						},
					},
				},
				{
					LongRunningToolIDs: []string{"dup-1"},
					LLMResponse: model.LLMResponse{
						Partial: false,
						Content: &genai.Content{
							Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "dup-1", Name: "ask", Args: map[string]any{"a": float64(1)}}}},
						},
					},
				},
			},
			want: []pendingInterrupt{
				{id: "dup-1", name: "ask", args: map[string]any{"a": float64(1)}},
			},
		},
		{
			// A partial-only event must never surface an
			// interrupt on its own; the settled prompt always
			// comes from the final aggregated event.
			name: "partial-only event does not surface an interrupt",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"p-1"},
					LLMResponse: model.LLMResponse{
						Partial: true,
						Content: &genai.Content{
							Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "p-1", Name: "ask"}}},
						},
					},
				},
			},
			want: nil,
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
// workflow-input dispatch: a JSON object reply is returned
// verbatim (the operator submitted the final shape); scalars,
// arrays, and raw text are wrapped under "payload".
func TestBuildInterruptResponse_WorkflowInput(t *testing.T) {
	tests := []struct {
		name         string
		userInput    string
		wantResponse map[string]any
	}{
		{
			name:         "raw text is wrapped under payload",
			userInput:    "approve\n",
			wantResponse: map[string]any{"payload": "approve"},
		},
		{
			name:         "JSON object is returned verbatim (no wrapper)",
			userInput:    `{"approved": true, "days": 3}` + "\n",
			wantResponse: map[string]any{"approved": true, "days": float64(3)},
		},
		{
			name:         "JSON scalar is wrapped under payload",
			userInput:    "42\n",
			wantResponse: map[string]any{"payload": float64(42)},
		},
		{
			name:         "JSON array is wrapped under payload",
			userInput:    `[1, 2, "three"]` + "\n",
			wantResponse: map[string]any{"payload": []any{float64(1), float64(2), "three"}},
		},
		{
			name:         "trailing CR is stripped",
			userInput:    "approve\r\n",
			wantResponse: map[string]any{"payload": "approve"},
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
			if !reflect.DeepEqual(part.FunctionResponse.Response, tc.wantResponse) {
				t.Errorf("Response = %#v, want %#v",
					part.FunctionResponse.Response, tc.wantResponse)
			}
		})
	}
}

// TestBuildInterruptResponse_GenericFallback verifies the catch-all
// path used for any long-running call name the launcher does not
// specifically know about.
func TestBuildInterruptResponse_GenericFallback(t *testing.T) {
	tests := []struct {
		name         string
		userInput    string
		wantResponse map[string]any
	}{
		{
			name:         "raw text is wrapped under result",
			userInput:    "some_value\n",
			wantResponse: map[string]any{"result": "some_value"},
		},
		{
			name:         "JSON object is returned verbatim (no wrapper)",
			userInput:    `{"foo": "bar"}` + "\n",
			wantResponse: map[string]any{"foo": "bar"},
		},
		{
			name:         "JSON scalar is wrapped under result",
			userInput:    "42\n",
			wantResponse: map[string]any{"result": float64(42)},
		},
		{
			name:         "JSON array is wrapped under result",
			userInput:    `[1, 2]` + "\n",
			wantResponse: map[string]any{"result": []any{float64(1), float64(2)}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := pendingInterrupt{id: "g", name: "some_unknown_kind"}
			part := buildInterruptResponse(p, tc.userInput)
			if !reflect.DeepEqual(part.FunctionResponse.Response, tc.wantResponse) {
				t.Errorf("Response = %#v, want %#v",
					part.FunctionResponse.Response, tc.wantResponse)
			}
		})
	}
}

// TestRenderWorkflowInputPrompt_PayloadFormatting locks in the
// payload-rendering contract:
//
//   - a string payload prints raw (no surrounding quotes, no
//     escapes) so a value that survived a JSON roundtrip stays
//     human-readable;
//   - an object payload prints as JSON aligned under the
//     "  Payload: " label: top-level keys are indented 4 spaces
//     (2-space label column + 2-space JSON indent) and the closing
//     brace sits at the 2-space label column. Drop the prefix and
//     keys/brace collapse left of the label, producing visually
//     broken output.
func TestRenderWorkflowInputPrompt_PayloadFormatting(t *testing.T) {
	tests := []struct {
		name         string
		args         map[string]any
		wantContains []string
		wantAbsent   []string
	}{
		{
			name: "string payload prints raw, no escaped quotes",
			args: map[string]any{
				"message": "approve?",
				"payload": "hello world",
			},
			wantContains: []string{
				"Agent -> approve?\n",
				"  Payload: hello world\n",
			},
			wantAbsent: []string{`"hello world"`, `\"`},
		},
		{
			name: "object payload is indented under the Payload label",
			args: map[string]any{
				"message": "ok?",
				"payload": map[string]any{"user": "alice", "days": float64(5)},
			},
			// Expected exact form. prefix="  " is appended at the
			// start of every continuation line; indent="  " adds
			// one more level for object keys. So keys land at
			// column 4 ("    \"days\"") and the closing brace
			// at column 2 ("  }"):
			//   Agent -> ok?
			//     Payload: {
			//         "days": 5,
			//         "user": "alice"
			//       }
			// (Go map keys serialise in sorted order, so "days"
			// precedes "user".)
			wantContains: []string{
				"  Payload: {\n",
				"\n    \"days\": 5",
				"\n    \"user\": \"alice\"",
				"\n  }\n",
			},
			// Without prefix="  ", keys would land flush at
			// column 2 (one indent level, no prefix) and the
			// closing brace at column 0; assert neither happens.
			wantAbsent: []string{
				"\n  \"days\"",
				"\n}\n",
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := captureStdout(t, func() {
				renderWorkflowInputPrompt(tc.args)
			})
			for _, s := range tc.wantContains {
				if !strings.Contains(out, s) {
					t.Errorf("output missing %q\nfull output:\n%s", s, out)
				}
			}
			for _, s := range tc.wantAbsent {
				if strings.Contains(out, s) {
					t.Errorf("output contains forbidden %q\nfull output:\n%s", s, out)
				}
			}
		})
	}
}
