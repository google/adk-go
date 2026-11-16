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

package workflow

import (
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// rootOutputSchema requires name and fills tag.
const rootOutputSchema = `{
	"type": "object",
	"required": ["name"],
	"properties": {
		"name": {"type": "string"},
		"tag": {"type": "string", "default": "ok"}
	}
}`

// rootInputSchema requires user and fills role.
const rootInputSchema = `{
	"type": "object",
	"required": ["user"],
	"properties": {
		"user": {"type": "string"},
		"role": {"type": "string", "default": "guest"}
	}
}`

// collect consumes seq and returns its events and its first error.
func collect(seq iter.Seq2[*session.Event, error]) ([]*session.Event, error) {
	var events []*session.Event
	var firstErr error
	for ev, err := range seq {
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		events = append(events, ev)
	}
	return events, firstErr
}

// fixedNode returns a schema-free node that records its input in log
// and outputs output.
func fixedNode(name string, output any, log *inputLog) Node {
	return NewDynamicNode(name, func(_ agent.Context, in any, _ func(*session.Event) error) (any, error) {
		log.record(name, in)
		return output, nil
	}, NodeConfig{})
}

// echoNode returns a schema-free node that records its input in log
// and outputs it.
func echoNode(name string, log *inputLog) Node {
	return NewDynamicNode(name, func(_ agent.Context, in any, _ func(*session.Event) error) (any, error) {
		log.record(name, in)
		return in, nil
	}, NodeConfig{})
}

// askNode returns a node that pauses for the interrupt "approval" and
// outputs nothing until it is answered.
func askNode(name string) Node {
	return NewDynamicNode(name, func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		return ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "approval", Message: "approve"})
	}, NodeConfig{})
}

func userText(ctx *MockInvocationContext, text string) *MockInvocationContext {
	ctx.userContent = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: text}}}
	return ctx
}

func hasInterrupt(events []*session.Event, id string) bool {
	for _, ev := range events {
		if slices.Contains(ev.LongRunningToolIDs, id) {
			return true
		}
	}
	return false
}

func TestWorkflow_WithOutputSchema_ValidatesAndCoercesTerminalAndPause(t *testing.T) {
	out := resolveSchemaJSON(t, rootOutputSchema)

	t.Run("valid_terminal_output", func(t *testing.T) {
		w, err := New("wf", []Edge{{From: Start, To: fixedNode("finish", map[string]any{"name": "done"}, &inputLog{})}}, WithOutputSchema(out))
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		events, err := collect(w.Run(newMockCtx(t)))
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"name": "done"}, outputsByNode(events)["finish"]); diff != "" {
			t.Errorf("terminal output (-want +got):\n%s", diff)
		}
	})

	// A required field that has a default is filled before validation,
	// as on node outputs.
	t.Run("required_field_with_default", func(t *testing.T) {
		required := resolveSchemaJSON(t, `{
			"type": "object",
			"required": ["name", "tag"],
			"properties": {
				"name": {"type": "string"},
				"tag": {"type": "string", "default": "ok"}
			}
		}`)
		w, err := New("wf", []Edge{{From: Start, To: fixedNode("finish", map[string]any{"name": "done"}, &inputLog{})}}, WithOutputSchema(required))
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		if _, err := collect(w.Run(newMockCtx(t))); err != nil {
			t.Fatalf("Run: %v", err)
		}
	})

	t.Run("invalid_terminal_output", func(t *testing.T) {
		w, err := New("wf", []Edge{{From: Start, To: fixedNode("finish", map[string]any{"age": 10}, &inputLog{})}}, WithOutputSchema(out))
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		events, err := collect(w.Run(newMockCtx(t)))

		if !errors.Is(err, ErrNodeFailed) {
			t.Fatalf("Run error = %v, want one wrapping ErrNodeFailed", err)
		}
		// The Go Adapter classifies a schema failure by this prefix,
		// which names the workflow, not its terminal node.
		if prefix := `workflow: dynamic child failed: output validation failed for node "wf": `; !strings.HasPrefix(err.Error(), prefix) {
			t.Errorf("Run error = %q, want prefix %q", err, prefix)
		}
		// The terminal node's own event is yielded before the failure.
		if diff := cmp.Diff(map[string]any{"age": 10}, outputsByNode(events)["finish"]); diff != "" {
			t.Errorf("terminal output (-want +got):\n%s", diff)
		}
	})

	// While a node waits, the run has not finished: an invalid output
	// of another terminal node is not validated yet.
	t.Run("waiting_node_defers_validation", func(t *testing.T) {
		w, err := New("wf", []Edge{
			{From: Start, To: askNode("ask")},
			{From: Start, To: fixedNode("side", map[string]any{"age": 10}, &inputLog{})},
		}, WithOutputSchema(out))
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		events, err := collect(w.Run(newMockCtx(t)))
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if !hasInterrupt(events, "approval") {
			t.Error("Run yielded no interrupt event for approval")
		}
	})

	t.Run("resume_validates_terminal_output", func(t *testing.T) {
		for _, tc := range []struct {
			name     string
			response map[string]any
			wantErr  error
		}{
			{name: "invalid", response: map[string]any{"age": 10}, wantErr: ErrNodeFailed},
			{name: "valid", response: map[string]any{"name": "done"}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				w, err := New("wf", []Edge{{From: Start, To: askNode("ask")}}, WithOutputSchema(out))
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				events, err := collect(w.Run(newMockCtx(t)))
				if err != nil {
					t.Fatalf("Run: %v", err)
				}
				if !hasInterrupt(events, "approval") {
					t.Fatal("Run yielded no interrupt event for approval")
				}
				state, err := w.ReconstructRunState(fakeSession{events: sliceEvents(events)}, "")
				if err != nil {
					t.Fatalf("ReconstructRunState: %v", err)
				}

				_, err = collect(w.Resume(agent.Promote(newMockCtx(t)), state, map[string]any{"approval": tc.response}))

				if tc.wantErr == nil && err != nil {
					t.Fatalf("Resume: %v", err)
				}
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("Resume error = %v, want one wrapping %v", err, tc.wantErr)
				}
			})
		}
	})
}

func TestWorkflow_WithInputSchema_RootNestedAndConcurrent(t *testing.T) {
	in := resolveSchemaJSON(t, rootInputSchema)
	out := resolveSchemaJSON(t, rootOutputSchema)

	t.Run("accessors", func(t *testing.T) {
		w, err := New("wf", []Edge{{From: Start, To: echoNode("echo", &inputLog{})}}, WithInputSchema(in), WithOutputSchema(out))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if w.InputSchema() != in || w.OutputSchema() != out {
			t.Error("InputSchema and OutputSchema do not return the schemas passed to New")
		}
		plain, err := New("plain", []Edge{{From: Start, To: echoNode("echo", &inputLog{})}})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if plain.InputSchema() != nil || plain.OutputSchema() != nil {
			t.Error("a workflow built without schema options reports a schema")
		}
	})

	t.Run("valid_user_message", func(t *testing.T) {
		log := &inputLog{}
		w, err := New("wf", []Edge{{From: Start, To: echoNode("echo", log)}}, WithInputSchema(in))
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		if _, err := collect(w.Run(userText(newMockCtx(t), `{"user":"u1"}`))); err != nil {
			t.Fatalf("Run: %v", err)
		}
		want := map[string][]any{"echo": {map[string]any{"user": "u1", "role": "guest"}}}
		if diff := cmp.Diff(want, log.recorded()); diff != "" {
			t.Errorf("inputs (-want +got):\n%s", diff)
		}
	})

	for _, text := range []string{`{"wrong": 1}`, `not-json`} {
		t.Run("invalid_user_message/"+text, func(t *testing.T) {
			log := &inputLog{}
			w, err := New("wf", []Edge{{From: Start, To: echoNode("echo", log)}}, WithInputSchema(in))
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			events, err := collect(w.Run(userText(newMockCtx(t), text)))

			if !errors.Is(err, ErrInputValidation) {
				t.Fatalf("Run error = %v, want one wrapping ErrInputValidation", err)
			}
			if len(events) != 0 || len(log.recorded()) != 0 {
				t.Errorf("Run yielded %d events and ran %d nodes, want none", len(events), len(log.recorded()))
			}
		})
	}

	// A nested workflow after Start receives the START seed itself and
	// validates the user message against its own input schema.
	t.Run("nested_start_successor", func(t *testing.T) {
		log := &inputLog{}
		inner, err := NewWorkflowNodeWithSchemas("inner", []Edge{{From: Start, To: echoNode("echo", log)}}, in, nil)
		if err != nil {
			t.Fatalf("NewWorkflowNodeWithSchemas: %v", err)
		}
		w, err := New("outer", []Edge{{From: Start, To: inner}})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		if _, err := collect(w.Run(userText(newMockCtx(t), `{"user":"u1"}`))); err != nil {
			t.Fatalf("Run: %v", err)
		}
		want := map[string][]any{"echo": {map[string]any{"user": "u1", "role": "guest"}}}
		if diff := cmp.Diff(want, log.recorded()); diff != "" {
			t.Errorf("inputs (-want +got):\n%s", diff)
		}
	})

	t.Run("concurrent_runs", func(t *testing.T) {
		w, err := New("wf", []Edge{{From: Start, To: echoNode("echo", &inputLog{})}}, WithInputSchema(in))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		var wg sync.WaitGroup
		for i := range 8 {
			wg.Go(func() {
				user := fmt.Sprintf("u%d", i)
				events, err := collect(w.Run(userText(newMockCtx(t), fmt.Sprintf(`{"user":%q}`, user))))
				if err != nil {
					t.Errorf("Run %s: %v", user, err)
					return
				}
				if diff := cmp.Diff(map[string]any{"user": user, "role": "guest"}, outputsByNode(events)["echo"]); diff != "" {
					t.Errorf("Run %s output (-want +got):\n%s", user, diff)
				}
			})
		}
		wg.Wait()
	})
}

// A root input schema validates the user message as Content. When the
// schema keeps the Content, each successor of Start receives the form it
// receives without a schema.
func TestWorkflow_WithInputSchema_KeepsUserContent(t *testing.T) {
	contentSchema, err := jsonschema.For[*genai.Content](nil)
	if err != nil {
		t.Fatalf("jsonschema.For: %v", err)
	}
	contentResolved, err := contentSchema.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	tests := []struct {
		name   string
		schema *jsonschema.Resolved
	}{
		{name: "empty_schema", schema: resolveSchemaJSON(t, `{}`)},
		{name: "content_schema", schema: contentResolved},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			log := &inputLog{}
			content := NewDynamicNode("content", func(_ agent.Context, in any, _ func(*session.Event) error) (any, error) {
				log.record("content", in)
				return nil, nil
			}, NodeConfig{ContentInput: true})
			w, err := New("wf", []Edge{{From: Start, To: content}, {From: Start, To: fixedNode("text", nil, log)}}, WithInputSchema(tc.schema))
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			if _, err := collect(w.Run(userText(newMockCtx(t), "hello"))); err != nil {
				t.Fatalf("Run: %v", err)
			}
			want := map[string][]any{
				"content": {&genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hello"}}}},
				"text":    {"hello"},
			}
			if diff := cmp.Diff(want, log.recorded()); diff != "" {
				t.Errorf("inputs (-want +got):\n%s", diff)
			}
		})
	}
}

// historyInvocationID is the invocation of the histories below.
const historyInvocationID = "inv-1"

// userMessageEvent is the stored user message text.
func userMessageEvent(text string) *session.Event {
	ev := &session.Event{Author: "user", InvocationID: historyInvocationID}
	ev.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: text}}}
	return ev
}

// answeredHistory is the history of a run whose user message is text,
// whose node ask paused for "approval", and whose user answered it.
func answeredHistory(text string) sliceEvents {
	raised := &session.Event{Author: "ask", InvocationID: historyInvocationID, LongRunningToolIDs: []string{"approval"}}
	reply := &session.Event{Author: "user", InvocationID: historyInvocationID}
	reply.Content = genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"response": "approved"}, genai.RoleUser)
	reply.Content.Parts[0].FunctionResponse.ID = "approval"
	return sliceEvents{userMessageEvent(text), raised, reply}
}

// reconstructSchema requires x as a string and fills tag.
const reconstructSchema = `{
	"type": "object",
	"required": ["x"],
	"properties": {
		"x": {"type": "string"},
		"tag": {"type": "string", "default": "d"}
	}
}`

// newReentryWorkflow returns a workflow whose only node ask re-runs on
// resume with the input it first received from Start.
func newReentryWorkflow(t *testing.T) *Workflow {
	t.Helper()
	rerun := true
	ask := NewDynamicNode("ask", func(_ agent.Context, in any, _ func(*session.Event) error) (any, error) {
		return in, nil
	}, NodeConfig{RerunOnResume: &rerun})
	w, err := New("wf", []Edge{{From: Start, To: ask}}, WithInputSchema(resolveSchemaJSON(t, reconstructSchema)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return w
}

func TestWorkflow_ReconstructRunState_RootInputSchema(t *testing.T) {
	t.Run("invalid_seed", func(t *testing.T) {
		w := newReentryWorkflow(t)

		_, err := w.ReconstructRunState(fakeSession{events: answeredHistory(`{"x": 1}`)}, historyInvocationID)

		if !errors.Is(err, ErrInputValidation) {
			t.Fatalf("ReconstructRunState error = %v, want one wrapping ErrInputValidation", err)
		}
	})

	// A failed node is reconstructed for retry, so its history is
	// validated even with no interrupt.
	t.Run("failed_node_history", func(t *testing.T) {
		w := newReentryWorkflow(t)
		failed := &session.Event{Author: "ask", InvocationID: historyInvocationID}
		failed.ErrorCode = workflowNodeFailureCode

		_, err := w.ReconstructRunState(fakeSession{events: sliceEvents{userMessageEvent(`{"x": 1}`), failed}}, historyInvocationID)

		if !errors.Is(err, ErrInputValidation) {
			t.Fatalf("ReconstructRunState error = %v, want one wrapping ErrInputValidation", err)
		}
	})

	t.Run("valid_seed_is_conformed", func(t *testing.T) {
		w := newReentryWorkflow(t)

		state, err := w.ReconstructRunState(fakeSession{events: answeredHistory(`{"x": "a"}`)}, historyInvocationID)
		if err != nil {
			t.Fatalf("ReconstructRunState: %v", err)
		}

		ns := nodeState(t, state, "ask")
		if ns.Status != NodePending {
			t.Fatalf("ask status = %v, want NodePending", ns.Status)
		}
		if diff := cmp.Diff(map[string]any{"x": "a", "tag": "d"}, ns.Input); diff != "" {
			t.Errorf("re-entry input (-want +got):\n%s", diff)
		}
	})
}

func TestWorkflow_ReconstructRunState_NothingToResumeSkipsValidation(t *testing.T) {
	w := newReentryWorkflow(t)
	history := sliceEvents{
		userMessageEvent(`{"x": 1}`),
		{Author: "ask", InvocationID: historyInvocationID, Output: "result"},
	}

	state, err := w.ReconstructRunState(fakeSession{events: history}, historyInvocationID)

	if err != nil || state != nil {
		t.Fatalf("ReconstructRunState = (%v, %v), want (nil, nil)", state, err)
	}
}
