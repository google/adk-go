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

package workflow_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// Profile is the Go form of the "profile" property of schema S.
type Profile struct {
	Age int `json:"age"`
}

// stateSchemaS returns schema S: title "PipelineSchema", counter (integer,
// minimum 1), name (string), optional_field (string or null), profile (object
// with an integer age), and required [counter, name].
func stateSchemaS(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	one := 1.0
	s := &jsonschema.Schema{
		Title: "PipelineSchema",
		Type:  "object",
		Properties: map[string]*jsonschema.Schema{
			"counter":        {Type: "integer", Minimum: &one},
			"name":           {Type: "string"},
			"optional_field": {Types: []string{"string", "null"}},
			"profile": {
				Type:       "object",
				Properties: map[string]*jsonschema.Schema{"age": {Type: "integer"}},
			},
		},
		Required: []string{"counter", "name"},
	}
	r, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve(schema S) error = %v", err)
	}
	return r
}

// runStateWorkflow runs wf as the root agent of a runner over an in-memory
// session that starts with initial, and returns every event the caller saw,
// the stored session state after the run, and the run's error.
func runStateWorkflow(t *testing.T, wf *workflow.Workflow, initial map[string]any) (events []*session.Event, stored map[string]any, err error) {
	t.Helper()
	return runStateWorkflowCtx(t, t.Context(), wf, initial)
}

func runStateWorkflowCtx(t *testing.T, ctx context.Context, wf *workflow.Workflow, initial map[string]any) (events []*session.Event, stored map[string]any, err error) {
	t.Helper()
	svc := session.InMemoryService()
	r, sessionID := newStateRunner(t, svc, wf, initial)
	for ev, runErr := range r.Run(ctx, "u", sessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			if err == nil {
				err = runErr
			}
			continue
		}
		events = append(events, ev)
	}
	return events, storedState(t, svc, sessionID), err
}

func newStateRunner(t *testing.T, svc session.Service, wf *workflow.Workflow, initial map[string]any) (*runner.Runner, string) {
	t.Helper()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", State: initial})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	a, err := agent.New(agent.Config{Name: "wf", Run: wf.Run})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r, created.Session.ID()
}

func storedState(t *testing.T, svc session.Service, sessionID string) map[string]any {
	t.Helper()
	got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: sessionID})
	if err != nil {
		t.Fatalf("session Get() error = %v", err)
	}
	return maps.Collect(got.Session.State().All())
}

// stateNode is a custom node whose Run is given by the test.
type stateNode struct {
	workflow.BaseNode
	run func(ctx agent.Context, yield func(*session.Event, error) bool)
}

func newStateNode(name string, run func(ctx agent.Context, yield func(*session.Event, error) bool)) *stateNode {
	return &stateNode{BaseNode: workflow.NewBaseNode(name, "", workflow.NodeConfig{}), run: run}
}

func (n *stateNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.run(ctx, yield)
	}
}

// outputEvent builds an event with the given output and state delta.
func outputEvent(ctx agent.Context, output any, delta map[string]any) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Output = output
	maps.Copy(ev.Actions.StateDelta, delta)
	return ev
}

func newStateWorkflow(t *testing.T, node workflow.Node, opts ...workflow.Option) *workflow.Workflow {
	t.Helper()
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node), opts...)
	if err != nil {
		t.Fatalf("workflow.New() error = %v", err)
	}
	return wf
}

func asStateSchemaError(t *testing.T, err error) *workflow.StateSchemaError {
	t.Helper()
	var se *workflow.StateSchemaError
	if !errors.As(err, &se) {
		t.Fatalf("run error = %v, want a *workflow.StateSchemaError", err)
	}
	return se
}

// eventsWithKey returns the caller-visible events whose state delta has key.
func eventsWithKey(events []*session.Event, key string) []*session.Event {
	var out []*session.Event
	for _, ev := range events {
		if _, ok := ev.Actions.StateDelta[key]; ok {
			out = append(out, ev)
		}
	}
	return out
}

// eventsWithArtifact returns the caller-visible events whose artifact delta has
// name.
func eventsWithArtifact(events []*session.Event, name string) []*session.Event {
	var out []*session.Event
	for _, ev := range events {
		if _, ok := ev.Actions.ArtifactDelta[name]; ok {
			out = append(out, ev)
		}
	}
	return out
}

func eventWithOutput(t *testing.T, events []*session.Event, output any) *session.Event {
	t.Helper()
	for _, ev := range events {
		if ev.Output == output {
			return ev
		}
	}
	t.Fatalf("no caller-visible event has Output %v", output)
	return nil
}

func hasOutput(events []*session.Event, output any) bool {
	for _, ev := range events {
		if ev.Output == output {
			return true
		}
	}
	return false
}

func TestStateSchemaError_UndeclaredKey(t *testing.T) {
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		return "", ctx.State().Set("unknown_key", "hunter2")
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	_, _, err := runStateWorkflow(t, wf, nil)

	se := asStateSchemaError(t, err)
	want := &workflow.StateSchemaError{
		Key:      "unknown_key",
		Schema:   "PipelineSchema",
		Reason:   workflow.StateKeyUndeclared,
		Declared: []string{"counter", "name", "optional_field", "profile"},
	}
	if diff := cmp.Diff(want, se); diff != "" {
		t.Errorf("StateSchemaError mismatch (-want +got):\n%s", diff)
	}
	if strings.Contains(err.Error(), "hunter2") {
		t.Errorf("run error %q contains the rejected value", err.Error())
	}
}

// Each StateSchemaError owns its Declared slice: a caller that edits one
// error's list does not change the list of a later error.
func TestStateSchemaError_DeclaredIsACopy(t *testing.T) {
	var second error
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		var first *workflow.StateSchemaError
		if !errors.As(ctx.State().Set("x", 1), &first) {
			return "", errors.New("Set(x) was not refused")
		}
		first.Declared[0] = "edited"
		second = ctx.State().Set("y", 1)
		return "ok", nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	if _, _, err := runStateWorkflow(t, wf, nil); err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	se := asStateSchemaError(t, second)
	if diff := cmp.Diff([]string{"counter", "name", "optional_field", "profile"}, se.Declared); diff != "" {
		t.Errorf("second error's Declared mismatch (-want +got):\n%s", diff)
	}
}

func TestStateSchemaError_ErrorText(t *testing.T) {
	declared := []string{"counter", "name", "optional_field"}
	tests := []struct {
		name string
		err  *workflow.StateSchemaError
		want string
	}{
		{
			name: "undeclared with title",
			err:  &workflow.StateSchemaError{Key: "unknown_key", Schema: "PipelineSchema", Reason: workflow.StateKeyUndeclared, Declared: declared},
			want: `workflow: state key "unknown_key" is not declared in the state schema "PipelineSchema"; declared: [counter name optional_field]`,
		},
		{
			name: "undeclared without title",
			err:  &workflow.StateSchemaError{Key: "unknown_key", Reason: workflow.StateKeyUndeclared, Declared: declared},
			want: `workflow: state key "unknown_key" is not declared in the state schema; declared: [counter name optional_field]`,
		},
		{
			name: "invalid with type and title",
			err:  &workflow.StateSchemaError{Key: "counter", Schema: "PipelineSchema", Reason: workflow.StateValueInvalid, Type: "integer"},
			want: `workflow: value for state key "counter" does not satisfy type integer in the state schema "PipelineSchema"`,
		},
		{
			name: "invalid without type",
			err:  &workflow.StateSchemaError{Key: "tag", Schema: "PipelineSchema", Reason: workflow.StateValueInvalid},
			want: `workflow: value for state key "tag" does not satisfy its property in the state schema "PipelineSchema"`,
		},
		{
			name: "invalid with type, no title",
			err:  &workflow.StateSchemaError{Key: "counter", Reason: workflow.StateValueInvalid, Type: "integer"},
			want: `workflow: value for state key "counter" does not satisfy type integer in the state schema`,
		},
		{
			name: "invalid without type or title",
			err:  &workflow.StateSchemaError{Key: "tag", Reason: workflow.StateValueInvalid},
			want: `workflow: value for state key "tag" does not satisfy its property in the state schema`,
		},
		{
			name: "zero reason",
			err:  &workflow.StateSchemaError{Key: "k", Schema: "PipelineSchema"},
			want: `workflow: state key "k" refused by the state schema`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.err.Error(); got != tc.want {
				t.Errorf("Error() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestStateSchema_ValueRules(t *testing.T) {
	enumSchema := func(t *testing.T) *jsonschema.Resolved {
		t.Helper()
		s := &jsonschema.Schema{
			Title:      "PipelineSchema",
			Properties: map[string]*jsonschema.Schema{"tag": {Enum: []any{"a", "b"}}},
		}
		r, err := s.Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve(enum schema) error = %v", err)
		}
		return r
	}
	unconstrainedSchema := func(t *testing.T) *jsonschema.Resolved {
		t.Helper()
		s := &jsonschema.Schema{
			Title:      "PipelineSchema",
			Properties: map[string]*jsonschema.Schema{"free": {}},
		}
		r, err := s.Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve(unconstrained schema) error = %v", err)
		}
		return r
	}
	tests := []struct {
		name   string
		schema func(*testing.T) *jsonschema.Resolved
		key    string
		value  any
		// wantGoType is the %T of the value read back after an accepted write.
		wantGoType string
		// wantErr is the Error() text of a refused write; empty for an accepted one.
		wantErr  string
		wantType string
	}{
		{name: "int", key: "counter", value: 5, wantGoType: "int"},
		{name: "float64 integral", key: "counter", value: float64(5), wantGoType: "float64"},
		{name: "minimum boundary", key: "counter", value: 1, wantGoType: "int"},
		{
			name: "below minimum", key: "counter", value: 0, wantType: "integer",
			wantErr: `workflow: value for state key "counter" does not satisfy type integer in the state schema "PipelineSchema"`,
		},
		{
			name: "fraction", key: "counter", value: 5.5, wantType: "integer",
			wantErr: `workflow: value for state key "counter" does not satisfy type integer in the state schema "PipelineSchema"`,
		},
		{
			name: "string for integer", key: "counter", value: "secret-value", wantType: "integer",
			wantErr: `workflow: value for state key "counter" does not satisfy type integer in the state schema "PipelineSchema"`,
		},
		{name: "nil for nullable", key: "optional_field", value: nil, wantGoType: "<nil>"},
		{
			name: "int for type list", key: "optional_field", value: 5, wantType: "string|null",
			wantErr: `workflow: value for state key "optional_field" does not satisfy type string|null in the state schema "PipelineSchema"`,
		},
		{
			name: "nil for string", key: "name", value: nil, wantType: "string",
			wantErr: `workflow: value for state key "name" does not satisfy type string in the state schema "PipelineSchema"`,
		},
		{
			name: "value with no JSON form", key: "name", value: make(chan int), wantType: "string",
			wantErr: `workflow: value for state key "name" does not satisfy type string in the state schema "PipelineSchema"`,
		},
		{
			name: "value with no JSON form, unconstrained property", schema: unconstrainedSchema, key: "free", value: make(chan int),
			wantErr: `workflow: value for state key "free" does not satisfy its property in the state schema "PipelineSchema"`,
		},
		{name: "struct", key: "profile", value: Profile{Age: 3}, wantGoType: "workflow_test.Profile"},
		{
			name: "outside enum", schema: enumSchema, key: "tag", value: "c",
			wantErr: `workflow: value for state key "tag" does not satisfy its property in the state schema "PipelineSchema"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			schema := stateSchemaS(t)
			if tc.schema != nil {
				schema = tc.schema(t)
			}
			var gotGoType string
			node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
				if err := ctx.State().Set(tc.key, tc.value); err != nil {
					return "", err
				}
				v, err := ctx.State().Get(tc.key)
				if err != nil {
					return "", err
				}
				gotGoType = fmt.Sprintf("%T", v)
				return "ok", nil
			}, workflow.NodeConfig{})
			wf := newStateWorkflow(t, node, workflow.WithStateSchema(schema))

			_, _, err := runStateWorkflow(t, wf, nil)

			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("run error = %v, want nil", err)
				}
				if gotGoType != tc.wantGoType {
					t.Errorf("value read back has type %s, want %s", gotGoType, tc.wantGoType)
				}
				return
			}
			se := asStateSchemaError(t, err)
			want := &workflow.StateSchemaError{Key: tc.key, Schema: "PipelineSchema", Reason: workflow.StateValueInvalid, Type: tc.wantType}
			if diff := cmp.Diff(want, se); diff != "" {
				t.Errorf("StateSchemaError mismatch (-want +got):\n%s", diff)
			}
			if got := err.Error(); got != tc.wantErr {
				t.Errorf("run error = %q, want %q", got, tc.wantErr)
			}
		})
	}
}

func TestStateSchema_NoSchemaAcceptsAnyEventDelta(t *testing.T) {
	node := newStateNode("emitter", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		yield(outputEvent(ctx, "done", map[string]any{"anything": map[string]any{"a": 1}, "counter": "x"}), nil)
	})
	wf := newStateWorkflow(t, node)

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if diff := cmp.Diff(map[string]any{"a": 1}, stored["anything"]); diff != "" {
		t.Errorf("stored anything mismatch (-want +got):\n%s", diff)
	}
	if got := stored["counter"]; got != "x" {
		t.Errorf("stored counter = %v, want x", got)
	}
}

func TestStateSchema_PartialWriteIsNotRejected(t *testing.T) {
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		return "ok", ctx.State().Set("counter", 5)
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if got := stored["counter"]; got != 5 {
		t.Errorf("stored counter = %v, want 5", got)
	}
	if _, ok := stored["name"]; ok {
		t.Errorf("stored state has name = %v, want it absent", stored["name"])
	}
}

func TestStateSchema_FailedNodeKeepsEarlierValidWrites(t *testing.T) {
	errBoom := errors.New("boom")
	tests := []struct {
		name    string
		fail    func(ctx agent.Context) error
		checkEr func(t *testing.T, err error)
	}{
		{
			name: "refused write",
			fail: func(ctx agent.Context) error { return ctx.State().Set("unknown_key", "v") },
			checkEr: func(t *testing.T, err error) {
				if se := asStateSchemaError(t, err); se.Key != "unknown_key" {
					t.Errorf("StateSchemaError.Key = %q, want unknown_key", se.Key)
				}
			},
		},
		{
			name: "error",
			fail: func(agent.Context) error { return errBoom },
			checkEr: func(t *testing.T, err error) {
				if !errors.Is(err, errBoom) {
					t.Errorf("run error = %v, want errBoom", err)
				}
			},
		},
		{
			name: "panic",
			fail: func(agent.Context) error { panic("kaboom") },
			checkEr: func(t *testing.T, err error) {
				if err == nil || !strings.Contains(err.Error(), "panicked") {
					t.Errorf("run error = %v, want a panic error", err)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
				if err := ctx.State().Set("counter", 1); err != nil {
					return "", err
				}
				return "", tc.fail(ctx)
			}, workflow.NodeConfig{})
			wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

			events, stored, err := runStateWorkflow(t, wf, nil)

			tc.checkEr(t, err)
			if got := stored["counter"]; got != 1 {
				t.Errorf("stored counter = %v, want 1", got)
			}
			if _, ok := stored["unknown_key"]; ok {
				t.Errorf("stored state has unknown_key")
			}
			if got := eventsWithKey(events, "unknown_key"); len(got) != 0 {
				t.Errorf("%d caller-visible events carry unknown_key, want 0", len(got))
			}
			withCounter := eventsWithKey(events, "counter")
			if len(withCounter) != 1 {
				t.Fatalf("%d caller-visible events carry counter, want 1", len(withCounter))
			}
			ev := withCounter[0]
			if diff := cmp.Diff(map[string]any{"counter": 1}, ev.Actions.StateDelta); diff != "" {
				t.Errorf("state-only event StateDelta mismatch (-want +got):\n%s", diff)
			}
			if ev.Output != nil || ev.Content != nil {
				t.Errorf("state-only event has Output %v and Content %v, want both nil", ev.Output, ev.Content)
			}
		})
	}
}

func TestStateSchema_OffendingEventIsNotEmitted(t *testing.T) {
	node := newStateNode("emitter", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if err := ctx.State().Set("counter", 1); err != nil {
			yield(nil, err)
			return
		}
		yield(outputEvent(ctx, "done", map[string]any{"b_bad": "value", "a_bad": "value"}), nil)
	})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	events, stored, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "a_bad" {
		t.Errorf("StateSchemaError.Key = %q, want a_bad (the first refused key in sorted order)", se.Key)
	}
	if hasOutput(events, "done") {
		t.Errorf("the refused event reached the caller")
	}
	if got := len(eventsWithKey(events, "a_bad")) + len(eventsWithKey(events, "b_bad")); got != 0 {
		t.Errorf("%d caller-visible events carry a refused key, want 0", got)
	}
	if _, ok := stored["a_bad"]; ok {
		t.Errorf("stored state has a_bad")
	}
	if got := stored["counter"]; got != 1 {
		t.Errorf("stored counter = %v, want 1", got)
	}
}

// A partial event's state delta is checked like any other event's.
func TestStateSchema_PartialEventDeltaChecked(t *testing.T) {
	node := newStateNode("emitter", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		ev := outputEvent(ctx, nil, map[string]any{"unknown": 1})
		ev.Partial = true
		if !yield(ev, nil) {
			return
		}
		yield(outputEvent(ctx, "done", nil), nil)
	})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	_, _, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
}

func TestStateSchema_ColonKeysBypass(t *testing.T) {
	node := newStateNode("emitter", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if err := ctx.State().Set("users:x", "v"); err != nil {
			yield(nil, err)
			return
		}
		yield(outputEvent(ctx, "done", map[string]any{"adk_oauth_state:wf@1/n@1": "s", "temp:t": true}), nil)
	})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if got := stored["users:x"]; got != "v" {
		t.Errorf("stored users:x = %v, want v", got)
	}
	if got := stored["adk_oauth_state:wf@1/n@1"]; got != "s" {
		t.Errorf("stored adk_oauth_state:wf@1/n@1 = %v, want s", got)
	}
}

// placed is what TestNodeWrites_Placement compares of one caller-visible event.
type placed struct {
	Output  any
	Text    string
	Partial bool
	Delta   map[string]any
}

func TestNodeWrites_Placement(t *testing.T) {
	set := func(t *testing.T, ctx agent.Context, key string, value any) {
		t.Helper()
		if err := ctx.State().Set(key, value); err != nil {
			t.Errorf("State().Set(%q) error = %v", key, err)
		}
	}
	textEvent := func(ctx agent.Context, text string, partial bool) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Content = genai.NewContentFromText(text, genai.RoleModel)
		ev.Partial = partial
		return ev
	}
	tests := []struct {
		name string
		run  func(t *testing.T, ctx agent.Context, yield func(*session.Event, error) bool)
		want []placed
	}{
		{
			name: "no event",
			run: func(t *testing.T, ctx agent.Context, _ func(*session.Event, error) bool) {
				set(t, ctx, "counter", 7)
			},
			want: []placed{{Delta: map[string]any{"counter": 7}}},
		},
		{
			name: "write after the last event",
			run: func(t *testing.T, ctx agent.Context, yield func(*session.Event, error) bool) {
				if !yield(outputEvent(ctx, "a", nil), nil) {
					return
				}
				set(t, ctx, "name", "late")
			},
			want: []placed{{Output: "a"}, {Delta: map[string]any{"name": "late"}}},
		},
		{
			name: "partial event carries nothing",
			run: func(t *testing.T, ctx agent.Context, yield func(*session.Event, error) bool) {
				set(t, ctx, "counter", 1)
				if !yield(textEvent(ctx, "part", true), nil) {
					return
				}
				yield(outputEvent(ctx, "b", nil), nil)
			},
			want: []placed{{Text: "part", Partial: true}, {Output: "b", Delta: map[string]any{"counter": 1}}},
		},
		{
			name: "sent once",
			run: func(t *testing.T, ctx agent.Context, yield func(*session.Event, error) bool) {
				set(t, ctx, "counter", 1)
				if !yield(outputEvent(ctx, "a", nil), nil) {
					return
				}
				yield(textEvent(ctx, "b", false), nil)
			},
			want: []placed{{Output: "a", Delta: map[string]any{"counter": 1}}, {Text: "b"}},
		},
		{
			name: "last write wins",
			run: func(t *testing.T, ctx agent.Context, yield func(*session.Event, error) bool) {
				set(t, ctx, "counter", 2)
				set(t, ctx, "counter", 3)
				yield(outputEvent(ctx, "a", nil), nil)
			},
			want: []placed{{Output: "a", Delta: map[string]any{"counter": 3}}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node := newStateNode("n", func(ctx agent.Context, yield func(*session.Event, error) bool) {
				tc.run(t, ctx, yield)
			})
			wf := newStateWorkflow(t, node)

			events, _, err := runStateWorkflow(t, wf, nil)
			if err != nil {
				t.Fatalf("run error = %v, want nil", err)
			}

			var got []placed
			for _, ev := range events {
				if ev.Output == nil && ev.Content == nil && !ev.Partial && len(ev.Actions.StateDelta) == 0 {
					continue // the scheduler's completion record
				}
				p := placed{Output: ev.Output, Partial: ev.Partial, Delta: ev.Actions.StateDelta}
				if ev.Content != nil && len(ev.Content.Parts) > 0 {
					p.Text = ev.Content.Parts[0].Text
				}
				got = append(got, p)
			}
			if diff := cmp.Diff(tc.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("caller-visible events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNodeContext_DeltaOnlyEntryIsPersisted(t *testing.T) {
	var seenDelta, seenValue any
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		ctx.Actions().StateDelta["in_delta"] = 2
		var err error
		if seenDelta, err = ctx.State().Get("in_delta"); err != nil {
			return "", err
		}
		if seenValue, err = ctx.State().Get("in_value"); err != nil {
			return "", err
		}
		return "done", nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node)

	_, stored, err := runStateWorkflow(t, wf, map[string]any{"in_value": 1})
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if got := stored["in_delta"]; got != 2 {
		t.Errorf("stored in_delta = %v, want 2", got)
	}
	if seenDelta != 2 || seenValue != 1 {
		t.Errorf("node read in_delta = %v and in_value = %v, want 2 and 1", seenDelta, seenValue)
	}
}

// awaitBoth blocks until both callers arrived, or fails after five seconds.
func awaitBoth(arrived *sync.WaitGroup) error {
	arrived.Done()
	done := make(chan struct{})
	go func() {
		arrived.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-time.After(5 * time.Second):
		return errors.New("the other node did not start within 5s")
	}
}

func TestNodeContext_FanOutKeepsDeltasApart(t *testing.T) {
	var arrived sync.WaitGroup
	arrived.Add(2)
	a := workflow.NewFunctionNode("A", func(ctx agent.Context, _ any) (string, error) {
		if err := awaitBoth(&arrived); err != nil {
			return "", err
		}
		return "a", ctx.State().Set("counter", 1)
	}, workflow.NodeConfig{})
	b := workflow.NewFunctionNode("B", func(ctx agent.Context, _ any) (string, error) {
		if err := awaitBoth(&arrived); err != nil {
			return "", err
		}
		return "b", ctx.State().Set("name", "b")
	}, workflow.NodeConfig{})
	end := workflow.NewJoinNode("end")
	wf, err := workflow.New("wf", []workflow.Edge{
		{From: workflow.Start, To: a},
		{From: workflow.Start, To: b},
		{From: a, To: end},
		{From: b, To: end},
	})
	if err != nil {
		t.Fatalf("workflow.New() error = %v", err)
	}

	events, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if stored["counter"] != 1 || stored["name"] != "b" {
		t.Errorf("stored counter = %v and name = %v, want 1 and b", stored["counter"], stored["name"])
	}
	if diff := cmp.Diff(map[string]any{"counter": 1}, eventWithOutput(t, events, "a").Actions.StateDelta); diff != "" {
		t.Errorf("A's event StateDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"name": "b"}, eventWithOutput(t, events, "b").Actions.StateDelta); diff != "" {
		t.Errorf("B's event StateDelta mismatch (-want +got):\n%s", diff)
	}
}

// childFailureWriter returns a node that writes counter = 1 and then fails
// with the error a dynamic node returns when its child failed, as long as fail
// reports true; otherwise it runs after.
func childFailureWriter(fail func() bool, after func(ctx agent.Context) (string, error), cfg workflow.NodeConfig) workflow.Node {
	return workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		if !fail() {
			return after(ctx)
		}
		if err := ctx.State().Set("counter", 1); err != nil {
			return "", err
		}
		return "", &workflow.NodeRunError{
			ChildName: "child",
			ChildPath: "wf@1/writer@1/child@1",
			Cause:     fmt.Errorf("%w: child fails", workflow.ErrNodeFailed),
		}
	}, cfg)
}

func TestRunNode_CancelDiscardsPending(t *testing.T) {
	t.Run("caller cancels", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		node := workflow.NewFunctionNode("writer", func(nctx agent.Context, _ any) (string, error) {
			if err := nctx.State().Set("counter", 1); err != nil {
				return "", err
			}
			cancel()
			<-nctx.Done()
			return "", nctx.Err()
		}, workflow.NodeConfig{})
		wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

		events, stored, err := runStateWorkflowCtx(t, ctx, wf, nil)

		if !errors.Is(err, context.Canceled) {
			t.Errorf("run error = %v, want context.Canceled", err)
		}
		if _, ok := stored["counter"]; ok {
			t.Errorf("stored counter = %v, want it absent", stored["counter"])
		}
		if got := eventsWithKey(events, "counter"); len(got) != 0 {
			t.Errorf("%d caller-visible events carry counter, want 0", len(got))
		}
	})
	t.Run("node returns a cancellation", func(t *testing.T) {
		node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
			if err := ctx.State().Set("counter", 1); err != nil {
				return "", err
			}
			return "", fmt.Errorf("lookup: %w", context.Canceled)
		}, workflow.NodeConfig{})
		wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

		// The scheduler records the node as canceled; the run itself does not fail.
		events, stored, _ := runStateWorkflow(t, wf, nil)

		if _, ok := stored["counter"]; ok {
			t.Errorf("stored counter = %v, want it absent", stored["counter"])
		}
		if got := eventsWithKey(events, "counter"); len(got) != 0 {
			t.Errorf("%d caller-visible events carry counter, want 0", len(got))
		}
	})
	t.Run("dynamic child fails", func(t *testing.T) {
		node := childFailureWriter(func() bool { return true }, nil, workflow.NodeConfig{})
		wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

		events, stored, err := runStateWorkflow(t, wf, nil)

		if !errors.Is(err, workflow.ErrNodeFailed) {
			t.Errorf("run error = %v, want ErrNodeFailed", err)
		}
		if _, ok := stored["counter"]; ok {
			t.Errorf("stored counter = %v, want it absent", stored["counter"])
		}
		if got := eventsWithKey(events, "counter"); len(got) != 0 {
			t.Errorf("%d caller-visible events carry counter, want 0", len(got))
		}
	})
}

func TestRunNode_RetryStartsFreshPending(t *testing.T) {
	retry := workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 2, InitialDelay: time.Millisecond}}
	secondAttempt := func(ctx agent.Context) (string, error) {
		return "ok", ctx.State().Set("name", "x")
	}
	t.Run("first attempt returns an error", func(t *testing.T) {
		var attempts atomic.Int32
		node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
			if attempts.Add(1) > 1 {
				return secondAttempt(ctx)
			}
			if err := ctx.State().Set("counter", 1); err != nil {
				return "", err
			}
			return "", errors.New("first attempt fails")
		}, retry)
		wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

		events, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if stored["counter"] != 1 || stored["name"] != "x" {
			t.Errorf("stored counter = %v and name = %v, want 1 and x", stored["counter"], stored["name"])
		}
		if got := eventsWithKey(events, "counter"); len(got) != 1 {
			t.Errorf("%d caller-visible events carry counter, want 1", len(got))
		}
		if diff := cmp.Diff(map[string]any{"name": "x"}, eventWithOutput(t, events, "ok").Actions.StateDelta); diff != "" {
			t.Errorf("second attempt's event StateDelta mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("first attempt fails with a child failure", func(t *testing.T) {
		var attempts atomic.Int32
		node := childFailureWriter(func() bool { return attempts.Add(1) == 1 }, secondAttempt, retry)
		wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

		events, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if _, ok := stored["counter"]; ok {
			t.Errorf("stored counter = %v, want it absent", stored["counter"])
		}
		if diff := cmp.Diff(map[string]any{"name": "x"}, eventWithOutput(t, events, "ok").Actions.StateDelta); diff != "" {
			t.Errorf("second attempt's event StateDelta mismatch (-want +got):\n%s", diff)
		}
	})
}

// Run returns after the caller stops reading: the scheduler keeps reading the
// node event queue until every started node has completed, so a node's
// trailing state-only event and its completion, both sent without a select on
// cancellation, never block.
func TestRunNode_CallerBreakDoesNotBlockOnPending(t *testing.T) {
	node := newStateNode("n", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if !yield(outputEvent(ctx, "a", nil), nil) {
			return
		}
		if err := ctx.State().Set("counter", 1); err != nil {
			yield(nil, err)
		}
	})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))
	r, sessionID := newStateRunner(t, session.InMemoryService(), wf, nil)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for range r.Run(t.Context(), "u", sessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) { //nolint:errcheck // the test checks that Run returns once the caller stops at its first item
			break
		}
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return within 5s after the caller stopped reading")
	}
}

func TestNodeContext_RenderUIWidgetAcceptedInNode(t *testing.T) {
	var first, second error
	node := workflow.NewFunctionNode("renderer", func(ctx agent.Context, _ any) (string, error) {
		first = agent.RenderUIWidget(ctx, session.UIWidget{ID: "w1", Provider: "mcp"})
		second = agent.RenderUIWidget(ctx, session.UIWidget{ID: "w1", Provider: "mcp"})
		return "done", nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node)

	if _, _, err := runStateWorkflow(t, wf, nil); err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if first != nil {
		t.Errorf("first RenderUIWidget() error = %v, want nil", first)
	}
	const want = "UI widget with ID 'w1' already exists in the current event actions."
	if second == nil || second.Error() != want {
		t.Errorf("second RenderUIWidget() error = %v, want %q", second, want)
	}
}

func TestNodeContext_MergeArtifactDeltaFlushed(t *testing.T) {
	var mergeErr error
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		mergeErr = agent.MergeArtifactDelta(ctx, map[string]int64{"a.pdf": 3})
		return "x", nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node)

	events, _, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if mergeErr != nil {
		t.Errorf("MergeArtifactDelta() error = %v, want nil", mergeErr)
	}
	if diff := cmp.Diff(map[string]int64{"a.pdf": 3}, eventWithOutput(t, events, "x").Actions.ArtifactDelta); diff != "" {
		t.Errorf("ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
}

// An artifact delta recorded in a node rides on one event only; a later event
// of the same node carries none.
func TestNodeContext_ArtifactDeltaFlushedOnce(t *testing.T) {
	node := newStateNode("emitter", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if err := agent.MergeArtifactDelta(ctx, map[string]int64{"a.pdf": 3}); err != nil {
			yield(nil, err)
			return
		}
		if !yield(outputEvent(ctx, "a", nil), nil) {
			return
		}
		yield(outputEvent(ctx, nil, map[string]any{"k": 1}), nil)
	})
	wf := newStateWorkflow(t, node)

	events, _, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if diff := cmp.Diff(map[string]int64{"a.pdf": 3}, eventWithOutput(t, events, "a").Actions.ArtifactDelta); diff != "" {
		t.Errorf("first event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
	second := eventsWithKey(events, "k")
	if len(second) != 1 {
		t.Fatalf("%d caller-visible events carry k, want 1", len(second))
	}
	if got := second[0].Actions.ArtifactDelta; len(got) != 0 {
		t.Errorf("second event ArtifactDelta = %v, want none", got)
	}
}

// A node whose only pending change is an artifact delta, and which yields no
// event, sends the delta on a trailing event.
func TestNodeContext_ArtifactOnlyPendingFlushed(t *testing.T) {
	node := newStateNode("saver", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if err := agent.MergeArtifactDelta(ctx, map[string]int64{"f.txt": 1}); err != nil {
			yield(nil, err)
		}
	})
	wf := newStateWorkflow(t, node)

	events, _, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	carried := eventsWithArtifact(events, "f.txt")
	if len(carried) != 1 {
		t.Fatalf("%d caller-visible events carry artifact f.txt, want 1", len(carried))
	}
	if diff := cmp.Diff(map[string]int64{"f.txt": 1}, carried[0].Actions.ArtifactDelta); diff != "" {
		t.Errorf("ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
}

func TestNew_SchemaWithSiblingRef(t *testing.T) {
	s := &jsonschema.Schema{
		Properties: map[string]*jsonschema.Schema{
			"a": {Type: "integer"},
			"b": {Ref: "#/properties/a"},
		},
	}
	resolved, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		return "", ctx.State().Set("b", "x")
	}, workflow.NodeConfig{})
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node), workflow.WithStateSchema(resolved))
	if err != nil {
		t.Fatalf("workflow.New() error = %v, want nil", err)
	}

	_, _, err = runStateWorkflow(t, wf, nil)

	se := asStateSchemaError(t, err)
	if se.Key != "b" || se.Reason != workflow.StateValueInvalid {
		t.Errorf("StateSchemaError Key = %q and Reason = %v, want b and StateValueInvalid", se.Key, se.Reason)
	}
}

// A property name with a JSON Pointer or URI special character is validated
// like any other.
func TestStateSchema_PropertyNamesAreEscaped(t *testing.T) {
	s, err := (&jsonschema.Schema{
		Title: "SpecialNames",
		Type:  "object",
		Properties: map[string]*jsonschema.Schema{
			"a/b":     {Type: "integer"},
			"a~b":     {Type: "integer"},
			"100%":    {Type: "integer"},
			"with sp": {Type: "integer"},
		},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		for k, v := range map[string]int{"a/b": 1, "a~b": 2, "100%": 3, "with sp": 4} {
			if err := ctx.State().Set(k, v); err != nil {
				return "", err
			}
		}
		return "", ctx.State().Set("a/b", "text")
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(s))

	_, stored, err := runStateWorkflow(t, wf, nil)

	se := asStateSchemaError(t, err)
	if se.Key != "a/b" || se.Reason != workflow.StateValueInvalid || se.Type != "integer" {
		t.Errorf("StateSchemaError Key = %q, Reason = %v, Type = %q, want a/b, StateValueInvalid, integer", se.Key, se.Reason, se.Type)
	}
	// The refused write is the last statement, so the four valid ones are stored.
	want := map[string]any{"a/b": 1, "a~b": 2, "100%": 3, "with sp": 4}
	if diff := cmp.Diff(want, stored); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
}

func TestStateSchema_FunctionNodeDirectDeltaWriteValidated(t *testing.T) {
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		ctx.Actions().StateDelta["unknown"] = "v"
		return "done", nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	events, _, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
	if hasOutput(events, "done") {
		t.Errorf("the refused event reached the caller")
	}
}

func TestStateSchema_ForgedPathEventIsChecked(t *testing.T) {
	node := newStateNode("emitter", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		ev := outputEvent(ctx, nil, map[string]any{"unknown": 1})
		ev.NodeInfo = &session.NodeInfo{Path: "wf@1/other@1"}
		yield(ev, nil)
	})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	events, stored, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
	if got := eventsWithKey(events, "unknown"); len(got) != 0 {
		t.Errorf("%d caller-visible events carry unknown, want 0", len(got))
	}
	if _, ok := stored["unknown"]; ok {
		t.Errorf("stored state has unknown")
	}
}

func TestStateSchema_CustomNodeDirectWriteNotValidated(t *testing.T) {
	node := newStateNode("emitter", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		ctx.Actions().StateDelta["unknown"] = "v"
		yield(outputEvent(ctx, "done", nil), nil)
	})
	wf := newStateWorkflow(t, node, workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if got := stored["unknown"]; got != "v" {
		t.Errorf("stored unknown = %v, want v", got)
	}
}

func TestStateSchema_ZeroResolvedSchemaRefusesEveryPlainKey(t *testing.T) {
	var plainErr, prefixedErr error
	node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		plainErr = ctx.State().Set("k", 1)
		prefixedErr = ctx.State().Set("app:k", 1)
		return "ok", nil
	}, workflow.NodeConfig{})
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node), workflow.WithStateSchema(&jsonschema.Resolved{}))
	if err != nil {
		t.Fatalf("workflow.New() error = %v, want nil", err)
	}

	if _, _, err := runStateWorkflow(t, wf, nil); err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	var se *workflow.StateSchemaError
	if !errors.As(plainErr, &se) {
		t.Fatalf("Set(%q) error = %v, want a *workflow.StateSchemaError", "k", plainErr)
	}
	if se.Reason != workflow.StateKeyUndeclared || len(se.Declared) != 0 {
		t.Errorf("StateSchemaError Reason = %v and Declared = %v, want StateKeyUndeclared and none", se.Reason, se.Declared)
	}
	if prefixedErr != nil {
		t.Errorf("Set(%q) error = %v, want nil", "app:k", prefixedErr)
	}
}

// Workflows built at the same time from one resolved schema each validate
// against it without touching the others; under -race, a build that wrote into
// the shared schema fails this test.
func TestNew_SharedSchemaBuiltConcurrently(t *testing.T) {
	schema := stateSchemaS(t)
	const n = 8
	wfs := make([]*workflow.Workflow, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Go(func() {
			node := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
				return "ok", ctx.State().Set("counter", 2)
			}, workflow.NodeConfig{})
			wfs[i], errs[i] = workflow.New("wf", workflow.Chain(workflow.Start, node), workflow.WithStateSchema(schema))
		})
	}
	wg.Wait()

	for i := range n {
		if errs[i] != nil {
			t.Fatalf("workflow.New() #%d error = %v, want nil", i, errs[i])
		}
		_, stored, err := runStateWorkflow(t, wfs[i], nil)
		if err != nil {
			t.Fatalf("run of workflow #%d error = %v, want nil", i, err)
		}
		if stored["counter"] != 2 {
			t.Errorf("workflow #%d stored counter = %v, want 2", i, stored["counter"])
		}
	}
}
