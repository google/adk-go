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
	"slices"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// counterSchema declares one state key, counter, as an integer.
func counterSchema(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	s := &jsonschema.Schema{
		Title:      "PipelineSchema",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"counter": {Type: "integer"}},
	}
	r, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	return r
}

// newSessionMockCtx returns a mock invocation context over a new in-memory
// session, and the service that stores it.
func newSessionMockCtx(t *testing.T) (*MockInvocationContext, session.Service) {
	t.Helper()
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	return &MockInvocationContext{Context: t.Context(), sess: created.Session}, svc
}

func TestResume_ResumedNodeStillValidates(t *testing.T) {
	ask := newHitlNode("ask", func(agent.Context, any, func(*session.Event, error) bool) {})
	writer := NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		return "", ctx.State().Set("unknown", 1)
	}, NodeConfig{})
	wf, err := New("wf", []Edge{{From: Start, To: ask}, {From: ask, To: writer}}, WithStateSchema(counterSchema(t)))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	state := NewRunState()
	state.Nodes["ask"] = &NodeState{Status: NodeWaiting, Interrupts: []string{"ask"}}
	mock, _ := newSessionMockCtx(t)

	var runErr error
	for _, err := range wf.Resume(agent.Promote(mock), state, map[string]any{"ask": "yes"}) {
		if err != nil && runErr == nil {
			runErr = err
		}
	}

	var se *StateSchemaError
	if !errors.As(runErr, &se) {
		t.Fatalf("Resume error = %v, want a *StateSchemaError", runErr)
	}
	if se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
}

func TestResume_WriteBeforePauseSurvives(t *testing.T) {
	tests := []struct {
		name        string
		writeBefore bool
	}{
		{name: "write before the request", writeBefore: true},
		{name: "write after the request", writeBefore: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			writeCounter := func(ctx agent.Context) {
				if err := ctx.State().Set("counter", 1); err != nil {
					t.Errorf("State().Set(counter) error = %v", err)
				}
			}
			ask := newHitlNode("ask", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				if tc.writeBefore {
					writeCounter(ctx)
				}
				if !yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "approval", Message: "approve?"}), nil) {
					return
				}
				if !tc.writeBefore {
					writeCounter(ctx)
				}
			})
			var resumedWith any
			done := NewFunctionNode("done", func(_ agent.Context, in any) (string, error) {
				resumedWith = in
				return "finished", nil
			}, NodeConfig{})
			wf, err := New("wf", []Edge{{From: Start, To: ask}, {From: ask, To: done}}, WithStateSchema(counterSchema(t)))
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			mock, svc := newSessionMockCtx(t)

			var history sliceEvents
			for ev, err := range wf.Run(mock) {
				if err != nil {
					t.Fatalf("Run error = %v", err)
				}
				if ev.Partial {
					continue
				}
				if err := svc.AppendEvent(t.Context(), mock.sess, ev); err != nil {
					t.Fatalf("AppendEvent() error = %v", err)
				}
				history = append(history, ev)
			}

			got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: mock.sess.ID()})
			if err != nil {
				t.Fatalf("session Get() error = %v", err)
			}
			if v, err := got.Session.State().Get("counter"); err != nil || v != 1 {
				t.Errorf("stored counter = %v (error %v), want 1", v, err)
			}
			state, err := wf.ReconstructRunState(fakeSession{events: history}, "")
			if err != nil {
				t.Fatalf("ReconstructRunState() error = %v", err)
			}
			ns := nodeState(t, state, "ask")
			if ns.Status != NodeWaiting || !slices.Equal(ns.Interrupts, []string{"approval"}) {
				t.Fatalf("ask reconstructed as status %v with interrupts %v, want NodeWaiting with [approval]", ns.Status, ns.Interrupts)
			}
			for _, err := range wf.Resume(agent.Promote(mock), state, map[string]any{"approval": "yes"}) {
				if err != nil {
					t.Fatalf("Resume error = %v", err)
				}
			}
			if resumedWith != "yes" {
				t.Errorf("successor received %v, want the response yes", resumedWith)
			}
		})
	}
}
