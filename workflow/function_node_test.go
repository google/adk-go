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
	"errors"
	"testing"

	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// fakeEmitter records every call into the EventEmitter for assertions.
type fakeEmitter struct {
	outputs    []any
	events     []*session.Event
	requests   []workflow.RequestInput
	stateDelta []map[string]any
	artDelta   []map[string]int64
	failOnIdx  int // 0-based; if outputs reaches this size, return err
	err        error
}

func (e *fakeEmitter) Event(ev *session.Event) error {
	e.events = append(e.events, ev)
	return nil
}
func (e *fakeEmitter) Output(v any) error {
	if e.failOnIdx > 0 && len(e.outputs) == e.failOnIdx-1 {
		return errors.New("yield failed")
	}
	e.outputs = append(e.outputs, v)
	return nil
}
func (e *fakeEmitter) RequestInput(r workflow.RequestInput) error {
	e.requests = append(e.requests, r)
	return nil
}
func (e *fakeEmitter) StateDelta(m map[string]any) error {
	e.stateDelta = append(e.stateDelta, m)
	return nil
}
func (e *fakeEmitter) ArtifactDelta(m map[string]int64) error {
	e.artDelta = append(e.artDelta, m)
	return nil
}

type req struct {
	Q string `json:"q"`
}
type resp struct {
	A string `json:"a"`
}

func TestFunc_HappyPath(t *testing.T) {
	n := workflow.Func("classify",
		func(_ *workflow.NodeContext, in req) (resp, error) {
			return resp{A: "answer:" + in.Q}, nil
		})
	em := &fakeEmitter{}
	if err := n.RunImpl(&workflow.NodeContext{}, req{Q: "hi"}, em); err != nil {
		t.Fatalf("RunImpl: %v", err)
	}
	if len(em.outputs) != 1 {
		t.Fatalf("outputs = %d, want 1", len(em.outputs))
	}
	r, ok := em.outputs[0].(resp)
	if !ok || r.A != "answer:hi" {
		t.Errorf("output = %+v", em.outputs[0])
	}
}

func TestFunc_PropagatesError(t *testing.T) {
	want := errors.New("boom")
	n := workflow.Func("err",
		func(_ *workflow.NodeContext, _ req) (resp, error) { return resp{}, want })
	em := &fakeEmitter{}
	err := n.RunImpl(&workflow.NodeContext{}, req{Q: "x"}, em)
	if !errors.Is(err, want) {
		t.Errorf("err = %v, want wraps boom", err)
	}
	if len(em.outputs) != 0 {
		t.Errorf("outputs should be empty on error, got %v", em.outputs)
	}
}

func TestFunc_CoercesMapInput_WithExplicitSchema(t *testing.T) {
	n := workflow.Func("schema_in",
		func(_ *workflow.NodeContext, in req) (resp, error) {
			return resp{A: in.Q}, nil
		},
		workflow.WithInputSchema(workflow.JSONSchemaFor[req]()))
	em := &fakeEmitter{}
	if err := n.RunImpl(&workflow.NodeContext{}, map[string]any{"q": "hello"}, em); err != nil {
		t.Fatalf("RunImpl: %v", err)
	}
	if got := em.outputs[0].(resp).A; got != "hello" {
		t.Errorf("answer = %q", got)
	}
}

func TestFunc_PassthroughTypedInput(t *testing.T) {
	n := workflow.Func("typed",
		func(_ *workflow.NodeContext, in req) (resp, error) {
			return resp{A: in.Q}, nil
		})
	em := &fakeEmitter{}
	if err := n.RunImpl(&workflow.NodeContext{}, req{Q: "ok"}, em); err != nil {
		t.Fatalf("RunImpl: %v", err)
	}
	if got := em.outputs[0].(resp).A; got != "ok" {
		t.Errorf("answer = %q", got)
	}
}

func TestFunc_NilInputCoercesToZero(t *testing.T) {
	n := workflow.Func("zero",
		func(_ *workflow.NodeContext, in req) (resp, error) {
			if in.Q != "" {
				t.Errorf("expected empty input, got %q", in.Q)
			}
			return resp{A: "default"}, nil
		})
	em := &fakeEmitter{}
	if err := n.RunImpl(&workflow.NodeContext{}, nil, em); err != nil {
		t.Fatalf("RunImpl: %v", err)
	}
	if got := em.outputs[0].(resp).A; got != "default" {
		t.Errorf("answer = %q", got)
	}
}

func TestFuncStream_EmitsMultipleOutputs(t *testing.T) {
	n := workflow.FuncStream("stream",
		func(_ *workflow.NodeContext, in req, yield func(resp) error) error {
			for i := 0; i < 3; i++ {
				if err := yield(resp{A: in.Q}); err != nil {
					return err
				}
			}
			return nil
		})
	em := &fakeEmitter{}
	if err := n.RunImpl(&workflow.NodeContext{}, req{Q: "tick"}, em); err != nil {
		t.Fatalf("RunImpl: %v", err)
	}
	if len(em.outputs) != 3 {
		t.Errorf("outputs = %d, want 3", len(em.outputs))
	}
}

func TestFunc_NodeOptions_SetSpec(t *testing.T) {
	n := workflow.Func("with_opts",
		func(_ *workflow.NodeContext, _ req) (resp, error) { return resp{}, nil },
		workflow.WithDescription("does a thing"),
		workflow.WithRerunOnResume(true),
		workflow.WithWaitForOutput(true),
	)
	if n.Description() != "does a thing" {
		t.Errorf("Description = %q", n.Description())
	}
	if !n.Spec().RerunOnResume {
		t.Error("RerunOnResume should be true")
	}
	if !n.Spec().WaitForOutput {
		t.Error("WaitForOutput should be true")
	}
}
