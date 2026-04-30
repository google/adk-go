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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// TestWorkflow_EndToEnd_SequentialThreeNodes runs a 3-node sequential
// workflow through the Runner and verifies that each node's output is
// observed downstream and that the final value is what the last node
// produced.
func TestWorkflow_EndToEnd_SequentialThreeNodes(t *testing.T) {
	type val struct {
		N int `json:"n"`
	}

	a := workflow.Func("step_a",
		func(_ *workflow.NodeContext, in val) (val, error) {
			return val{N: in.N + 1}, nil
		})
	b := workflow.Func("step_b",
		func(_ *workflow.NodeContext, in val) (val, error) {
			return val{N: in.N * 10}, nil
		})
	c := workflow.Func("step_c",
		func(_ *workflow.NodeContext, in val) (val, error) {
			return val{N: in.N - 5}, nil
		})

	wf, err := workflow.New(workflow.Config{
		Name: "math_wf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, a),
			workflow.Connect(a, b),
			workflow.Connect(b, c),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	wfAgent, err := wf.AsAgent()
	if err != nil {
		t.Fatalf("AsAgent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "test",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}

	// Send initial input encoded as JSON in the user content. FunctionNode
	// coerces *genai.Content -> val via the JSON path; without an explicit
	// schema we forward typed values, so we wire START's input through the
	// node context directly here. Simplest: provide the typed value via a
	// custom front node that ignores input.
	_ = a
	startInput := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (val, error) { return val{N: 3}, nil })
	wf2, err := workflow.New(workflow.Config{
		Name: "math_wf2",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, startInput),
			workflow.Connect(startInput, a),
			workflow.Connect(a, b),
			workflow.Connect(b, c),
		},
	})
	if err != nil {
		t.Fatalf("New2: %v", err)
	}
	wfAgent2, _ := wf2.AsAgent()
	r2, _ := runner.New(runner.Config{
		AppName:           "test",
		Agent:             wfAgent2,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})

	ctx := context.Background()
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	var got *val
	for ev, err := range r2.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run err: %v", err)
		}
		if ev.Author == "step_c" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			v := ev.Actions.NodeInfo.Output.(val)
			got = &v
		}
	}
	if got == nil {
		t.Fatal("did not observe step_c output event")
	}
	// 3 -> +1=4 -> *10=40 -> -5=35
	if got.N != 35 {
		t.Errorf("final = %d, want 35", got.N)
	}
	_ = r
}

// TestWorkflow_PropagatesErrorFromNode verifies that an error returned by
// a node aborts the workflow and surfaces an error event.
func TestWorkflow_PropagatesErrorFromNode(t *testing.T) {
	wantMsg := "boom"
	bad := workflow.Func("bad",
		func(_ *workflow.NodeContext, _ any) (struct{}, error) {
			return struct{}{}, &simpleErr{wantMsg}
		})
	wf, err := workflow.New(workflow.Config{
		Name:  "errwf",
		Edges: []workflow.Edge{workflow.Connect(workflow.START, bad)},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	wfAgent, _ := wf.AsAgent()
	r, _ := runner.New(runner.Config{
		AppName:           "test",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "x"}}}
	sawErr := false
	for _, err := range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			sawErr = true
		}
	}
	if !sawErr {
		t.Error("expected an error event")
	}
}

type simpleErr struct{ msg string }

func (e *simpleErr) Error() string { return e.msg }
