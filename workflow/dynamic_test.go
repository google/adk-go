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
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// TestWorkflow_DynamicNode_RunsChildAndForwardsOutput verifies that a
// node calling ctx.RunNode runs the child synchronously and observes the
// child's output.
func TestWorkflow_DynamicNode_RunsChildAndForwardsOutput(t *testing.T) {
	var childCalls atomic.Int32
	child := workflow.Func("child",
		func(_ *workflow.NodeContext, in int) (int, error) {
			childCalls.Add(1)
			return in * 2, nil
		})

	parent := workflow.Func("parent",
		func(ctx *workflow.NodeContext, _ any) (int, error) {
			out, err := ctx.RunNode(child, 21)
			if err != nil {
				return 0, err
			}
			return out.(int), nil
		})

	wf, err := workflow.New(workflow.Config{
		Name:  "dwf",
		Edges: []workflow.Edge{workflow.Connect(workflow.START, parent)},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)

	var got int
	for _, ev := range events {
		if ev.Author == "parent" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			got = ev.Actions.NodeInfo.Output.(int)
		}
	}
	if got != 42 {
		t.Errorf("parent output = %d, want 42", got)
	}
	if childCalls.Load() != 1 {
		t.Errorf("childCalls = %d, want 1", childCalls.Load())
	}
}

// TestWorkflow_DynamicNode_DedupsAcrossResumes verifies that on resume,
// a previously-completed dynamic node returns its cached output without
// re-executing.
func TestWorkflow_DynamicNode_DedupsAcrossResumes(t *testing.T) {
	var childCalls atomic.Int32
	child := workflow.Func("child",
		func(_ *workflow.NodeContext, in int) (int, error) {
			childCalls.Add(1)
			return in + 1, nil
		})

	// Parent runs child, then either returns or fails depending on flag.
	failParent := atomic.Bool{}
	failParent.Store(true)
	parent := workflow.Func("parent",
		func(ctx *workflow.NodeContext, _ any) (int, error) {
			out, err := ctx.RunNode(child, 10)
			if err != nil {
				return 0, err
			}
			if failParent.Load() {
				// Force the parent (and workflow) to fail AFTER the child
				// completes so the child's completion event lands in the
				// session.
				return 0, &simpleErr{"forced"}
			}
			return out.(int), nil
		})

	wf, err := workflow.New(workflow.Config{
		Name:  "dedup",
		Edges: []workflow.Edge{workflow.Connect(workflow.START, parent)},
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

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	for range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
	}
	if childCalls.Load() != 1 {
		t.Fatalf("after run 1: childCalls = %d, want 1", childCalls.Load())
	}

	// Flip parent to succeed and resume. The dynamic child should NOT
	// re-run because its completion event is in the session.
	failParent.Store(false)
	for range r.Resume(context.Background(), "u", "s", agent.RunConfig{}) {
	}
	if got := childCalls.Load(); got != 1 {
		t.Errorf("after resume: childCalls = %d, want 1 (cached output)", got)
	}
}

// TestWorkflow_HITL_RequestInputEmitsLongRunningTool verifies that a node
// emitting RequestInput produces an event with a FunctionCall named
// adk_request_input and a populated LongRunningToolIDs.
func TestWorkflow_HITL_RequestInputEmitsLongRunningTool(t *testing.T) {
	asker := workflow.Func("asker",
		func(_ *workflow.NodeContext, _ any) (any, error) {
			return nil, nil
		})
	// Wrap asker through a custom inline node that calls em.RequestInput.
	hitl := &hitlNode{interruptID: "intr-1", prompt: "what next?"}
	if err := hitl.SetMetadata("hitl", "", workflow.NodeSpec{}); err != nil {
		t.Fatalf("SetMetadata: %v", err)
	}
	wf, err := workflow.New(workflow.Config{
		Name: "hitlwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, asker),
			workflow.Connect(asker, hitl),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)

	var sawCall bool
	for _, ev := range events {
		if ev.Author != "hitl" {
			continue
		}
		if len(ev.LongRunningToolIDs) == 0 {
			continue
		}
		if ev.LongRunningToolIDs[0] != "intr-1" {
			t.Errorf("LongRunningToolIDs[0] = %q, want intr-1", ev.LongRunningToolIDs[0])
		}
		if ev.Content == nil || len(ev.Content.Parts) == 0 || ev.Content.Parts[0].FunctionCall == nil {
			t.Errorf("expected FunctionCall part on hitl event, got %+v", ev.Content)
			continue
		}
		fc := ev.Content.Parts[0].FunctionCall
		if fc.Name != "adk_request_input" {
			t.Errorf("FunctionCall.Name = %q, want adk_request_input", fc.Name)
		}
		if fc.ID != "intr-1" {
			t.Errorf("FunctionCall.ID = %q, want intr-1", fc.ID)
		}
		sawCall = true
	}
	if !sawCall {
		t.Error("hitl node did not produce a long-running FunctionCall event")
	}
}

// hitlNode demonstrates the real-world HITL pattern: emit RequestInput
// on the first invocation; on resume read the user's response via
// ResumeInput and emit it as the node's Output so successors can run.
type hitlNode struct {
	workflow.Base
	interruptID string
	prompt      string
}

func (h *hitlNode) RunImpl(ctx *workflow.NodeContext, _ any, em workflow.EventEmitter) error {
	if v, ok := ctx.ResumeInput(h.interruptID); ok {
		return em.Output(v)
	}
	return em.RequestInput(workflow.RequestInput{
		Prompt:      h.prompt,
		InterruptID: h.interruptID,
	})
}

// TestWorkflow_HITL_ResumeWithFunctionResponse verifies that on resume,
// a user-supplied FunctionResponse populates ctx.ResumeInput so the node
// can act on the answer. Pairs with the encoding test above.
func TestWorkflow_HITL_ResumeWithFunctionResponse(t *testing.T) {
	answered := atomic.Bool{}
	asker := &hitlNode{interruptID: "ask1", prompt: "yes/no?"}
	if err := asker.SetMetadata("asker", "", workflow.NodeSpec{}); err != nil {
		t.Fatalf("SetMetadata: %v", err)
	}
	consumer := workflow.Func("consumer",
		func(ctx *workflow.NodeContext, _ any) (string, error) {
			v, ok := ctx.ResumeInput("ask1")
			if !ok {
				return "no_answer", nil
			}
			answered.Store(true)
			// FunctionResponse.Response is a map[string]any in genai; check loosely.
			if m, isMap := v.(map[string]any); isMap {
				if a, has := m["answer"].(string); has {
					return a, nil
				}
			}
			return "got_response", nil
		})
	// asker has WithRerunOnResume(false) by default — its prior Output is
	// replayed (none). The consumer runs after asker on resume because in
	// Phase 4 we stripped the asker from the queue if it produced an
	// Output event. Without that, asker still re-runs which is fine: it
	// re-emits the interrupt but the response is already in resumeInputs.
	wf, err := workflow.New(workflow.Config{
		Name: "hitlrwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, asker),
			workflow.Connect(asker, consumer),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	wfAgent, _ := wf.AsAgent()
	sessSvc := session.InMemoryService()
	r, _ := runner.New(runner.Config{
		AppName:           "t",
		Agent:             wfAgent,
		SessionService:    sessSvc,
		AutoCreateSession: true,
	})

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	for range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
	}

	// Append the user's FunctionResponse to the session. In a real
	// runtime, this would arrive via r.Run with msg=FunctionResponse.
	// For test simplicity we use Run with the response as msg.
	respMsg := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       "ask1",
				Name:     "adk_request_input",
				Response: map[string]any{"answer": "yes"},
			},
		}},
	}
	var consumerOut string
	for ev := range r.Run(context.Background(), "u", "s", respMsg, agent.RunConfig{}) {
		if ev.Author == "consumer" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			consumerOut = ev.Actions.NodeInfo.Output.(string)
		}
	}
	if !answered.Load() {
		t.Error("consumer did not see the resume input")
	}
	if consumerOut != "yes" {
		t.Errorf("consumer output = %q, want yes", consumerOut)
	}
}
