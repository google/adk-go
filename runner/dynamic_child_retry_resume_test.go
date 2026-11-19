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

package runner_test

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// TestRunner_RetriedChild_PauseAfterFailureResumesAsPaused checks that a
// dynamic child that failed one attempt and asked for input on the next is
// resumed as a paused child: on the answering turn it runs again from
// attempt 1 with its whole retry budget, reads the answer, and its output
// becomes the root's.
//
// Fails if the earlier attempt's failure record makes the resume treat the
// child as failed, or if the attempt count carries over to the next turn.
func TestRunner_RetriedChild_PauseAfterFailureResumesAsPaused(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	var (
		mu       sync.Mutex
		calls    int
		attempts []int
	)
	rerun := true
	retry := &workflow.RetryConfig{MaxAttempts: 3, BackoffFactor: 1, ShouldRetry: func(error) bool { return true }}
	child := &funcNode{
		BaseNode: workflow.NewBaseNode("child", "", workflow.NodeConfig{RerunOnResume: &rerun, RetryConfig: retry}),
		run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			mu.Lock()
			calls++
			call := calls
			attempts = append(attempts, agent.AttemptCount(ctx))
			mu.Unlock()
			switch call {
			case 1, 3:
				yield(nil, errors.New("attempt failed"))
			case 2:
				yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "q1", Message: "?"}), nil)
			default:
				ans, _ := ctx.ResumedInput("q1")
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = fmt.Sprintf("answer:%v", ans)
				yield(ev, nil)
			}
		},
	}
	parent := workflow.NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
		return workflow.RunNode[any](ctx, child, nil)
	}, workflow.NodeConfig{})
	wf, err := workflow.NewWorkflowNode("wf", workflow.Chain(workflow.Start, parent))
	if err != nil {
		t.Fatalf("NewWorkflowNode() error = %v", err)
	}
	r := newNodeRootRunner(t, wf, svc)

	turn1, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}))
	if len(errs) > 0 {
		t.Fatalf("turn 1 Run() errors = %v", errs)
	}
	var asked bool
	for _, ev := range turn1 {
		if ev.RequestedInput != nil && ev.RequestedInput.InterruptID == "q1" {
			asked = true
		}
	}
	if !asked {
		t.Fatal("turn 1 did not ask for input q1")
	}
	mu.Lock()
	turn1Attempts := attempts
	attempts = nil
	mu.Unlock()
	if diff := cmp.Diff([]int{1, 2}, turn1Attempts); diff != "" {
		t.Errorf("turn 1 attempt numbers mismatch (-want +got):\n%s", diff)
	}

	answer := genai.NewContentFromFunctionResponse(workflow.WorkflowInputFunctionCallName, map[string]any{"result": "yes"}, genai.RoleUser)
	answer.Parts[0].FunctionResponse.ID = "q1"
	turn2, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, answer, agent.RunConfig{}))
	if len(errs) > 0 {
		t.Fatalf("turn 2 Run() errors = %v", errs)
	}
	mu.Lock()
	turn2Attempts := attempts
	mu.Unlock()
	if diff := cmp.Diff([]int{1, 2}, turn2Attempts); diff != "" {
		t.Errorf("turn 2 attempt numbers mismatch (-want +got):\n%s", diff)
	}
	var rootOutputs []any
	for _, ev := range turn2 {
		if ev.Output != nil && ev.NodeInfo != nil && ev.NodeInfo.Path == "wf@1" {
			rootOutputs = append(rootOutputs, ev.Output)
		}
	}
	if diff := cmp.Diff([]any{"answer:yes"}, rootOutputs); diff != "" {
		t.Errorf("turn 2 outputs of the root wf@1 mismatch (-want +got):\n%s", diff)
	}
}
