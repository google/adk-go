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

package adka2a

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func TestExecutor_WorkflowRetryOutcome(t *testing.T) {
	for _, exhausted := range []bool{false, true} {
		name := "recovered"
		if exhausted {
			name = "exhausted"
		}
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			node := workflow.NewFunctionNode("flaky", func(agent.Context, any) (string, error) {
				if calls.Add(1) == 1 || exhausted {
					return "", errors.New("scripted failure")
				}
				return "ok", nil
			}, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 2, InitialDelay: time.Millisecond}})
			wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: []workflow.Edge{{From: workflow.Start, To: node}}})
			if err != nil {
				t.Fatal("workflow construction failed")
			}
			executor := NewExecutor(ExecutorConfig{RunnerConfig: runner.Config{AppName: "wf", Agent: wf, SessionService: session.InMemoryService()}})
			req := &a2asrv.ExecutorContext{TaskID: a2a.NewTaskID(), ContextID: a2a.NewContextID(), Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("start"))}
			var finalState a2a.TaskState
			for event, err := range executor.Execute(t.Context(), req) {
				if err != nil {
					t.Fatal("executor failed to dispatch workflow events")
				}
				if status, ok := event.(*a2a.TaskStatusUpdateEvent); ok {
					finalState = status.Status.State
				}
			}
			want := a2a.TaskStateCompleted
			if exhausted {
				want = a2a.TaskStateFailed
			}
			if finalState != want || calls.Load() != 2 {
				t.Fatal("A2A final state did not reflect the recovered workflow outcome")
			}
		})
	}
}
