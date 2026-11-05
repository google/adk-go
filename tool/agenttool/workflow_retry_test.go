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

package agenttool_test

import (
	"errors"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/workflow"
)

func TestAgentTool_WorkflowRecoveredFailure(t *testing.T) {
	for _, mode := range []string{"retry", "fallback", "exhausted"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int32
			failure := errors.New("scripted failure")
			flaky := workflow.NewFunctionNode("flaky", func(agent.Context, any) (string, error) {
				if calls.Add(1) == 1 || mode != "retry" {
					return "", failure
				}
				return "ok", nil
			}, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 2, InitialDelay: time.Millisecond}})
			var node workflow.Node = flaky
			if mode == "fallback" {
				child := workflow.NewFunctionNode("child", func(agent.Context, any) (string, error) { calls.Add(1); return "", failure }, workflow.NodeConfig{})
				node = workflow.NewDynamicNode("parent", func(ctx agent.Context, in any, _ func(*session.Event) error) (string, error) {
					if _, err := workflow.RunNode[string](ctx, child, in); !errors.Is(err, failure) {
						return "", errors.New("expected child failure")
					}
					return "ok", nil
				}, workflow.NodeConfig{})
			}
			// AgentTool consumes content, not the workflow's typed Output.
			report := workflow.NewDynamicNode("report", func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
				if in != "ok" {
					return nil, errors.New("unexpected recovered result")
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = genai.NewContentFromText("ok", genai.RoleModel)
				return nil, emit(ev)
			}, workflow.NodeConfig{})
			wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: []workflow.Edge{{From: workflow.Start, To: node}, {From: node, To: report}}})
			if err != nil {
				t.Fatal("workflow construction failed")
			}
			impl, ok := agenttool.New(wf, nil).(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agent tool did not implement FunctionTool")
			}
			result, err := impl.Run(createToolContext(t, wf), map[string]any{"request": "start"})
			if mode == "exhausted" {
				if !errors.Is(err, failure) || calls.Load() != 2 {
					t.Fatal("exhausted retry did not preserve the final error")
				}
				return
			}
			if err != nil {
				t.Fatal("recovered workflow failed its agent tool call")
			}
			if !reflect.DeepEqual(result, map[string]any{"result": "ok"}) {
				t.Fatalf("recovered workflow result has unexpected shape (fields: %d, result type: %T)", len(result), result["result"])
			}
			wantCalls := int32(2)
			if mode == "fallback" {
				wantCalls = 1
			}
			if calls.Load() != wantCalls {
				t.Fatal("agent tool stopped consuming before recovery")
			}
		})
	}
}
