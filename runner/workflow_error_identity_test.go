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
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

type workflowBodyError struct{}

func (*workflowBodyError) Error() string { return "scripted body failure" }

func TestRunner_WorkflowErrorIdentity(t *testing.T) {
	sentinel := errors.New("scripted sentinel")
	for _, tc := range []struct {
		name string
		err  error
	}{
		{name: "sentinel", err: sentinel},
		{name: "typed", err: &workflowBodyError{}},
		{name: "user_wrapper", err: fmt.Errorf("scripted wrapper: %w", sentinel)},
		{name: "node_run_error", err: &workflow.NodeRunError{ChildName: "child", Cause: sentinel}},
	} {
		for _, mode := range []string{"recovered", "exhausted", "no_retry"} {
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				var attempts atomic.Int32
				predicateCalls := 0
				cfg := workflow.NodeConfig{}
				if mode != "no_retry" {
					cfg.RetryConfig = &workflow.RetryConfig{MaxAttempts: 2, ShouldRetry: func(err error) bool {
						predicateCalls++
						if err != tc.err {
							t.Errorf("retry predicate received a different error object (got %T, want %T)", err, tc.err)
						}
						if tc.name == "typed" {
							_, ok := err.(*workflowBodyError)
							return ok
						}
						return err == tc.err
					}}
				}
				n := workflow.NewDynamicNode("body", func(agent.Context, any, func(*session.Event) error) (any, error) {
					if attempts.Add(1) == 1 || mode != "recovered" {
						return nil, tc.err
					}
					return "ok", nil
				}, cfg)
				r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: n}})
				var got error
				var completed bool
				for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
					if err != nil {
						got = err
					} else if ev.Output == "ok" {
						completed = true
					}
				}
				wantAttempts, wantPredicates := int32(2), 1
				if mode == "no_retry" {
					wantAttempts, wantPredicates = 1, 0
				}
				if attempts.Load() != wantAttempts || predicateCalls != wantPredicates {
					t.Fatal("error identity changed retry behavior")
				}
				if mode == "recovered" {
					if got != nil || !completed {
						t.Fatal("identity-based retry did not recover")
					}
				} else if got != tc.err {
					t.Fatalf("final error identity changed (got %T, want %T)", got, tc.err)
				}
			})
		}
	}
}

type opaqueInventoryNode struct {
	workflow.BaseNode
	node workflow.Node
}

func (n *opaqueInventoryNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return n.node.Run(ctx, input)
}

func TestRunner_WorkflowOpaqueWrapperRetry(t *testing.T) {
	for _, placement := range []string{"static", "dynamic"} {
		t.Run(placement, func(t *testing.T) {
			var chargeCalls, bodyCalls atomic.Int32
			failure := errors.New("scripted wrapped body failure")
			charge := workflow.NewFunctionNode("charge", func(agent.Context, any) (string, error) {
				chargeCalls.Add(1)
				return "charged", nil
			}, workflow.NodeConfig{})
			body := workflow.NewDynamicNode("body", func(ctx agent.Context, in any, _ func(*session.Event) error) (string, error) {
				out, err := workflow.RunNode[string](ctx, charge, in, workflow.WithRunID("stable"))
				if err != nil {
					return "", err
				}
				if bodyCalls.Add(1) == 1 {
					return "", failure
				}
				return out, nil
			}, workflow.NodeConfig{})
			cfg := workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 2, ShouldRetry: func(err error) bool {
				return err == failure
			}}}
			if placement == "dynamic" {
				cfg = workflow.NodeConfig{}
			}
			var root workflow.Node = &opaqueInventoryNode{BaseNode: workflow.NewBaseNode("body", "", cfg), node: body}
			if placement == "dynamic" {
				wrapped := root
				root = workflow.NewDynamicNode("parent", func(ctx agent.Context, in any, _ func(*session.Event) error) (string, error) {
					return workflow.RunNode[string](ctx, wrapped, in, workflow.WithRunID("stable"))
				}, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 2}})
			}
			r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: root}})
			completed := false
			for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatal("custom wrapper did not recover")
				}
				completed = completed || ev.Output == "charged"
			}
			// Direct forwarding is opaque to the executor. Unknown inventory
			// must invalidate the whole wrapper, not trust its child's report.
			if !completed || chargeCalls.Load() != 2 || bodyCalls.Load() != 2 {
				t.Fatalf("opaque wrapper did not conservatively retry (completed=%v, charge=%d, body=%d)", completed, chargeCalls.Load(), bodyCalls.Load())
			}
		})
	}
}

type filteringInventoryNode struct {
	workflow.BaseNode
	node workflow.Node
}

func (n *filteringInventoryNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for ev, err := range n.node.Run(ctx, input) {
			if ev != nil && ev.CustomMetadata["adk.workflow.node_outcome"] == "failed" {
				continue
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

func TestRunner_WorkflowOpaqueWrapperPartialSibling(t *testing.T) {
	var firstCalls, secondCalls atomic.Int32
	firstFailure := errors.New("scripted first failure")
	secondFailure := errors.New("scripted second failure")
	first := workflow.NewFunctionNode("first", func(agent.Context, any) (string, error) {
		if firstCalls.Add(1) == 1 {
			return "", firstFailure
		}
		return "recovered-first", nil
	}, workflow.NodeConfig{})
	second := workflow.NewEmittingFunctionNode("second", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
		if secondCalls.Add(1) == 1 {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = "incomplete-second"
			if err := emit(ev); err != nil {
				return "", err
			}
			return "", secondFailure
		}
		return "recovered-second", nil
	}, workflow.NodeConfig{})
	body := workflow.NewDynamicNode("body", func(ctx agent.Context, in any, _ func(*session.Event) error) (string, error) {
		_, firstErr := workflow.RunNode[string](ctx, first, in, workflow.WithRunID("stable"))
		out, secondErr := workflow.RunNode[string](ctx, second, in, workflow.WithRunID("stable"))
		if firstErr != nil {
			return "", firstErr
		}
		return out, secondErr
	}, workflow.NodeConfig{})
	wrapper := &filteringInventoryNode{BaseNode: workflow.NewBaseNode("body", "", workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 2}}), node: body}
	r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: wrapper}})
	recovered := false
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("wrapper failed to retry")
		}
		recovered = recovered || ev.Output == "recovered-second"
	}
	if firstCalls.Load() != 2 || secondCalls.Load() != 2 || !recovered {
		t.Fatalf("opaque wrapper trusted incomplete sibling inventory (first=%d second=%d recovered=%v)", firstCalls.Load(), secondCalls.Load(), recovered)
	}
}
