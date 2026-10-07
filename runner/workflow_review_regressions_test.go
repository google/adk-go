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
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func TestRunner_WorkflowHITL_NormalizedReentryResponse(t *testing.T) {
	for _, tc := range []struct {
		name     string
		response map[string]any
		schema   *jsonschema.Schema
		want     any
	}{
		{name: "result", response: map[string]any{"result": "approved"}, want: "approved"},
		{name: "result_with_schema", response: map[string]any{"result": "approved"}, schema: &jsonschema.Schema{Type: "string"}, want: "approved"},
		{name: "json_payload", response: map[string]any{"payload": "42"}, want: float64(42)},
		{name: "business_map", response: map[string]any{"result": "approved", "other": true}, want: map[string]any{"result": "approved", "other": true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			ask := workflow.NewDynamicNode("ask", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				calls.Add(1)
				answer, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "approval", Message: "approve", ResponseSchema: tc.schema})
				if err == nil && !reflect.DeepEqual(answer, tc.want) {
					t.Error("re-entry received a response with the wrong shape")
				}
				return answer, err
			}, workflow.NodeConfig{})
			r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: ask}})
			for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatal("initial run failed")
				}
			}
			msg := genai.NewContentFromFunctionResponse(workflow.WorkflowInputFunctionCallName, tc.response, genai.RoleUser)
			msg.Parts[0].FunctionResponse.ID = "approval"
			for _, err := range r.Run(t.Context(), "u", "s", msg, agent.RunConfig{}) {
				if err != nil {
					t.Fatal("normalized response failed to resume")
				}
			}
			if calls.Load() != 2 {
				t.Fatal("asker did not re-enter")
			}
		})
	}
}

func TestRunner_WorkflowHITL_DelegatedOutputAfterInterrupt(t *testing.T) {
	var calls, downstreamCalls atomic.Int32
	raised := make(chan struct{})
	compute := workflow.NewFunctionNode("compute", func(agent.Context, any) (string, error) { return "computed", nil }, workflow.NodeConfig{})
	ask := workflow.NewDynamicNode("ask", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		return workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "approval", Message: "approve"})
	}, workflow.NodeConfig{})
	p := workflow.NewDynamicNode("p", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		calls.Add(1)
		askDone := make(chan error, 1)
		go func() {
			_, err := workflow.RunNode[any](ctx, ask, in)
			askDone <- err
		}()
		select {
		case <-raised:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		if _, err := workflow.RunNode[string](ctx, compute, in, workflow.WithUseAsOutput()); err != nil {
			return nil, err
		}
		if err := <-askDone; err != nil {
			return nil, err
		}
		return nil, nil
	}, workflow.NodeConfig{})
	d := workflow.NewFunctionNode("d", func(agent.Context, any) (string, error) {
		downstreamCalls.Add(1)
		return "done", nil
	}, workflow.NodeConfig{})
	r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: p}, {From: p, To: d}})
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("initial run failed")
		}
		if ev.RequestedInput != nil {
			close(raised)
		}
	}
	msg := resumeContent("approval", workflow.WorkflowInputFunctionCallName, "approved")
	for _, err := range r.Run(t.Context(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatal("delegated output prevented re-entry")
		}
	}
	if calls.Load() != 2 || downstreamCalls.Load() != 1 {
		t.Fatal("resume did not execute the parent and its successor")
	}
	for _, err := range r.Run(t.Context(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil && !errors.Is(err, workflow.ErrNothingToResume) {
			t.Fatal("duplicate resume returned an unexpected error")
		}
	}
	if calls.Load() != 2 || downstreamCalls.Load() != 1 {
		t.Fatal("duplicate answer repeated completed work")
	}
}

func TestRunner_WorkflowHITL_CachedDelegationAcrossResume(t *testing.T) {
	compute := workflow.NewFunctionNode("compute", func(agent.Context, any) (string, error) { return "computed", nil }, workflow.NodeConfig{})
	p := workflow.NewDynamicNode("p", func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
		if _, err := workflow.RunNode[string](ctx, compute, in, workflow.WithUseAsOutput()); err != nil {
			return nil, err
		}
		if _, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "p1", Message: "approve"}); err != nil {
			return nil, err
		}
		_, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "p2", Message: "approve 2"})
		return nil, err
	}, workflow.NodeConfig{})
	b := joinResumeAgent(t, "b", "B", true)
	bn := joinResumeAgentNode(t, b)
	join := workflow.NewJoinNode("join")
	r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: p}, {From: workflow.Start, To: bn}, {From: p, To: join}, {From: bn, To: join}}, b)
	for _, err := range r.Run(t.Context(), "u", "review-delegation", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
	}
	for _, err := range r.Run(t.Context(), "u", "review-delegation", resumeContent("p1", workflow.WorkflowInputFunctionCallName, "approved P"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
	}
	for _, err := range r.Run(t.Context(), "u", "review-delegation", resumeContent("p2", workflow.WorkflowInputFunctionCallName, "approved P2"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
	}
	var sawJoin bool
	for ev, err := range r.Run(t.Context(), "u", "review-delegation", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved B"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
		if out, ok := ev.Output.(map[string]any); ok {
			sawJoin = true
			if !reflect.DeepEqual(out, map[string]any{"p": "computed", "b": "approved B"}) {
				t.Fatal("join lost the cached delegated result")
			}
		}
	}
	if !sawJoin {
		t.Fatal("join not emitted")
	}
}

func TestRunner_WorkflowHITL_NestedCachedDelegationAcrossResume(t *testing.T) {
	compute := workflow.NewFunctionNode("compute", func(agent.Context, any) (string, error) { return "computed", nil }, workflow.NodeConfig{})
	mid := workflow.NewDynamicNode("mid", func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
		if _, err := workflow.RunNode[string](ctx, compute, in, workflow.WithUseAsOutput()); err != nil {
			return nil, err
		}
		if _, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "p1", Message: "approve"}); err != nil {
			return nil, err
		}
		_, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "p2", Message: "approve 2"})
		return nil, err
	}, workflow.NodeConfig{})
	p := workflow.NewDynamicNode("p", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		return workflow.RunNode[any](ctx, mid, in, workflow.WithUseAsOutput())
	}, workflow.NodeConfig{})
	b := joinResumeAgent(t, "b", "B", true)
	bn := joinResumeAgentNode(t, b)
	join := workflow.NewJoinNode("join")
	r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: p}, {From: workflow.Start, To: bn}, {From: p, To: join}, {From: bn, To: join}}, b)
	for _, err := range r.Run(t.Context(), "u", "review-delegation", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
	}
	for _, err := range r.Run(t.Context(), "u", "review-delegation", resumeContent("p1", workflow.WorkflowInputFunctionCallName, "approved P"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
	}
	for _, err := range r.Run(t.Context(), "u", "review-delegation", resumeContent("p2", workflow.WorkflowInputFunctionCallName, "approved P2"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
	}
	var sawJoin bool
	for ev, err := range r.Run(t.Context(), "u", "review-delegation", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved B"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
		if out, ok := ev.Output.(map[string]any); ok {
			sawJoin = true
			if !reflect.DeepEqual(out, map[string]any{"p": "computed", "b": "approved B"}) {
				t.Fatal("join lost the nested delegated result")
			}
		}
	}
	if !sawJoin {
		t.Fatal("join not emitted")
	}
}
