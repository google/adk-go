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

package testutil

import (
	"errors"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// NewInputWorkflow returns a graph workflow agent named name with two nodes.
// Node agent "b" asks for input with interrupt ID "b", which pauses a
// resumable invocation; once answered (see InputResponse) it yields the text
// "B", and node agent "d" then yields the text "D". Both node agents are in
// the workflow's SubAgents, so a shell agent over the workflow narrows its
// resume to it when the user answers b.
func NewInputWorkflow(t *testing.T, name string) agent.Agent {
	t.Helper()
	b, err := agent.New(agent.Config{Name: "b", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			emit := func(ev *session.Event) error {
				if !yield(ev, nil) {
					return errors.New("consumer stopped")
				}
				return nil
			}
			if _, err := workflow.ResumeOrRequestInput(agent.Promote(ctx), emit, session.RequestInput{InterruptID: "b", Message: "approve"}); err != nil {
				return
			}
			yield(&session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("B", genai.RoleModel)}}, nil)
		}
	}})
	if err != nil {
		t.Fatalf("agent.New(b) error = %v", err)
	}
	d, err := agent.New(agent.Config{Name: "d", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "d"
			ev.Content = genai.NewContentFromText("D", genai.RoleModel)
			yield(ev, nil)
		}
	}})
	if err != nil {
		t.Fatalf("agent.New(d) error = %v", err)
	}
	bn, err := workflow.NewAgentNode(b, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(b) error = %v", err)
	}
	dn, err := workflow.NewAgentNode(d, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(d) error = %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{
		Name: name, SubAgents: []agent.Agent{b, d},
		Edges: []workflow.Edge{{From: workflow.Start, To: bn}, {From: bn, To: dn}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(%q) error = %v", name, err)
	}
	return wf
}

// InputResponse returns the user content that answers the workflow input
// request with interrupt ID interruptID with payload.
func InputResponse(interruptID string, payload any) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{
			ID:       interruptID,
			Name:     workflow.WorkflowInputFunctionCallName,
			Response: map[string]any{"payload": payload},
		},
	}}}
}
