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
	"context"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// cfcNamedModel is a model with a fixed name that answers every unary call
// with "unary reply". It has no live connection.
type cfcNamedModel struct{ name string }

func (m *cfcNamedModel) Name() string { return m.name }

func (m *cfcNamedModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("unary reply", genai.RoleModel)}, nil)
	}
}

// answerBeforeModel ends every model call with "callback reply" before the
// model is reached, so a run that passes the CFC check needs no live
// connection.
func answerBeforeModel(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
	return &model.LLMResponse{Content: genai.NewContentFromText("callback reply", genai.RoleModel)}, nil
}

func cfcLLMAgent(t *testing.T, name string, m model.LLM, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:                 name,
		Model:                m,
		SubAgents:            subAgents,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{answerBeforeModel},
	})
	if err != nil {
		t.Fatalf("llmagent.New(%q) error = %v", name, err)
	}
	return a
}

// contentEvents returns the description of each event of evs that has content.
func contentEvents(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		if ev.Content != nil {
			out = append(out, describeEvent(ev))
		}
	}
	return out
}

// errTexts returns the text of each error.
func errTexts(errs []error) []string {
	var out []string
	for _, err := range errs {
		out = append(out, err.Error())
	}
	return out
}

func TestRunner_CFCModelCheck(t *testing.T) {
	t.Run("refuses a root model without the gemini-2 prefix before storing the message", func(t *testing.T) {
		for _, name := range []string{"gemini-3-pro", "models/gemini-2.0-flash"} {
			t.Run(name, func(t *testing.T) {
				svc := session.InMemoryService()
				newNodeTestSession(t, t.Context(), svc)
				r := newResumeRunner(t, cfcLLMAgent(t, "root_agent", &cfcNamedModel{name: name}), svc, false)

				events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{SupportCFC: true}))

				want := []string{"CFC is not supported for model: " + name + " in agent: root_agent"}
				if diff := cmp.Diff(want, errTexts(errs)); diff != "" {
					t.Errorf("Run() errors mismatch (-want +got):\n%s", diff)
				}
				if len(events) != 0 {
					t.Errorf("Run() yielded %d events, want 0", len(events))
				}
				if n := storedNodeTestSession(t, t.Context(), svc).Events().Len(); n != 0 {
					t.Errorf("stored session has %d events, want 0", n)
				}
			})
		}
	})

	t.Run("checks the model the root inherits from its static parent", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		root := cfcLLMAgent(t, "root_agent", nil)
		cfcLLMAgent(t, "parent_agent", &cfcNamedModel{name: "claude-3-sonnet"}, root)
		r := newResumeRunner(t, root, svc, false)

		_, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{SupportCFC: true}))

		want := []string{"CFC is not supported for model: claude-3-sonnet in agent: root_agent"}
		if diff := cmp.Diff(want, errTexts(errs)); diff != "" {
			t.Errorf("Run() errors mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("refuses a resumed invocation before storing its state delta", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv-1", "hi"), seedText("inv-1", "root_agent", "earlier"))
		r := newResumeRunner(t, cfcLLMAgent(t, "root_agent", &cfcNamedModel{name: "gemini-3-pro"}), svc, true)

		events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, nil, agent.RunConfig{SupportCFC: true},
			runner.WithInvocationID("inv-1"), runner.WithStateDelta(map[string]any{"k": "v"})))

		want := []string{"CFC is not supported for model: gemini-3-pro in agent: root_agent"}
		if diff := cmp.Diff(want, errTexts(errs)); diff != "" {
			t.Errorf("Run() errors mismatch (-want +got):\n%s", diff)
		}
		if len(events) != 0 {
			t.Errorf("Run() yielded %d events, want 0", len(events))
		}
		stored := storedNodeTestSession(t, t.Context(), svc)
		if n := stored.Events().Len(); n != 2 {
			t.Errorf("stored session has %d events, want the 2 seeded", n)
		}
		if got, err := stored.State().Get("k"); err == nil {
			t.Errorf("state k = %v, want it not stored", got)
		}
	})

	t.Run("passes a gemini-2 root model", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		r := newResumeRunner(t, cfcLLMAgent(t, "root_agent", &cfcNamedModel{name: "gemini-2.5-flash"}), svc, false)

		events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{SupportCFC: true}))

		if len(errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", errs)
		}
		if diff := cmp.Diff([]string{"root_agent text callback reply"}, contentEvents(events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("does not check the LLM sub-agent of a root that is not an LLM agent", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
			Name:      "pipeline",
			SubAgents: []agent.Agent{cfcLLMAgent(t, "step_agent", &cfcNamedModel{name: "gemini-3-pro"})},
		}})
		if err != nil {
			t.Fatalf("sequentialagent.New() error = %v", err)
		}
		r := newResumeRunner(t, root, svc, false)

		events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{SupportCFC: true}))

		if len(errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", errs)
		}
		if diff := cmp.Diff([]string{"step_agent text callback reply"}, contentEvents(events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("does not check a root that is a workflow node", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		fn := workflow.NewFunctionNode("fn", func(agent.Context, any) (string, error) { return "node ran", nil }, workflow.NodeConfig{})
		r, err := runner.New(runner.Config{AppName: nodeTestApp, Node: fn, SessionService: svc})
		if err != nil {
			t.Fatalf("runner.New() error = %v", err)
		}

		events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{SupportCFC: true}))

		if len(errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", errs)
		}
		var outputs []any
		for _, ev := range events {
			outputs = append(outputs, ev.Output)
		}
		if diff := cmp.Diff([]any{"node ran"}, outputs); diff != "" {
			t.Errorf("Run() event outputs mismatch (-want +got):\n%s", diff)
		}
	})
}
