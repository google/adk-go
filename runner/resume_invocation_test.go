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
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

func newResumeRunner(t *testing.T, root agent.Agent, svc session.Service, resumable bool, plugins ...*plugin.Plugin) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          root,
		SessionService: svc,
		Resumable:      resumable,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r
}

// seedResumeSession creates the node test session and appends evs to it.
func seedResumeSession(t *testing.T, svc session.Service, evs ...*session.Event) {
	t.Helper()
	newNodeTestSession(t, t.Context(), svc)
	for _, ev := range evs {
		appendSeedEvent(t, t.Context(), svc, ev)
	}
}

func resumeSeed(inv, author string, content *genai.Content) *session.Event {
	ev := session.NewEvent(context.Background(), inv)
	ev.Author = author
	ev.Content = content
	return ev
}

func seedUser(inv, text string) *session.Event {
	return resumeSeed(inv, "user", genai.NewContentFromText(text, genai.RoleUser))
}

func seedText(inv, author, text string) *session.Event {
	return resumeSeed(inv, author, genai.NewContentFromText(text, genai.RoleModel))
}

func seedEnd(inv, author string) *session.Event {
	ev := resumeSeed(inv, author, nil)
	ev.Actions.EndOfAgent = new(true)
	return ev
}

func seedAgentState(inv, author string, state map[string]any) *session.Event {
	ev := resumeSeed(inv, author, nil)
	ev.Actions.AgentState = state
	return ev
}

func seedLongRunningCall(inv, author, id, name string) *session.Event {
	ev := resumeSeed(inv, author, fnCallContent(id, name, nil))
	ev.LongRunningToolIDs = []string{id}
	return ev
}

func fnCallContent(id, name string, args map[string]any) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: args}},
	}}
}

func responseContent(id, name string, response map[string]any) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{ID: id, Name: name, Response: response}},
	}}
}

func transferCallContent(target string) *genai.Content {
	return fnCallContent("", "transfer_to_agent", map[string]any{"agent_name": target})
}

func seedTransferCall(inv, author, target, id string) *session.Event {
	return resumeSeed(inv, author, fnCallContent(id, "transfer_to_agent", map[string]any{"agent_name": target}))
}

func seedTransferResponse(inv, author, target, id string) *session.Event {
	ev := resumeSeed(inv, author, responseContent(id, "transfer_to_agent", map[string]any{"result": nil}))
	ev.Actions.TransferToAgent = target
	return ev
}

// describeEvent names ev as "<author> <kind>", where kind is "end" for an
// end-of-agent event, "call <name>", "response <name>" or "text <text>".
func describeEvent(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + " end"
	}
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				return ev.Author + " call " + p.FunctionCall.Name
			case p.FunctionResponse != nil:
				return ev.Author + " response " + p.FunctionResponse.Name
			}
		}
		return ev.Author + " text " + eventText(ev)
	}
	return ev.Author + " empty"
}

func describeEvents(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		out = append(out, describeEvent(ev))
	}
	return out
}

func sessionEvents(t *testing.T, svc session.Service) []*session.Event {
	t.Helper()
	var out []*session.Event
	for ev := range storedNodeTestSession(t, t.Context(), svc).Events().All() {
		out = append(out, ev)
	}
	return out
}

func scriptedLlmAgent(t *testing.T, name string, m model.LLM, cfg llmagent.Config) agent.Agent {
	t.Helper()
	cfg.Name = name
	cfg.Model = m
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q) error = %v", name, err)
	}
	return a
}

func scripted(contents ...*genai.Content) *scriptedModel {
	return &scriptedModel{responses: contents}
}

func modelText(text string) *genai.Content {
	return genai.NewContentFromText(text, genai.RoleModel)
}

// shellAgent returns a custom agent that counts its runs and ranges over the
// events of child, as Python's custom root agents do in the tests ported here.
func shellAgent(t *testing.T, name string, runs *atomic.Int32, child agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:      name,
		SubAgents: []agent.Agent{child},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				for ev, err := range child.Run(ctx) {
					if !yield(ev, err) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

// recordingRoot returns a custom agent that counts its runs, records the
// user content and invocation ID it sees, and yields one text event.
type recordingRoot struct {
	runs         atomic.Int32
	mu           sync.Mutex
	userContents []string
	invocations  []string
}

func (r *recordingRoot) agent(t *testing.T, name string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:      name,
		SubAgents: subAgents,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				r.runs.Add(1)
				r.mu.Lock()
				text := ""
				if uc := ctx.UserContent(); uc != nil && len(uc.Parts) > 0 {
					text = uc.Parts[0].Text
				}
				r.userContents = append(r.userContents, text)
				r.invocations = append(r.invocations, ctx.InvocationID())
				r.mu.Unlock()
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = modelText(name + " reply")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

func runTurn(t *testing.T, r *runner.Runner, msg *genai.Content, opts ...runner.RunOption) []*session.Event {
	t.Helper()
	events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, msg, agent.RunConfig{}, opts...))
	if len(errs) > 0 {
		t.Fatalf("Run() errors = %v", errs)
	}
	return events
}

func endOfAgentCount(evs []*session.Event) int {
	n := 0
	for _, ev := range evs {
		if ev.Actions.EndOfAgent != nil {
			n++
		}
	}
	return n
}

func TestRun_NonResumable_WritesNoEndOfAgent(t *testing.T) {
	tests := []struct {
		name string
		root func(t *testing.T) agent.Agent
		want []string
	}{
		{
			name: "text turn",
			root: func(t *testing.T) agent.Agent {
				return scriptedLlmAgent(t, "root_agent", scripted(modelText("hello")), llmagent.Config{})
			},
			want: []string{"root_agent text hello"},
		},
		{
			name: "transfer turn",
			root: func(t *testing.T) agent.Agent {
				sub := scriptedLlmAgent(t, "sub_agent_1", scripted(modelText("hi")), llmagent.Config{})
				return scriptedLlmAgent(t, "root_agent", scripted(transferCallContent("sub_agent_1")), llmagent.Config{SubAgents: []agent.Agent{sub}})
			},
			want: []string{"root_agent call transfer_to_agent", "root_agent response transfer_to_agent", "sub_agent_1 text hi"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			newNodeTestSession(t, t.Context(), svc)
			r := newResumeRunner(t, tt.root(t), svc, false)

			events := runTurn(t, r, userText("hi"))

			if diff := cmp.Diff(tt.want, describeEvents(events)); diff != "" {
				t.Errorf("events (-want +got):\n%s", diff)
			}
			if n := endOfAgentCount(sessionEvents(t, svc)); n != 0 {
				t.Errorf("session holds %d events with EndOfAgent set, want 0", n)
			}
		})
	}
}

func TestRun_Resumable_EndOfAgentEventFields(t *testing.T) {
	t.Run("LlmAgent root", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		r := newResumeRunner(t, scriptedLlmAgent(t, "root_agent", scripted(modelText("hello")), llmagent.Config{}), svc, true)

		events := runTurn(t, r, userText("hi"))

		if len(events) != 2 {
			t.Fatalf("Run() yielded %v, want 2 events", describeEvents(events))
		}
		marker := events[1]
		if marker.Author != "root_agent" || marker.InvocationID != events[0].InvocationID || marker.Branch != "" {
			t.Errorf("marker author, invocation, branch = %q, %q, %q; want %q, %q, %q",
				marker.Author, marker.InvocationID, marker.Branch, "root_agent", events[0].InvocationID, "")
		}
		if marker.Content != nil || marker.Actions.EndOfAgent == nil || !*marker.Actions.EndOfAgent ||
			marker.Actions.AgentState != nil || len(marker.Actions.StateDelta) != 0 {
			t.Errorf("marker content, actions = %v, %+v; want no content, EndOfAgent true and nothing else", marker.Content, marker.Actions)
		}
	})
	t.Run("custom root runs the LlmAgent on a branch", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		child := scriptedLlmAgent(t, "child", scripted(modelText("hello")), llmagent.Config{})
		shell, err := agent.New(agent.Config{
			Name:      "shell",
			SubAgents: []agent.Agent{child},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				branch := "shell.child"
				return child.Run(ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &branch}))
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		r := newResumeRunner(t, shell, svc, true)

		events := runTurn(t, r, userText("hi"))

		if diff := cmp.Diff([]string{"child text hello", "child end"}, describeEvents(events)); diff != "" {
			t.Fatalf("events (-want +got):\n%s", diff)
		}
		if got := events[1].Branch; got != "shell.child" {
			t.Errorf("marker branch = %q, want %q", got, "shell.child")
		}
	})
}

func TestRun_Resumable_LongRunningToolPausesWithoutEndOfAgent(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	type args struct{}
	longRunning, err := functiontool.New(functiontool.Config{Name: "ask_human", Description: "asks", IsLongRunning: true},
		func(agent.Context, args) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "looks up"},
		func(agent.Context, args) (map[string]any, error) { return map[string]any{"v": 1}, nil })
	if err != nil {
		t.Fatal(err)
	}
	both := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "ask_human", Args: map[string]any{}}},
		{FunctionCall: &genai.FunctionCall{ID: "fc-2", Name: "lookup", Args: map[string]any{}}},
	}}
	root := scriptedLlmAgent(t, "root_agent", scripted(both, modelText("done")), llmagent.Config{Tools: []tool.Tool{longRunning, ordinary}})
	r := newResumeRunner(t, root, svc, true)

	events := runTurn(t, r, userText("hi"))

	paused := 0
	for _, ev := range events {
		if len(ev.LongRunningToolIDs) > 0 && len(ev.Content.Parts) == 2 {
			paused++
		}
	}
	if paused != 1 {
		t.Errorf("stream %v has %d call events with LongRunningToolIDs, want 1", describeEvents(events), paused)
	}
	if n := endOfAgentCount(events); n != 0 {
		t.Errorf("stream %v has %d end-of-agent events, want 0", describeEvents(events), n)
	}
	if n := endOfAgentCount(sessionEvents(t, svc)); n != 0 {
		t.Errorf("session has %d end-of-agent events, want 0", n)
	}
}

// failingModel is a model whose every call fails.
type failingModel struct{}

func (failingModel) Name() string { return "failing" }

func (failingModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) { yield(nil, errors.New("model down")) }
}

func TestRun_Resumable_FlowErrorWritesNoEndOfAgent(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	worker := scriptedLlmAgent(t, "worker", failingModel{}, llmagent.Config{})
	var runs atomic.Int32
	r := newResumeRunner(t, shellAgent(t, "shell", &runs, worker), svc, true)

	_, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}))

	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "model down") {
		t.Errorf("Run() errors = %v, want one error containing %q", errs, "model down")
	}
	if n := endOfAgentCount(sessionEvents(t, svc)); n != 0 {
		t.Errorf("session has %d end-of-agent events, want 0", n)
	}
}

func TestRun_ResumeByID_LlmAgentRoot_UnansweredLongRunningCallKeepsAgentOpen(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	type args struct{}
	longRunning, err := functiontool.New(functiontool.Config{Name: "ask_human", Description: "asks", IsLongRunning: true},
		func(agent.Context, args) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	root := scriptedLlmAgent(t, "root_agent", scripted(fnCallContent("lr-1", "ask_human", map[string]any{}), modelText("later")),
		llmagent.Config{Tools: []tool.Tool{longRunning}})
	r := newResumeRunner(t, root, svc, true)
	first := runTurn(t, r, userText("hi"))
	if diff := cmp.Diff([]string{"root_agent call ask_human"}, describeEvents(first)); diff != "" {
		t.Fatalf("run 1 events (-want +got):\n%s", diff)
	}

	// The unanswered call of run 1 is still the last event of the
	// invocation, so the resumed run pauses without calling the model, and the
	// agent does not end.
	events := runTurn(t, r, nil, runner.WithInvocationID(first[0].InvocationID))

	if diff := cmp.Diff([]string(nil), describeEvents(events)); diff != "" {
		t.Errorf("run 2 events (-want +got):\n%s", diff)
	}
	if n := endOfAgentCount(sessionEvents(t, svc)); n != 0 {
		t.Errorf("session has %d end-of-agent events, want 0", n)
	}
}

// A long-running call followed by the response of a tool called beside it
// still pauses the resumed invocation: the call is the second-to-last event,
// so the agent writes no end-of-agent event.
func TestRun_ResumeByID_LlmAgentRoot_LongRunningCallBeforeSiblingResponseKeepsAgentOpen(t *testing.T) {
	svc := session.InMemoryService()
	calls := resumeSeed("inv-1", "root_agent", &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "ask_human", Args: map[string]any{}}},
		{FunctionCall: &genai.FunctionCall{ID: "fc-2", Name: "lookup", Args: map[string]any{}}},
	}})
	calls.LongRunningToolIDs = []string{"lr-1"}
	seedResumeSession(t, svc, seedUser("inv-1", "hi"), calls,
		resumeSeed("inv-1", "root_agent", responseContent("fc-2", "lookup", map[string]any{"v": 1})))
	type args struct{}
	longRunning, err := functiontool.New(functiontool.Config{Name: "ask_human", Description: "asks", IsLongRunning: true},
		func(agent.Context, args) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "looks up"},
		func(agent.Context, args) (map[string]any, error) { return map[string]any{"v": 1}, nil })
	if err != nil {
		t.Fatal(err)
	}
	root := scriptedLlmAgent(t, "root_agent", scripted(modelText("later")), llmagent.Config{Tools: []tool.Tool{longRunning, ordinary}})
	r := newResumeRunner(t, root, svc, true)

	events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

	if diff := cmp.Diff([]string(nil), describeEvents(events)); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	if n := endOfAgentCount(sessionEvents(t, svc)); n != 0 {
		t.Errorf("session has %d end-of-agent events, want 0", n)
	}
}

func TestRun_Resumable_TransferOrder(t *testing.T) {
	t.Run("node path: the transferring agent ends before the target runs", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		sub := scriptedLlmAgent(t, "sub", scripted(transferCallContent("root_agent")), llmagent.Config{})
		rootModel := scripted(transferCallContent("sub"), modelText("root final"))
		root := scriptedLlmAgent(t, "root_agent", rootModel, llmagent.Config{SubAgents: []agent.Agent{sub}})
		r := newResumeRunner(t, root, svc, true)

		events := runTurn(t, r, userText("hi"))

		want := []string{
			"root_agent call transfer_to_agent", "root_agent response transfer_to_agent", "root_agent end",
			"sub call transfer_to_agent", "sub response transfer_to_agent", "sub end",
			"root_agent text root final", "root_agent end",
		}
		if diff := cmp.Diff(want, describeEvents(events)); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})
	t.Run("classic path: the transferring agent ends after the target", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		helper := scriptedLlmAgent(t, "helper", scripted(modelText("hi")), llmagent.Config{})
		coordinator := scriptedLlmAgent(t, "coordinator", scripted(transferCallContent("helper")), llmagent.Config{SubAgents: []agent.Agent{helper}})
		var runs atomic.Int32
		r := newResumeRunner(t, shellAgent(t, "shell", &runs, coordinator), svc, true)

		events := runTurn(t, r, userText("go"))

		want := []string{
			"coordinator call transfer_to_agent", "coordinator response transfer_to_agent",
			"helper text hi", "helper end", "coordinator end",
		}
		if diff := cmp.Diff(want, describeEvents(events)); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})
}

func TestRun_Resumable_NextTurnSkipsEndOfAgentAndAgentStateEvents(t *testing.T) {
	stateEvent := func(inv, author string) *session.Event {
		ev := resumeSeed(inv, author, nil)
		ev.Actions.AgentState = map[string]any{}
		return ev
	}
	tests := []struct {
		name string
		tail []*session.Event
	}{
		{"end-of-agent events", []*session.Event{seedEnd("inv-1", "sub_agent_1"), seedEnd("inv-1", "root_agent")}},
		{"agent-state event", []*session.Event{stateEvent("inv-1", "root_agent")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, append([]*session.Event{seedUser("inv-1", "hi"), seedText("inv-1", "sub_agent_1", "earlier")}, tt.tail...)...)
			subModel := scripted(modelText("sub answer"))
			sub := scriptedLlmAgent(t, "sub_agent_1", subModel, llmagent.Config{})
			rootModel := scripted(modelText("root answer"))
			root := scriptedLlmAgent(t, "root_agent", rootModel, llmagent.Config{SubAgents: []agent.Agent{sub}})
			r := newResumeRunner(t, root, svc, true)

			events := runTurn(t, r, userText("next"))

			if diff := cmp.Diff([]string{"sub_agent_1 text sub answer", "sub_agent_1 end"}, describeEvents(events)); diff != "" {
				t.Errorf("events (-want +got):\n%s", diff)
			}
			if rootModel.call != 0 {
				t.Errorf("root_agent's model was called %d times, want 0", rootModel.call)
			}
		})
	}
}

func TestRun_Resumable_AnsweredCallRouting(t *testing.T) {
	t.Run("call author outside the tree", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc,
			resumeSeed("inv-1", "ghost", fnCallContent("fc-1", "tool_fn", nil)),
			resumeSeed("inv-1", "user", responseContent("fc-1", "tool_fn", map[string]any{"ok": true})),
		)
		root := scriptedLlmAgent(t, "root_agent", scripted(modelText("root answer")), llmagent.Config{})
		r := newResumeRunner(t, root, svc, true)

		events := runTurn(t, r, userText("Continue"))

		if diff := cmp.Diff([]string{"root_agent text root answer", "root_agent end"}, describeEvents(events)); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})
	t.Run("agent-authored answer on a runner that is not resumable", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc,
			resumeSeed("inv-1", "sub_agent", fnCallContent("fc-1", "tool_fn", nil)),
			resumeSeed("inv-1", "sub_agent", responseContent("fc-1", "tool_fn", map[string]any{"ok": true})),
		)
		sub := scriptedLlmAgent(t, "sub_agent", scripted(modelText("sub answer")), llmagent.Config{
			DisallowTransferToParent: true, DisallowTransferToPeers: true,
		})
		root := scriptedLlmAgent(t, "root_agent", scripted(modelText("root answer")), llmagent.Config{SubAgents: []agent.Agent{sub}})
		r := newResumeRunner(t, root, svc, false)

		events := runTurn(t, r, userText("Continue"))

		if diff := cmp.Diff([]string{"root_agent text root answer"}, describeEvents(events)); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})
}

func TestRun_Resumable_CustomRootPlainMessageDoesNotRouteToPriorAgentResponse(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc,
		resumeSeed("inv-1", "sub_agent", fnCallContent("fc-1", "tool_fn", nil)),
		resumeSeed("inv-1", "sub_agent", responseContent("fc-1", "tool_fn", map[string]any{"ok": true})),
	)
	subModel := scripted(modelText("sub answer"))
	sub := scriptedLlmAgent(t, "sub_agent", subModel, llmagent.Config{})
	rec := &recordingRoot{}
	r := newResumeRunner(t, rec.agent(t, "root", sub), svc, true)

	runTurn(t, r, userText("Continue"))

	if got := rec.runs.Load(); got != 1 {
		t.Errorf("root ran %d times, want 1", got)
	}
	if subModel.call != 0 {
		t.Errorf("sub_agent's model was called %d times, want 0", subModel.call)
	}
}

func TestRun_WithInvocationID_NewMessageOnNonResumableRunner(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	rec := &recordingRoot{}
	r := newResumeRunner(t, rec.agent(t, "root"), svc, false)

	events := runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-custom"))

	if len(events) != 1 || events[0].InvocationID != "inv-custom" {
		t.Errorf("events %v with invocation IDs %v, want one event under %q", describeEvents(events), invocationIDs(events), "inv-custom")
	}
	if got := rec.runs.Load(); got != 1 {
		t.Errorf("root ran %d times, want 1", got)
	}
	stored := sessionEvents(t, svc)
	if len(stored) == 0 || stored[0].Author != "user" || stored[0].InvocationID != "inv-custom" {
		t.Errorf("first stored event = %v under %v, want the user event under %q", describeEvents(stored), invocationIDs(stored), "inv-custom")
	}
}

func invocationIDs(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		out = append(out, ev.InvocationID)
	}
	return out
}

func TestRun_ResumeByID_Classic_Routing(t *testing.T) {
	t.Run("ended root yields nothing", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv-1", "hi"), seedText("inv-1", "root_agent", "earlier"), seedEnd("inv-1", "root_agent"))
		rec := &recordingRoot{}
		r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)

		events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if len(events) != 0 {
			t.Errorf("Run() yielded %v, want nothing", describeEvents(events))
		}
		if got := rec.runs.Load(); got != 0 {
			t.Errorf("root ran %d times, want 0", got)
		}
	})
	t.Run("root that has not ended runs", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv-1", "hi"), seedText("inv-1", "root_agent", "earlier"))
		rec := &recordingRoot{}
		r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)

		runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if got := rec.runs.Load(); got != 1 {
			t.Errorf("root ran %d times, want 1", got)
		}
	})
	t.Run("answered call routes to its author", func(t *testing.T) {
		svc := session.InMemoryService()
		call := resumeSeed("inv-1", "worker", fnCallContent("fc1", "ask_human", nil))
		call.LongRunningToolIDs = []string{"fc1"}
		seedResumeSession(t, svc,
			seedUser("inv-1", "go"),
			call,
			resumeSeed("inv-1", "user", responseContent("fc1", "ask_human", map[string]any{"ok": true})),
		)
		worker := scriptedLlmAgent(t, "worker", scripted(modelText("done")), llmagent.Config{})
		var runs atomic.Int32
		r := newResumeRunner(t, shellAgent(t, "shell", &runs, worker), svc, true)

		events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if diff := cmp.Diff([]string{"worker text done", "worker end"}, describeEvents(events)); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if got := runs.Load(); got != 0 {
			t.Errorf("shell ran %d times, want 0", got)
		}
	})
	t.Run("root reopened by an agent state runs", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv-1", "go"), seedText("inv-1", "root_agent", "earlier"),
			seedEnd("inv-1", "root_agent"), seedAgentState("inv-1", "root_agent", map[string]any{"k": "v"}))
		rec := &recordingRoot{}
		r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)

		runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if got := rec.runs.Load(); got != 1 {
			t.Errorf("root ran %d times, want 1", got)
		}
	})
	t.Run("root that recorded content runs instead of an answered call's author", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc,
			seedUser("inv-1", "go"),
			seedText("inv-1", "root_agent", "earlier"),
			resumeSeed("inv-1", "worker", fnCallContent("fc1", "tool_fn", nil)),
			resumeSeed("inv-1", "user", responseContent("fc1", "tool_fn", map[string]any{"ok": true})),
		)
		workerModel := scripted(modelText("worker again"))
		worker := scriptedLlmAgent(t, "worker", workerModel, llmagent.Config{})
		rec := &recordingRoot{}
		r := newResumeRunner(t, rec.agent(t, "root_agent", worker), svc, true)

		runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if got := rec.runs.Load(); got != 1 {
			t.Errorf("root ran %d times, want 1", got)
		}
		if workerModel.call != 0 {
			t.Errorf("worker's model was called %d times, want 0", workerModel.call)
		}
	})
}

// A function response resumes the invocation of the call it answers, with or
// without the caller's id, so an agent that already ended there does not run
// again (adk-python Runner.run_async, _resolve_invocation_id).
func TestRun_Resumable_FunctionResponseResumesItsInvocation(t *testing.T) {
	tests := []struct {
		name string
		opts []runner.RunOption
	}{
		{"no caller id", nil},
		{"caller id", []runner.RunOption{runner.WithInvocationID("inv-1")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc,
				seedUser("inv-1", "go"),
				resumeSeed("inv-1", "worker", fnCallContent("fc1", "tool_fn", nil)),
				resumeSeed("inv-1", "worker", responseContent("fc1", "tool_fn", map[string]any{"ok": true})),
				seedText("inv-1", "worker", "done"),
				seedEnd("inv-1", "worker"),
			)
			workerModel := scripted(modelText("again"))
			worker := scriptedLlmAgent(t, "worker", workerModel, llmagent.Config{})
			var runs atomic.Int32
			r := newResumeRunner(t, shellAgent(t, "shell", &runs, worker), svc, true)

			events := runTurn(t, r, responseContent("fc1", "tool_fn", map[string]any{"ok": true}), tt.opts...)

			if len(events) != 0 {
				t.Errorf("Run() yielded %v, want nothing", describeEvents(events))
			}
			if workerModel.call != 0 {
				t.Errorf("worker's model was called %d times, want 0", workerModel.call)
			}
		})
	}
}

func TestRun_ResumeByID_Classic_RefusesWithoutEventsOrUserMessage(t *testing.T) {
	tests := []struct {
		name    string
		seed    []*session.Event
		wantErr string
	}{
		{"no events", nil, "session s has no events to resume"},
		{"no user message of the invocation", []*session.Event{seedUser("inv-other", "hi")}, "no user message available for resuming invocation: inv-1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, tt.seed...)
			rec := &recordingRoot{}
			r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)

			_, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, nil, agent.RunConfig{},
				runner.WithInvocationID("inv-1"), runner.WithStateDelta(map[string]any{"k": "v"})))

			if len(errs) != 1 || errs[0].Error() != tt.wantErr {
				t.Errorf("Run() errors = %v, want [%s]", errs, tt.wantErr)
			}
			if got := len(sessionEvents(t, svc)); got != len(tt.seed) {
				t.Errorf("session holds %d events, want %d", got, len(tt.seed))
			}
			if got := rec.runs.Load(); got != 0 {
				t.Errorf("root ran %d times, want 0", got)
			}
		})
	}
}

func TestRun_ResumeByID_Classic_UserContentAndID(t *testing.T) {
	t.Run("no message", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv-1", "original"), seedText("inv-1", "root_agent", "earlier"))
		rec := &recordingRoot{}
		r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)

		runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if diff := cmp.Diff([]string{"original"}, rec.userContents); diff != "" {
			t.Errorf("user contents (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"inv-1"}, rec.invocations); diff != "" {
			t.Errorf("invocation IDs (-want +got):\n%s", diff)
		}
	})
	t.Run("new message", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv-1", "original"), seedText("inv-1", "root_agent", "earlier"))
		rec := &recordingRoot{}
		r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)

		runTurn(t, r, userText("again"), runner.WithInvocationID("inv-1"))

		if diff := cmp.Diff([]string{"again"}, rec.userContents); diff != "" {
			t.Errorf("user contents (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"inv-1"}, rec.invocations); diff != "" {
			t.Errorf("invocation IDs (-want +got):\n%s", diff)
		}
	})
}

func TestRun_ResumeByID_LlmAgentRoot_RestoresUserContentWithoutUserEvent(t *testing.T) {
	for _, resumable := range []bool{false, true} {
		t.Run(fmt.Sprintf("resumable=%v", resumable), func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, seedUser("inv-1", "original"), seedText("inv-1", "root_agent", "earlier"))
			var seen []string
			root := scriptedLlmAgent(t, "root_agent", scripted(modelText("again")), llmagent.Config{
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
					text := ""
					if uc := ctx.UserContent(); uc != nil && len(uc.Parts) > 0 {
						text = uc.Parts[0].Text
					}
					seen = append(seen, text)
					return nil, nil
				}},
			})
			var userMessageCallbacks atomic.Int32
			p, err := plugin.New(plugin.Config{
				Name: "counter",
				OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
					userMessageCallbacks.Add(1)
					return nil, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			r := newResumeRunner(t, root, svc, resumable, p)

			events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

			if diff := cmp.Diff([]string{"original"}, seen); diff != "" {
				t.Errorf("user contents (-want +got):\n%s", diff)
			}
			if got := userMessageCallbacks.Load(); got != 0 {
				t.Errorf("on-user-message callback ran %d times, want 0", got)
			}
			users := 0
			for _, ev := range sessionEvents(t, svc) {
				if ev.Author == "user" {
					users++
				}
			}
			if users != 1 {
				t.Errorf("session holds %d user events, want 1", users)
			}
			for _, ev := range events {
				if ev.InvocationID != "inv-1" {
					t.Errorf("event %q has invocation ID %q, want %q", describeEvent(ev), ev.InvocationID, "inv-1")
				}
			}
		})
	}
}

func TestRun_ResumeByID_LlmAgentRoot_DoesNotShortCircuitEndedAgent(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc, seedUser("inv-1", "hi"), seedText("inv-1", "root_agent", "earlier"), seedEnd("inv-1", "root_agent"))
	root := scriptedLlmAgent(t, "root_agent", scripted(modelText("again")), llmagent.Config{})
	r := newResumeRunner(t, root, svc, true)

	events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

	if diff := cmp.Diff([]string{"root_agent text again", "root_agent end"}, describeEvents(events)); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
}

func TestRun_ResumeByID_StateDelta(t *testing.T) {
	roots := []struct {
		name string
		root func(t *testing.T) agent.Agent
	}{
		{"classic", func(t *testing.T) agent.Agent { return (&recordingRoot{}).agent(t, "root_agent") }},
		{"node", func(t *testing.T) agent.Agent {
			return scriptedLlmAgent(t, "root_agent", scripted(modelText("again")), llmagent.Config{})
		}},
	}
	for _, rt := range roots {
		t.Run(rt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, seedUser("inv-1", "hi"), seedText("inv-1", "root_agent", "earlier"))
			r := newResumeRunner(t, rt.root(t), svc, true)

			runTurn(t, r, nil, runner.WithInvocationID("inv-1"), runner.WithStateDelta(map[string]any{"resumed_key": "resumed_value"}))

			stored := storedNodeTestSession(t, t.Context(), svc)
			delta := stored.Events().At(2)
			if delta.Author != "user" || delta.InvocationID != "inv-1" || delta.Content != nil || delta.Branch != "" {
				t.Errorf("first new event: author %q, invocation %q, content %v, branch %q; want %q, %q, nil, %q",
					delta.Author, delta.InvocationID, delta.Content, delta.Branch, "user", "inv-1", "")
			}
			if diff := cmp.Diff(map[string]any{"resumed_key": "resumed_value"}, delta.Actions.StateDelta); diff != "" {
				t.Errorf("first new event state delta (-want +got):\n%s", diff)
			}
			if got, err := stored.State().Get("resumed_key"); err != nil || got != "resumed_value" {
				t.Errorf("state resumed_key = %v, %v; want %q", got, err, "resumed_value")
			}
		})
	}
}

func TestRun_ResumeByID_Classic_RebuildsAgentStates(t *testing.T) {
	transferToHelper := []*session.Event{
		seedUser("inv-1", "go"),
		seedTransferCall("inv-1", "coordinator", "helper", "t-1"),
		seedTransferResponse("inv-1", "coordinator", "helper", "t-1"),
	}
	tests := []struct {
		name           string
		history        []*session.Event
		coordinator    *scriptedModel
		helper, deep   *scriptedModel
		want           []string
		wantErr        string
		wantModelCalls [3]int // coordinator, helper, deep
		// siblings makes deep a sub-agent of coordinator, beside helper,
		// instead of a sub-agent of helper.
		siblings bool
	}{
		{
			name:           "resumes the agent's transfer target",
			history:        transferToHelper,
			coordinator:    scripted(modelText("unused")),
			helper:         scripted(modelText("helper done")),
			deep:           scripted(modelText("unused")),
			want:           []string{"helper text helper done", "helper end", "coordinator end"},
			wantModelCalls: [3]int{0, 1, 0},
		},
		{
			name:           "an ended agent starts over",
			history:        append(slicesClone(transferToHelper), seedEnd("inv-1", "coordinator"), seedText("inv-1", "helper", "partial")),
			coordinator:    scripted(modelText("fresh")),
			helper:         scripted(modelText("unused")),
			deep:           scripted(modelText("unused")),
			want:           []string{"coordinator text fresh", "coordinator end"},
			wantModelCalls: [3]int{1, 0, 0},
		},
		{
			name: "content after an end reopens the agent",
			history: append([]*session.Event{
				seedUser("inv-1", "go"), seedText("inv-1", "coordinator", "a"), seedEnd("inv-1", "coordinator"),
			}, transferToHelper[1:]...),
			coordinator:    scripted(modelText("unused")),
			helper:         scripted(modelText("helper done")),
			deep:           scripted(modelText("unused")),
			want:           []string{"helper text helper done", "helper end", "coordinator end"},
			wantModelCalls: [3]int{0, 1, 0},
		},
		{
			name: "chained transfers resume down the chain",
			history: append(slicesClone(transferToHelper),
				seedTransferCall("inv-1", "helper", "deep", "t-2"),
				seedTransferResponse("inv-1", "helper", "deep", "t-2"),
				seedText("inv-1", "deep", "partial"),
			),
			coordinator:    scripted(modelText("unused")),
			helper:         scripted(modelText("unused")),
			deep:           scripted(modelText("deep done")),
			want:           []string{"deep text deep done", "deep end", "helper end", "coordinator end"},
			wantModelCalls: [3]int{0, 0, 1},
		},
		{
			name: "a sub-agent's transfer to a peer is not the agent's own",
			history: append(slicesClone(transferToHelper),
				seedTransferCall("inv-1", "helper", "deep", "t-2"),
				seedTransferResponse("inv-1", "helper", "deep", "t-2"),
				seedText("inv-1", "deep", "partial"),
			),
			coordinator:    scripted(modelText("unused")),
			helper:         scripted(modelText("unused")),
			deep:           scripted(modelText("deep done")),
			want:           []string{"deep text deep done", "deep end", "helper end", "coordinator end"},
			wantModelCalls: [3]int{0, 0, 1},
			siblings:       true,
		},
		{
			name: "the agent's own last event stops the search",
			history: append(slicesClone(transferToHelper),
				seedText("inv-1", "helper", "h"), seedEnd("inv-1", "helper"), seedText("inv-1", "coordinator", "back"),
			),
			coordinator:    scripted(modelText("fresh")),
			helper:         scripted(modelText("unused")),
			deep:           scripted(modelText("unused")),
			want:           []string{"coordinator text fresh", "coordinator end"},
			wantModelCalls: [3]int{1, 0, 0},
		},
		{
			name: "a user response that answers no call is an error",
			history: append(slicesClone(transferToHelper),
				seedText("inv-1", "helper", "h"),
				resumeSeed("inv-1", "user", responseContent("unknown", "tool_fn", map[string]any{})),
			),
			coordinator: scripted(modelText("unused")),
			helper:      scripted(modelText("unused")),
			deep:        scripted(modelText("unused")),
			wantErr:     "no agent to transfer to for resuming agent from function response coordinator",
		},
		{
			name: "an agent state alone makes a user response that answers no call an error",
			history: []*session.Event{
				seedUser("inv-1", "go"),
				seedAgentState("inv-1", "coordinator", map[string]any{}),
				resumeSeed("inv-1", "user", responseContent("unknown", "tool_fn", map[string]any{})),
			},
			coordinator: scripted(modelText("unused")),
			helper:      scripted(modelText("unused")),
			deep:        scripted(modelText("unused")),
			wantErr:     "no agent to transfer to for resuming agent from function response coordinator",
		},
		{
			name: "the user's answer to the agent's own call continues the agent",
			history: append(slicesClone(transferToHelper),
				seedText("inv-1", "helper", "h"),
				seedLongRunningCall("inv-1", "coordinator", "lr-1", "ask_human"),
				resumeSeed("inv-1", "user", responseContent("lr-1", "ask_human", map[string]any{"ok": true})),
			),
			coordinator:    scripted(modelText("coordinator done")),
			helper:         scripted(modelText("unused")),
			deep:           scripted(modelText("unused")),
			want:           []string{"coordinator text coordinator done", "coordinator end"},
			wantModelCalls: [3]int{1, 0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, tt.history...)
			deep := scriptedLlmAgent(t, "deep", tt.deep, llmagent.Config{})
			helperSubs, coordinatorSubs := []agent.Agent{deep}, []agent.Agent(nil)
			if tt.siblings {
				helperSubs, coordinatorSubs = nil, []agent.Agent{deep}
			}
			helper := scriptedLlmAgent(t, "helper", tt.helper, llmagent.Config{SubAgents: helperSubs})
			coordinator := scriptedLlmAgent(t, "coordinator", tt.coordinator, llmagent.Config{SubAgents: append([]agent.Agent{helper}, coordinatorSubs...)})
			var runs atomic.Int32
			r := newResumeRunner(t, shellAgent(t, "shell", &runs, coordinator), svc, true)

			events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, nil, agent.RunConfig{}, runner.WithInvocationID("inv-1")))

			if tt.wantErr != "" {
				if len(errs) != 1 || errs[0].Error() != tt.wantErr {
					t.Errorf("Run() errors = %v, want [%s]", errs, tt.wantErr)
				}
				return
			}
			if len(errs) > 0 {
				t.Fatalf("Run() errors = %v", errs)
			}
			if diff := cmp.Diff(tt.want, describeEvents(events)); diff != "" {
				t.Errorf("events (-want +got):\n%s", diff)
			}
			if got := [3]int{tt.coordinator.call, tt.helper.call, tt.deep.call}; got != tt.wantModelCalls {
				t.Errorf("model calls (coordinator, helper, deep) = %v, want %v", got, tt.wantModelCalls)
			}
		})
	}
}

func slicesClone(evs []*session.Event) []*session.Event {
	return append([]*session.Event(nil), evs...)
}

func TestRun_ResumeByID_Classic_PausedSubAgentDoesNotEndCoordinator(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc,
		seedUser("inv-1", "go"),
		seedTransferCall("inv-1", "coordinator", "helper", "t-1"),
		seedTransferResponse("inv-1", "coordinator", "helper", "t-1"),
	)
	type args struct{}
	ask, err := functiontool.New(functiontool.Config{Name: "ask_human", Description: "asks", IsLongRunning: true},
		func(agent.Context, args) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	helper := scriptedLlmAgent(t, "helper", scripted(fnCallContent("lr-1", "ask_human", map[string]any{})), llmagent.Config{Tools: []tool.Tool{ask}})
	coordinator := scriptedLlmAgent(t, "coordinator", scripted(modelText("unused")), llmagent.Config{SubAgents: []agent.Agent{helper}})
	var runs atomic.Int32
	r := newResumeRunner(t, shellAgent(t, "shell", &runs, coordinator), svc, true)

	events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

	sawPause := false
	for _, ev := range events {
		if ev.Author == "helper" && len(ev.LongRunningToolIDs) > 0 {
			sawPause = true
		}
	}
	if !sawPause {
		t.Errorf("stream %v has no long-running call from helper", describeEvents(events))
	}
	if n := endOfAgentCount(events); n != 0 {
		t.Errorf("stream %v has %d end-of-agent events, want 0", describeEvents(events), n)
	}
}

func TestRun_ResumeByID_ConcurrentRuns(t *testing.T) {
	svc := session.InMemoryService()
	rec := &recordingRoot{}
	r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)
	const n = 8
	for i := range n {
		sid := fmt.Sprintf("s-%d", i)
		created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: nodeTestApp, UserID: nodeTestUser, SessionID: sid})
		if err != nil {
			t.Fatal(err)
		}
		inv := fmt.Sprintf("inv-%d", i)
		for _, ev := range []*session.Event{seedUser(inv, "hi"), seedText(inv, "root_agent", "earlier")} {
			if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
				t.Fatal(err)
			}
		}
	}
	var wg sync.WaitGroup
	errs := make([]string, n)
	for i := range n {
		wg.Go(func() {
			inv := fmt.Sprintf("inv-%d", i)
			events, runErrs := collectRun(r.Run(t.Context(), nodeTestUser, fmt.Sprintf("s-%d", i), nil, agent.RunConfig{}, runner.WithInvocationID(inv)))
			if len(runErrs) > 0 || len(events) != 1 || events[0].InvocationID != inv {
				errs[i] = fmt.Sprintf("run %d: events under %v, errors %v; want one event under %q", i, invocationIDs(events), runErrs, inv)
			}
		})
	}
	wg.Wait()
	if msg := strings.TrimSpace(strings.Join(errs, "\n")); msg != "" {
		t.Error(msg)
	}
	if got := rec.runs.Load(); got != n {
		t.Errorf("root ran %d times, want %d", got, n)
	}
}

func TestRun_Resumable_ConsumerStopsEarly_NoEndOfAgentPersisted(t *testing.T) {
	tests := []struct {
		name string
		root func(t *testing.T) agent.Agent
		want []string
	}{
		{
			name: "node path",
			root: func(t *testing.T) agent.Agent {
				return scriptedLlmAgent(t, "root_agent", scripted(modelText("hello")), llmagent.Config{})
			},
			want: []string{"user text hi", "root_agent text hello"},
		},
		{
			name: "classic path",
			root: func(t *testing.T) agent.Agent {
				var runs atomic.Int32
				return shellAgent(t, "shell", &runs, scriptedLlmAgent(t, "child", scripted(modelText("hello")), llmagent.Config{}))
			},
			want: []string{"user text hi", "child text hello"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			newNodeTestSession(t, t.Context(), svc)
			r := newResumeRunner(t, tt.root(t), svc, true)

			for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				if ev != nil {
					break
				}
			}

			if diff := cmp.Diff(tt.want, describeEvents(sessionEvents(t, svc))); diff != "" {
				t.Errorf("session events (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRun_RoutingSkipsRewoundEvents(t *testing.T) {
	svc := session.InMemoryService()
	rewind := resumeSeed("inv-2", "user", nil)
	rewind.Actions.RewindBeforeInvocationID = "inv-1"
	seedResumeSession(t, svc, seedUser("inv-1", "hi"), seedText("inv-1", "sub_agent_1", "earlier"), rewind)
	subModel := scripted(modelText("sub answer"))
	sub := scriptedLlmAgent(t, "sub_agent_1", subModel, llmagent.Config{})
	root := scriptedLlmAgent(t, "root_agent", scripted(modelText("root answer")), llmagent.Config{SubAgents: []agent.Agent{sub}})
	r := newResumeRunner(t, root, svc, false)

	events := runTurn(t, r, userText("next"))

	if diff := cmp.Diff([]string{"root_agent text root answer"}, describeEvents(events)); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	if subModel.call != 0 {
		t.Errorf("sub_agent_1's model was called %d times, want 0", subModel.call)
	}
}

func TestRun_WithInvocationID_EmptyStartsNewInvocation(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	rec := &recordingRoot{}
	r := newResumeRunner(t, rec.agent(t, "root_agent"), svc, true)

	runTurn(t, r, userText("one"), runner.WithInvocationID(""))
	runTurn(t, r, userText("two"), runner.WithInvocationID(""))

	if len(rec.invocations) != 2 || rec.invocations[0] == rec.invocations[1] ||
		!strings.HasPrefix(rec.invocations[0], "e-") || !strings.HasPrefix(rec.invocations[1], "e-") {
		t.Errorf("invocation IDs = %v, want two different IDs starting with %q", rec.invocations, "e-")
	}
}

func TestRunLive_RefusesInvocationID(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r := newResumeRunner(t, scriptedLlmAgent(t, "root_agent", scripted(modelText("hello")), llmagent.Config{}), svc, true)

	_, _, err := r.RunLive(t.Context(), nodeTestUser, "missing-session", agent.LiveRunConfig{}, runner.WithInvocationID("inv-1"))

	if err == nil || err.Error() != "RunLive does not support WithInvocationID; resume an invocation with Run" {
		t.Errorf("RunLive() error = %v, want the WithInvocationID refusal", err)
	}
}

// TestRun_ResumeByID_NodeRoot_AnswersInterruptWithOriginalUserContent runs
// u3336's shape: the turn that resumes by ID carries the function response,
// the node sees the stored message as its user content, and the interrupt
// counts as answered.
func TestRun_ResumeByID_NodeRoot_AnswersInterruptWithOriginalUserContent(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	var mu sync.Mutex
	var userContents []string
	n := newFuncNode("node", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		mu.Lock()
		userContents = append(userContents, ctx.UserContent().Parts[0].Text)
		mu.Unlock()
		ev := session.NewEvent(ctx, ctx.InvocationID())
		if _, ok := ctx.ResumedInput("fc-1"); ok {
			ev.Output = "resumed"
			yield(ev, nil)
			return
		}
		ev.Content = fnCallContent("fc-1", "tool", map[string]any{})
		ev.LongRunningToolIDs = []string{"fc-1"}
		yield(ev, nil)
	})
	r := newNodeRootRunner(t, n, svc)

	turn1 := runTurn(t, r, userText("original text"))
	turn2 := runTurn(t, r, responseContent("fc-1", "tool", map[string]any{"v": 1}), runner.WithInvocationID(turn1[0].InvocationID))

	if diff := cmp.Diff([]string{"original text", "original text"}, userContents); diff != "" {
		t.Errorf("node user contents (-want +got):\n%s", diff)
	}
	var outputs []any
	for _, ev := range turn2 {
		if len(ev.LongRunningToolIDs) > 0 {
			t.Errorf("turn 2 yielded a new long-running call: %q", describeEvent(ev))
		}
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}
	if diff := cmp.Diff([]any{"resumed"}, outputs); diff != "" {
		t.Errorf("turn 2 outputs (-want +got):\n%s", diff)
	}
}

func TestRun_PluginReplacementKeepsResumabilityFields(t *testing.T) {
	tests := []struct {
		name        string
		original    session.EventActions
		replacement session.EventActions
		want        session.EventActions
	}{
		{
			name:        "replacement sets the fields",
			replacement: session.EventActions{EndOfAgent: new(true), AgentState: map[string]any{"step": 1}, RewindBeforeInvocationID: "inv-0"},
			want:        session.EventActions{EndOfAgent: new(true), AgentState: map[string]any{"step": 1}, RewindBeforeInvocationID: "inv-0"},
		},
		{
			name:     "replacement leaves them unset",
			original: session.EventActions{EndOfAgent: new(false), AgentState: map[string]any{}, RewindBeforeInvocationID: "inv-0"},
			want:     session.EventActions{EndOfAgent: new(false), AgentState: map[string]any{}, RewindBeforeInvocationID: "inv-0"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			newNodeTestSession(t, t.Context(), svc)
			root, err := agent.New(agent.Config{Name: "root_agent", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Content = modelText("original")
					ev.Actions = tt.original
					yield(ev, nil)
				}
			}})
			if err != nil {
				t.Fatal(err)
			}
			p, err := plugin.New(plugin.Config{
				Name: "replacer",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					return &session.Event{LLMResponse: model.LLMResponse{Content: modelText("replaced")}, Actions: tt.replacement}, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			r := newResumeRunner(t, root, svc, false, p)

			runTurn(t, r, userText("hi"))

			stored := sessionEvents(t, svc)
			if diff := cmp.Diff(resumabilityOf(tt.want), resumabilityOf(stored[len(stored)-1].Actions)); diff != "" {
				t.Errorf("stored actions (-want +got):\n%s", diff)
			}
		})
	}
}

type resumability struct {
	EndOfAgent               *bool
	AgentState               map[string]any
	RewindBeforeInvocationID string
}

func resumabilityOf(a session.EventActions) resumability {
	return resumability{a.EndOfAgent, a.AgentState, a.RewindBeforeInvocationID}
}

// TestRun_NonResumableReplayUnexecutedSiblingPersistsResponseAndCallsModelOnce
// answers one call of a parallel call event whose other call never ran: the
// flow runs the missing call once, the session stores its response, and the
// next step sees that response and calls the model once rather than
// replaying the call again.
func TestRun_NonResumableReplayUnexecutedSiblingPersistsResponseAndCallsModelOnce(t *testing.T) {
	svc := session.InMemoryService()
	var askRuns, fetchRuns atomic.Int32
	type args struct{}
	ask, err := functiontool.New(functiontool.Config{Name: "ask", Description: "asks"},
		func(agent.Context, args) (map[string]any, error) {
			askRuns.Add(1)
			return map[string]any{"r": 1}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	fetch, err := functiontool.New(functiontool.Config{Name: "fetch", Description: "fetches"},
		func(agent.Context, args) (map[string]any, error) {
			fetchRuns.Add(1)
			return map[string]any{"r": 2}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	seedResumeSession(t, svc, resumeSeed("inv-1", "agent", &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "ask", Args: map[string]any{}}},
		{FunctionCall: &genai.FunctionCall{ID: "c2", Name: "fetch", Args: map[string]any{}}},
	}}))
	m := scripted(modelText("done"))
	root := scriptedLlmAgent(t, "agent", m, llmagent.Config{Tools: []tool.Tool{ask, fetch}})
	r := newResumeRunner(t, root, svc, false)

	runTurn(t, r, responseContent("c1", "ask", map[string]any{"r": 1}))

	if got := fetchRuns.Load(); got != 1 {
		t.Errorf("fetch ran %d times, want 1", got)
	}
	if got := askRuns.Load(); got != 0 {
		t.Errorf("ask ran %d times, want 0", got)
	}
	if m.call != 1 {
		t.Errorf("model called %d times, want 1", m.call)
	}
	want := []string{"agent call ask", "user response ask", "agent response fetch", "agent text done"}
	if diff := cmp.Diff(want, describeEvents(sessionEvents(t, svc))); diff != "" {
		t.Errorf("session events mismatch (-want +got):\n%s", diff)
	}
}
