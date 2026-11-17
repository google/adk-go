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

package generatortool_test

import (
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/generatortool"
)

const (
	appName   = "test_app"
	userID    = "test_user"
	sessionID = "s1"
)

// queryArgs is the argument type of most test tools.
type queryArgs struct {
	Query string `json:"query"`
}

// harnessConfig is the agent and runner a test runs: an LLM agent named
// root_agent with tools, a scripted model, and the runner's resumability.
type harnessConfig struct {
	tools     []tool.Tool
	responses []*genai.Content
	resumable bool
	before    []llmagent.BeforeToolCallback
	after     []llmagent.AfterToolCallback
	onError   []llmagent.OnToolErrorCallback
}

// harness is a runner over an in-memory session and artifact service whose
// root agent answers from a scripted testutil.MockModel.
type harness struct {
	t        *testing.T
	model    *testutil.MockModel
	runner   *runner.Runner
	sessions session.Service
}

func newHarness(t *testing.T, cfg harnessConfig) *harness {
	t.Helper()
	m := &testutil.MockModel{Responses: cfg.responses}
	a, err := llmagent.New(llmagent.Config{
		Name:                 "root_agent",
		Model:                m,
		Tools:                cfg.tools,
		BeforeToolCallbacks:  cfg.before,
		AfterToolCallbacks:   cfg.after,
		OnToolErrorCallbacks: cfg.onError,
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: appName, Agent: a, SessionService: sessions, ArtifactService: artifact.InMemoryService(), Resumable: cfg.resumable})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if _, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	return &harness{t: t, model: m, runner: r, sessions: sessions}
}

// run sends msg and returns the events the run yielded, up to its first
// error, and that error.
func (h *harness) run(msg *genai.Content, opts ...runner.RunOption) ([]*session.Event, error) {
	h.t.Helper()
	return h.runCtx(h.t.Context(), msg, opts...)
}

func (h *harness) runCtx(ctx context.Context, msg *genai.Content, opts ...runner.RunOption) ([]*session.Event, error) {
	h.t.Helper()
	var events []*session.Event
	for ev, err := range h.runner.Run(ctx, userID, sessionID, msg, agent.RunConfig{}, opts...) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

// mustRun is run for a turn that must end without an error.
func (h *harness) mustRun(msg *genai.Content, opts ...runner.RunOption) []*session.Event {
	h.t.Helper()
	events, err := h.run(msg, opts...)
	if err != nil {
		h.t.Fatalf("Run() error = %v", err)
	}
	return events
}

// session returns the stored session.
func (h *harness) session() session.Session {
	h.t.Helper()
	resp, err := h.sessions.Get(h.t.Context(), &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		h.t.Fatalf("session Get() error = %v", err)
	}
	return resp.Session
}

// events returns the stored events after the first user message.
func (h *harness) events() []*session.Event {
	h.t.Helper()
	var out []*session.Event
	for ev := range h.session().Events().All() {
		out = append(out, ev)
	}
	if len(out) > 0 && out[0].Author == "user" {
		out = out[1:]
	}
	return out
}

func (h *harness) state(key string) any {
	h.t.Helper()
	v, err := h.session().State().Get(key)
	if err != nil {
		h.t.Fatalf("State().Get(%q) error = %v", key, err)
	}
	return v
}

func (h *harness) modelRequests() int { return len(h.model.Requests) }

func userText(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleUser) }

func modelText(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleModel) }

// calls is a model turn that calls tools.
func calls(fcs ...*genai.FunctionCall) *genai.Content {
	parts := make([]*genai.Part, len(fcs))
	for i, fc := range fcs {
		parts[i] = &genai.Part{FunctionCall: fc}
	}
	return &genai.Content{Role: genai.RoleModel, Parts: parts}
}

func call(id, name string, args map[string]any) *genai.FunctionCall {
	return &genai.FunctionCall{ID: id, Name: name, Args: args}
}

// userResponse is a user turn that answers the function call id.
func userResponse(id, name string, response map[string]any) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{ID: id, Name: name, Response: response},
	}}}
}

// responses returns the function responses named name in events, in order.
func responses(events []*session.Event, name string) []*genai.FunctionResponse {
	var out []*genai.FunctionResponse
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name == name {
				out = append(out, p.FunctionResponse)
			}
		}
	}
	return out
}

// functionCalls returns the function calls named name in events.
func functionCalls(events []*session.Event, name string) []*genai.FunctionCall {
	var out []*genai.FunctionCall
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil && p.FunctionCall.Name == name {
				out = append(out, p.FunctionCall)
			}
		}
	}
	return out
}

// authoredBy returns the events whose author is author.
func authoredBy(events []*session.Event, author string) []*session.Event {
	var out []*session.Event
	for _, ev := range events {
		if ev.Author == author {
			out = append(out, ev)
		}
	}
	return out
}

// textOf is the concatenated text of ev's parts.
func textOf(ev *session.Event) string {
	if ev.Content == nil {
		return ""
	}
	var s string
	for _, p := range ev.Content.Parts {
		s += p.Text
	}
	return s
}

// pathOf is ev's node path, or "" without node info.
func pathOf(ev *session.Event) string {
	if ev.NodeInfo == nil {
		return ""
	}
	return ev.NodeInfo.Path
}

// message is an event whose content is one text part.
func message(text string) *session.Event {
	return &session.Event{LLMResponse: model.LLMResponse{Content: modelText(text)}}
}

func mustNew[TArgs, TResult any](t *testing.T, cfg functiontool.Config, h generatortool.Func[TArgs, TResult]) tool.Tool {
	t.Helper()
	tl, err := generatortool.New(cfg, h)
	if err != nil {
		t.Fatalf("generatortool.New() error = %v", err)
	}
	return tl
}

// evSummary is the part of an event the tests compare with literals.
type evSummary struct {
	Author, Branch, Path string
	Call                 string // name of the first function call
	Text                 string
	Output               any
	Response             map[string]any // the first function response
	NoContent            bool
}

func summarize(events []*session.Event) []evSummary {
	if len(events) == 0 {
		return nil
	}
	out := make([]evSummary, len(events))
	for i, ev := range events {
		s := evSummary{Author: ev.Author, Branch: ev.Branch, Path: pathOf(ev), Text: textOf(ev), Output: ev.Output, NoContent: ev.Content == nil}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.FunctionCall != nil && s.Call == "" {
					s.Call = p.FunctionCall.Name
				}
				if p.FunctionResponse != nil && s.Response == nil {
					s.Response = p.FunctionResponse.Response
				}
			}
		}
		out[i] = s
	}
	return out
}

// setState writes key in ctx's state and reports a failure on t.
func setState(t *testing.T, ctx agent.Context, key string, value any) {
	if err := ctx.State().Set(key, value); err != nil {
		t.Errorf("State().Set(%q) error = %v", key, err)
	}
}
