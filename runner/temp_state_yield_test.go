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
	"maps"
	"slices"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// These tests cover what Run and RunLive yield for an event the runner stored:
// the state delta has no "temp:" key, as the event Get returns, while the
// session value the rest of the invocation reads keeps the value.

const (
	tkApp  = "tk_app"
	tkUser = "tk_user"
	tkSess = "tk_sess"
)

// tkDelta is the state delta every producer in this file puts on its event.
func tkDelta() map[string]any { return map[string]any{"temp:k": "v", "k2": "w"} }

// tkSpy records what the plugin callbacks see: the state delta of every event
// before it is stored, and "temp:k" in the session value once the run ends.
type tkSpy struct {
	mu       sync.Mutex
	onEvent  []map[string]any
	afterRun any
}

// plugin returns the spy's plugin. mutate, when not nil, edits each event
// before the spy looks at it, as an OnEvent plugin that plants a key does.
func (s *tkSpy) plugin(t *testing.T, mutate func(*session.Event)) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "tk_spy",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if mutate != nil {
				mutate(ev)
			}
			s.mu.Lock()
			defer s.mu.Unlock()
			s.onEvent = append(s.onEvent, maps.Clone(ev.Actions.StateDelta))
			return nil, nil
		},
		AfterRunCallback: func(ic agent.InvocationContext) {
			v, _ := ic.Session().State().Get("temp:k")
			s.mu.Lock()
			defer s.mu.Unlock()
			s.afterRun = v
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// tkPlant is the mutate function of an OnEvent plugin that puts tkDelta
// on an event of its choosing.
func tkPlant(match func(*session.Event) bool) func(*session.Event) {
	return func(ev *session.Event) {
		if match(ev) {
			maps.Copy(ev.Actions.StateDelta, tkDelta())
		}
	}
}

func tkNew(t *testing.T, cfg runner.Config, plugins ...*plugin.Plugin) (*runner.Runner, session.Service) {
	t.Helper()
	svc := session.InMemoryService()
	cfg.AppName = tkApp
	cfg.SessionService = svc
	cfg.AutoCreateSession = true
	cfg.PluginConfig = runner.PluginConfig{Plugins: plugins}
	r, err := runner.New(cfg)
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r, svc
}

func tkRun(t *testing.T, r *runner.Runner, msg *genai.Content, opts ...runner.RunOption) []*session.Event {
	t.Helper()
	var got []*session.Event
	for ev, err := range r.Run(t.Context(), tkUser, tkSess, msg, agent.RunConfig{}, opts...) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		got = append(got, ev)
	}
	return got
}

func tkStored(t *testing.T, svc session.Service) []*session.Event {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: tkApp, UserID: tkUser, SessionID: tkSess})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return slices.Collect(resp.Session.Events().All())
}

// tkDeltas returns the state delta of each event. An empty delta is {}.
func tkDeltas(evs []*session.Event) []map[string]any {
	out := make([]map[string]any, 0, len(evs))
	for _, ev := range evs {
		out = append(out, ev.Actions.StateDelta)
	}
	return out
}

// tkAuthored returns the events written by author.
func tkAuthored(evs []*session.Event, author string) []*session.Event {
	var out []*session.Event
	for _, ev := range evs {
		if ev.Author == author {
			out = append(out, ev)
		}
	}
	return out
}

var tkEqual = cmpopts.EquateEmpty()

func tkText(s string) *genai.Content { return genai.NewContentFromText(s, genai.RoleModel) }

// tkAgentRoot is a custom agent that yields one event with the given state
// delta and, after the yield returns, records what its context's session holds
// for "temp:k".
type tkAgentRoot struct {
	agent.Agent
	held      *session.Event
	readAfter any
}

func tkNewAgentRoot(t *testing.T, delta map[string]any, partial bool) *tkAgentRoot {
	t.Helper()
	root := &tkAgentRoot{}
	a, err := agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "root"
			ev.Content = tkText("answer")
			ev.Partial = partial
			ev.Actions.StateDelta = delta
			root.held = ev
			if !yield(ev, nil) {
				return
			}
			root.readAfter, _ = ctx.Session().State().Get("temp:k")
		}
	}})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	root.Agent = a
	return root
}

// tkNode is a node root that yields one event and then reads it, as a
// producer that keeps using its event after the yield does.
type tkNode struct {
	workflow.BaseNode
	delta     map[string]any
	partial   bool
	held      *session.Event
	readAfter any
}

func tkNewNode(delta map[string]any, partial bool) *tkNode {
	return &tkNode{BaseNode: workflow.NewBaseNode("producer", "", workflow.NodeConfig{}), delta: delta, partial: partial}
}

func (n *tkNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = n.Name()
		ev.Content = tkText("chunk")
		ev.Partial = n.partial
		ev.NodeInfo = &session.NodeInfo{Path: n.Name()}
		ev.Actions.StateDelta = n.delta
		n.held = ev
		if !yield(ev, nil) {
			return
		}
		n.readAfter, _ = ctx.State().Get("temp:k")
	}
}

func TestRun_ClassicRootYieldsStoredEventWithoutTempKeys(t *testing.T) {
	root := tkNewAgentRoot(t, tkDelta(), false)
	spy := &tkSpy{}
	r, svc := tkNew(t, runner.Config{Agent: root}, spy.plugin(t, nil))

	got := tkRun(t, r, userText("hi"))

	want := []map[string]any{{"k2": "w"}}
	if diff := cmp.Diff(want, tkDeltas(got), tkEqual); diff != "" {
		t.Errorf("yielded state deltas (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, tkDeltas(tkAuthored(tkStored(t, svc), "root")), tkEqual); diff != "" {
		t.Errorf("stored state deltas (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]map[string]any{{"temp:k": "v", "k2": "w"}}, spy.onEvent); diff != "" {
		t.Errorf("state deltas the OnEvent plugin saw (-want +got):\n%s", diff)
	}
	if root.readAfter != "v" {
		t.Errorf(`agent read temp:k = %v after its yield, want "v"`, root.readAfter)
	}
	if spy.afterRun != "v" {
		t.Errorf(`after-run callback read temp:k = %v, want "v"`, spy.afterRun)
	}
	if diff := cmp.Diff(map[string]any{"temp:k": "v", "k2": "w"}, root.held.Actions.StateDelta); diff != "" {
		t.Errorf("producer's own event after the run (-want +got):\n%s", diff)
	}
}

func TestRun_YieldedDeltaDropsOnlyTempPrefixedKeys(t *testing.T) {
	tests := []struct {
		name  string
		delta map[string]any
		want  map[string]any
	}{
		{
			name:  "only_temp_keys",
			delta: map[string]any{"temp:a": 1},
			want:  map[string]any{},
		},
		{
			name:  "prefix_match_only",
			delta: map[string]any{"app:temp:x": 1, "user:temp:y": 2, "app:a": 3, "user:u": 4, "temp": 5, "Temp:z": 6, "temp:": 7, "temp:q:r": 8, "k": 9, " temp:s": 10, "n": nil},
			want:  map[string]any{"app:temp:x": 1, "user:temp:y": 2, "app:a": 3, "user:u": 4, "temp": 5, "Temp:z": 6, "k": 9, " temp:s": 10, "n": nil},
		},
		{
			name:  "nil_delta",
			delta: nil,
			want:  nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r, _ := tkNew(t, runner.Config{Agent: tkNewAgentRoot(t, tc.delta, false)})

			got := tkRun(t, r, userText("hi"))

			if len(got) != 1 {
				t.Fatalf("Run() yielded %d events, want 1", len(got))
			}
			if diff := cmp.Diff(tc.want, got[0].Actions.StateDelta); diff != "" {
				t.Errorf("yielded state delta (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRun_PartialEventKeepsTempKeys(t *testing.T) {
	tests := []struct {
		name string
		cfg  func(t *testing.T) runner.Config
	}{
		{"classic_root", func(t *testing.T) runner.Config { return runner.Config{Agent: tkNewAgentRoot(t, tkDelta(), true)} }},
		{"node_root", func(*testing.T) runner.Config { return runner.Config{Node: tkNewNode(tkDelta(), true)} }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r, svc := tkNew(t, tc.cfg(t))

			got := tkRun(t, r, userText("hi"))

			if diff := cmp.Diff([]map[string]any{{"temp:k": "v", "k2": "w"}}, tkDeltas(got)); diff != "" {
				t.Errorf("yielded state deltas (-want +got):\n%s", diff)
			}
			if n := len(tkStored(t, svc)); n != 1 {
				t.Errorf("session holds %d events, want 1 (the user message only)", n)
			}
		})
	}
}

// tkTool is a function tool that sets "temp:tool" and "k2" through its
// context's state.
func tkTool(t *testing.T) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: "writer", Description: "Writes state."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			if err := ctx.State().Set("temp:tool", "tv"); err != nil {
				return nil, err
			}
			if err := ctx.State().Set("k2", "w"); err != nil {
				return nil, err
			}
			return map[string]any{"ok": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return tl
}

// tkModel answers its n-th request with its n-th content.
type tkModel struct {
	mu        sync.Mutex
	responses []*genai.Content
	n         int
}

func (m *tkModel) Name() string { return "tk" }

func (m *tkModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		c := m.responses[min(m.n, len(m.responses)-1)]
		m.n++
		m.mu.Unlock()
		yield(&model.LLMResponse{Content: c}, nil)
	}
}

func TestRun_NodeRootYieldsStoredEventWithoutTempKeys(t *testing.T) {
	t.Run("config_node", func(t *testing.T) {
		node := tkNewNode(tkDelta(), false)
		spy := &tkSpy{}
		r, svc := tkNew(t, runner.Config{Node: node}, spy.plugin(t, nil))

		got := tkRun(t, r, userText("hi"))

		want := []map[string]any{{"k2": "w"}}
		if diff := cmp.Diff(want, tkDeltas(got), tkEqual); diff != "" {
			t.Errorf("yielded state deltas (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(want, tkDeltas(tkAuthored(tkStored(t, svc), "producer")), tkEqual); diff != "" {
			t.Errorf("stored state deltas (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]map[string]any{{"temp:k": "v", "k2": "w"}}, spy.onEvent); diff != "" {
			t.Errorf("state deltas the OnEvent plugin saw (-want +got):\n%s", diff)
		}
		if node.readAfter != "v" || spy.afterRun != "v" {
			t.Errorf(`temp:k read after the yield = %v, in the after-run callback = %v; want "v" for both`, node.readAfter, spy.afterRun)
		}
		if diff := cmp.Diff(map[string]any{"temp:k": "v", "k2": "w"}, node.held.Actions.StateDelta); diff != "" {
			t.Errorf("producer's own event after the run (-want +got):\n%s", diff)
		}
	})

	t.Run("llm_agent_output_key_and_tool_state", func(t *testing.T) {
		llm := &tkModel{responses: []*genai.Content{
			{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "writer", Args: map[string]any{}}}}},
			tkText("done"),
		}}
		a, err := llmagent.New(llmagent.Config{Name: "root", Model: llm, Tools: []tool.Tool{tkTool(t)}, OutputKey: "temp:out"})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		spy := &tkSpy{}
		r, svc := tkNew(t, runner.Config{Agent: a}, spy.plugin(t, nil))

		got := tkRun(t, r, userText("hi"))

		// The call, the tool's response, the final answer.
		want := []map[string]any{{}, {"k2": "w"}, {}}
		if diff := cmp.Diff(want, tkDeltas(got), tkEqual); diff != "" {
			t.Errorf("yielded state deltas (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(want, tkDeltas(tkAuthored(tkStored(t, svc), "root")), tkEqual); diff != "" {
			t.Errorf("stored state deltas (-want +got):\n%s", diff)
		}
		wantSeen := []map[string]any{{}, {"temp:tool": "tv", "k2": "w"}, {"temp:out": "done"}}
		if diff := cmp.Diff(wantSeen, spy.onEvent, tkEqual); diff != "" {
			t.Errorf("state deltas the OnEvent plugin saw (-want +got):\n%s", diff)
		}
	})
}

// tkLiveAgent is a live agent whose session yields the events of stream.
type tkLiveAgent struct {
	agent.Agent
	stream func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error]
}

type tkLiveSession struct{}

func (tkLiveSession) Send(agent.LiveRequest) error { return nil }
func (tkLiveSession) Close() error                 { return nil }

func (a *tkLiveAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return tkLiveSession{}, a.stream(ctx), nil
}

// tkLiveEvent describes one live event of the kinds the table below needs.
// Every event carries tkDelta unless bare is set.
type tkLiveEvent struct {
	partial       bool
	transcription string // "" for none; the event's OutputTranscription text
	call          bool   // holds a function call
	audio         bool   // holds inline audio
	bare          bool   // no state delta
}

func (e tkLiveEvent) build(ctx agent.InvocationContext) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = "live"
	ev.Partial = e.partial
	if !e.bare {
		ev.Actions.StateDelta = tkDelta()
	}
	if e.transcription != "" {
		ev.OutputTranscription = &genai.Transcription{Text: e.transcription}
	}
	var parts []*genai.Part
	if e.call {
		parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}})
	}
	if e.audio {
		parts = append(parts, &genai.Part{InlineData: &genai.Blob{MIMEType: "audio/pcm", Data: []byte{1, 2}}})
	}
	if len(parts) > 0 {
		ev.Content = &genai.Content{Role: genai.RoleModel, Parts: parts}
	}
	return ev
}

func tkNewLive(t *testing.T, stream func(agent.InvocationContext) iter.Seq2[*session.Event, error], plugins ...*plugin.Plugin) (*runner.Runner, session.Service) {
	t.Helper()
	base, err := agent.New(agent.Config{Name: "live"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: tkApp, UserID: tkUser, SessionID: tkSess}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:        tkApp,
		Agent:          &tkLiveAgent{Agent: base, stream: stream},
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r, svc
}

func tkRunLive(t *testing.T, r *runner.Runner) []*session.Event {
	t.Helper()
	_, stream, err := r.RunLive(t.Context(), tkUser, tkSess, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	var got []*session.Event
	for ev, err := range stream {
		if err != nil {
			t.Fatalf("RunLive() stream error = %v", err)
		}
		got = append(got, ev)
	}
	return got
}

func TestRunner_RunLive_YieldedEventsHaveNoTempKeys(t *testing.T) {
	both := map[string]any{"temp:k": "v", "k2": "w"}
	kept := map[string]any{"k2": "w"}
	none := map[string]any{}
	tests := []struct {
		name       string
		events     []tkLiveEvent
		wantYield  []map[string]any
		wantStored []map[string]any
		wantState  any // "temp:k" in the session value after the run
	}{
		{
			name:       "transcription_end",
			events:     []tkLiveEvent{{transcription: "Hello"}},
			wantYield:  []map[string]any{kept},
			wantStored: []map[string]any{kept},
			wantState:  "v",
		},
		{
			// The call is held back until the transcription ends, then stored and yielded.
			name: "buffered_after_transcription",
			events: []tkLiveEvent{
				{partial: true, transcription: "He", bare: true},
				{call: true},
				{transcription: "Hello", bare: true},
			},
			wantYield:  []map[string]any{none, none, kept},
			wantStored: []map[string]any{none, kept},
			wantState:  "v",
		},
		{
			name:       "plain_event",
			events:     []tkLiveEvent{{call: true}},
			wantYield:  []map[string]any{kept},
			wantStored: []map[string]any{kept},
			wantState:  "v",
		},
		{
			// The stream ends while the call is still held back.
			name: "flush_at_end",
			events: []tkLiveEvent{
				{partial: true, transcription: "He", bare: true},
				{call: true},
			},
			wantYield:  []map[string]any{none, kept},
			wantStored: []map[string]any{kept},
			wantState:  "v",
		},
		{
			name:       "partial_keeps_delta",
			events:     []tkLiveEvent{{partial: true, call: true}},
			wantYield:  []map[string]any{both},
			wantStored: []map[string]any{},
		},
		{
			name:       "inline_media_keeps_delta",
			events:     []tkLiveEvent{{audio: true}},
			wantYield:  []map[string]any{both},
			wantStored: []map[string]any{},
		},
		{
			name:       "inline_media_final_transcription_keeps_delta",
			events:     []tkLiveEvent{{transcription: "Hello", audio: true}},
			wantYield:  []map[string]any{both},
			wantStored: []map[string]any{},
		},
		{
			name: "buffered_partial_keeps_delta",
			events: []tkLiveEvent{
				{partial: true, transcription: "He", bare: true},
				{partial: true, call: true},
				{transcription: "Hello", bare: true},
			},
			wantYield:  []map[string]any{none, none, both},
			wantStored: []map[string]any{none},
		},
		{
			name: "buffered_inline_media_at_end_keeps_delta",
			events: []tkLiveEvent{
				{partial: true, transcription: "He", bare: true},
				{call: true, audio: true},
			},
			wantYield:  []map[string]any{none, both},
			wantStored: []map[string]any{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spy := &tkSpy{}
			r, svc := tkNewLive(t, func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, e := range tc.events {
						if !yield(e.build(ctx), nil) {
							return
						}
					}
				}
			}, spy.plugin(t, nil))

			got := tkRunLive(t, r)

			if diff := cmp.Diff(tc.wantYield, tkDeltas(got), tkEqual); diff != "" {
				t.Errorf("yielded state deltas (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantStored, tkDeltas(tkStored(t, svc)), tkEqual); diff != "" {
				t.Errorf("stored state deltas (-want +got):\n%s", diff)
			}
			if tc.wantState != nil && spy.afterRun != tc.wantState {
				t.Errorf("after-run callback read temp:k = %v, want %v", spy.afterRun, tc.wantState)
			}
		})
	}
}

type tkReadingNode struct {
	workflow.BaseNode
	readTempX *any
}

func (n *tkReadingNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		*n.readTempX, _ = ctx.State().Get("temp:x")
	}
}

func TestRun_YieldedUserEventHasNoTempKeys(t *testing.T) {
	var readTempX any
	tests := []struct {
		name string
		cfg  func(t *testing.T) runner.Config
		msg  *genai.Content
	}{
		{
			name: "config_node_with_message",
			cfg: func(*testing.T) runner.Config {
				return runner.Config{Node: &tkReadingNode{BaseNode: workflow.NewBaseNode("reader", "", workflow.NodeConfig{}), readTempX: &readTempX}}
			},
			msg: userText("hi"),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			readTempX = nil
			callerDelta := map[string]any{"temp:x": 1, "k": 2}
			r, svc := tkNew(t, tc.cfg(t))

			got := tkRun(t, r, tc.msg, runner.WithStateDelta(callerDelta), runner.WithYieldUserMessage())

			if len(got) == 0 || got[0].Author != "user" {
				t.Fatalf("first yielded event = %+v, want the user event", got)
			}
			if diff := cmp.Diff(map[string]any{"k": 2}, got[0].Actions.StateDelta); diff != "" {
				t.Errorf("yielded user event state delta (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(map[string]any{"k": 2}, tkStored(t, svc)[0].Actions.StateDelta); diff != "" {
				t.Errorf("stored user event state delta (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(map[string]any{"temp:x": 1, "k": 2}, callerDelta); diff != "" {
				t.Errorf("caller's WithStateDelta map after the run (-want +got):\n%s", diff)
			}
			if readTempX != 1 {
				t.Errorf("root read temp:x = %v during the run, want 1", readTempX)
			}
		})
	}
}

// tkReplyPlugin answers every run with a cached reply from its before-run
// callback.
func tkReplyPlugin(t *testing.T) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name:              "tk_reply",
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) { return tkText("cached"), nil },
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// tkIdleAgent is a custom agent that must never run: the before-run reply
// ends the run first.
func tkIdleAgent(t *testing.T) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: "root", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(func(*session.Event, error) bool) { t.Error("agent ran after a before-run reply") }
	}})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return a
}

func TestRun_BeforeRunReplyYieldsEventWithoutTempKeys(t *testing.T) {
	always := func(*session.Event) bool { return true }
	plant := tkPlant(always)
	plantPartial := func(ev *session.Event) { plant(ev); ev.Partial = true }
	nodeRoot := func(*testing.T) runner.Config { return runner.Config{Node: tkNewNode(nil, false)} }
	tests := []struct {
		name       string
		cfg        func(t *testing.T) runner.Config
		mutate     func(*session.Event)
		wantYield  map[string]any
		wantStored []map[string]any // deltas of the reply and, before it, the user message
		wantState  any
	}{
		{
			name:       "classic_root",
			cfg:        func(t *testing.T) runner.Config { return runner.Config{Agent: tkIdleAgent(t)} },
			mutate:     plant,
			wantYield:  map[string]any{"k2": "w"},
			wantStored: []map[string]any{{}, {"k2": "w"}},
			wantState:  "v",
		},
		{
			name:       "node_root",
			cfg:        nodeRoot,
			mutate:     plant,
			wantYield:  map[string]any{"k2": "w"},
			wantStored: []map[string]any{{}, {"k2": "w"}},
			wantState:  "v",
		},
		{
			name:       "partial_reply_keeps_delta",
			cfg:        func(t *testing.T) runner.Config { return runner.Config{Agent: tkIdleAgent(t)} },
			mutate:     plantPartial,
			wantYield:  map[string]any{"temp:k": "v", "k2": "w"},
			wantStored: []map[string]any{{}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spy := &tkSpy{}
			r, svc := tkNew(t, tc.cfg(t), tkReplyPlugin(t), spy.plugin(t, tc.mutate))

			got := tkRun(t, r, userText("hi"))

			if len(got) != 1 {
				t.Fatalf("Run() yielded %d events, want the reply only", len(got))
			}
			if diff := cmp.Diff(tc.wantYield, got[0].Actions.StateDelta); diff != "" {
				t.Errorf("yielded reply state delta (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantStored, tkDeltas(tkStored(t, svc)), tkEqual); diff != "" {
				t.Errorf("stored state deltas (-want +got):\n%s", diff)
			}
			if tc.wantState != nil && spy.afterRun != tc.wantState {
				t.Errorf("after-run callback read temp:k = %v, want %v", spy.afterRun, tc.wantState)
			}
		})
	}

	t.Run("live", func(t *testing.T) {
		spy := &tkSpy{}
		r, svc := tkNewLive(t, func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) { t.Error("live agent ran after a before-run reply") }
		}, tkReplyPlugin(t), spy.plugin(t, plant))

		got := tkRunLive(t, r)

		if diff := cmp.Diff([]map[string]any{{"k2": "w"}}, tkDeltas(got), tkEqual); diff != "" {
			t.Errorf("yielded state deltas (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]map[string]any{{"k2": "w"}}, tkDeltas(tkStored(t, svc)), tkEqual); diff != "" {
			t.Errorf("stored state deltas (-want +got):\n%s", diff)
		}
		if spy.afterRun != "v" {
			t.Errorf("after-run callback read temp:k = %v, want v", spy.afterRun)
		}
	})
}
