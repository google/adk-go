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

package runner

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"slices"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/workflow"
)

const cmApp, cmUser = "cm_app", "cm_user"

// cmCustomAgent returns a custom (non-LLM) root agent named "custom" that
// yields the events newEvents builds. Runner.Run drives it through its agent
// loop.
func cmCustomAgent(t *testing.T, newEvents func(ctx agent.InvocationContext) []*session.Event) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "custom",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for _, ev := range newEvents(ctx) {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return a
}

// cmEvent builds a model text event authored by the "custom" agent.
func cmEvent(ctx agent.InvocationContext, text string, metadata map[string]any) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = "custom"
	ev.Content = genai.NewContentFromText(text, genai.RoleModel)
	ev.CustomMetadata = metadata
	return ev
}

// cmSingleEventAgent returns a custom root agent whose only event is text with
// a copy of metadata.
func cmSingleEventAgent(t *testing.T, text string, metadata map[string]any) agent.Agent {
	t.Helper()
	return cmCustomAgent(t, func(ctx agent.InvocationContext) []*session.Event {
		return []*session.Event{cmEvent(ctx, text, maps.Clone(metadata))}
	})
}

// cmModel answers every request with text carrying a copy of metadata.
type cmModel struct {
	text     string
	metadata map[string]any
}

func (m *cmModel) Name() string { return "cm" }

func (m *cmModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:        genai.NewContentFromText(m.text, genai.RoleModel),
			CustomMetadata: maps.Clone(m.metadata),
		}, nil)
	}
}

// cmLLMAgent returns a chat LlmAgent root whose model answers text with
// metadata. Runner.Run drives it through runNode's node loop.
func cmLLMAgent(t *testing.T, text string, metadata map[string]any) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &cmModel{text: text, metadata: metadata}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

// cmRootKinds lists one root agent per Runner.Run event loop. Each root's only
// answer is text carrying metadata.
var cmRootKinds = []struct {
	name    string
	newRoot func(t *testing.T, text string, metadata map[string]any) agent.Agent
}{
	{name: "agent loop", newRoot: cmSingleEventAgent},
	{name: "node loop", newRoot: cmLLMAgent},
}

// cmLiveAgent returns a live root agent whose RunLive stream yields the events
// newEvents builds.
func cmLiveAgent(t *testing.T, newEvents func(ctx agent.InvocationContext) []*session.Event) agent.Agent {
	t.Helper()
	return &mockLiveAgent{
		Agent: must(agent.New(agent.Config{Name: "custom"})),
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
				for _, ev := range newEvents(ctx) {
					if !yield(ev, nil) {
						return
					}
				}
			}, nil
		},
	}
}

func cmNewRunner(t *testing.T, a agent.Agent, plugins ...*plugin.Plugin) (*Runner, session.Service) {
	t.Helper()
	svc := session.InMemoryService()
	r, err := New(Config{
		AppName:           cmApp,
		Agent:             a,
		SessionService:    svc,
		AutoCreateSession: true,
		PluginConfig:      PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r, svc
}

func cmOnEventPlugin(t *testing.T, onEvent plugin.OnEventCallback) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{Name: "on_event", OnEventCallback: onEvent})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// cmCollect drains a run, failing the test on any error.
func cmCollect(t *testing.T, seq iter.Seq2[*session.Event, error]) []*session.Event {
	t.Helper()
	var out []*session.Event
	for ev, err := range seq {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		out = append(out, ev)
	}
	return out
}

func cmStored(t *testing.T, svc session.Service, sessionID string) []*session.Event {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: cmApp, UserID: cmUser, SessionID: sessionID})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return slices.Collect(resp.Session.Events().All())
}

// cmByText returns the first event whose content text is text.
func cmByText(t *testing.T, events []*session.Event, text string) *session.Event {
	t.Helper()
	for _, ev := range events {
		if textOfContent(ev.Content) == text {
			return ev
		}
	}
	t.Fatalf("no event with text %q among %d events", text, len(events))
	return nil
}

func cmCheck(t *testing.T, what string, got, want map[string]any) {
	t.Helper()
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("%s: CustomMetadata mismatch (-want +got):\n%s", what, diff)
	}
}

func cmCheckNil(t *testing.T, what string, got map[string]any) {
	t.Helper()
	if got != nil {
		t.Errorf("%s: CustomMetadata = %v, want nil", what, got)
	}
}

// cmSetInPlace writes into the event's own map, the way a plugin edits an
// event in place, and reports a missing map instead of panicking on it.
func cmSetInPlace(t *testing.T, ev *session.Event, key string, value any) {
	t.Helper()
	if ev.CustomMetadata == nil {
		t.Errorf("event %s reached OnEventCallback with nil CustomMetadata", ev.ID)
		return
	}
	ev.CustomMetadata[key] = value
}

func cmUserText() *genai.Content { return genai.NewContentFromText("question", genai.RoleUser) }

func TestRunner_RunConfigCustomMetadata_OutputEvents(t *testing.T) {
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{"req_id": "r-1"}}

	t.Run("agent loop", func(t *testing.T) {
		a := cmCustomAgent(t, func(ctx agent.InvocationContext) []*session.Event {
			partial := cmEvent(ctx, "an", map[string]any{"chunk": 1})
			partial.Partial = true
			final := cmEvent(ctx, "answer", map[string]any{"done": true})
			return []*session.Event{partial, final}
		})
		r, svc := cmNewRunner(t, a)

		got := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg))

		if len(got) != 2 {
			t.Fatalf("yielded %d events, want 2", len(got))
		}
		cmCheck(t, "yielded partial event", got[0].CustomMetadata, map[string]any{"req_id": "r-1", "chunk": 1})
		cmCheck(t, "yielded final event", got[1].CustomMetadata, map[string]any{"req_id": "r-1", "done": true})
		stored := cmStored(t, svc, "s")
		if len(stored) != 2 {
			t.Fatalf("stored %d events, want 2 (user and final; the partial event is not persisted)", len(stored))
		}
		cmCheck(t, "stored user event", stored[0].CustomMetadata, map[string]any{"req_id": "r-1"})
		cmCheck(t, "stored final event", stored[1].CustomMetadata, map[string]any{"req_id": "r-1", "done": true})
	})

	t.Run("node loop", func(t *testing.T) {
		r, svc := cmNewRunner(t, cmLLMAgent(t, "hi", nil))

		got := cmByText(t, cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg)), "hi")
		stored := cmByText(t, cmStored(t, svc, "s"), "hi")

		cmCheck(t, "yielded model event", got.CustomMetadata, map[string]any{"req_id": "r-1"})
		cmCheck(t, "stored model event", stored.CustomMetadata, map[string]any{"req_id": "r-1"})
		if stored.ID != got.ID {
			t.Fatalf("stored event ID = %q, want %q", stored.ID, got.ID)
		}
	})
}

func TestRunner_RunLive_RunConfigCustomMetadata_OutputEvents(t *testing.T) {
	transcription := func(ctx agent.InvocationContext, id string, partial bool) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.ID, ev.Author, ev.Partial = id, "custom", partial
		ev.OutputTranscription = &genai.Transcription{Text: id}
		return ev
	}
	toolCall := func(ctx agent.InvocationContext, id, tool string) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.ID, ev.Author = id, "custom"
		ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: tool}}}}
		ev.CustomMetadata = map[string]any{"tool": tool}
		return ev
	}
	a := cmLiveAgent(t, func(ctx agent.InvocationContext) []*session.Event {
		text := cmEvent(ctx, "d", map[string]any{"turn": "m1"})
		text.ID = "d"
		return []*session.Event{
			transcription(ctx, "a", true),
			toolCall(ctx, "b", "t1"), // buffered until "c" ends the transcription
			transcription(ctx, "c", false),
			text,
			transcription(ctx, "e", true),
			toolCall(ctx, "f", "t2"), // buffered until the stream ends
		}
	})
	r, svc := cmNewRunner(t, a)
	want := map[string]map[string]any{
		"a": {"live_id": "L-1"},
		"b": {"live_id": "L-1", "tool": "t1"},
		"c": {"live_id": "L-1"},
		"d": {"live_id": "L-1", "turn": "m1"},
		"e": {"live_id": "L-1"},
		"f": {"live_id": "L-1", "tool": "t2"},
	}

	_, seq, err := r.RunLive(t.Context(), cmUser, "s", agent.LiveRunConfig{CustomMetadata: map[string]any{"live_id": "L-1"}})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	got := cmCollect(t, seq)

	var gotIDs []string
	for _, ev := range got {
		gotIDs = append(gotIDs, ev.ID)
		cmCheck(t, "yielded event "+ev.ID, ev.CustomMetadata, want[ev.ID])
	}
	if diff := cmp.Diff([]string{"a", "c", "b", "d", "e", "f"}, gotIDs); diff != "" {
		t.Errorf("yielded event IDs mismatch (-want +got):\n%s", diff)
	}
	var storedIDs []string
	for _, ev := range cmStored(t, svc, "s") {
		storedIDs = append(storedIDs, ev.ID)
		cmCheck(t, "stored event "+ev.ID, ev.CustomMetadata, want[ev.ID])
	}
	if diff := cmp.Diff([]string{"c", "b", "d", "f"}, storedIDs); diff != "" {
		t.Errorf("stored event IDs mismatch (-want +got):\n%s", diff)
	}
}

func TestRunner_RunConfigCustomMetadata_EventPrecedence(t *testing.T) {
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{"shared": "from-run", "run_only": "r-val"}}
	eventMetadata := map[string]any{"shared": "from-event", "event_only": "e-val"}
	want := map[string]any{"shared": "from-event", "run_only": "r-val", "event_only": "e-val"}

	for _, kind := range cmRootKinds {
		t.Run(kind.name, func(t *testing.T) {
			r, svc := cmNewRunner(t, kind.newRoot(t, "answer", eventMetadata))

			got := cmByText(t, cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg)), "answer")

			cmCheck(t, "yielded event", got.CustomMetadata, want)
			stored := cmStored(t, svc, "s")
			cmCheck(t, "stored user event", stored[0].CustomMetadata, map[string]any{"shared": "from-run", "run_only": "r-val"})
			cmCheck(t, "stored event", cmByText(t, stored, "answer").CustomMetadata, want)
		})
	}
}

func TestRunner_RunConfigCustomMetadata_YieldUserMessageOnNodePath(t *testing.T) {
	r, svc := cmNewRunner(t, cmLLMAgent(t, "hi", nil))
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{"turn_id": "t-42", "__adk_internal_secret": "drop"}}

	got := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg, WithYieldUserMessage()))

	if got[0].Author != "user" {
		t.Fatalf("yielded event 0 author = %q, want user", got[0].Author)
	}
	cmCheck(t, "yielded user event", got[0].CustomMetadata, map[string]any{"turn_id": "t-42"})
	stored := cmStored(t, svc, "s")
	if stored[0].Author != "user" {
		t.Fatalf("stored event 0 author = %q, want user", stored[0].Author)
	}
	cmCheck(t, "stored user event", stored[0].CustomMetadata, map[string]any{"turn_id": "t-42"})
}

// cmSilentNode is a workflow node that finishes without output, so the
// workflow records its completion in a completion event's own metadata.
type cmSilentNode struct {
	workflow.BaseNode
}

func (n *cmSilentNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(func(*session.Event, error) bool) {}
}

func TestRunner_RunConfigCustomMetadata_DropsGoInternalKeys(t *testing.T) {
	node := &cmSilentNode{BaseNode: workflow.NewBaseNode("silent", "", workflow.NodeConfig{})}
	wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, node)})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	r, svc := cmNewRunner(t, wf)
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{
		"request_id":                      "req-9",
		"a2a:task_id":                     "task-1",
		"a2a:context_id":                  "ctx-1",
		"__adk_internalx":                 "kept",
		"adk:before_run_replyx":           "kept",
		"adk.workflowx":                   "kept",
		"__adk_internal_x":                "drop",
		"adk:before_run_reply":            true,
		"adk.workflow.node_completed":     false,
		"adk.workflow.node_outcome":       "failed",
		"adk.workflow.delegated_output":   "spoof",
		"adk.workflow.failed_child_paths": []any{"bad"},
		"adk.workflow.base_path.wf":       "spoof@1",
	}}
	callerKeys := map[string]any{
		"request_id":            "req-9",
		"a2a:task_id":           "task-1",
		"a2a:context_id":        "ctx-1",
		"__adk_internalx":       "kept",
		"adk:before_run_replyx": "kept",
		"adk.workflowx":         "kept",
	}

	cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg))

	stored := cmStored(t, svc, "s")
	if len(stored) != 2 {
		t.Fatalf("stored %d events, want 2 (user and node completion)", len(stored))
	}
	cmCheck(t, "stored user event", stored[0].CustomMetadata, callerKeys)
	cmCheck(t, "stored completion event", stored[1].CustomMetadata, map[string]any{
		"request_id":                  "req-9",
		"a2a:task_id":                 "task-1",
		"a2a:context_id":              "ctx-1",
		"__adk_internalx":             "kept",
		"adk:before_run_replyx":       "kept",
		"adk.workflowx":               "kept",
		"adk.workflow.node_completed": true,
		"adk.workflow.base_path.wf":   "wf@1", // the workflow's own value, not the caller's "spoof@1"
	})
}

func TestRunner_RunConfigCustomMetadata_InternalOnlyIsNoOp(t *testing.T) {
	tests := []struct {
		name     string
		metadata *map[string]any
	}{
		{name: "nil"},
		{name: "empty", metadata: &map[string]any{}},
		{name: "internal only", metadata: &map[string]any{"__adk_internal_a": 1, "adk:before_run_reply": true, "adk.workflow.node_completed": true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := cmCustomAgent(t, func(ctx agent.InvocationContext) []*session.Event {
				return []*session.Event{
					cmEvent(ctx, "e0", nil),
					cmEvent(ctx, "e1", map[string]any{"own": "val"}),
				}
			})
			r, svc := cmNewRunner(t, a)

			got := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{CustomMetadata: tt.metadata}))

			cmCheckNil(t, "yielded event 0", cmByText(t, got, "e0").CustomMetadata)
			cmCheck(t, "yielded event 1", cmByText(t, got, "e1").CustomMetadata, map[string]any{"own": "val"})
			stored := cmStored(t, svc, "s")
			cmCheckNil(t, "stored user event", stored[0].CustomMetadata)
			cmCheckNil(t, "stored event 0", cmByText(t, stored, "e0").CustomMetadata)
			cmCheck(t, "stored event 1", cmByText(t, stored, "e1").CustomMetadata, map[string]any{"own": "val"})
		})
	}
}

func TestRunner_RunConfigCustomMetadata_PluginReplacement(t *testing.T) {
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{"run_k": "from_run", "shared_k": "from_run", "secret": "x"}}
	eventMetadata := map[string]any{"agent_k": "from_agent", "shared_k": "from_agent"}
	editInPlace := func(t *testing.T, ev *session.Event) {
		cmSetInPlace(t, ev, "run_k", "from_plugin_inplace")
		cmSetInPlace(t, ev, "plugin_k", "p1")
		delete(ev.CustomMetadata, "secret")
	}
	tests := []struct {
		name string
		// onEvent is the plugin's answer for the agent's event. repl is a map
		// the subtest owns, for a replacement to carry.
		onEvent func(t *testing.T, ev *session.Event, repl map[string]any) *session.Event
		want    map[string]any
	}{
		{
			name: "S1 nil return keeps in-place edits",
			onEvent: func(t *testing.T, ev *session.Event, _ map[string]any) *session.Event {
				editInPlace(t, ev)
				return nil
			},
			want: map[string]any{"run_k": "from_plugin_inplace", "agent_k": "from_agent", "shared_k": "from_agent", "plugin_k": "p1"},
		},
		{
			name: "S2 same event is merged again",
			onEvent: func(t *testing.T, ev *session.Event, _ map[string]any) *session.Event {
				editInPlace(t, ev)
				return ev
			},
			want: map[string]any{"run_k": "from_plugin_inplace", "secret": "x", "agent_k": "from_agent", "shared_k": "from_agent", "plugin_k": "p1"},
		},
		{
			name: "S3 replacement with its own metadata",
			onEvent: func(_ *testing.T, ev *session.Event, repl map[string]any) *session.Event {
				return &session.Event{Author: ev.Author, LLMResponse: model.LLMResponse{Content: ev.Content, CustomMetadata: repl}}
			},
			want: map[string]any{"run_k": "from_run", "secret": "x", "shared_k": "from_replacement", "repl_k": "r1"},
		},
		{
			name: "S4 replacement without metadata",
			onEvent: func(_ *testing.T, ev *session.Event, _ map[string]any) *session.Event {
				return &session.Event{Author: ev.Author, LLMResponse: model.LLMResponse{Content: ev.Content}}
			},
			want: map[string]any{"run_k": "from_run", "secret": "x", "agent_k": "from_agent", "shared_k": "from_agent"},
		},
		{
			name: "S5 replacement with empty map",
			onEvent: func(_ *testing.T, ev *session.Event, _ map[string]any) *session.Event {
				return &session.Event{Author: ev.Author, LLMResponse: model.LLMResponse{Content: ev.Content, CustomMetadata: map[string]any{}}}
			},
			want: map[string]any{"run_k": "from_run", "shared_k": "from_run", "secret": "x"},
		},
	}
	for _, kind := range cmRootKinds {
		for _, tt := range tests {
			t.Run(kind.name+"/"+tt.name, func(t *testing.T) {
				repl := map[string]any{"shared_k": "from_replacement", "repl_k": "r1"}
				var atEntry map[string]any
				p := cmOnEventPlugin(t, func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if textOfContent(ev.Content) != "answer" {
						return nil, nil
					}
					atEntry = maps.Clone(ev.CustomMetadata)
					return tt.onEvent(t, ev, repl), nil
				})
				r, svc := cmNewRunner(t, kind.newRoot(t, "answer", eventMetadata), p)

				got := cmByText(t, cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg)), "answer")

				cmCheck(t, "event at OnEventCallback entry", atEntry, map[string]any{"run_k": "from_run", "secret": "x", "agent_k": "from_agent", "shared_k": "from_agent"})
				cmCheck(t, "yielded event", got.CustomMetadata, tt.want)
				cmCheck(t, "stored event", cmByText(t, cmStored(t, svc, "s"), "answer").CustomMetadata, tt.want)
				cmCheck(t, "plugin's replacement map after the run", repl, map[string]any{"shared_k": "from_replacement", "repl_k": "r1"})
			})
		}
	}
}

func TestRunner_RunConfigCustomMetadata_BeforeRunReply(t *testing.T) {
	runMetadata := map[string]any{"trace": "t-1", "shared": "from_run", "adk:before_run_reply": false}
	tests := []struct {
		name    string
		onEvent func(t *testing.T, ev *session.Event) *session.Event
		want    map[string]any
	}{
		{
			name: "A nil return",
			onEvent: func(t *testing.T, ev *session.Event) *session.Event {
				cmSetInPlace(t, ev, "plugin_seen", true)
				return nil
			},
			want: map[string]any{"trace": "t-1", "shared": "from_run", "plugin_seen": true, "adk:before_run_reply": true},
		},
		{
			name: "B replacement",
			onEvent: func(_ *testing.T, ev *session.Event) *session.Event {
				return &session.Event{Author: ev.Author, LLMResponse: model.LLMResponse{
					Content:        ev.Content,
					CustomMetadata: map[string]any{"shared": "from_replacement", "adk:before_run_reply": false},
				}}
			},
			want: map[string]any{"trace": "t-1", "shared": "from_replacement", "adk:before_run_reply": true},
		},
	}
	// Each path runs the plugin against its root and returns the yielded
	// reply and the stored events. wantUser says whether a user event is
	// stored ahead of the reply.
	paths := []struct {
		name     string
		wantUser bool
		run      func(t *testing.T, p *plugin.Plugin) (*session.Event, []*session.Event)
	}{
		{name: "Run agent path", wantUser: true, run: func(t *testing.T, p *plugin.Plugin) (*session.Event, []*session.Event) {
			r, svc := cmNewRunner(t, cmSingleEventAgent(t, "unreached", nil), p)
			got := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{CustomMetadata: &runMetadata}))
			return got[0], cmStored(t, svc, "s")
		}},
		{name: "Run node path", wantUser: true, run: func(t *testing.T, p *plugin.Plugin) (*session.Event, []*session.Event) {
			r, svc := cmNewRunner(t, cmLLMAgent(t, "unreached", nil), p)
			got := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{CustomMetadata: &runMetadata}))
			return got[0], cmStored(t, svc, "s")
		}},
		{name: "RunLive", run: func(t *testing.T, p *plugin.Plugin) (*session.Event, []*session.Event) {
			r, svc := cmNewRunner(t, cmLiveAgent(t, func(agent.InvocationContext) []*session.Event { return nil }), p)
			_, seq, err := r.RunLive(t.Context(), cmUser, "s", agent.LiveRunConfig{CustomMetadata: runMetadata})
			if err != nil {
				t.Fatalf("RunLive() error = %v", err)
			}
			got := cmCollect(t, seq)
			return got[0], cmStored(t, svc, "s")
		}},
	}
	for _, path := range paths {
		for _, tt := range tests {
			t.Run(path.name+"/"+tt.name, func(t *testing.T) {
				var atEntry map[string]any
				p, err := plugin.New(plugin.Config{
					Name: "short_circuit",
					BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
						return genai.NewContentFromText("short-circuit", genai.RoleModel), nil
					},
					OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
						atEntry = maps.Clone(ev.CustomMetadata)
						return tt.onEvent(t, ev), nil
					},
				})
				if err != nil {
					t.Fatalf("plugin.New() error = %v", err)
				}

				reply, stored := path.run(t, p)

				cmCheck(t, "reply at OnEventCallback entry", atEntry, map[string]any{"trace": "t-1", "shared": "from_run"})
				cmCheck(t, "yielded reply", reply.CustomMetadata, tt.want)
				cmCheck(t, "stored reply", cmByText(t, stored, "short-circuit").CustomMetadata, tt.want)
				if path.wantUser {
					cmCheck(t, "stored user event", stored[0].CustomMetadata, map[string]any{"trace": "t-1", "shared": "from_run"})
				}
			})
		}
	}
}

func TestRunner_RunConfigCustomMetadata_CallerMapUntouched(t *testing.T) {
	t.Run("Run snapshot", func(t *testing.T) {
		a := cmCustomAgent(t, func(ctx agent.InvocationContext) []*session.Event {
			return []*session.Event{cmEvent(ctx, "e0", nil), cmEvent(ctx, "e1", nil)}
		})
		p := cmOnEventPlugin(t, func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if textOfContent(ev.Content) == "e0" {
				cmSetInPlace(t, ev, "only_ev0", true)
			}
			return nil, nil
		})
		r, svc := cmNewRunner(t, a, p)
		callerMap := map[string]any{"k": "v1", "__adk_internal_drop": "keep_in_caller"}

		seq := r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{CustomMetadata: &callerMap})
		callerMap["k"] = "mutated_after_call"
		callerMap["late"] = "x"
		cmCollect(t, seq)

		stored := cmStored(t, svc, "s")
		cmCheck(t, "stored user event", stored[0].CustomMetadata, map[string]any{"k": "v1"})
		cmCheck(t, "stored event 0", cmByText(t, stored, "e0").CustomMetadata, map[string]any{"k": "v1", "only_ev0": true})
		cmCheck(t, "stored event 1", cmByText(t, stored, "e1").CustomMetadata, map[string]any{"k": "v1"})
		cmCheck(t, "caller's map", callerMap, map[string]any{"k": "mutated_after_call", "__adk_internal_drop": "keep_in_caller", "late": "x"})
	})

	t.Run("RunLive snapshot", func(t *testing.T) {
		a := cmLiveAgent(t, func(ctx agent.InvocationContext) []*session.Event {
			return []*session.Event{cmEvent(ctx, "answer", nil)}
		})
		r, _ := cmNewRunner(t, a)
		liveMap := map[string]any{"k": "v1"}

		_, seq, err := r.RunLive(t.Context(), cmUser, "s", agent.LiveRunConfig{CustomMetadata: liveMap})
		if err != nil {
			t.Fatalf("RunLive() error = %v", err)
		}
		liveMap["k"] = "mutated_after_call"
		got := cmCollect(t, seq)

		cmCheck(t, "yielded event", cmByText(t, got, "answer").CustomMetadata, map[string]any{"k": "v1"})
	})

	t.Run("concurrent runs", func(t *testing.T) {
		r, svc := cmNewRunner(t, cmSingleEventAgent(t, "answer", nil))
		shared := map[string]any{"k": "v1", "__adk_internal_drop": "x"}

		var wg sync.WaitGroup
		for i := range 4 {
			wg.Go(func() {
				for _, err := range r.Run(t.Context(), cmUser, fmt.Sprintf("s%d", i), cmUserText(), agent.RunConfig{CustomMetadata: &shared}) {
					if err != nil {
						t.Errorf("run %d failed: %v", i, err)
					}
				}
			})
		}
		wg.Wait()

		for i := range 4 {
			stored := cmStored(t, svc, fmt.Sprintf("s%d", i))
			if len(stored) != 2 {
				t.Fatalf("session s%d stored %d events, want 2", i, len(stored))
			}
			cmCheck(t, fmt.Sprintf("session s%d user event", i), stored[0].CustomMetadata, map[string]any{"k": "v1"})
			cmCheck(t, fmt.Sprintf("session s%d agent event", i), stored[1].CustomMetadata, map[string]any{"k": "v1"})
		}
		cmCheck(t, "shared caller map", shared, map[string]any{"k": "v1", "__adk_internal_drop": "x"})
	})
}

func TestRunner_RunLive_RunConfigCustomMetadata_OmitsUserContent(t *testing.T) {
	a := cmLiveAgent(t, func(ctx agent.InvocationContext) []*session.Event {
		return []*session.Event{cmEvent(ctx, "answer", nil)}
	})
	r, svc := cmNewRunner(t, a)

	sess, seq, err := r.RunLive(t.Context(), cmUser, "s", agent.LiveRunConfig{CustomMetadata: map[string]any{"live_k": "v"}})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	if err := sess.Send(agent.LiveRequest{Content: genai.NewContentFromText("hi live", genai.RoleUser)}); err != nil {
		t.Fatalf("Send() error = %v", err)
	}
	cmCollect(t, seq)

	stored := cmStored(t, svc, "s")
	if len(stored) != 2 {
		t.Fatalf("stored %d events, want 2 (user content and model event)", len(stored))
	}
	if stored[0].Author != "user" {
		t.Fatalf("stored event 0 author = %q, want user", stored[0].Author)
	}
	cmCheckNil(t, "stored user content event", stored[0].CustomMetadata)
	cmCheck(t, "stored model event", stored[1].CustomMetadata, map[string]any{"live_k": "v"})
}

func TestRunner_RunConfigCustomMetadata_CompactionSummaryUntouched(t *testing.T) {
	svc := session.InMemoryService()
	r, err := New(Config{
		AppName:           cmApp,
		Agent:             cmSingleEventAgent(t, "answer", nil),
		SessionService:    svc,
		AutoCreateSession: true,
		Compaction: &compaction.Config{
			CompactionInterval: 1,
			OverlapSize:        0,
			Summarizer:         &recordingSummarizer{summary: "SUMMARY"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{"run_k": "v"}}

	cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg))
	cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), cfg))

	var summaries, others []*session.Event
	for _, ev := range cmStored(t, svc, "s") {
		if ev.Actions.Compaction != nil {
			summaries = append(summaries, ev)
		} else {
			others = append(others, ev)
		}
	}
	if len(summaries) == 0 {
		t.Fatal("no compaction summary was stored")
	}
	for _, ev := range summaries {
		cmCheckNil(t, "stored compaction summary", ev.CustomMetadata)
	}
	if len(others) != 4 {
		t.Fatalf("stored %d user and agent events, want 4", len(others))
	}
	for _, ev := range others {
		cmCheck(t, "stored "+ev.Author+" event", ev.CustomMetadata, map[string]any{"run_k": "v"})
	}
}
