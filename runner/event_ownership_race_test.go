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

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// The race detector is what fails most of these tests without the runner's
// ownership rule for partial events, so run them with -race and a high
// -count.

const ownershipApp = "ownership_app"

// partialOnlyModel answers every request with one partial response and
// nothing else, so the LLM flow's step ends on a partial event, which the
// flow reads again (lastEvent.IsFinalResponse) after yielding it.
type partialOnlyModel struct{}

func (partialOnlyModel) Name() string { return "partial-only" }

func (partialOnlyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("chunk", genai.RoleModel), Partial: true}, nil)
	}
}

// yieldThenReadNode yields one event and then reads every field of it, as a
// producer that keeps using its event after the yield does. It keeps the
// event so the test can inspect what the producer holds once the run is over.
type yieldThenReadNode struct {
	workflow.BaseNode
	partial bool
	held    *session.Event
}

func (n *yieldThenReadNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = n.Name()
		ev.Content = genai.NewContentFromText("chunk", genai.RoleModel)
		ev.Partial = n.partial
		// The node names itself, so the workflow engine has no path to stamp.
		ev.NodeInfo = &session.NodeInfo{Path: n.Name()}
		if !yield(ev, nil) {
			return
		}
		snapshot := *ev
		_ = snapshot.IsFinalResponse()
		n.held = ev
	}
}

// inPlacePlugin edits every event in place and returns nil: it replaces the
// content and plants a compaction record.
func inPlacePlugin(t *testing.T) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "in_place",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			ev.Content = genai.NewContentFromText("edited", genai.RoleModel)
			ev.Actions.Compaction = &session.EventCompaction{
				CompactedContent: genai.NewContentFromText("planted", genai.RoleModel),
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

func newOwnershipRunner(t *testing.T, cfg runner.Config) *runner.Runner {
	t.Helper()
	cfg.AppName = ownershipApp
	cfg.SessionService = session.InMemoryService()
	cfg.AutoCreateSession = true
	r, err := runner.New(cfg)
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r
}

// ownershipEvent is what the caller sees of one event: its text, whether it is
// partial, whether it carries a compaction record, and its run metadata.
type ownershipEvent struct {
	text       string
	partial    bool
	compaction bool
	tag        any
}

func ownershipView(ev *session.Event) ownershipEvent {
	got := ownershipEvent{partial: ev.Partial, compaction: ev.Actions.Compaction != nil, tag: ev.CustomMetadata["tag"]}
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			got.text += p.Text
		}
	}
	return got
}

func runOwnership(t *testing.T, r *runner.Runner) []ownershipEvent {
	t.Helper()
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{"tag": "run"}}
	var got []ownershipEvent
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("hi", genai.RoleUser), cfg) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		got = append(got, ownershipView(ev))
	}
	return got
}

func TestRunner_LlmAgentRootPartialEvent(t *testing.T) {
	tests := []struct {
		name    string
		plugins []*plugin.Plugin
		want    ownershipEvent
	}{
		{
			name: "no plugin",
			want: ownershipEvent{text: "chunk", partial: true, tag: "run"},
		},
		{
			name:    "plugin edits the event in place",
			plugins: []*plugin.Plugin{inPlacePlugin(t)},
			want:    ownershipEvent{text: "edited", partial: true, tag: "run"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := llmagent.New(llmagent.Config{Name: "root", Model: partialOnlyModel{}})
			if err != nil {
				t.Fatalf("llmagent.New() error = %v", err)
			}
			r := newOwnershipRunner(t, runner.Config{Agent: a, PluginConfig: runner.PluginConfig{Plugins: tt.plugins}})

			got := runOwnership(t, r)

			if len(got) != 1 || got[0] != tt.want {
				t.Errorf("yielded events = %+v, want [%+v]", got, tt.want)
			}
		})
	}
}

func TestRunner_NodeRootPartialEventStaysWithProducer(t *testing.T) {
	tests := []struct {
		name    string
		plugins []*plugin.Plugin
		want    ownershipEvent
	}{
		{
			name: "no plugin",
			want: ownershipEvent{text: "chunk", partial: true, tag: "run"},
		},
		{
			name:    "plugin edits the event in place",
			plugins: []*plugin.Plugin{inPlacePlugin(t)},
			want:    ownershipEvent{text: "edited", partial: true, tag: "run"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &yieldThenReadNode{BaseNode: workflow.NewBaseNode("producer", "", workflow.NodeConfig{}), partial: true}
			r := newOwnershipRunner(t, runner.Config{Node: n, PluginConfig: runner.PluginConfig{Plugins: tt.plugins}})

			got := runOwnership(t, r)

			if len(got) != 1 || got[0] != tt.want {
				t.Errorf("yielded events = %+v, want [%+v]", got, tt.want)
			}
			// The runner and its plugins wrote to their own copy, so the
			// producer's event is as the producer left it.
			wantHeld := ownershipEvent{text: "chunk", partial: true}
			if gotHeld := ownershipView(n.held); gotHeld != wantHeld {
				t.Errorf("producer's event after the run = %+v, want %+v", gotHeld, wantHeld)
			}
		})
	}
}

// A non-partial event is handed over: its producer waits until the runner has
// processed it, so edits made in place are on the producer's event too, as
// they are on the classic path and in adk-python, where both sides hold one
// object.
func TestRunner_NodeRootNonPartialEventIsHandedOver(t *testing.T) {
	n := &yieldThenReadNode{BaseNode: workflow.NewBaseNode("producer", "", workflow.NodeConfig{})}
	r := newOwnershipRunner(t, runner.Config{Node: n, PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{inPlacePlugin(t)}}})

	got := runOwnership(t, r)

	want := ownershipEvent{text: "edited", tag: "run"}
	if len(got) == 0 || got[0] != want {
		t.Errorf("first yielded event = %+v, want %+v (all: %+v)", got, want, got)
	}
	wantHeld := ownershipEvent{text: "edited", tag: "run"}
	if gotHeld := ownershipView(n.held); gotHeld != wantHeld {
		t.Errorf("producer's event after the run = %+v, want %+v", gotHeld, wantHeld)
	}
}
