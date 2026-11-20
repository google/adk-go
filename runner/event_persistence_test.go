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
	"iter"
	"maps"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/workflow"
)

var (
	epT0 = time.Date(2026, time.March, 2, 10, 0, 0, 0, time.UTC)
	epT1 = time.Date(2026, time.March, 3, 11, 0, 0, 0, time.UTC)
)

// epNode is a workflow node that yields the events newEvents builds.
type epNode struct {
	workflow.BaseNode
	newEvents func(ctx agent.Context) []*session.Event
}

func (n *epNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for _, ev := range n.newEvents(ctx) {
			if !yield(ev, nil) {
				return
			}
		}
	}
}

// epNodeRunner returns a runner whose root is a bare node (Config.Node) that
// yields the events newEvents builds. Runner.Run drives it through runNode's
// node loop.
func epNodeRunner(t *testing.T, newEvents func(ctx agent.Context) []*session.Event, plugins ...*plugin.Plugin) (*Runner, session.Service) {
	t.Helper()
	svc := session.InMemoryService()
	r, err := New(Config{
		AppName:           cmApp,
		Node:              &epNode{BaseNode: workflow.NewBaseNode("node", "", workflow.NodeConfig{}), newEvents: newEvents},
		SessionService:    svc,
		AutoCreateSession: true,
		PluginConfig:      PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r, svc
}

// epRunLive drains a RunLive stream that is sent no request.
func epRunLive(t *testing.T, r *Runner) []*session.Event {
	t.Helper()
	_, seq, err := r.RunLive(t.Context(), cmUser, "s", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	return cmCollect(t, seq)
}

// epSession returns the session's events other than user events, and its
// state.
func epSession(t *testing.T, svc session.Service) ([]*session.Event, map[string]any) {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: cmApp, UserID: cmUser, SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var events []*session.Event
	for ev := range resp.Session.Events().All() {
		if ev.Author != genai.RoleUser {
			events = append(events, ev)
		}
	}
	return events, maps.Collect(resp.Session.State().All())
}

// epNewEvent builds an event authored "custom" and lets fill set the rest.
func epNewEvent(ctx context.Context, invocationID string, fill func(*session.Event)) *session.Event {
	ev := session.NewEvent(ctx, invocationID)
	ev.Author = "custom"
	fill(ev)
	return ev
}

// epLoop runs a root on one runner loop. The root yields one event, which
// epNewEvent builds with fill. run returns the yielded events, the stored
// events other than user events, and the session state.
type epLoop struct {
	name string
	run  func(t *testing.T, fill func(*session.Event), plugins ...*plugin.Plugin) (yielded, stored []*session.Event, state map[string]any)
}

var (
	epClassicLoop = epLoop{name: "classic loop", run: func(t *testing.T, fill func(*session.Event), plugins ...*plugin.Plugin) ([]*session.Event, []*session.Event, map[string]any) {
		a := cmCustomAgent(t, func(ctx agent.InvocationContext) []*session.Event {
			return []*session.Event{epNewEvent(ctx, ctx.InvocationID(), fill)}
		})
		r, svc := cmNewRunner(t, a, plugins...)
		yielded := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{}))
		stored, state := epSession(t, svc)
		return yielded, stored, state
	}}
	epNodeLoop = epLoop{name: "node loop", run: func(t *testing.T, fill func(*session.Event), plugins ...*plugin.Plugin) ([]*session.Event, []*session.Event, map[string]any) {
		r, svc := epNodeRunner(t, func(ctx agent.Context) []*session.Event {
			return []*session.Event{epNewEvent(ctx, ctx.InvocationID(), fill)}
		}, plugins...)
		yielded := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{}))
		stored, state := epSession(t, svc)
		return yielded, stored, state
	}}
	epLiveLoop = epLoop{name: "live loop", run: func(t *testing.T, fill func(*session.Event), plugins ...*plugin.Plugin) ([]*session.Event, []*session.Event, map[string]any) {
		a := cmLiveAgent(t, func(ctx agent.InvocationContext) []*session.Event {
			return []*session.Event{epNewEvent(ctx, ctx.InvocationID(), fill)}
		})
		r, svc := cmNewRunner(t, a, plugins...)
		yielded := epRunLive(t, r)
		stored, state := epSession(t, svc)
		return yielded, stored, state
	}}
)

// epOnlyEvent returns the only event in events.
func epOnlyEvent(t *testing.T, what string, events []*session.Event) *session.Event {
	t.Helper()
	if len(events) != 1 {
		t.Fatalf("%s: got %d events, want 1", what, len(events))
	}
	return events[0]
}

// epFullEvent returns an event with every field of session.Event,
// model.LLMResponse and session.EventActions set, except Partial,
// PartialSet and Actions.Compaction.
func epFullEvent() *session.Event {
	return &session.Event{
		ID:                 "orig-id",
		InvocationID:       "orig-inv",
		Timestamp:          epT0,
		Author:             "orig_agent",
		Branch:             "b1",
		IsolationScope:     "iso1",
		LongRunningToolIDs: []string{"lrt-1"},
		Routes:             []string{"route-a"},
		RequestedInput:     &session.RequestInput{InterruptID: "int-1", Message: "approve?"},
		Output:             map[string]any{"k": "v"},
		NodeInfo:           &session.NodeInfo{Path: "p1"},
		Actions: session.EventActions{
			StateDelta:                 map[string]any{"sk": "sv"},
			ArtifactDelta:              map[string]int64{"f": 1},
			SkipSummarization:          true,
			TransferToAgent:            "target",
			Escalate:                   true,
			RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{"tc1": {Hint: "h"}},
			RequestedAuthConfigs:       map[string]toolauth.Config{"ac1": {AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"}, CredentialKey: "ck1"}},
			SetModelResponse:           map[string]any{"answer": "orig"},
		},
		LLMResponse: model.LLMResponse{
			Content:                 genai.NewContentFromText("orig", genai.RoleModel),
			CustomMetadata:          map[string]any{"ek": "ev"},
			TurnComplete:            true,
			Interrupted:             true,
			ErrorCode:               "E1",
			ErrorMessage:            "msg1",
			FinishReason:            genai.FinishReasonStop,
			ModelVersion:            "v1",
			SessionResumptionHandle: "h1",
			AvgLogprobs:             -0.5,
			UsageMetadata:           &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 3},
			CitationMetadata:        &genai.CitationMetadata{},
			GroundingMetadata:       &genai.GroundingMetadata{},
			LogprobsResult:          &genai.LogprobsResult{},
			InputTranscription:      &genai.Transcription{Text: "in", Finished: true},
			OutputTranscription:     &genai.Transcription{Text: "out", Finished: true},
		},
	}
}

func TestRunner_OnEventReplacement_MergesFieldsAcrossLoops(t *testing.T) {
	cases := []struct {
		name string
		// replacement builds the plugin's return value, a new struct on each
		// call, so the test can compare the returned struct after the run.
		replacement func() *session.Event
		// original edits epFullEvent into the event the custom agent
		// yields; nil yields epFullEvent as it is.
		original func(*session.Event)
		// want edits epFullEvent into the event the custom-agent row expects.
		want func(*session.Event)
		// wantState is the custom-agent row's session state.
		wantState map[string]any
		// wantReplyAuthor is the before-run reply's author; empty means the
		// root's default author.
		wantReplyAuthor string
	}{
		{
			name: "sparse_content_only",
			replacement: func() *session.Event {
				return &session.Event{ID: "plugin-id", InvocationID: "plugin-inv", Timestamp: epT1, LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromText("replaced", genai.RoleModel),
				}}
			},
			want:      func(ev *session.Event) { ev.Content = genai.NewContentFromText("replaced", genai.RoleModel) },
			wantState: map[string]any{"sk": "sv"},
		},
		{
			name: "author_override",
			replacement: func() *session.Event {
				return &session.Event{Author: "custom_author", LLMResponse: model.LLMResponse{ErrorCode: "E2"}}
			},
			want: func(ev *session.Event) {
				ev.Author = "custom_author"
				ev.ErrorCode = "E2"
			},
			wantState:       map[string]any{"sk": "sv"},
			wantReplyAuthor: "custom_author",
		},
		{
			name: "actions_and_metadata_override",
			replacement: func() *session.Event {
				return &session.Event{
					Actions:     session.EventActions{StateDelta: map[string]any{"sk2": "sv2"}},
					LLMResponse: model.LLMResponse{CustomMetadata: map[string]any{"pk": "pv"}},
				}
			},
			want: func(ev *session.Event) {
				ev.Actions.StateDelta = map[string]any{"sk2": "sv2"}
				ev.CustomMetadata = map[string]any{"pk": "pv"}
			},
			wantState: map[string]any{"sk2": "sv2"},
		},
		{
			name: "set_model_response_override",
			replacement: func() *session.Event {
				return &session.Event{Actions: session.EventActions{SetModelResponse: map[string]any{"answer": 1}}}
			},
			want:      func(ev *session.Event) { ev.Actions.SetModelResponse = map[string]any{"answer": 1} },
			wantState: map[string]any{"sk": "sv"},
		},
		{
			// The original carries neither flag, so only the merge can set
			// them.
			name: "flags_branch_error_output_override",
			original: func(ev *session.Event) {
				ev.Actions.Escalate = false
				ev.TurnComplete = false
			},
			replacement: func() *session.Event {
				return &session.Event{
					Branch:      "b2",
					Output:      map[string]any{"k2": "v2"},
					Actions:     session.EventActions{Escalate: true},
					LLMResponse: model.LLMResponse{TurnComplete: true, ErrorMessage: "msg2"},
				}
			},
			want: func(ev *session.Event) {
				ev.Branch = "b2"
				ev.Output = map[string]any{"k2": "v2"}
				ev.ErrorMessage = "msg2"
			},
			wantState: map[string]any{"sk": "sv"},
		},
	}

	for _, tc := range cases {
		t.Run("custom agent root/"+tc.name, func(t *testing.T) {
			var returned *session.Event
			p := cmOnEventPlugin(t, func(agent.InvocationContext, *session.Event) (*session.Event, error) {
				returned = tc.replacement()
				return returned, nil
			})
			original := epFullEvent()
			if tc.original != nil {
				tc.original(original)
			}
			a := cmCustomAgent(t, func(agent.InvocationContext) []*session.Event { return []*session.Event{original} })
			r, svc := cmNewRunner(t, a, p)

			yielded := epOnlyEvent(t, "yielded", cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{})))
			stored, state := epSession(t, svc)

			want := epFullEvent()
			tc.want(want)
			if diff := cmp.Diff(want, yielded); diff != "" {
				t.Errorf("yielded event mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(want, epOnlyEvent(t, "stored", stored)); diff != "" {
				t.Errorf("stored event mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantState, state); diff != "" {
				t.Errorf("session state mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.replacement(), returned); diff != "" {
				t.Errorf("plugin's returned event was mutated (-want +got):\n%s", diff)
			}
			// The agent owns its event's maps after the yield returns; the
			// merged event must not share its state delta.
			original.Actions.StateDelta["late"] = "x"
			if diff := cmp.Diff(want.Actions.StateDelta, yielded.Actions.StateDelta); diff != "" {
				t.Errorf("yielded StateDelta after the agent wrote its own (-want +got):\n%s", diff)
			}
		})
	}

	roots := []struct {
		name       string
		wantAuthor string
		newRunner  func(t *testing.T, p *plugin.Plugin) (*Runner, session.Service)
	}{
		{name: "before-run reply on agent root", wantAuthor: "custom", newRunner: func(t *testing.T, p *plugin.Plugin) (*Runner, session.Service) {
			return cmNewRunner(t, cmSingleEventAgent(t, "unreached", nil), p)
		}},
		{name: "before-run reply on node root", wantAuthor: genai.RoleModel, newRunner: func(t *testing.T, p *plugin.Plugin) (*Runner, session.Service) {
			return epNodeRunner(t, func(agent.Context) []*session.Event { return nil }, p)
		}},
	}
	for _, root := range roots {
		for _, tc := range cases {
			t.Run(root.name+"/"+tc.name, func(t *testing.T) {
				var seenID, seenInvocationID string
				p, err := plugin.New(plugin.Config{
					Name: "short_circuit",
					BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
						return genai.NewContentFromText("orig", genai.RoleModel), nil
					},
					OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
						seenID, seenInvocationID = ev.ID, ev.InvocationID
						return tc.replacement(), nil
					},
				})
				if err != nil {
					t.Fatalf("plugin.New() error = %v", err)
				}
				r, svc := root.newRunner(t, p)
				ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return epT0 })

				yielded := epOnlyEvent(t, "yielded", cmCollect(t, r.Run(ctx, cmUser, "s", cmUserText(), agent.RunConfig{})))
				stored, _ := epSession(t, svc)

				wantAuthor := root.wantAuthor
				if tc.wantReplyAuthor != "" {
					wantAuthor = tc.wantReplyAuthor
				}
				for what, ev := range map[string]*session.Event{"yielded": yielded, "stored": epOnlyEvent(t, "stored", stored)} {
					if !ev.Timestamp.Equal(epT0) {
						t.Errorf("%s reply Timestamp = %v, want %v", what, ev.Timestamp, epT0)
					}
					if ev.ID != seenID || ev.ID == "plugin-id" {
						t.Errorf("%s reply ID = %q, want the original's %q", what, ev.ID, seenID)
					}
					if ev.InvocationID != seenInvocationID || ev.InvocationID == "plugin-inv" {
						t.Errorf("%s reply InvocationID = %q, want the original's %q", what, ev.InvocationID, seenInvocationID)
					}
					if ev.Author != wantAuthor {
						t.Errorf("%s reply Author = %q, want %q", what, ev.Author, wantAuthor)
					}
					if ev.CustomMetadata["adk:before_run_reply"] != true {
						t.Errorf("%s reply CustomMetadata[adk:before_run_reply] = %v, want true", what, ev.CustomMetadata["adk:before_run_reply"])
					}
				}
			})
		}
	}
}

func TestRunner_OnEventReplacement_PreservesInternalMetadataAndClearsExplicitEmpty(t *testing.T) {
	eventMetadata := map[string]any{"public_k": "pub", "__adk_internal_k": "sec1", "adk.workflow.node_completed": true}
	tests := []struct {
		name        string
		replacement func() *session.Event
		want        map[string]any
	}{
		{
			name: "replacement_sets_new_map_with_clash",
			replacement: func() *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{CustomMetadata: map[string]any{"repl_k": "rv", "__adk_internal_k": "spoof"}}}
			},
			want: map[string]any{"repl_k": "rv", "__adk_internal_k": "sec1", "adk.workflow.node_completed": true},
		},
		{
			name: "replacement_sets_empty_non_nil_map",
			replacement: func() *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{CustomMetadata: map[string]any{}}}
			},
			want: map[string]any{"__adk_internal_k": "sec1", "adk.workflow.node_completed": true},
		},
		{
			name: "replacement_leaves_nil_map",
			replacement: func() *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("new", genai.RoleModel)}}
			},
			want: map[string]any{"public_k": "pub", "__adk_internal_k": "sec1", "adk.workflow.node_completed": true},
		},
	}
	for _, kind := range cmRootKinds {
		for _, tt := range tests {
			t.Run(kind.name+"/"+tt.name, func(t *testing.T) {
				p := cmOnEventPlugin(t, func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if textOfContent(ev.Content) != "answer" {
						return nil, nil
					}
					return tt.replacement(), nil
				})
				r, svc := cmNewRunner(t, kind.newRoot(t, "answer", eventMetadata), p)

				got := epOnlyEvent(t, "yielded", cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{})))
				storedEvents, _ := epSession(t, svc)
				stored := epOnlyEvent(t, "stored", storedEvents)

				cmCheck(t, "yielded event", got.CustomMetadata, tt.want)
				cmCheck(t, "stored event", stored.CustomMetadata, tt.want)
			})
		}
	}
}

func TestRunner_OnEventReplacement_PartialDowngradeAndUpgrade(t *testing.T) {
	type outcome struct {
		partial bool
		stored  int
		state   map[string]any
	}
	persisted := outcome{partial: false, stored: 1, state: map[string]any{"k": "v"}}
	notStored := outcome{partial: true, stored: 0, state: map[string]any{}}

	tests := []struct {
		name          string
		originPartial bool
		onEvent       func(ev *session.Event) *session.Event
		want          outcome
		loops         []epLoop
	}{
		{
			name:          "downgrade_partial_to_non_partial",
			originPartial: true,
			onEvent: func(*session.Event) *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("final", genai.RoleModel), Partial: false, PartialSet: true}}
			},
			want:  persisted,
			loops: []epLoop{epClassicLoop, epLiveLoop},
		},
		{
			name:          "downgrade_copy_with_partialset",
			originPartial: true,
			onEvent: func(ev *session.Event) *session.Event {
				out := *ev
				out.Partial = false
				out.PartialSet = true
				out.Content = genai.NewContentFromText("final", genai.RoleModel)
				return &out
			},
			want:  persisted,
			loops: []epLoop{epClassicLoop, epLiveLoop},
		},
		{
			name:          "partial_inherited_when_unset",
			originPartial: true,
			onEvent: func(*session.Event) *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("still-partial", genai.RoleModel)}}
			},
			want:  notStored,
			loops: []epLoop{epClassicLoop, epLiveLoop},
		},
		{
			name:          "upgrade_non_partial_to_partial",
			originPartial: false,
			onEvent: func(*session.Event) *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("chunk", genai.RoleModel), Partial: true}}
			},
			want:  notStored,
			loops: []epLoop{epClassicLoop, epLiveLoop, epNodeLoop},
		},
	}
	for _, tt := range tests {
		for _, loop := range tt.loops {
			t.Run(loop.name+"/"+tt.name, func(t *testing.T) {
				p := cmOnEventPlugin(t, func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					return tt.onEvent(ev), nil
				})

				yielded, stored, state := loop.run(t, func(ev *session.Event) {
					ev.Partial = tt.originPartial
					ev.Content = genai.NewContentFromText("orig", genai.RoleModel)
					ev.Actions.StateDelta = map[string]any{"k": "v"}
				}, p)

				got := epOnlyEvent(t, "yielded", yielded)
				if got.Partial != tt.want.partial {
					t.Errorf("yielded Partial = %v, want %v", got.Partial, tt.want.partial)
				}
				if len(stored) != tt.want.stored {
					t.Fatalf("stored %d events, want %d", len(stored), tt.want.stored)
				}
				if tt.want.stored == 1 && (stored[0].ID != got.ID || !stored[0].Timestamp.Equal(got.Timestamp)) {
					t.Errorf("stored event (ID %q, Timestamp %v), want the yielded (ID %q, Timestamp %v)", stored[0].ID, stored[0].Timestamp, got.ID, got.Timestamp)
				}
				if diff := cmp.Diff(tt.want.state, state); diff != "" {
					t.Errorf("session state mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}

	t.Run("before-run reply/upgrade_non_partial_to_partial", func(t *testing.T) {
		p, err := plugin.New(plugin.Config{
			Name: "short_circuit",
			BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
				return genai.NewContentFromText("orig", genai.RoleModel), nil
			},
			OnEventCallback: func(agent.InvocationContext, *session.Event) (*session.Event, error) {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("chunk", genai.RoleModel), Partial: true}}, nil
			},
		})
		if err != nil {
			t.Fatalf("plugin.New() error = %v", err)
		}
		r, svc := cmNewRunner(t, cmSingleEventAgent(t, "unreached", nil), p)

		got := epOnlyEvent(t, "yielded", cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{})))
		stored, _ := epSession(t, svc)

		if !got.Partial {
			t.Errorf("yielded reply Partial = false, want true")
		}
		if len(stored) != 0 {
			t.Errorf("stored %d replies, want 0", len(stored))
		}
	})
}

func TestRunner_RunLive_MediaFilterAfterPlugins(t *testing.T) {
	inline := func(mime string) *genai.Part {
		return &genai.Part{InlineData: &genai.Blob{MIMEType: mime, Data: []byte("raw")}}
	}
	modelContent := func(parts ...*genai.Part) *genai.Content { return &genai.Content{Role: genai.RoleModel, Parts: parts} }
	tests := []struct {
		name    string
		content *genai.Content
		// pluginContent, when set, is what the plugin puts on the event in
		// place of its content.
		pluginContent *genai.Content
		wantStored    int
	}{
		{name: "model_audio_skipped", content: modelContent(inline("audio/pcm;rate=24000"))},
		{name: "model_video_uppercase_skipped", content: modelContent(inline("VIDEO/MP4"))},
		{name: "model_image_mixedcase_skipped", content: modelContent(inline("Image/Png"))},
		{name: "non_media_inline_pdf_persisted", content: modelContent(inline("application/pdf")), wantStored: 1},
		{name: "non_media_inline_text_persisted", content: modelContent(inline("text/plain")), wantStored: 1},
		{name: "file_data_audio_persisted", content: modelContent(&genai.Part{FileData: &genai.FileData{FileURI: "artifact://a.pcm", MIMEType: "audio/pcm"}}), wantStored: 1},
		{name: "plugin_turns_text_into_audio_skipped", content: genai.NewContentFromText("hello", genai.RoleModel), pluginContent: modelContent(inline("audio/pcm"))},
		{name: "plugin_turns_audio_into_text_persisted", content: modelContent(inline("audio/pcm")), pluginContent: genai.NewContentFromText("transcript", genai.RoleModel), wantStored: 1},
	}
	paths := []struct {
		name string
		run  func(t *testing.T, content *genai.Content, onEvent plugin.OnEventCallback) *Runner
	}{
		{name: "live agent stream", run: func(t *testing.T, content *genai.Content, onEvent plugin.OnEventCallback) *Runner {
			a := cmLiveAgent(t, func(ctx agent.InvocationContext) []*session.Event {
				return []*session.Event{epNewEvent(ctx, ctx.InvocationID(), func(ev *session.Event) { ev.Content = content })}
			})
			r, _ := cmNewRunner(t, a, cmOnEventPlugin(t, onEvent))
			return r
		}},
		{name: "before-run reply", run: func(t *testing.T, content *genai.Content, onEvent plugin.OnEventCallback) *Runner {
			p, err := plugin.New(plugin.Config{
				Name:              "short_circuit",
				BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) { return content, nil },
				OnEventCallback:   onEvent,
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			r, _ := cmNewRunner(t, cmLiveAgent(t, func(agent.InvocationContext) []*session.Event { return nil }), p)
			return r
		}},
	}
	for _, path := range paths {
		for _, tt := range tests {
			t.Run(path.name+"/"+tt.name, func(t *testing.T) {
				onEvent := func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if tt.pluginContent != nil {
						ev.Content = tt.pluginContent
					}
					return nil, nil
				}
				r := path.run(t, tt.content, onEvent)
				svc := r.sessionService

				got := epOnlyEvent(t, "yielded", epRunLive(t, r))
				stored, _ := epSession(t, svc)

				wantContent := tt.content
				if tt.pluginContent != nil {
					wantContent = tt.pluginContent
				}
				if diff := cmp.Diff(wantContent, got.Content); diff != "" {
					t.Errorf("yielded content mismatch (-want +got):\n%s", diff)
				}
				if len(stored) != tt.wantStored {
					t.Fatalf("stored %d events, want %d", len(stored), tt.wantStored)
				}
				if tt.wantStored == 1 {
					if diff := cmp.Diff(wantContent, stored[0].Content); diff != "" {
						t.Errorf("stored content mismatch (-want +got):\n%s", diff)
					}
					if stored[0].ID != got.ID || !stored[0].Timestamp.Equal(got.Timestamp) {
						t.Errorf("stored event (ID %q, Timestamp %v), want the yielded (ID %q, Timestamp %v)", stored[0].ID, stored[0].Timestamp, got.ID, got.Timestamp)
					}
				}
			})
		}
	}

	// The live agent stream also stores events through its transcription
	// branch and its two flushes of buffered tool events; each applies the
	// same filter.
	transcription := func(ctx agent.InvocationContext, id string, partial bool, parts ...*genai.Part) *session.Event {
		return epNewEvent(ctx, ctx.InvocationID(), func(ev *session.Event) {
			ev.ID, ev.Partial = id, partial
			ev.OutputTranscription = &genai.Transcription{Text: id, Finished: !partial}
			if len(parts) > 0 {
				ev.Content = modelContent(parts...)
			}
		})
	}
	toolCallWithAudio := func(ctx agent.InvocationContext, id string) *session.Event {
		return epNewEvent(ctx, ctx.InvocationID(), func(ev *session.Event) {
			ev.ID = id
			ev.Content = modelContent(&genai.Part{FunctionCall: &genai.FunctionCall{Name: "t"}}, inline("audio/pcm"))
		})
	}
	streamTests := []struct {
		name        string
		events      func(ctx agent.InvocationContext) []*session.Event
		wantYielded []string
		wantStored  []string
	}{
		{
			name: "transcription_with_model_audio_skipped",
			events: func(ctx agent.InvocationContext) []*session.Event {
				return []*session.Event{transcription(ctx, "a", false, inline("audio/pcm"))}
			},
			wantYielded: []string{"a"},
		},
		{
			name: "buffered_tool_event_with_model_audio_skipped_at_transcription_end",
			events: func(ctx agent.InvocationContext) []*session.Event {
				return []*session.Event{transcription(ctx, "a", true), toolCallWithAudio(ctx, "b"), transcription(ctx, "c", false)}
			},
			wantYielded: []string{"a", "c", "b"},
			wantStored:  []string{"c"},
		},
		{
			name: "buffered_tool_event_with_model_audio_skipped_at_stream_end",
			events: func(ctx agent.InvocationContext) []*session.Event {
				return []*session.Event{transcription(ctx, "a", true), toolCallWithAudio(ctx, "b")}
			},
			wantYielded: []string{"a", "b"},
		},
	}
	for _, tt := range streamTests {
		t.Run("live agent stream/"+tt.name, func(t *testing.T) {
			r, svc := cmNewRunner(t, cmLiveAgent(t, tt.events))

			var gotYielded, gotStored []string
			for _, ev := range epRunLive(t, r) {
				gotYielded = append(gotYielded, ev.ID)
			}
			stored, _ := epSession(t, svc)
			for _, ev := range stored {
				gotStored = append(gotStored, ev.ID)
			}

			if diff := cmp.Diff(tt.wantYielded, gotYielded); diff != "" {
				t.Errorf("yielded event IDs mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantStored, gotStored); diff != "" {
				t.Errorf("stored event IDs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRunner_OnEventCallback_RegistrationOrderFirstWins(t *testing.T) {
	type origin struct {
		id, invocationID, author string
		timestamp                time.Time
	}
	newPlugins := func(t *testing.T, beforeRun bool, seen *origin, p2Called *bool) []*plugin.Plugin {
		cfg1 := plugin.Config{
			Name: "p1",
			OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
				if textOfContent(ev.Content) != "orig" {
					return nil, nil
				}
				*seen = origin{id: ev.ID, invocationID: ev.InvocationID, author: ev.Author, timestamp: ev.Timestamp}
				return &session.Event{LLMResponse: model.LLMResponse{
					Content:        genai.NewContentFromText("from-p1", genai.RoleModel),
					CustomMetadata: map[string]any{"p": "1"},
				}}, nil
			},
		}
		if beforeRun {
			cfg1.BeforeRunCallback = func(agent.InvocationContext) (*genai.Content, error) {
				return genai.NewContentFromText("orig", genai.RoleModel), nil
			}
		}
		p1, err := plugin.New(cfg1)
		if err != nil {
			t.Fatalf("plugin.New(p1) error = %v", err)
		}
		p2, err := plugin.New(plugin.Config{
			Name: "p2",
			OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
				if textOfContent(ev.Content) != "orig" {
					return nil, nil
				}
				*p2Called = true
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("from-p2", genai.RoleModel)}}, nil
			},
		})
		if err != nil {
			t.Fatalf("plugin.New(p2) error = %v", err)
		}
		return []*plugin.Plugin{p1, p2}
	}
	check := func(t *testing.T, seen origin, p2Called bool, yielded, stored []*session.Event) {
		t.Helper()
		if p2Called {
			t.Error("p2's OnEventCallback ran after p1 returned an event")
		}
		for what, ev := range map[string]*session.Event{"yielded": epOnlyEvent(t, "yielded", yielded), "stored": epOnlyEvent(t, "stored", stored)} {
			if got := textOfContent(ev.Content); got != "from-p1" {
				t.Errorf("%s text = %q, want %q", what, got, "from-p1")
			}
			if ev.CustomMetadata["p"] != "1" {
				t.Errorf("%s CustomMetadata[p] = %v, want 1", what, ev.CustomMetadata["p"])
			}
			got := origin{id: ev.ID, invocationID: ev.InvocationID, author: ev.Author, timestamp: ev.Timestamp}
			if got.id != seen.id || got.invocationID != seen.invocationID || got.author != seen.author || !got.timestamp.Equal(seen.timestamp) {
				t.Errorf("%s identity = %+v, want the original's %+v", what, got, seen)
			}
		}
	}

	for _, loop := range []epLoop{epClassicLoop, epNodeLoop, epLiveLoop} {
		t.Run(loop.name, func(t *testing.T) {
			var seen origin
			var p2Called bool

			yielded, stored, _ := loop.run(t, func(ev *session.Event) {
				ev.Content = genai.NewContentFromText("orig", genai.RoleModel)
			}, newPlugins(t, false, &seen, &p2Called)...)

			check(t, seen, p2Called, yielded, stored)
		})
	}
	t.Run("before-run reply", func(t *testing.T) {
		var seen origin
		var p2Called bool
		r, svc := cmNewRunner(t, cmSingleEventAgent(t, "unreached", nil), newPlugins(t, true, &seen, &p2Called)...)

		yielded := cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{}))
		stored, _ := epSession(t, svc)

		check(t, seen, p2Called, yielded, stored)
	})
}

func TestRunner_MessageAsOutput_ClearedAfterPlugin(t *testing.T) {
	fill := func(partial bool) func(*session.Event) {
		return func(ev *session.Event) {
			ev.Partial = partial
			ev.NodeInfo = &session.NodeInfo{MessageAsOutput: true}
			ev.Output = "answer"
			ev.Content = genai.NewContentFromText("answer", genai.RoleModel)
		}
	}
	for _, loop := range []epLoop{epClassicLoop, epNodeLoop, epLiveLoop} {
		t.Run(loop.name+"/downgraded_in_place_is_cleared", func(t *testing.T) {
			p := cmOnEventPlugin(t, func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
				ev.Partial = false
				return nil, nil
			})

			yielded, stored, _ := loop.run(t, fill(true), p)

			for what, ev := range map[string]*session.Event{"yielded": epOnlyEvent(t, "yielded", yielded), "stored": epOnlyEvent(t, "stored", stored)} {
				if ev.Output != nil {
					t.Errorf("%s Output = %v, want nil", what, ev.Output)
				}
				if got := textOfContent(ev.Content); got != "answer" {
					t.Errorf("%s text = %q, want %q", what, got, "answer")
				}
			}
		})

		t.Run(loop.name+"/upgraded_in_place_keeps_output", func(t *testing.T) {
			var seenOutput any
			p := cmOnEventPlugin(t, func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
				seenOutput = ev.Output
				ev.Partial = true
				return nil, nil
			})

			yielded, stored, _ := loop.run(t, fill(false), p)

			if seenOutput != "answer" {
				t.Errorf("plugin saw Output = %v, want %q", seenOutput, "answer")
			}
			if got := epOnlyEvent(t, "yielded", yielded).Output; got != "answer" {
				t.Errorf("yielded Output = %v, want %q", got, "answer")
			}
			if len(stored) != 0 {
				t.Errorf("stored %d events, want 0", len(stored))
			}
		})
	}
}
