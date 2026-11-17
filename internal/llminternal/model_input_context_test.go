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

package llminternal_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

const micAgentName = "mic_agent"

// micRun describes one run of ContentsRequestProcessor with a
// RunConfig.ModelInputContext.
type micRun struct {
	agentName         string // micAgentName when empty
	includeContents   llmagent.IncludeContents
	events            []*session.Event
	userContent       *genai.Content
	modelInputContext []*genai.Content
	// initial is what req.Contents holds before the processor runs.
	initial []*genai.Content
}

// process runs ContentsRequestProcessor for r and returns the request it built.
func (r micRun) process(t *testing.T) *model.LLMRequest {
	t.Helper()
	name := r.agentName
	if name == "" {
		name = micAgentName
	}
	a := utils.Must(llmagent.New(llmagent.Config{
		Name:            name,
		Model:           &testModel{},
		IncludeContents: r.includeContents,
	}))
	rc := &agent.RunConfig{}
	if r.modelInputContext != nil {
		rc.ModelInputContext = &r.modelInputContext
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:       a,
		Session:     &fakeSession{events: r.events},
		UserContent: r.userContent,
		RunConfig:   rc,
	})
	req := &model.LLMRequest{Contents: r.initial}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("ContentsRequestProcessor() error = %v", err)
		}
	}
	return req
}

func micText(text string, role genai.Role) *genai.Content {
	return genai.NewContentFromText(text, role)
}

func micEvent(author string, c *genai.Content) *session.Event {
	return &session.Event{Author: author, LLMResponse: model.LLMResponse{Content: c}}
}

func micUser(text string) *session.Event { return micEvent("user", micText(text, genai.RoleUser)) }

func micModel(text string) *session.Event {
	return micEvent(micAgentName, micText(text, genai.RoleModel))
}

func micCTX() []*genai.Content { return []*genai.Content{micText("CTX", genai.RoleUser)} }

func TestContentsRequestProcessor_ModelInputContext_BeforeLastEqualUserMessage(t *testing.T) {
	tests := []struct {
		name string
		run  micRun
		want []*genai.Content
	}{
		{
			name: "after history",
			run: micRun{
				events:            []*session.Event{micUser("first"), micModel("reply"), micUser("second")},
				userContent:       micText("second", genai.RoleUser),
				modelInputContext: micCTX(),
			},
			want: []*genai.Content{
				micText("first", genai.RoleUser),
				micText("reply", genai.RoleModel),
				micText("CTX", genai.RoleUser),
				micText("second", genai.RoleUser),
			},
		},
		{
			name: "repeated message, last one",
			run: micRun{
				events:            []*session.Event{micUser("yes"), micModel("ok"), micUser("yes")},
				userContent:       micText("yes", genai.RoleUser),
				modelInputContext: micCTX(),
			},
			want: []*genai.Content{
				micText("yes", genai.RoleUser),
				micText("ok", genai.RoleModel),
				micText("CTX", genai.RoleUser),
				micText("yes", genai.RoleUser),
			},
		},
		{
			name: "two entries keep their order",
			run: micRun{
				events:            []*session.Event{micUser("q")},
				userContent:       micText("q", genai.RoleUser),
				modelInputContext: []*genai.Content{micText("C1", genai.RoleUser), micText("C2", genai.RoleUser)},
			},
			want: []*genai.Content{
				micText("C1", genai.RoleUser),
				micText("C2", genai.RoleUser),
				micText("q", genai.RoleUser),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.run.process(t).Contents
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestContentsRequestProcessor_ModelInputContext_FrontWhenNoAnchor(t *testing.T) {
	history := func() []*session.Event {
		return []*session.Event{micUser("q1"), micModel("a1"), micUser("q2")}
	}
	wantHistory := []*genai.Content{
		micText("CTX", genai.RoleUser),
		micText("q1", genai.RoleUser),
		micText("a1", genai.RoleModel),
		micText("q2", genai.RoleUser),
	}
	tests := []struct {
		name string
		run  micRun
		want []*genai.Content
	}{
		{
			name: "no user content",
			run:  micRun{events: history(), modelInputContext: micCTX()},
			want: wantHistory,
		},
		{
			name: "user content not in history",
			run: micRun{
				events:            history(),
				userContent:       micText("other question", genai.RoleUser),
				modelInputContext: micCTX(),
			},
			want: wantHistory,
		},
		{
			name: "no history",
			run:  micRun{modelInputContext: micCTX()},
			want: []*genai.Content{micText("CTX", genai.RoleUser)},
		},
		{
			name: "user content has only an empty part",
			run: micRun{
				events: []*session.Event{
					micUser("q1"),
					micModel("done"),
					micEvent("user", &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{}}}),
				},
				userContent:       &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{}}},
				modelInputContext: micCTX(),
			},
			want: []*genai.Content{
				micText("CTX", genai.RoleUser),
				micText("q1", genai.RoleUser),
				micText("done", genai.RoleModel),
				micText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", genai.RoleUser),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.run.process(t).Contents
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestContentsRequestProcessor_ModelInputContext_NearMissIsNotAnchor(t *testing.T) {
	tests := []struct {
		name string
		run  micRun
		want []*genai.Content
	}{
		{
			name: "same text, model role",
			run: micRun{
				events:            []*session.Event{micUser("earlier"), micModel("Q"), micUser("later")},
				userContent:       micText("Q", genai.RoleUser),
				modelInputContext: micCTX(),
			},
			want: []*genai.Content{
				micText("CTX", genai.RoleUser),
				micText("earlier", genai.RoleUser),
				micText("Q", genai.RoleModel),
				micText("later", genai.RoleUser),
			},
		},
		{
			name: "user content has an extra part",
			run: micRun{
				events: []*session.Event{micModel("x"), micUser("Q")},
				userContent: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
					{Text: "Q"}, {Text: "extra"},
				}},
				modelInputContext: micCTX(),
			},
			want: []*genai.Content{
				micText("CTX", genai.RoleUser),
				micText("x", genai.RoleModel),
				micText("Q", genai.RoleUser),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.run.process(t).Contents
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestContentsRequestProcessor_ModelInputContext_IncludeContentsNone(t *testing.T) {
	t.Run("current turn is the user message", func(t *testing.T) {
		run := micRun{
			includeContents:   llmagent.IncludeContentsNone,
			events:            []*session.Event{micUser("old"), micModel("x"), micUser("q")},
			userContent:       micText("q", genai.RoleUser),
			modelInputContext: micCTX(),
		}
		want := []*genai.Content{micText("CTX", genai.RoleUser), micText("q", genai.RoleUser)}
		if diff := cmp.Diff(want, run.process(t).Contents); diff != "" {
			t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("current turn starts at another agent", func(t *testing.T) {
		run := micRun{
			agentName:       "agent2",
			includeContents: llmagent.IncludeContentsNone,
			events: []*session.Event{
				micUser("Original"),
				micEvent("agent1", micText("Agent1 response: XYZ", genai.RoleModel)),
			},
			userContent:       micText("Original", genai.RoleUser),
			modelInputContext: micCTX(),
		}
		got := run.process(t).Contents
		if len(got) != 2 {
			t.Fatalf("len(req.Contents) = %d, want 2; contents: %v", len(got), got)
		}
		if diff := cmp.Diff(micText("CTX", genai.RoleUser), got[0]); diff != "" {
			t.Errorf("req.Contents[0] mismatch (-want +got):\n%s", diff)
		}
		if got[1].Role != genai.RoleUser {
			t.Errorf("req.Contents[1].Role = %q, want %q", got[1].Role, genai.RoleUser)
		}
		var text strings.Builder
		for _, p := range got[1].Parts {
			text.WriteString(p.Text)
		}
		if !strings.Contains(text.String(), "Agent1 response: XYZ") {
			t.Errorf("req.Contents[1] text = %q, want it to contain %q", text.String(), "Agent1 response: XYZ")
		}
	})
}

func TestContentsRequestProcessor_ModelInputContext_ClientCallIDMessageGoesFirst(t *testing.T) {
	response := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: id, Name: "t", Response: map[string]any{"r": 1},
		}}}}
	}
	call := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: id, Name: "t",
		}}}}
	}
	run := micRun{
		events:            []*session.Event{micUser("start"), micEvent(micAgentName, call("adk-1")), micEvent("user", response("adk-1"))},
		userContent:       response("adk-1"),
		modelInputContext: micCTX(),
	}
	want := []*genai.Content{
		micText("CTX", genai.RoleUser),
		micText("start", genai.RoleUser),
		call(""),
		response(""),
	}
	if diff := cmp.Diff(want, run.process(t).Contents); diff != "" {
		t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
	}
}

func TestContentsRequestProcessor_ModelInputContext_EmptyPartInUserMessage(t *testing.T) {
	withEmptyPart := func() *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "Q"}, {}}}
	}
	run := micRun{
		events:            []*session.Event{micModel("x"), micEvent("user", withEmptyPart())},
		userContent:       withEmptyPart(),
		modelInputContext: micCTX(),
	}
	want := []*genai.Content{
		micText("x", genai.RoleModel),
		micText("CTX", genai.RoleUser),
		micText("Q", genai.RoleUser),
	}
	if diff := cmp.Diff(want, run.process(t).Contents); diff != "" {
		t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
	}
}

func TestContentsRequestProcessor_ModelInputContext_LeavesUserContentUnchanged(t *testing.T) {
	userContent := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "Q"}, {}}}
	run := micRun{
		events:            []*session.Event{micEvent("user", &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "Q"}, {}}})},
		userContent:       userContent,
		modelInputContext: micCTX(),
	}

	run.process(t)

	if got := len(userContent.Parts); got != 2 {
		t.Errorf("len(invocation user content parts) = %d, want 2 (the processor must not drop the empty part from the caller's content)", got)
	}
}

func TestContentsRequestProcessor_ModelInputContext_NestedValuesAreCopied(t *testing.T) {
	callerContent := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		Name:     "t",
		Response: map[string]any{"outer": map[string]any{"inner": "orig"}},
	}}}}
	run := micRun{
		events:            []*session.Event{micUser("q")},
		userContent:       micText("q", genai.RoleUser),
		modelInputContext: []*genai.Content{callerContent},
	}
	req := run.process(t)
	if fr := req.Contents[0].Parts[0].FunctionResponse; fr == nil {
		t.Fatalf("req.Contents[0] = %v, want the copy of ModelInputContext[0]", req.Contents[0])
	}

	req.Contents[0].Parts[0].FunctionResponse.Response["outer"].(map[string]any)["inner"] = "mutated"

	if got := callerContent.Parts[0].FunctionResponse.Response["outer"].(map[string]any)["inner"]; got != "orig" {
		t.Errorf("caller's ModelInputContext[0] inner value = %v, want %q", got, "orig")
	}
}

func TestContentsRequestProcessor_ModelInputContext_IndexIsRelativeToHistory(t *testing.T) {
	run := micRun{
		events:            []*session.Event{micUser("a"), micModel("b"), micUser("c")},
		userContent:       micText("c", genai.RoleUser),
		modelInputContext: micCTX(),
		initial:           []*genai.Content{micText("PRE", genai.RoleUser)},
	}
	want := []*genai.Content{
		micText("PRE", genai.RoleUser),
		micText("a", genai.RoleUser),
		micText("b", genai.RoleModel),
		micText("CTX", genai.RoleUser),
		micText("c", genai.RoleUser),
	}
	if diff := cmp.Diff(want, run.process(t).Contents); diff != "" {
		t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
	}
}
