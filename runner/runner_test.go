// Copyright 2025 Google LLC
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
	"bytes"
	"context"
	"errors"
	"fmt"
	"iter"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

func TestRunner_findAgentToRun(t *testing.T) {
	t.Parallel()

	appName, userID, sessionID := "test", "userID", "sessionID"

	agentTree := agentTree(t)

	tests := []struct {
		name        string
		rootAgent   agent.Agent
		session     session.Session
		userMessage *genai.Content
		wantAgent   agent.Agent
		wantErr     bool
	}{
		{
			name: "last event from agent allowing transfer",
			session: createSession(t, t.Context(), appName, userID, sessionID, []*session.Event{
				{
					Author: "allows_transfer_agent",
				},
			}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.allowsTransferAgent,
		},
		{
			name: "last event from agent not allowing transfer",
			session: createSession(t, t.Context(), appName, userID, sessionID, []*session.Event{
				{
					Author: "no_transfer_agent",
				},
			}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.root,
		},
		{
			name:      "no events from agents, call root",
			session:   createSession(t, t.Context(), appName, userID, sessionID, []*session.Event{}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.root,
		},
		{
			name: "last event from user with function response",
			session: createSession(t, t.Context(), appName, userID, sessionID, []*session.Event{
				{
					Author: agentTree.noTransferAgent.Name(),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										Name: "fn_name",
										ID:   "fn_id",
									},
								},
							},
						},
					},
				},
				{
					Author: agentTree.root.Name(),
				},
			}),
			userMessage: genai.NewContentFromParts([]*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					Name: "fn_name",
					ID:   "fn_id",
				},
			}}, genai.RoleUser),
			rootAgent: agentTree.root,
			wantAgent: agentTree.noTransferAgent,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Runner{
				rootAgent: tt.rootAgent,
			}
			gotAgent, err := r.findAgentToRun(tt.session, tt.userMessage)
			if (err != nil) != tt.wantErr {
				t.Errorf("Runner.findAgentToRun() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantAgent != gotAgent {
				t.Errorf("Runner.findAgentToRun() = %+v, want %+v", gotAgent.Name(), tt.wantAgent.Name())
			}
		})
	}
}

func Test_isTransferrableAcrossAgentTree(t *testing.T) {
	tests := []struct {
		name  string
		agent agent.Agent
		want  bool
	}{
		{
			name: "disallow for agent with DisallowTransferToParent",
			agent: must(llmagent.New(llmagent.Config{
				Name:                     "test",
				DisallowTransferToParent: true,
			})),
			want: false,
		},
		{
			name: "disallow for non-LLM agent",
			agent: must(agent.New(agent.Config{
				Name: "test",
			})),
			want: false,
		},
		{
			name: "allow for the default LLM agent",
			agent: must(llmagent.New(llmagent.Config{
				Name: "test",
			})),
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner, err := New(Config{
				AppName:        "testApp",
				Agent:          tt.agent,
				SessionService: session.InMemoryService(),
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := runner.isTransferableAcrossAgentTree(tt.agent); got != tt.want {
				t.Errorf("isTransferrableAcrossAgentTree() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRunner_SaveInputBlobsAsArtifacts(t *testing.T) {
	ctx := t.Context()
	appName := "testApp"
	userID := "testUser"
	sessionID := "testSession"

	sessionService := session.InMemoryService()
	artifactService := artifact.InMemoryService()

	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// no-op, we are testing logic before agent run.
			}
		},
	}))

	r, err := New(Config{
		AppName:        appName,
		Agent:          testAgent,
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	r.artifactService = artifactService

	_, err = sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}

	// Blob data, the message is saved only when inline data is present.
	blobData := []byte("this is not blob data - René Magritte")
	msg := &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("here is a file"),
			{InlineData: &genai.Blob{MIMEType: "application/octet-stream", Data: blobData}},
		},
		Role: genai.RoleUser,
	}

	cfg := agent.RunConfig{
		SaveInputBlobsAsArtifacts: true,
	}

	// Consume the iterator from Run. The agent itself does nothing, but the runner
	// will save the artifact before calling the agent.
	for _, err := range r.Run(ctx, userID, sessionID, msg, cfg) {
		if err != nil {
			t.Fatalf("r.Run() returned an error: %v", err)
		}
	}

	listResp, err := artifactService.List(ctx, &artifact.ListRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("artifactService.List() error = %v", err)
	}
	if len(listResp.FileNames) != 1 {
		t.Fatalf("expected 1 artifact, got %d", len(listResp.FileNames))
	}
	savedFileName := listResp.FileNames[0]

	if !strings.HasPrefix(savedFileName, "artifact_") {
		t.Errorf("saved file name should start with 'artifact_', got %q", savedFileName)
	}

	loadResp, err := artifactService.Load(ctx, &artifact.LoadRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  savedFileName,
	})
	if err != nil {
		t.Fatalf("artifactService.Load() error = %v", err)
	}

	if !bytes.Equal(loadResp.Part.InlineData.Data, blobData) {
		t.Errorf("loaded artifact data does not match original blob data")
	}

	getResponse, err := sessionService.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("sessionService.Get() error = %v", err)
	}

	events := getResponse.Session.Events()
	if events.Len() == 0 {
		t.Fatal("no events in session")
	}
	userEvent := events.At(0)
	if userEvent.Author != "user" {
		t.Fatalf("expected first event to be from user, got %s", userEvent.Author)
	}

	// The part with InlineData should be replaced.
	if len(userEvent.LLMResponse.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts in user message event, got %d", len(userEvent.LLMResponse.Content.Parts))
	}
	partWithBlob := userEvent.LLMResponse.Content.Parts[1]
	if partWithBlob.InlineData != nil {
		t.Errorf("InlineData was not removed from the message part in the session")
	}
	expectedText := fmt.Sprintf("Uploaded file: %s. It has been saved to the artifacts", savedFileName)
	if partWithBlob.Text != expectedText {
		t.Errorf("unexpected text in placeholder part. got %q, want %q", partWithBlob.Text, expectedText)
	}
}

// TestRunner_PluginModifiesUserMessage guards that a plugin modifying the
// user message still yields a full run context.
func TestRunner_PluginModifiesUserMessage(t *testing.T) {
	ctx := t.Context()
	appName := "testApp"
	userID := "testUser"
	sessionID := "testSession"

	var gotMessage *genai.Content
	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// Accessors that nil-deref on a callback-only context.
				_ = ctx.Agent().Name()
				_ = ctx.Session().ID()
				gotMessage = ctx.UserContent()
			}
		},
	}))

	modifierPlugin, err := plugin.New(plugin.Config{
		Name: "message_modifier",
		OnUserMessageCallback: func(_ agent.InvocationContext, _ *genai.Content) (*genai.Content, error) {
			return &genai.Content{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{genai.NewPartFromText("modified")},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}

	sessionService := session.InMemoryService()
	r, err := New(Config{
		AppName:        appName,
		Agent:          testAgent,
		SessionService: sessionService,
		PluginConfig:   PluginConfig{Plugins: []*plugin.Plugin{modifierPlugin}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if _, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromText("original")}}
	for _, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run() returned an error: %v", err)
		}
	}

	if gotMessage == nil {
		t.Fatal("agent did not observe a user message")
	}
	if len(gotMessage.Parts) != 1 || gotMessage.Parts[0].Text != "modified" {
		t.Errorf("agent saw %v, want the plugin-modified message", gotMessage)
	}
}

// recordingModel records each request it receives and replies with a fixed
// text.
type recordingModel struct{ reqs []*model.LLMRequest }

func (*recordingModel) Name() string { return "recording" }

func (m *recordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.reqs = append(m.reqs, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel)}, nil)
	}
}

// TestRunner_InboundMessageIsUserTurn guards that the inbound message is
// stored and sent to the model as a user turn, whatever Role the caller set.
// An empty Role used to be dropped when the request was built, and a "model"
// Role made the caller's text look like the model's own earlier turn.
func TestRunner_InboundMessageIsUserTurn(t *testing.T) {
	for _, tc := range []struct {
		name string
		role string
	}{
		{name: "empty role", role: ""},
		{name: "model role", role: genai.RoleModel},
		{name: "user role", role: genai.RoleUser},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			m := &recordingModel{}
			a := must(llmagent.New(llmagent.Config{Name: "a", Model: m}))
			sessionService := session.InMemoryService()
			r, err := New(Config{AppName: "app", Agent: a, SessionService: sessionService})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
				t.Fatalf("sessionService.Create() error = %v", err)
			}

			msg := &genai.Content{Role: tc.role, Parts: []*genai.Part{genai.NewPartFromText("hello model")}}
			for _, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run() returned an error: %v", err)
				}
			}

			// The caller's struct is left alone.
			if msg.Role != tc.role {
				t.Errorf("caller's msg.Role = %q, want it unchanged (%q)", msg.Role, tc.role)
			}

			// The model sees the message as a user turn.
			if len(m.reqs) != 1 {
				t.Fatalf("model got %d requests, want 1", len(m.reqs))
			}
			var sent []string
			for _, c := range m.reqs[0].Contents {
				for _, p := range c.Parts {
					if p.Text == "hello model" {
						sent = append(sent, c.Role)
					}
				}
			}
			if len(sent) != 1 || sent[0] != genai.RoleUser {
				t.Errorf("model saw the inbound text with roles %q, want exactly [%q]", sent, genai.RoleUser)
			}

			// The stored event is a user turn too.
			got, err := sessionService.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
			if err != nil {
				t.Fatalf("sessionService.Get() error = %v", err)
			}
			first := got.Session.Events().At(0)
			if first.Author != "user" || first.Content == nil || first.Content.Role != genai.RoleUser {
				t.Errorf("stored inbound event = author %q content %+v, want author user and role %q", first.Author, first.Content, genai.RoleUser)
			}
		})
	}
}

// TestRunner_SaveInputBlobsDoesNotMutateCallerMessage guards that replacing
// inline blobs with placeholders edits the stored copy, not the Parts of the
// caller's message or of a message an OnUserMessageCallback plugin returned,
// on both run paths.
func TestRunner_SaveInputBlobsDoesNotMutateCallerMessage(t *testing.T) {
	customRoot := func() agent.Agent {
		return must(agent.New(agent.Config{
			Name: "test_agent",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			},
		}))
	}
	llmRoot := func() agent.Agent {
		return must(llmagent.New(llmagent.Config{Name: "test_agent", Model: &recordingModel{}}))
	}

	for _, tc := range []struct {
		name      string
		root      func() agent.Agent
		viaPlugin bool
	}{
		{name: "classic_path", root: customRoot},
		{name: "node_path", root: llmRoot},
		{name: "plugin_message/classic", root: customRoot, viaPlugin: true},
		{name: "plugin_message/node", root: llmRoot, viaPlugin: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			blob := &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream", Data: []byte("data")}}
			// watched is the message whose Parts must survive the run: the
			// caller's own message, or the one the plugin returns.
			watched := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromText("file"), blob}}
			msg := watched
			var plugins []*plugin.Plugin
			if tc.viaPlugin {
				msg = &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromText("hi")}}
				p, err := plugin.New(plugin.Config{
					Name: "returns_blob",
					OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
						return watched, nil
					},
				})
				if err != nil {
					t.Fatalf("plugin.New() error = %v", err)
				}
				plugins = append(plugins, p)
			}

			sessionService := session.InMemoryService()
			r, err := New(Config{AppName: "app", Agent: tc.root(), SessionService: sessionService, PluginConfig: PluginConfig{Plugins: plugins}})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			r.artifactService = artifact.InMemoryService()
			if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
				t.Fatalf("sessionService.Create() error = %v", err)
			}

			for _, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{SaveInputBlobsAsArtifacts: true}) {
				if err != nil {
					t.Fatalf("r.Run() returned an error: %v", err)
				}
			}
			if watched.Parts[1] != blob || blob.InlineData == nil {
				t.Errorf("message was modified: Parts[1] = %+v, want the original blob part", watched.Parts[1])
			}
		})
	}
}

// TestRunner_InboundRoleNormalizedForContextAndPlugin guards the two other
// places the inbound message flows: the invocation context's UserContent, and
// a message returned by an OnUserMessageCallback plugin.
func TestRunner_InboundRoleNormalizedForContextAndPlugin(t *testing.T) {
	ctx := t.Context()
	var gotUserContent *genai.Content
	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) { gotUserContent = ic.UserContent() }
		},
	}))
	sessionService := session.InMemoryService()
	r, err := New(Config{AppName: "app", Agent: testAgent, SessionService: sessionService})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}
	msg := &genai.Content{Parts: []*genai.Part{genai.NewPartFromText("hi")}}
	for _, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run() returned an error: %v", err)
		}
	}
	if gotUserContent == nil || gotUserContent.Role != genai.RoleUser {
		t.Errorf("ctx.UserContent() = %+v, want Role %q", gotUserContent, genai.RoleUser)
	}

	// A plugin that returns a message with no Role still yields a user turn.
	roleless, err := plugin.New(plugin.Config{
		Name: "roleless",
		OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
			return &genai.Content{Parts: []*genai.Part{genai.NewPartFromText("from plugin")}}, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	r2, err := New(Config{AppName: "app", Agent: testAgent, SessionService: sessionService, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{roleless}}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s2"}); err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}
	gotUserContent = nil
	for _, err := range r2.Run(ctx, "u", "s2", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r2.Run() returned an error: %v", err)
		}
	}
	if gotUserContent == nil || gotUserContent.Role != genai.RoleUser || gotUserContent.Parts[0].Text != "from plugin" {
		t.Errorf("ctx.UserContent() with plugin = %+v, want the plugin message with Role %q", gotUserContent, genai.RoleUser)
	}
	got, err := sessionService.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s2"})
	if err != nil {
		t.Fatalf("sessionService.Get() error = %v", err)
	}
	first := got.Session.Events().At(0)
	if first.Content == nil || first.Content.Role != genai.RoleUser || first.Content.Parts[0].Text != "from plugin" {
		t.Errorf("stored event content = %+v, want the plugin message with Role %q", first.Content, genai.RoleUser)
	}
}

// TestRunner_RejectsUserFunctionCall guards that a user message carrying a
// function call ends the run with an error before the message or its input
// blobs are stored or any agent runs, on both the legacy agent path and the
// node path.
func TestRunner_RejectsUserFunctionCall(t *testing.T) {
	functionCall := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "some_tool", Args: map[string]any{"key": "value"}}}
	inlineData := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89, 0x50}}}
	for _, tc := range []struct {
		name  string
		llm   bool
		parts []*genai.Part
	}{
		{name: "legacy path", parts: []*genai.Part{functionCall}},
		{name: "node path", llm: true, parts: []*genai.Part{functionCall}},
		{name: "node path, call after text", llm: true, parts: []*genai.Part{genai.NewPartFromText("hi"), functionCall}},
		{name: "legacy path, call after inline data", parts: []*genai.Part{inlineData, functionCall}},
		{name: "node path, call after inline data", llm: true, parts: []*genai.Part{inlineData, functionCall}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			ran := false
			m := &recordingModel{}
			var a agent.Agent
			if tc.llm {
				a = must(llmagent.New(llmagent.Config{Name: "test_agent", Model: m}))
			} else {
				a = must(agent.New(agent.Config{
					Name: "test_agent",
					Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) { ran = true }
					},
				}))
			}
			sessionService := session.InMemoryService()
			artifactService := artifact.InMemoryService()
			r, err := New(Config{AppName: "app", Agent: a, SessionService: sessionService, ArtifactService: artifactService})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
				t.Fatalf("sessionService.Create() error = %v", err)
			}

			msg := &genai.Content{Role: genai.RoleUser, Parts: tc.parts}
			var runErr error
			for _, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{SaveInputBlobsAsArtifacts: true}) {
				if err != nil {
					runErr = err
				}
			}

			if runErr == nil || runErr.Error() != "user message cannot contain function calls" {
				t.Errorf("r.Run() error = %v, want %q", runErr, "user message cannot contain function calls")
			}
			if ran || len(m.reqs) != 0 {
				t.Errorf("agent ran = %t, model requests = %d, want no agent run and no model request", ran, len(m.reqs))
			}
			got, err := sessionService.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
			if err != nil {
				t.Fatalf("sessionService.Get() error = %v", err)
			}
			if n := got.Session.Events().Len(); n != 0 {
				t.Errorf("session has %d events, want 0", n)
			}
			saved, err := artifactService.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "u", SessionID: "s"})
			if err != nil {
				t.Fatalf("artifactService.List() error = %v", err)
			}
			if len(saved.FileNames) != 0 {
				t.Errorf("saved artifacts = %v, want none", saved.FileNames)
			}
		})
	}
}

// TestRunner_UserFunctionCallCheckedAfterPlugin guards that the function call
// check applies to the message an OnUserMessageCallback plugin returns, not to
// the message the caller sent.
func TestRunner_UserFunctionCallCheckedAfterPlugin(t *testing.T) {
	functionCall := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "some_tool"}}
	for _, tc := range []struct {
		name     string
		sent     []*genai.Part
		returned []*genai.Part
		wantErr  bool
	}{
		{
			name:     "plugin removes the call",
			sent:     []*genai.Part{functionCall},
			returned: []*genai.Part{genai.NewPartFromText("from plugin")},
		},
		{
			name:     "plugin adds a call",
			sent:     []*genai.Part{genai.NewPartFromText("hi")},
			returned: []*genai.Part{functionCall},
			wantErr:  true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			ran := false
			testAgent := must(agent.New(agent.Config{
				Name: "test_agent",
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(func(*session.Event, error) bool) { ran = true }
				},
			}))
			replacer, err := plugin.New(plugin.Config{
				Name: "replacer",
				OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
					return &genai.Content{Role: genai.RoleUser, Parts: tc.returned}, nil
				},
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			sessionService := session.InMemoryService()
			r, err := New(Config{AppName: "app", Agent: testAgent, SessionService: sessionService, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{replacer}}})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
				t.Fatalf("sessionService.Create() error = %v", err)
			}

			var runErr error
			for _, err := range r.Run(ctx, "u", "s", &genai.Content{Role: genai.RoleUser, Parts: tc.sent}, agent.RunConfig{}) {
				if err != nil {
					runErr = err
				}
			}

			if gotErr := errors.Is(runErr, errUserFunctionCall); gotErr != tc.wantErr {
				t.Errorf("r.Run() error = %v, want errUserFunctionCall %t", runErr, tc.wantErr)
			}
			if ran == tc.wantErr {
				t.Errorf("agent ran = %t, want %t", ran, !tc.wantErr)
			}
		})
	}
}

// TestRunner_UserMessageWithNilPartRuns guards that the function call check
// skips a nil part instead of dereferencing it.
func TestRunner_UserMessageWithNilPartRuns(t *testing.T) {
	ctx := t.Context()
	ran := false
	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) { ran = true }
		},
	}))
	sessionService := session.InMemoryService()
	r, err := New(Config{AppName: "app", Agent: testAgent, SessionService: sessionService})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{nil, genai.NewPartFromText("hi")}}
	for _, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run() returned an error: %v", err)
		}
	}
	if !ran {
		t.Error("agent did not run")
	}
}

// creates agentTree for tests and returns references to the agents
func agentTree(t *testing.T) agentTreeStruct {
	t.Helper()

	sub1 := must(llmagent.New(llmagent.Config{
		Name:                     "no_transfer_agent",
		DisallowTransferToParent: true,
	}))
	sub2 := must(llmagent.New(llmagent.Config{
		Name: "allows_transfer_agent",
	}))
	parent := must(llmagent.New(llmagent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{sub1, sub2},
	}))

	return agentTreeStruct{
		root:                parent,
		noTransferAgent:     sub1,
		allowsTransferAgent: sub2,
	}
}

type agentTreeStruct struct {
	root, noTransferAgent, allowsTransferAgent agent.Agent
}

func must[T agent.Agent](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}

// TestBuildRunnerNode_AllAgentKinds verifies buildRunnerNode wraps every
// agent kind (not just LlmAgent) and names the node after the agent.
func TestBuildRunnerNode_AllAgentKinds(t *testing.T) {
	t.Parallel()

	noop := func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {}
	}

	tests := []struct {
		name  string
		agent agent.Agent
	}{
		{
			name:  "llm_agent",
			agent: must(llmagent.New(llmagent.Config{Name: "llm_agent", Model: &noopModel{}})),
		},
		{
			name:  "custom_agent",
			agent: must(agent.New(agent.Config{Name: "custom_agent", Run: noop})),
		},
		{
			name: "agent_with_subagents",
			agent: must(agent.New(agent.Config{
				Name: "agent_with_subagents",
				Run:  noop,
				SubAgents: []agent.Agent{
					must(agent.New(agent.Config{Name: "child", Run: noop})),
				},
			})),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := buildRunnerNode(tc.agent)
			if err != nil {
				t.Fatalf("buildRunnerNode(%T) error = %v, want nil", tc.agent, err)
			}
			if node == nil {
				t.Fatalf("buildRunnerNode(%T) returned nil node", tc.agent)
			}
			if got, want := node.Name(), tc.agent.Name(); got != want {
				t.Errorf("node.Name() = %q, want %q", got, want)
			}
		})
	}
}

// TestBuildRunnerNode_NilAgent verifies buildRunnerNode rejects a nil
// agent instead of panicking later in the node runtime.
func TestBuildRunnerNode_NilAgent(t *testing.T) {
	t.Parallel()

	if _, err := buildRunnerNode(nil); err == nil {
		t.Error("buildRunnerNode(nil) error = nil, want non-nil")
	}
}

// noopModel is a minimal model.LLM used to construct LlmAgents in tests
// that never call the model.
type noopModel struct{}

func (noopModel) Name() string { return "noop" }

func (noopModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func createSession(t *testing.T, ctx context.Context, sessionID, appName, userID string, events []*session.Event) session.Session {
	t.Helper()

	service := session.InMemoryService()

	resp, err := service.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, event := range events {
		if err := service.AppendEvent(ctx, resp.Session, event); err != nil {
			t.Fatal(err)
		}
	}

	return resp.Session
}

// TestNewInMemory verifies the convenience constructor wires all three
// in-memory services, enables session auto-creation, and runs end to end
// without any manual service or session setup.
func TestNewInMemory(t *testing.T) {
	t.Parallel()

	appName := "testApp"
	userID := "testUser"
	sessionID := "testSession"

	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	}))

	r, err := NewInMemory(appName, testAgent)
	if err != nil {
		t.Fatalf("NewInMemory() error = %v", err)
	}

	if r.sessionService == nil {
		t.Error("NewInMemory() sessionService = nil, want in-memory service")
	}
	if r.artifactService == nil {
		t.Error("NewInMemory() artifactService = nil, want in-memory service")
	}
	if r.memoryService == nil {
		t.Error("NewInMemory() memoryService = nil, want in-memory service")
	}
	if !r.autoCreateSession {
		t.Error("NewInMemory() autoCreateSession = false, want true")
	}

	// A run must succeed without pre-creating the session (auto-create on).
	ctx := t.Context()
	msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
	for _, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run() error = %v", err)
		}
	}

	if _, err := r.sessionService.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		t.Errorf("expected auto-created session to exist, got error: %v", err)
	}
}

func TestRunner_AutoCreateSession(t *testing.T) {
	t.Parallel()

	appName := "testApp"
	userID := "testUser"
	sessionID := "testSession"

	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// no-op, we are testing logic before agent run.
			}
		},
	}))

	tests := []struct {
		name              string
		autoCreateSession bool
		setupSession      bool
		wantErr           bool
	}{
		{
			name:              "auto_create_true_session_missing",
			autoCreateSession: true,
			setupSession:      false,
			wantErr:           false,
		},
		{
			name:              "auto_create_false_session_missing",
			autoCreateSession: false,
			setupSession:      false,
			wantErr:           true,
		},
		{
			name:              "auto_create_false_session_exists",
			autoCreateSession: false,
			setupSession:      true,
			wantErr:           false,
		},
		{
			name:              "auto_create_true_session_exists",
			autoCreateSession: true,
			setupSession:      true,
			wantErr:           false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()
			sessionService := session.InMemoryService()

			if tt.setupSession {
				_, err := sessionService.Create(ctx, &session.CreateRequest{
					AppName:   appName,
					UserID:    userID,
					SessionID: sessionID,
				})
				if err != nil {
					t.Fatalf("failed to setup session: %v", err)
				}
			}

			r, err := New(Config{
				AppName:           appName,
				Agent:             testAgent,
				SessionService:    sessionService,
				AutoCreateSession: tt.autoCreateSession,
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
			gotError := false
			for _, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
				if err != nil {
					gotError = true
				}
			}

			if gotError != tt.wantErr {
				t.Errorf("Runner.Run() error = %v, wantErr %v", gotError, tt.wantErr)
			}

			// If we expected success, verify session exists/persists
			if !tt.wantErr {
				_, err = sessionService.Get(ctx, &session.GetRequest{
					AppName:   appName,
					UserID:    userID,
					SessionID: sessionID,
				})
				if err != nil {
					t.Errorf("expected session to exist, but got error: %v", err)
				}
			}
		})
	}
}

// getErrSessionService fails Get with getErr and counts Create calls.
type getErrSessionService struct {
	session.Service
	getErr  error
	creates int
}

func (s *getErrSessionService) Get(context.Context, *session.GetRequest) (*session.GetResponse, error) {
	return nil, s.getErr
}

func (s *getErrSessionService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	s.creates++
	return s.Service.Create(ctx, req)
}

func TestRunner_AutoCreateSession_GetError(t *testing.T) {
	t.Parallel()

	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	}))
	errTransient := errors.New("connection refused")
	// A third-party service that returns its own "not found" without wrapping
	// session.ErrNotFound.
	errPlainNotFound := errors.New("not found")

	tests := []struct {
		name              string
		autoCreateSession bool
		getErr            error
		wantCreates       int
		wantErr           error
	}{
		{
			name:              "transient_error_is_returned_without_create",
			autoCreateSession: true,
			getErr:            errTransient,
			wantCreates:       0,
			wantErr:           errTransient,
		},
		{
			name:              "unwrapped_not_found_is_returned_without_create",
			autoCreateSession: true,
			getErr:            errPlainNotFound,
			wantCreates:       0,
			wantErr:           errPlainNotFound,
		},
		{
			name:              "wrapped_not_found_creates_once",
			autoCreateSession: true,
			getErr:            fmt.Errorf("%w: %q", session.ErrNotFound, "testSession"),
			wantCreates:       1,
		},
		{
			name:              "auto_create_false_returns_not_found",
			autoCreateSession: false,
			getErr:            session.ErrNotFound,
			wantCreates:       0,
			wantErr:           session.ErrNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := &getErrSessionService{Service: session.InMemoryService(), getErr: tt.getErr}
			r, err := New(Config{
				AppName:           "testApp",
				Agent:             testAgent,
				SessionService:    svc,
				AutoCreateSession: tt.autoCreateSession,
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			var gotErr error
			msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
			for _, err := range r.Run(t.Context(), "testUser", "testSession", msg, agent.RunConfig{}) {
				if err != nil {
					gotErr = err
				}
			}

			if !errors.Is(gotErr, tt.wantErr) {
				t.Errorf("Runner.Run() error = %v, want %v", gotErr, tt.wantErr)
			}
			if svc.creates != tt.wantCreates {
				t.Errorf("Create calls = %d, want %d", svc.creates, tt.wantCreates)
			}
		})
	}
}

// TestRunner_AutoCreateSession_EmptySessionID checks that an empty session ID
// still creates a session with a generated ID. Get rejects the empty ID with a
// validation error rather than session.ErrNotFound.
func TestRunner_AutoCreateSession_EmptySessionID(t *testing.T) {
	t.Parallel()

	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	}))
	svc := session.InMemoryService()
	r, err := New(Config{
		AppName:           "testApp",
		Agent:             testAgent,
		SessionService:    svc,
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
	for _, err := range r.Run(t.Context(), "testUser", "", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Runner.Run() with an empty session ID error = %v", err)
		}
	}

	resp, err := svc.List(t.Context(), &session.ListRequest{AppName: "testApp", UserID: "testUser"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(resp.Sessions) != 1 {
		t.Fatalf("List() returned %d sessions, want 1", len(resp.Sessions))
	}
	if resp.Sessions[0].ID() == "" {
		t.Error("created session has an empty ID, want a generated one")
	}
}

func TestRunner_NilEventYieldedTerminatesRun(t *testing.T) {
	nilAgent, err := agent.New(agent.Config{
		Name: "nil_flood",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for {
					if !yield(nil, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	sessionService := session.InMemoryService()
	_, err = sessionService.Create(context.Background(), &session.CreateRequest{
		AppName:   "test_app",
		UserID:    "user1",
		SessionID: "session1",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	r, err := New(Config{
		AppName:        "test_app",
		Agent:          nilAgent,
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	type result struct {
		err        error
		iterations int
	}
	done := make(chan result, 1)
	go func() {
		got := result{}
		for _, err := range r.Run(context.Background(), "user1", "session1", genai.NewContentFromText("hello", genai.RoleUser), agent.RunConfig{}) {
			got.err = err
			got.iterations++
			break
		}
		done <- got
	}()

	select {
	case got := <-done:
		if got.iterations != 1 {
			t.Fatalf("Run() iterations = %d, want 1", got.iterations)
		}
		if got.err == nil {
			t.Fatal("Run() yielded no error for a nil event")
		}
		if !strings.Contains(got.err.Error(), "nil_flood") {
			t.Fatalf("Run() error = %v, want the agent name", got.err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run() did not terminate after a nil event")
	}
}

// newTestNode returns a function node named name that returns no output.
func newTestNode(name string) workflow.Node {
	return workflow.NewFunctionNode(name, func(agent.Context, any) (any, error) { return nil, nil }, workflow.NodeConfig{})
}

func TestRunner_New_MutuallyExclusiveRoots(t *testing.T) {
	a := must(agent.New(agent.Config{Name: "a"}))

	r, err := New(Config{Agent: a, Node: newTestNode("myNode"), SessionService: session.InMemoryService()})

	if r != nil || err == nil || err.Error() != "only one of Agent or Node may be provided" {
		t.Errorf("New() = %v, %v; want nil, %q", r, err, "only one of Agent or Node may be provided")
	}
}

func TestRunner_New_NodeNameRequired(t *testing.T) {
	r, err := New(Config{Node: newTestNode(""), SessionService: session.InMemoryService()})

	if r != nil || err == nil || err.Error() != "node name must not be empty" {
		t.Errorf("New() = %v, %v; want nil, %q", r, err, "node name must not be empty")
	}
}

func TestRunner_New_NodeRootCompactionNeedsSummarizer(t *testing.T) {
	const want = "the Compaction config needs a Summarizer: root is a node, so no default model is available"

	r, err := New(Config{
		Node:           newTestNode("myNode"),
		SessionService: session.InMemoryService(),
		Compaction:     &compaction.Config{CompactionInterval: 2, OverlapSize: 1},
	})

	if r != nil || err == nil || err.Error() != want {
		t.Errorf("New() = %v, %v; want nil, %q", r, err, want)
	}
}

func TestRunner_New_DefaultsAppNameToNodeName(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	r, err := New(Config{Node: newTestNode("myNode"), SessionService: svc, AutoCreateSession: true})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	for _, err := range r.Run(ctx, "u", "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "myNode", UserID: "u", SessionID: "s"}); err != nil {
		t.Errorf("session under app name %q: Get() error = %v", "myNode", err)
	}
}

func TestRunner_RunLive_RefusedOnNodeRoot(t *testing.T) {
	const want = `runner root "myNode" is a node, not an agent`
	r, err := New(Config{AppName: "app", Node: newTestNode("myNode"), SessionService: session.InMemoryService()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// The session does not exist, so the refusal has to come before the
	// session is read.
	sess, events, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})

	if sess != nil || events != nil || err == nil || err.Error() != want {
		t.Errorf("RunLive() = %v, %v, %v; want nil, nil, %q", sess, events, err, want)
	}
}

// The abort tests below run roots that are still working when the caller
// closes the abort signal. A root waits abortBlockFor when the abort does not
// cancel its context, and a run must end within abortPrompt, so a missing
// cancellation fails a test instead of passing slowly.
const (
	abortBlockFor = 30 * time.Second
	abortPrompt   = 10 * time.Second
	abortedCode   = "INVOCATION_ABORTED"
)

// blockingCallAgent is a custom agent whose first run yields a function call
// to _slow_tool with ID call_1 and then waits for its context to end. Later
// runs yield the text "follow-up".
type blockingCallAgent struct {
	name     string
	yieldErr bool          // yield ctx.Err() when the wait ends by cancellation
	runs     atomic.Int32  // number of runs started
	blocked  chan struct{} // receives once, when the first run starts waiting
}

func newBlockingCallAgent(t *testing.T, name string, yieldErr bool) (*blockingCallAgent, agent.Agent) {
	t.Helper()
	b := &blockingCallAgent{name: name, yieldErr: yieldErr, blocked: make(chan struct{}, 1)}
	a, err := agent.New(agent.Config{Name: name, Run: b.run})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return b, a
}

func (b *blockingCallAgent) run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = b.name
		if b.runs.Add(1) > 1 {
			ev.Content = genai.NewContentFromText("follow-up", genai.RoleModel)
			yield(ev, nil)
			return
		}
		ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "_slow_tool"}},
		}}
		if !yield(ev, nil) {
			return
		}
		b.blocked <- struct{}{}
		select {
		case <-ctx.Done():
			if b.yieldErr {
				yield(nil, ctx.Err())
			}
		case <-time.After(abortBlockFor):
		}
	}
}

// slowToolProbe is a tool named _slow_tool that waits for its context to end.
type slowToolProbe struct {
	started chan struct{} // receives when a call starts waiting
	ctxErr  chan error    // receives ctx.Err() when the wait ends by cancellation
}

func newSlowTool(t *testing.T) (*slowToolProbe, tool.Tool) {
	t.Helper()
	p := &slowToolProbe{started: make(chan struct{}, 1), ctxErr: make(chan error, 1)}
	tl, err := functiontool.New(functiontool.Config{Name: "_slow_tool", Description: "waits"}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		p.started <- struct{}{}
		select {
		case <-ctx.Done():
			p.ctxErr <- ctx.Err()
			return nil, ctx.Err()
		case <-time.After(abortBlockFor):
			return map[string]any{"result": "done"}, nil
		}
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return p, tl
}

// callThenTextModel answers its first request with a call to toolName with ID
// call_1, and every later request with the text "Recovered". It records the
// requests it receives.
type callThenTextModel struct {
	toolName string
	mu       sync.Mutex
	reqs     []*model.LLMRequest
}

func (*callThenTextModel) Name() string { return "call-then-text" }

func (m *callThenTextModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.reqs = append(m.reqs, req)
	n := len(m.reqs)
	m.mu.Unlock()
	resp := &model.LLMResponse{Content: genai.NewContentFromText("Recovered", genai.RoleModel)}
	if n == 1 {
		resp = &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: m.toolName, Args: map[string]any{}}},
		}}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(resp, nil)
	}
}

func (m *callThenTextModel) requests() []*model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.reqs)
}

// newSlowToolAgent returns an LLM agent named name whose model calls
// _slow_tool once and then answers "Recovered".
func newSlowToolAgent(t *testing.T, name string) (*slowToolProbe, *callThenTextModel, agent.Agent) {
	t.Helper()
	probe, tl := newSlowTool(t)
	m := &callThenTextModel{toolName: "_slow_tool"}
	a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Tools: []tool.Tool{tl}})
	if err != nil {
		t.Fatalf("llmagent.New(%q) error = %v", name, err)
	}
	return probe, m, a
}

// newAbortRunner returns a runner for cfg under app "app" that creates its
// sessions, with an in-memory session service unless cfg names one.
func newAbortRunner(t *testing.T, cfg Config) (*Runner, session.Service) {
	t.Helper()
	cfg.AppName = "app"
	cfg.AutoCreateSession = true
	if cfg.SessionService == nil {
		cfg.SessionService = session.InMemoryService()
	}
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r, cfg.SessionService
}

// abortRun ranges one Run of user "u" with WithAbortSignal(sig).
type abortRun struct {
	ctx     context.Context           // t.Context() when nil
	session string                    // "s" when empty
	cfg     agent.RunConfig           //
	msg     *genai.Content            // "Run" when nil
	sig     chan struct{}             //
	closeOn func(*session.Event) bool // closes sig in the loop body at the first event it matches
	stop    bool                      // stops ranging right after closeOn closed sig
	opts    []RunOption               // added after WithAbortSignal
}

type abortResult struct {
	events []*session.Event
	errs   []error
	took   time.Duration
}

func (a abortRun) drive(t *testing.T, r *Runner) abortResult {
	t.Helper()
	ctx := a.ctx
	if ctx == nil {
		ctx = t.Context()
	}
	msg := a.msg
	if msg == nil {
		msg = genai.NewContentFromText("Run", genai.RoleUser)
	}
	sessionID := a.session
	if sessionID == "" {
		sessionID = "s"
	}
	var res abortResult
	closed := false
	start := time.Now()
	for ev, err := range r.Run(ctx, "u", sessionID, msg, a.cfg, append([]RunOption{WithAbortSignal(a.sig)}, a.opts...)...) {
		if err != nil {
			res.errs = append(res.errs, err)
			continue
		}
		res.events = append(res.events, ev)
		if !closed && a.closeOn != nil && a.closeOn(ev) {
			close(a.sig)
			closed = true
			if a.stop {
				break
			}
		}
	}
	res.took = time.Since(start)
	return res
}

// closeWhen closes sig from another goroutine once ready receives, as a caller
// that aborts from outside its range loop does.
func closeWhen(t *testing.T, ready <-chan struct{}, sig chan struct{}) {
	go func() {
		select {
		case <-ready:
			close(sig)
		case <-t.Context().Done():
		}
	}()
}

// hasCall reports whether ev holds a function call.
func hasCall(ev *session.Event) bool {
	return len(utils.FunctionCalls(ev.Content)) > 0
}

// describeAbortEvent names ev by author, error code and message, function
// calls, function responses and text, so a test can compare whole events with
// literals.
func describeAbortEvent(ev *session.Event) string {
	s := ev.Author
	if ev.ErrorCode != "" || ev.ErrorMessage != "" {
		s += fmt.Sprintf(" %s %q", ev.ErrorCode, ev.ErrorMessage)
	}
	if ev.Content == nil {
		return s + " no-content"
	}
	for _, fc := range utils.FunctionCalls(ev.Content) {
		s += fmt.Sprintf(" call %s %s", fc.ID, fc.Name)
	}
	for _, fr := range utils.FunctionResponses(ev.Content) {
		s += fmt.Sprintf(" response %s %s %v", fr.ID, fr.Name, fr.Response)
	}
	var text string
	for _, p := range ev.Content.Parts {
		text += p.Text
	}
	if text != "" {
		s += fmt.Sprintf(" text %q", text)
	}
	return s
}

func describeAbortEvents(evs []*session.Event) []string {
	out := []string{}
	for _, ev := range evs {
		out = append(out, describeAbortEvent(ev))
	}
	return out
}

// abortedEvents returns the events of evs that seal an aborted invocation.
func abortedEvents(evs []*session.Event) []*session.Event {
	var out []*session.Event
	for _, ev := range evs {
		if ev.ErrorCode == abortedCode {
			out = append(out, ev)
		}
	}
	return out
}

// storedAbortSession returns the events of session id of user "u" in app "app".
func storedAbortSession(t *testing.T, svc session.Service, id string) []*session.Event {
	t.Helper()
	resp, err := svc.Get(context.Background(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: id})
	if err != nil {
		t.Fatalf("Get(%q) error = %v", id, err)
	}
	return slices.Collect(resp.Session.Events().All())
}

// textModel answers every request with the text "hello" and counts the calls.
type textModel struct{ calls *atomic.Int32 }

func (textModel) Name() string { return "text" }

func (m textModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel)}, nil)
	}
}

func TestRunner_Abort_PreSignaledAndNilSignal(t *testing.T) {
	t.Parallel()
	roots := []struct {
		name  string
		build func(t *testing.T, runs *atomic.Int32) Config
	}{
		{
			name: "custom",
			build: func(t *testing.T, runs *atomic.Int32) Config {
				a, err := agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						runs.Add(1)
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "root"
						ev.Content = genai.NewContentFromText("hello", genai.RoleModel)
						yield(ev, nil)
					}
				}})
				if err != nil {
					t.Fatalf("agent.New() error = %v", err)
				}
				return Config{Agent: a}
			},
		},
		{
			name: "llm",
			build: func(t *testing.T, runs *atomic.Int32) Config {
				return Config{Agent: must(llmagent.New(llmagent.Config{Name: "root", Model: textModel{calls: runs}}))}
			},
		},
		{
			name: "node_root",
			build: func(t *testing.T, runs *atomic.Int32) Config {
				return Config{Node: workflow.NewFunctionNode("root", func(agent.Context, any) (string, error) {
					runs.Add(1)
					return "hello", nil
				}, workflow.NodeConfig{})}
			},
		},
	}
	signals := []struct {
		name       string
		sig        func() chan struct{}
		wantRuns   int32
		wantSealed []string // the abort events, yielded and stored
	}{
		{
			name:       "pre_closed",
			sig:        func() chan struct{} { ch := make(chan struct{}); close(ch); return ch },
			wantRuns:   0,
			wantSealed: []string{`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`},
		},
		{
			name:       "send_struct",
			sig:        func() chan struct{} { ch := make(chan struct{}, 1); ch <- struct{}{}; return ch },
			wantRuns:   0,
			wantSealed: []string{`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`},
		},
		{
			name:       "nil_channel",
			sig:        func() chan struct{} { return nil },
			wantRuns:   1,
			wantSealed: []string{},
		},
	}
	for _, root := range roots {
		for _, s := range signals {
			t.Run(root.name+"/"+s.name, func(t *testing.T) {
				var runs atomic.Int32
				r, svc := newAbortRunner(t, root.build(t, &runs))

				res := abortRun{sig: s.sig()}.drive(t, r)

				if len(res.errs) != 0 {
					t.Fatalf("Run() errors = %v, want none", res.errs)
				}
				if got := runs.Load(); got != s.wantRuns {
					t.Errorf("root ran %d times, want %d", got, s.wantRuns)
				}
				if diff := cmp.Diff(s.wantSealed, describeAbortEvents(abortedEvents(res.events))); diff != "" {
					t.Errorf("yielded abort events mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(s.wantSealed, describeAbortEvents(abortedEvents(storedAbortSession(t, svc, "s")))); diff != "" {
					t.Errorf("stored abort events mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

// TestRunner_Abort_PreSignaledResumeRunsNoNode checks that a turn that
// resumes a paused node root, with a signal that fired before the run, runs
// no node: a node that runs again on resume and ignores its context does not
// run, and the turn stores the user message and the seal. The workflow starts
// a node that runs again on resume before it looks at its context, so only
// the runner's own check before the workflow starts keeps it from running, as
// adk-python's runner cancels the root task before it has had a turn
// (runners.py, _consume_event_queue and _cleanup_root_task).
func TestRunner_Abort_PreSignaledResumeRunsNoNode(t *testing.T) {
	t.Parallel()
	signals := []struct {
		name string
		sig  func() chan struct{}
	}{
		{name: "pre_closed", sig: func() chan struct{} { ch := make(chan struct{}); close(ch); return ch }},
		{name: "send_struct", sig: func() chan struct{} { ch := make(chan struct{}, 1); ch <- struct{}{}; return ch }},
	}
	for _, s := range signals {
		t.Run(s.name, func(t *testing.T) {
			var resumedRuns atomic.Int32
			root := &abortGraphNode{
				BaseNode: workflow.NewBaseNode("root", "", workflow.NodeConfig{RerunOnResume: new(true)}),
				fn: func(ctx agent.Context, yield func(*session.Event, error) bool) {
					if _, ok := ctx.ResumedInput("approval"); ok {
						resumedRuns.Add(1)
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Output = "approved"
						yield(ev, nil)
						return
					}
					yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "approval", Message: "decide"}), nil)
				},
			}
			r, svc := newAbortRunner(t, Config{Node: root})
			paused := abortRun{sig: nil}.drive(t, r)
			if len(paused.errs) != 0 || len(paused.events) != 1 {
				t.Fatalf("first turn: errors = %v, events = %v, want no errors and the input request", paused.errs, describeAbortEvents(paused.events))
			}
			answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID:       "approval",
				Name:     workflow.WorkflowInputFunctionCallName,
				Response: map[string]any{"payload": "yes"},
			}}}}

			res := abortRun{msg: answer, sig: s.sig()}.drive(t, r)

			if len(res.errs) != 0 {
				t.Fatalf("Run() errors = %v, want none", res.errs)
			}
			if got := resumedRuns.Load(); got != 0 {
				t.Errorf("root ran %d times on resume, want 0", got)
			}
			want := []string{`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`}
			if diff := cmp.Diff(want, describeAbortEvents(res.events)); diff != "" {
				t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
			}
			stored := describeAbortEvents(storedAbortSession(t, svc, "s"))
			wantStored := []string{
				`user response approval adk_request_input map[payload:yes]`,
				`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
			}
			if len(stored) < 2 {
				t.Fatalf("stored events = %v, want at least the resume turn's 2", stored)
			}
			if diff := cmp.Diff(wantStored, stored[len(stored)-2:]); diff != "" {
				t.Errorf("stored events of the resume turn mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRunner_Abort_StopsAfterTheEventTheCallerAbortedOn checks that a signal
// closed while the caller handles an event stops the run before the next
// event is requested, even from a root that never looks at its context, as
// adk-python breaks after each event (runners.py, Runner.run_async's execute).
func TestRunner_Abort_StopsAfterTheEventTheCallerAbortedOn(t *testing.T) {
	t.Parallel()
	twoEvents := func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		for _, text := range []string{"first", "second"} {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "root"
			ev.Content = genai.NewContentFromText(text, genai.RoleModel)
			if !yield(ev, nil) {
				return
			}
		}
	}
	tests := []struct {
		name   string
		config Config
	}{
		{
			name: "custom",
			config: Config{Agent: must(agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) { twoEvents(ctx, yield) }
			}}))},
		},
		{
			name: "node_root",
			config: Config{Node: &abortGraphNode{BaseNode: workflow.NewBaseNode("root", "", workflow.NodeConfig{}), fn: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				twoEvents(ctx, yield)
			}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, _ := newAbortRunner(t, tt.config)
			first := func(*session.Event) bool { return true }

			res := abortRun{sig: make(chan struct{}), closeOn: first}.drive(t, r)

			if len(res.errs) != 0 {
				t.Errorf("Run() errors = %v, want none", res.errs)
			}
			want := []string{
				`root text "first"`,
				`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
			}
			if diff := cmp.Diff(want, describeAbortEvents(res.events)); diff != "" {
				t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRunner_Abort_ValueSentWhileTheCallerHandlesAnEventStopsTheRun checks
// that one value sent while the caller handles an event is seen by the check
// before the next event is requested, and that the run yields its seal, even
// when the goroutine that receives for the run's watch takes the value: a
// check after the send sees the signal, as a check after adk-python's
// Event.set does (runners.py, Runner.run_async's execute).
func TestRunner_Abort_ValueSentWhileTheCallerHandlesAnEventStopsTheRun(t *testing.T) {
	// One P, and two yields of it before the send: the goroutine that waits
	// on the signal for the run's watch has run and is parked on the channel,
	// so the send hands the value to it and leaves the channel empty, and it
	// cannot trip the state before the caller's goroutine blocks. On the node
	// path the scheduler then wakes that goroutine and the node's goroutine,
	// in an order the race detector randomizes, so each root runs 32 times.
	// Not parallel, as it changes GOMAXPROCS.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	// offered counts the events the root offers; a run that requests no
	// further event after "first" lets the root offer one.
	var offered atomic.Int32
	twoEvents := func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		for _, text := range []string{"first", "second"} {
			offered.Add(1)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "root"
			ev.Content = genai.NewContentFromText(text, genai.RoleModel)
			if !yield(ev, nil) {
				return
			}
		}
	}
	tests := []struct {
		name   string
		config Config
	}{
		{
			name: "custom",
			config: Config{Agent: must(agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) { twoEvents(ctx, yield) }
			}}))},
		},
		{
			name: "node_root",
			config: Config{Node: &abortGraphNode{BaseNode: workflow.NewBaseNode("root", "", workflow.NodeConfig{}), fn: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				twoEvents(ctx, yield)
			}}},
		},
	}
	want := []string{
		`root text "first"`,
		`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
	}
	wantStored := []string{
		`user text "Run"`,
		`root text "first"`,
		`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, svc := newAbortRunner(t, tt.config)
			for i := range 32 {
				offered.Store(0)
				sessionID := fmt.Sprintf("s%d", i)
				sig := make(chan struct{}, 1)

				var got []string
				for ev, err := range r.Run(t.Context(), "u", sessionID, genai.NewContentFromText("Run", genai.RoleUser), agent.RunConfig{}, WithAbortSignal(sig)) {
					if err != nil {
						t.Fatalf("iteration %d: Run() error = %v, want none", i, err)
					}
					got = append(got, describeAbortEvent(ev))
					if len(got) == 1 {
						runtime.Gosched()
						runtime.Gosched()
						sig <- struct{}{}
					}
				}

				if n := offered.Load(); n != 1 {
					t.Fatalf("iteration %d: root offered %d events, want 1: the run requested an event after the abort", i, n)
				}
				if diff := cmp.Diff(want, got); diff != "" {
					t.Fatalf("iteration %d: yielded events mismatch (-want +got):\n%s", i, diff)
				}
				if diff := cmp.Diff(wantStored, describeAbortEvents(storedAbortSession(t, svc, sessionID))); diff != "" {
					t.Fatalf("iteration %d: stored events mismatch (-want +got):\n%s", i, diff)
				}
			}
		})
	}
}

// TestRunner_Abort_FollowUpRunUnderAnAbortedRunIsAborted checks that a run
// started under the context of a run that one sent value aborted, with the
// same channel, is aborted too, also after the first run was sealed: it runs
// no agent and is sealed, as a closed channel aborts every run given it and
// as adk-python's abort event stays set for whoever holds it
// (agents/invocation_context.py, is_aborted).
func TestRunner_Abort_FollowUpRunUnderAnAbortedRunIsAborted(t *testing.T) {
	t.Parallel()
	var secondRuns atomic.Int32
	second, secondSvc := newAbortRunner(t, Config{Agent: must(agent.New(agent.Config{Name: "second", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			secondRuns.Add(1)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "second"
			ev.Content = genai.NewContentFromText("hello", genai.RoleModel)
			yield(ev, nil)
		}
	}}))})
	sig := make(chan struct{}, 1)
	var secondEvents []string
	var secondErrs []error
	p, err := plugin.New(plugin.Config{
		Name: "follow_up",
		// Runs after the first run was sealed, under its context.
		AfterRunCallback: func(ctx agent.InvocationContext) {
			for ev, err := range second.Run(ctx, "u", "s2", genai.NewContentFromText("again", genai.RoleUser), agent.RunConfig{}, WithAbortSignal(sig)) {
				if err != nil {
					secondErrs = append(secondErrs, err)
					continue
				}
				secondEvents = append(secondEvents, describeAbortEvent(ev))
			}
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	var firstRuns atomic.Int32
	first, _ := newAbortRunner(t, Config{
		Agent: must(agent.New(agent.Config{Name: "root", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) { firstRuns.Add(1) }
		}})),
		PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	sig <- struct{}{}

	res := abortRun{session: "s1", sig: sig}.drive(t, first)

	if len(res.errs) != 0 || len(secondErrs) != 0 {
		t.Fatalf("Run() errors = %v, follow-up run errors = %v, want none", res.errs, secondErrs)
	}
	if got := firstRuns.Load(); got != 0 {
		t.Errorf("first root ran %d times, want 0", got)
	}
	if got := secondRuns.Load(); got != 0 {
		t.Errorf("follow-up root ran %d times, want 0", got)
	}
	wantFirst := []string{`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`}
	if diff := cmp.Diff(wantFirst, describeAbortEvents(res.events)); diff != "" {
		t.Errorf("first run events mismatch (-want +got):\n%s", diff)
	}
	wantSecond := []string{`second INVOCATION_ABORTED "Invocation was aborted by client." no-content`}
	if diff := cmp.Diff(wantSecond, secondEvents); diff != "" {
		t.Errorf("follow-up run events mismatch (-want +got):\n%s", diff)
	}
	wantSecondStored := []string{
		`user text "again"`,
		`second INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
	}
	if diff := cmp.Diff(wantSecondStored, describeAbortEvents(storedAbortSession(t, secondSvc, "s2"))); diff != "" {
		t.Errorf("follow-up run stored events mismatch (-want +got):\n%s", diff)
	}
}

// TestRunner_Abort_ValueReceivedByAPollCancelsTheEngineContext checks that a
// value sent on the signal and received by the runner's poll after an event
// cancels the context the agent runs under, so work of the agent that waits
// on ctx.Done() returns, as every reader of adk-python's abort event sees it
// set (agents/invocation_context.py, is_aborted).
func TestRunner_Abort_ValueReceivedByAPollCancelsTheEngineContext(t *testing.T) {
	// One P: the abort watcher goroutine does not run before the run's
	// goroutine blocks, so the value the caller sends stays in the channel
	// until the runner's poll after the event receives it. Not parallel, as it
	// changes GOMAXPROCS.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	var workCanceled atomic.Bool
	root, err := agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			// Work the agent started, such as a tool, that ends when the run's
			// context is canceled.
			workDone := make(chan struct{})
			go func() {
				<-ctx.Done()
				close(workDone)
			}()
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "root"
			ev.Content = genai.NewContentFromText("hello", genai.RoleModel)
			yield(ev, nil)
			select {
			case <-workDone:
				workCanceled.Store(true)
			case <-time.After(5 * time.Second):
			}
		}
	}})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	r, _ := newAbortRunner(t, Config{Agent: root})
	sig := make(chan struct{}, 1)

	var got []string
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("Run", genai.RoleUser), agent.RunConfig{}, WithAbortSignal(sig)) {
		if err != nil {
			t.Fatalf("Run() error = %v, want none", err)
		}
		got = append(got, describeAbortEvent(ev))
		if len(got) == 1 {
			sig <- struct{}{}
		}
	}

	if !workCanceled.Load() {
		t.Error("the agent's work was not canceled within 5s of the poll that received the signal")
	}
	want := []string{
		`root text "hello"`,
		`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
	}
}

// TestRunner_Abort_SealsOnTheRunsBranch checks that the event that seals an
// aborted run with no open call is on the run's branch: the branch of the
// sub-agent whose call the new message answers, as adk-python passes
// ic.branch after restoring it from history (runners.py,
// Runner._synthesize_abort_events_if_needed and
// _setup_context_for_new_invocation).
func TestRunner_Abort_SealsOnTheRunsBranch(t *testing.T) {
	t.Parallel()
	_, worker := newBlockingCallAgent(t, "worker", false)
	root := must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{worker}}}))
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	call := session.NewEvent(t.Context(), "inv_0")
	call.Author = "worker"
	call.Branch = "root.worker"
	call.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "approve"}}}}
	call.LongRunningToolIDs = []string{"call_1"}
	if err := svc.AppendEvent(t.Context(), created.Session, call); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	r, _ := newAbortRunner(t, Config{Agent: root, SessionService: svc})
	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Name: "approve", Response: map[string]any{"ok": true}}}}}
	sig := make(chan struct{})
	close(sig)

	res := abortRun{msg: answer, sig: sig}.drive(t, r)

	if len(res.errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", res.errs)
	}
	var got []string
	for _, ev := range res.events {
		got = append(got, describeAbortEvent(ev)+" branch="+ev.Branch)
	}
	want := []string{`root INVOCATION_ABORTED "Invocation was aborted by client." no-content branch=root.worker`}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
	}
}

func TestRunner_Abort_CompositeAgentsDoNotYieldContextCanceledError(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		root func(t *testing.T, blocker, second agent.Agent) agent.Agent
		want []string
	}{
		{
			name: "sequential",
			root: func(t *testing.T, blocker, second agent.Agent) agent.Agent {
				return must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{blocker, second}}}))
			},
			want: []string{
				`blocker call call_1 _slow_tool`,
				`blocker INVOCATION_ABORTED "Invocation was aborted by client." response call_1 _slow_tool map[error:Invocation was aborted by client.]`,
			},
		},
		{
			name: "loop",
			root: func(t *testing.T, blocker, second agent.Agent) agent.Agent {
				return must(loopagent.New(loopagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{blocker, second}}, MaxIterations: 5}))
			},
			want: []string{
				`blocker call call_1 _slow_tool`,
				`blocker INVOCATION_ABORTED "Invocation was aborted by client." response call_1 _slow_tool map[error:Invocation was aborted by client.]`,
			},
		},
		{
			name: "parallel",
			root: func(t *testing.T, blocker, _ agent.Agent) agent.Agent {
				return must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{blocker}}}))
			},
			want: []string{
				`blocker call call_1 _slow_tool`,
				`blocker INVOCATION_ABORTED "Invocation was aborted by client." response call_1 _slow_tool map[error:Invocation was aborted by client.]`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, blocker := newBlockingCallAgent(t, "blocker", true)
			second, secondAgent := newBlockingCallAgent(t, "second", true)
			r, _ := newAbortRunner(t, Config{Agent: tt.root(t, blocker, secondAgent)})
			sig := make(chan struct{})
			closeWhen(t, b.blocked, sig)

			res := abortRun{sig: sig}.drive(t, r)

			if len(res.errs) != 0 {
				t.Errorf("Run() errors = %v, want none", res.errs)
			}
			if diff := cmp.Diff(tt.want, describeAbortEvents(res.events)); diff != "" {
				t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
			}
			if got := second.runs.Load(); got != 0 {
				t.Errorf("second sub-agent ran %d times, want 0", got)
			}
			if res.took > abortPrompt {
				t.Errorf("Run() took %v, want under %v", res.took, abortPrompt)
			}
		})
	}
}

// sealedCall1 describes the event that seals tool_agent's call call_1 to
// _slow_tool.
const sealedCall1 = `tool_agent INVOCATION_ABORTED "Invocation was aborted by client." response call_1 _slow_tool map[error:Invocation was aborted by client.]`

// abortCheckingService fails an AppendEvent whose context is done, as a
// context-aware session service does, and counts the abort events it is asked
// to store. When failAbort is set it returns failAbort for every abort event
// instead of storing it.
type abortCheckingService struct {
	session.Service
	failAbort    error
	abortAppends atomic.Int32
}

func (s *abortCheckingService) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if ev.ErrorCode == abortedCode {
		s.abortAppends.Add(1)
		if s.failAbort != nil {
			return s.failAbort
		}
	}
	return s.Service.AppendEvent(ctx, sess, ev)
}

// slowCallRoots are the roots named tool_agent that call _slow_tool with ID
// call_1 and then wait for their context to end: a custom agent on the classic
// loop and an LLM agent on the node loop. build returns the root and a channel
// that receives once the root waits.
func slowCallRoots() []struct {
	name  string
	build func(t *testing.T) (agent.Agent, <-chan struct{})
} {
	return []struct {
		name  string
		build func(t *testing.T) (agent.Agent, <-chan struct{})
	}{
		{
			name: "custom",
			build: func(t *testing.T) (agent.Agent, <-chan struct{}) {
				b, a := newBlockingCallAgent(t, "tool_agent", false)
				return a, b.blocked
			},
		},
		{
			name: "llm",
			build: func(t *testing.T) (agent.Agent, <-chan struct{}) {
				probe, _, a := newSlowToolAgent(t, "tool_agent")
				return a, probe.started
			},
		},
	}
}

// errorStrings returns the messages of errs.
func errorStrings(errs []error) []string {
	out := []string{}
	for _, err := range errs {
		out = append(out, err.Error())
	}
	return out
}

func TestRunner_Abort_NormalPathSealingUsesUncancelledContextAndPropagatesErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name             string
		failAbort        error // returned by AppendEvent for an abort event
		onEventErr       error // returned by OnEventCallback for an abort event
		wantErrs         []string
		wantSealed       []string // yielded abort events
		wantAbortAppends int32
	}{
		{
			name:             "ctx_alive_in_append",
			wantErrs:         []string{},
			wantSealed:       []string{sealedCall1},
			wantAbortAppends: 1,
		},
		{
			name:             "append_error_propagates",
			failAbort:        errors.New("simulated db failure"),
			wantErrs:         []string{"failed to add event to session: simulated db failure"},
			wantSealed:       []string{},
			wantAbortAppends: 1,
		},
		{
			name:             "plugin_error_propagates",
			onEventErr:       errors.New("plugin rejected abort event"),
			wantErrs:         []string{"Error in plugin 'reject' during 'on_event_callback' callback: plugin rejected abort event"},
			wantSealed:       []string{},
			wantAbortAppends: 0,
		},
	}
	for _, root := range slowCallRoots() {
		for _, tt := range tests {
			t.Run(root.name+"/"+tt.name, func(t *testing.T) {
				a, ready := root.build(t)
				svc := &abortCheckingService{Service: session.InMemoryService(), failAbort: tt.failAbort}
				p, err := plugin.New(plugin.Config{
					Name: "reject",
					OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
						if ev.ErrorCode == abortedCode {
							return nil, tt.onEventErr
						}
						return nil, nil
					},
				})
				if err != nil {
					t.Fatalf("plugin.New() error = %v", err)
				}
				r, _ := newAbortRunner(t, Config{Agent: a, SessionService: svc, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}})
				sig := make(chan struct{})
				// Closed from outside the range loop once the root waits, so the
				// abort cancels the context the root runs under before sealing.
				closeWhen(t, ready, sig)

				res := abortRun{sig: sig}.drive(t, r)

				if diff := cmp.Diff(tt.wantErrs, errorStrings(res.errs)); diff != "" {
					t.Errorf("Run() errors mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tt.wantSealed, describeAbortEvents(abortedEvents(res.events))); diff != "" {
					t.Errorf("yielded abort events mismatch (-want +got):\n%s", diff)
				}
				if got := svc.abortAppends.Load(); got != tt.wantAbortAppends {
					t.Errorf("AppendEvent got %d abort events, want %d", got, tt.wantAbortAppends)
				}
			})
		}
	}
}

// afterRunRecorder returns a plugin that counts its after-run calls and
// records the error code of the session's last event at each.
func afterRunRecorder(t *testing.T) (*plugin.Plugin, *atomic.Int32, *atomic.Value) {
	t.Helper()
	var calls atomic.Int32
	var lastCode atomic.Value
	p, err := plugin.New(plugin.Config{
		Name: "after_run_recorder",
		AfterRunCallback: func(ic agent.InvocationContext) {
			calls.Add(1)
			evs := ic.Session().Events()
			lastCode.Store(evs.At(evs.Len() - 1).ErrorCode)
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p, &calls, &lastCode
}

func TestRunner_Abort_EarlyBreakSealsSilentlyDropsAppendErrorAndRunsAfterRun(t *testing.T) {
	t.Parallel()
	roots := []struct {
		name       string
		config     func(t *testing.T) Config
		opts       []RunOption
		wantSealed []string // stored abort events when the append succeeds
	}{
		{
			name: "custom",
			config: func(t *testing.T) Config {
				_, a := newBlockingCallAgent(t, "tool_agent", false)
				return Config{Agent: a}
			},
			wantSealed: []string{sealedCall1},
		},
		{
			name: "llm",
			config: func(t *testing.T) Config {
				_, _, a := newSlowToolAgent(t, "tool_agent")
				return Config{Agent: a}
			},
			wantSealed: []string{sealedCall1},
		},
		{
			name:       "node_root_yield_user_message",
			config:     func(t *testing.T) Config { return Config{Node: newTestNode("root")} },
			opts:       []RunOption{WithYieldUserMessage()},
			wantSealed: []string{`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`},
		},
	}
	for _, root := range roots {
		t.Run(root.name+"/seals_on_break", func(t *testing.T) {
			p, afterRuns, lastCode := afterRunRecorder(t)
			cfg := root.config(t)
			cfg.PluginConfig = PluginConfig{Plugins: []*plugin.Plugin{p}}
			r, svc := newAbortRunner(t, cfg)

			res := abortRun{sig: make(chan struct{}), closeOn: func(*session.Event) bool { return true }, stop: true, opts: root.opts}.drive(t, r)

			if len(res.errs) != 0 {
				t.Errorf("Run() errors = %v, want none", res.errs)
			}
			if diff := cmp.Diff([]string{}, describeAbortEvents(abortedEvents(res.events))); diff != "" {
				t.Errorf("yielded abort events mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(root.wantSealed, describeAbortEvents(abortedEvents(storedAbortSession(t, svc, "s")))); diff != "" {
				t.Errorf("stored abort events mismatch (-want +got):\n%s", diff)
			}
			if got := afterRuns.Load(); got != 1 {
				t.Errorf("after-run callbacks ran %d times, want 1", got)
			}
			if got := lastCode.Load(); got != abortedCode {
				t.Errorf("after-run saw last event error code %q, want %q", got, abortedCode)
			}
		})
		t.Run(root.name+"/append_failure_dropped_on_break", func(t *testing.T) {
			p, afterRuns, _ := afterRunRecorder(t)
			svc := &abortCheckingService{Service: session.InMemoryService(), failAbort: errors.New("simulated db failure")}
			cfg := root.config(t)
			cfg.SessionService = svc
			cfg.PluginConfig = PluginConfig{Plugins: []*plugin.Plugin{p}}
			r, _ := newAbortRunner(t, cfg)

			res := abortRun{sig: make(chan struct{}), closeOn: func(*session.Event) bool { return true }, stop: true, opts: root.opts}.drive(t, r)

			if len(res.errs) != 0 {
				t.Errorf("Run() errors = %v, want none", res.errs)
			}
			if got := svc.abortAppends.Load(); got != 1 {
				t.Errorf("AppendEvent got %d abort events, want 1", got)
			}
			if diff := cmp.Diff([]string{}, describeAbortEvents(abortedEvents(storedAbortSession(t, svc, "s")))); diff != "" {
				t.Errorf("stored abort events mismatch (-want +got):\n%s", diff)
			}
			if got := afterRuns.Load(); got != 1 {
				t.Errorf("after-run callbacks ran %d times, want 1", got)
			}
		})
	}
}

// twoChunkModel answers every request with two responses, "chunk 1" and
// "chunk 2", in one model call.
type twoChunkModel struct{}

func (twoChunkModel) Name() string { return "two-chunk" }

func (twoChunkModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		for _, text := range []string{"chunk 1", "chunk 2"} {
			if !yield(&model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)}, nil) {
				return
			}
		}
	}
}

// callThenAfterModel answers every request with two responses in one call:
// a call to _slow_tool with ID call_1, then the text "after".
type callThenAfterModel struct{}

func (callThenAfterModel) Name() string { return "call-then-after" }

func (callThenAfterModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		call := &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "_slow_tool", Args: map[string]any{}}},
		}}}
		if !yield(call, nil) {
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("after", genai.RoleModel)}, nil)
	}
}

// requestResponses describes the function responses in the contents of req.
func requestResponses(req *model.LLMRequest) []string {
	out := []string{}
	for _, c := range req.Contents {
		for _, fr := range utils.FunctionResponses(c) {
			out = append(out, fmt.Sprintf("%s %s %v", fr.ID, fr.Name, fr.Response))
		}
	}
	return out
}

func TestRunner_Abort_FlowStopsBetweenChunksAndCancelsRunningTool(t *testing.T) {
	t.Parallel()
	t.Run("tool_cancelled_and_next_turn_recovers", func(t *testing.T) {
		probe, m, a := newSlowToolAgent(t, "tool_agent")
		r, svc := newAbortRunner(t, Config{Agent: a})
		sig := make(chan struct{})
		closeWhen(t, probe.started, sig)

		res := abortRun{sig: sig}.drive(t, r)

		if len(res.errs) != 0 {
			t.Errorf("Run() errors = %v, want none", res.errs)
		}
		if res.took > abortPrompt {
			t.Errorf("Run() took %v, want under %v", res.took, abortPrompt)
		}
		select {
		case err := <-probe.ctxErr:
			if !errors.Is(err, context.Canceled) {
				t.Errorf("tool saw ctx.Err() = %v, want %v", err, context.Canceled)
			}
		default:
			t.Error("the tool's context was not cancelled")
		}
		wantTurn1 := []string{`tool_agent call call_1 _slow_tool`, sealedCall1}
		if diff := cmp.Diff(wantTurn1, describeAbortEvents(res.events)); diff != "" {
			t.Errorf("turn 1 events mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{`user text "Run"`, `tool_agent call call_1 _slow_tool`, sealedCall1}, describeAbortEvents(storedAbortSession(t, svc, "s"))); diff != "" {
			t.Errorf("stored events after turn 1 mismatch (-want +got):\n%s", diff)
		}

		var turn2 []*session.Event
		for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("Next question", genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("turn 2: Run() error = %v", err)
			}
			turn2 = append(turn2, ev)
		}

		if diff := cmp.Diff([]string{`tool_agent text "Recovered"`}, describeAbortEvents(turn2)); diff != "" {
			t.Errorf("turn 2 events mismatch (-want +got):\n%s", diff)
		}
		reqs := m.requests()
		if len(reqs) != 2 {
			t.Fatalf("model got %d requests, want 2", len(reqs))
		}
		if diff := cmp.Diff([]string{"call_1 _slow_tool map[error:Invocation was aborted by client.]"}, requestResponses(reqs[1])); diff != "" {
			t.Errorf("turn 2 request function responses mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("response_after_cancelled_call_is_dropped", func(t *testing.T) {
		probe, tl := newSlowTool(t)
		inner := must(llmagent.New(llmagent.Config{Name: "tool_agent", Model: callThenAfterModel{}, Tools: []tool.Tool{tl}}))
		root := must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{inner}}}))
		r, svc := newAbortRunner(t, Config{Agent: root})
		sig := make(chan struct{})
		closeWhen(t, probe.started, sig)

		res := abortRun{sig: sig}.drive(t, r)

		if len(res.errs) != 0 {
			t.Errorf("Run() errors = %v, want none", res.errs)
		}
		if diff := cmp.Diff([]string{`tool_agent call call_1 _slow_tool`, sealedCall1}, describeAbortEvents(res.events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{`user text "Run"`, `tool_agent call call_1 _slow_tool`, sealedCall1}, describeAbortEvents(storedAbortSession(t, svc, "s"))); diff != "" {
			t.Errorf("stored events mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("stops_before_second_step", func(t *testing.T) {
		quick, err := functiontool.New(functiontool.Config{Name: "_quick_tool", Description: "returns"}, func(agent.Context, struct{}) (map[string]any, error) {
			return map[string]any{"result": "ok"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New() error = %v", err)
		}
		m := &callThenTextModel{toolName: "_quick_tool"}
		r, _ := newAbortRunner(t, Config{Agent: must(llmagent.New(llmagent.Config{Name: "tool_agent", Model: m, Tools: []tool.Tool{quick}}))})
		closeOnResponse := func(ev *session.Event) bool { return len(utils.FunctionResponses(ev.Content)) > 0 }

		res := abortRun{sig: make(chan struct{}), closeOn: closeOnResponse}.drive(t, r)

		if len(res.errs) != 0 {
			t.Errorf("Run() errors = %v, want none", res.errs)
		}
		want := []string{
			`tool_agent call call_1 _quick_tool`,
			`tool_agent response call_1 _quick_tool map[result:ok]`,
			`tool_agent INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
		}
		if diff := cmp.Diff(want, describeAbortEvents(res.events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
		if got := len(m.requests()); got != 1 {
			t.Errorf("model got %d requests, want 1", got)
		}
	})
	t.Run("stops_between_chunks", func(t *testing.T) {
		r, _ := newAbortRunner(t, Config{Agent: must(llmagent.New(llmagent.Config{Name: "test_agent", Model: twoChunkModel{}}))})
		byTestAgent := func(ev *session.Event) bool { return ev.Author == "test_agent" }

		res := abortRun{sig: make(chan struct{}), closeOn: byTestAgent}.drive(t, r)

		if len(res.errs) != 0 {
			t.Errorf("Run() errors = %v, want none", res.errs)
		}
		want := []string{
			`test_agent text "chunk 1"`,
			`test_agent INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
		}
		if diff := cmp.Diff(want, describeAbortEvents(res.events)); diff != "" {
			t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
		}
	})
}

// waitForCancel signals ready, waits for ctx to end and reports ctx.Err() on
// observed. It returns ctx.Err(), or nil when ctx outlives abortBlockFor.
func waitForCancel(ctx context.Context, ready chan<- struct{}, observed chan<- error) error {
	ready <- struct{}{}
	select {
	case <-ctx.Done():
		observed <- ctx.Err()
		return ctx.Err()
	case <-time.After(abortBlockFor):
		return nil
	}
}

func TestRunner_Abort_CallbacksAndCustomNodesObserveContextDone(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		build func(t *testing.T, ready chan<- struct{}, observed chan<- error) Config
	}{
		{
			name: "before_model_callback",
			build: func(t *testing.T, ready chan<- struct{}, observed chan<- error) Config {
				var calls atomic.Int32
				return Config{Agent: must(llmagent.New(llmagent.Config{
					Name:  "root",
					Model: textModel{calls: &calls},
					BeforeModelCallbacks: []llmagent.BeforeModelCallback{
						func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
							return nil, waitForCancel(ctx, ready, observed)
						},
					},
				}))}
			},
		},
		{
			name: "function_node",
			build: func(t *testing.T, ready chan<- struct{}, observed chan<- error) Config {
				return Config{Node: workflow.NewFunctionNode("root", func(ctx agent.Context, _ any) (any, error) {
					return nil, waitForCancel(ctx, ready, observed)
				}, workflow.NodeConfig{})}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ready := make(chan struct{}, 1)
			observed := make(chan error, 1)
			r, _ := newAbortRunner(t, tt.build(t, ready, observed))
			sig := make(chan struct{})
			closeWhen(t, ready, sig)

			res := abortRun{sig: sig}.drive(t, r)

			if len(res.errs) != 0 {
				t.Errorf("Run() errors = %v, want none", res.errs)
			}
			select {
			case err := <-observed:
				if !errors.Is(err, context.Canceled) {
					t.Errorf("ctx.Err() = %v, want %v", err, context.Canceled)
				}
			default:
				t.Error("the context was not cancelled")
			}
			want := []string{`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`}
			if diff := cmp.Diff(want, describeAbortEvents(res.events)); diff != "" {
				t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// abortGraphNode is a workflow node that runs fn.
type abortGraphNode struct {
	workflow.BaseNode
	fn func(ctx agent.Context, yield func(*session.Event, error) bool)
}

func (n *abortGraphNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) { n.fn(ctx, yield) }
}

func TestRunner_Abort_WorkflowStopsSchedulingAndCancelsParallelNodes(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		config func(t *testing.T, edges []workflow.Edge) Config
	}{
		{
			name: "workflowagent",
			config: func(t *testing.T, edges []workflow.Edge) Config {
				a, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: edges})
				if err != nil {
					t.Fatalf("workflowagent.New() error = %v", err)
				}
				return Config{Agent: a}
			},
		},
		{
			name: "node_root",
			config: func(t *testing.T, edges []workflow.Edge) Config {
				n, err := workflow.NewWorkflowNode("root", edges)
				if err != nil {
					t.Fatalf("workflow.NewWorkflowNode() error = %v", err)
				}
				return Config{Node: n}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bWaiting := make(chan struct{})
			bEnded := make(chan error, 1)
			var cRuns atomic.Int32
			fanoutA := &abortGraphNode{BaseNode: workflow.NewBaseNode("fanout_a", "", workflow.NodeConfig{}), fn: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				<-bWaiting
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "fanout_a"
				ev.Content = genai.NewContentFromText("a", genai.RoleModel)
				if !yield(ev, nil) {
					return
				}
				select {
				case <-ctx.Done():
				case <-time.After(abortBlockFor):
				}
			}}
			fanoutB := &abortGraphNode{BaseNode: workflow.NewBaseNode("fanout_b", "", workflow.NodeConfig{}), fn: func(ctx agent.Context, _ func(*session.Event, error) bool) {
				close(bWaiting)
				select {
				case <-ctx.Done():
					bEnded <- ctx.Err()
				case <-time.After(abortBlockFor):
					bEnded <- nil
				}
			}}
			successorC := workflow.NewFunctionNode("successor_c", func(agent.Context, any) (any, error) {
				cRuns.Add(1)
				return nil, nil
			}, workflow.NodeConfig{})
			edges := []workflow.Edge{
				{From: workflow.Start, To: fanoutA},
				{From: workflow.Start, To: fanoutB},
				{From: fanoutA, To: successorC},
			}
			r, _ := newAbortRunner(t, tt.config(t, edges))
			byFanoutA := func(ev *session.Event) bool { return ev.Author == "fanout_a" }

			res := abortRun{sig: make(chan struct{}), closeOn: byFanoutA}.drive(t, r)

			if len(res.errs) != 0 {
				t.Errorf("Run() errors = %v, want none", res.errs)
			}
			want := []string{
				`fanout_a text "a"`,
				`root INVOCATION_ABORTED "Invocation was aborted by client." no-content`,
			}
			if diff := cmp.Diff(want, describeAbortEvents(res.events)); diff != "" {
				t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
			}
			select {
			case err := <-bEnded:
				if !errors.Is(err, context.Canceled) {
					t.Errorf("fanout_b ended with ctx.Err() = %v, want %v", err, context.Canceled)
				}
			case <-time.After(abortPrompt):
				t.Error("fanout_b did not end")
			}
			if got := cRuns.Load(); got != 0 {
				t.Errorf("successor_c ran %d times, want 0", got)
			}
		})
	}
}

func TestRunner_Abort_CallerContextCancelVsAbortSignalAndPostRunSignal(t *testing.T) {
	t.Parallel()
	t.Run("signal_after_run_ends", func(t *testing.T) {
		root, err := agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "root"
				ev.Content = genai.NewContentFromText("ok", genai.RoleModel)
				yield(ev, nil)
			}
		}})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		r, svc := newAbortRunner(t, Config{Agent: root})
		sig := make(chan struct{})
		res := abortRun{sig: sig}.drive(t, r)
		if len(res.errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", res.errs)
		}

		close(sig)

		want := []string{`user text "Run"`, `root text "ok"`}
		if diff := cmp.Diff(want, describeAbortEvents(storedAbortSession(t, svc, "s"))); diff != "" {
			t.Errorf("stored events mismatch (-want +got):\n%s", diff)
		}
	})
	for _, root := range slowCallRoots() {
		t.Run("caller_ctx_and_abort_both_fire/"+root.name, func(t *testing.T) {
			a, _ := root.build(t)
			p, afterRuns, _ := afterRunRecorder(t)
			svc := &abortCheckingService{Service: session.InMemoryService()}
			r, _ := newAbortRunner(t, Config{Agent: a, SessionService: svc, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}})
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			sig := make(chan struct{})

			start := time.Now()
			for ev, err := range r.Run(ctx, "u", "s", genai.NewContentFromText("Run", genai.RoleUser), agent.RunConfig{}, WithAbortSignal(sig)) {
				if err == nil && hasCall(ev) {
					close(sig)
					cancel()
					break
				}
			}

			if took := time.Since(start); took > abortPrompt {
				t.Errorf("Run() took %v, want under %v", took, abortPrompt)
			}
			if diff := cmp.Diff([]string{sealedCall1}, describeAbortEvents(abortedEvents(storedAbortSession(t, svc, "s")))); diff != "" {
				t.Errorf("stored abort events mismatch (-want +got):\n%s", diff)
			}
			if got := afterRuns.Load(); got != 1 {
				t.Errorf("after-run callbacks ran %d times, want 1", got)
			}
		})
	}
	t.Run("concurrent_runs_isolated", func(t *testing.T) {
		release := make(chan struct{})
		root, err := agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "root"
				if ctx.Session().ID() == "s1" {
					ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "_slow_tool"}}}}
					if yield(ev, nil) {
						select {
						case <-ctx.Done():
						case <-time.After(abortBlockFor):
						}
					}
					return
				}
				select {
				case <-release:
				case <-ctx.Done():
					yield(nil, ctx.Err())
					return
				}
				ev.Content = genai.NewContentFromText("ok", genai.RoleModel)
				yield(ev, nil)
			}
		}})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		r, svc := newAbortRunner(t, Config{Agent: root})
		s2Done := make(chan abortResult, 1)
		go func() {
			s2Done <- abortRun{session: "s2", sig: make(chan struct{})}.drive(t, r)
		}()

		res1 := abortRun{session: "s1", sig: make(chan struct{}), closeOn: hasCall}.drive(t, r)
		close(release)
		res2 := <-s2Done

		if len(res1.errs) != 0 || len(res2.errs) != 0 {
			t.Errorf("Run() errors = %v and %v, want none", res1.errs, res2.errs)
		}
		wantS1 := []string{`root INVOCATION_ABORTED "Invocation was aborted by client." response call_1 _slow_tool map[error:Invocation was aborted by client.]`}
		if diff := cmp.Diff(wantS1, describeAbortEvents(abortedEvents(storedAbortSession(t, svc, "s1")))); diff != "" {
			t.Errorf("s1 stored abort events mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{`user text "Run"`, `root text "ok"`}, describeAbortEvents(storedAbortSession(t, svc, "s2"))); diff != "" {
			t.Errorf("s2 stored events mismatch (-want +got):\n%s", diff)
		}
	})
}
