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
	"fmt"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/model"
	"google.golang.org/adk/platform"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
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
	ctx := context.Background()
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
// inline blobs with placeholders edits the stored copy, not the caller's Parts.
func TestRunner_SaveInputBlobsDoesNotMutateCallerMessage(t *testing.T) {
	ctx := t.Context()
	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	}))
	sessionService := session.InMemoryService()
	r, err := New(Config{AppName: "app", Agent: testAgent, SessionService: sessionService})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	r.artifactService = artifact.InMemoryService()
	if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}

	blob := &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream", Data: []byte("data")}}
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromText("file"), blob}}
	for _, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{SaveInputBlobsAsArtifacts: true}) {
		if err != nil {
			t.Fatalf("r.Run() returned an error: %v", err)
		}
	}
	if msg.Parts[1] != blob || blob.InlineData == nil {
		t.Errorf("caller's message was modified: Parts[1] = %+v, want the original blob part", msg.Parts[1])
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

type scriptedModel struct {
	responses []*genai.Content
	call      int
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		i := m.call
		if i >= len(m.responses) {
			i = len(m.responses) - 1
		}
		m.call++
		yield(&model.LLMResponse{Content: m.responses[i]}, nil)
	}
}

func userText(text string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: text}}}
}

// TestRunner_StampsEventsYieldedWithoutTimestamp checks that an event built as
// a struct literal, with no timestamp, is stored with the run's current time,
// whether an agent yielded it or a plugin returned it in place of the agent's
// event, under both a sequential root and an LlmAgent root that transfers to
// the agent. An event that already carries a timestamp keeps it.
func TestRunner_StampsEventsYieldedWithoutTimestamp(t *testing.T) {
	const appName, userID, sessionID = "testApp", "testUser", "testSession"
	runTime := time.Date(2026, time.March, 2, 10, 0, 0, 0, time.UTC)
	ownTime := time.Date(2026, time.March, 1, 9, 0, 0, 0, time.UTC)

	newCustom := func(t *testing.T, own time.Time) agent.Agent {
		a, err := agent.New(agent.Config{
			Name: "custom",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(&session.Event{
						Author:      "custom",
						Timestamp:   own,
						LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("reply", genai.RoleModel)},
					}, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		return a
	}

	newSequentialRoot := func(t *testing.T, own time.Time) agent.Agent {
		a, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{newCustom(t, own)}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New() error = %v", err)
		}
		return a
	}
	newLlmRoot := func(t *testing.T, own time.Time) agent.Agent {
		m := &scriptedModel{responses: []*genai.Content{{
			Role: genai.RoleModel,
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "transfer-1", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "custom"},
			}}},
		}}}
		a, err := llmagent.New(llmagent.Config{Name: "llm_root", Model: m, SubAgents: []agent.Agent{newCustom(t, own)}})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		return a
	}
	tests := []struct {
		name string
		root func(t *testing.T, own time.Time) agent.Agent
		// own is the timestamp the custom agent gives its event; zero means
		// none.
		own time.Time
		// replace makes a plugin swap the custom agent's event for a struct
		// literal copy, which has no timestamp whatever the original carried.
		replace bool
		want    time.Time
	}{
		{name: "sequential agent root", root: newSequentialRoot, want: runTime},
		{name: "llm agent root", root: newLlmRoot, want: runTime},
		{name: "sequential agent root, plugin replacement", root: newSequentialRoot, replace: true, want: runTime},
		{name: "llm agent root, plugin replacement", root: newLlmRoot, replace: true, want: runTime},
		{name: "sequential agent root, own timestamp", root: newSequentialRoot, own: ownTime, want: ownTime},
		{name: "llm agent root, own timestamp", root: newLlmRoot, own: ownTime, want: ownTime},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return runTime })
			svc := session.InMemoryService()
			if _, err := svc.Create(ctx, &session.CreateRequest{
				AppName:   appName,
				UserID:    userID,
				SessionID: sessionID,
			}); err != nil {
				t.Fatalf("Create() error = %v", err)
			}
			// Plugins see the event as it will be stored, timestamp included.
			sawUnstamped := false
			p, err := plugin.New(plugin.Config{
				Name: "observe_and_replace",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev.Author != "custom" {
						return nil, nil
					}
					sawUnstamped = sawUnstamped || ev.Timestamp.IsZero()
					if !tt.replace {
						return nil, nil
					}
					return &session.Event{Author: ev.Author, LLMResponse: ev.LLMResponse}, nil
				},
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			r, err := New(Config{
				AppName:        appName,
				Agent:          tt.root(t, tt.own),
				SessionService: svc,
				PluginConfig:   PluginConfig{Plugins: []*plugin.Plugin{p}},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			for _, err := range r.Run(ctx, userID, sessionID, userText("go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
			}

			got, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			var sawReply bool
			for ev := range got.Session.Events().All() {
				if ev.Author != "custom" {
					continue
				}
				sawReply = true
				if !ev.Timestamp.Equal(tt.want) {
					t.Errorf("stored event from %q has timestamp %v, want %v", ev.Author, ev.Timestamp, tt.want)
				}
			}
			if !sawReply {
				t.Fatal("no event from the custom agent was stored")
			}
			if got.Session.LastUpdateTime().IsZero() {
				t.Error("session LastUpdateTime() is zero after the run")
			}
			if sawUnstamped {
				t.Error("OnEventCallback saw the custom agent's event with a zero timestamp")
			}
		})
	}
}

// TestRunner_StampsBeforeRunReplacement checks that a BeforeRun reply that an
// OnEvent plugin replaces with a struct literal is stored with a timestamp.
func TestRunner_StampsBeforeRunReplacement(t *testing.T) {
	const appName, userID, sessionID = "testApp", "testUser", "testSession"
	runTime := time.Date(2026, time.March, 2, 10, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return runTime })
	svc := session.InMemoryService()
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	a, err := agent.New(agent.Config{Name: "custom"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	p, err := plugin.New(plugin.Config{
		Name: "cache",
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			return genai.NewContentFromText("cached", genai.RoleModel), nil
		},
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev.Author != "custom" {
				return nil, nil
			}
			return &session.Event{Author: ev.Author, LLMResponse: ev.LLMResponse}, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	r, err := New(Config{
		AppName:        appName,
		Agent:          a,
		SessionService: svc,
		PluginConfig:   PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	for _, err := range r.Run(ctx, userID, sessionID, userText("go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var sawReply bool
	for ev := range got.Session.Events().All() {
		if ev.Author != "custom" {
			continue
		}
		sawReply = true
		if !ev.Timestamp.Equal(runTime) {
			t.Errorf("stored BeforeRun reply has timestamp %v, want %v", ev.Timestamp, runTime)
		}
	}
	if !sawReply {
		t.Fatal("the BeforeRun reply was not stored")
	}
	if got.Session.LastUpdateTime().IsZero() {
		t.Error("session LastUpdateTime() is zero after the run")
	}
}
