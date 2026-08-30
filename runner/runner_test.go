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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
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

func TestRunner_NilEventYieldedDoesNotPanic(t *testing.T) {
	t.Parallel()

	nilAgent, err := agent.New(agent.Config{
		Name: "nil_yielder",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	sessionService := session.InMemoryService()
	r, err := New(Config{
		AppName:           "test_app",
		Agent:             nilAgent,
		SessionService:    sessionService,
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	var count int
	msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
	for ev, err := range r.Run(t.Context(), "user1", "session1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if ev == nil {
			t.Errorf("unexpected nil event")
		}
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 events, got %d", count)
	}
}

// TestRunner_NilEventDoesNotTruncateStream pins that a skipped nil event only
// drops that event: everything after it must still reach the caller. Without
// this, replacing the skip with a break or a return still passes the suite.
func TestRunner_NilEventDoesNotTruncateStream(t *testing.T) {
	t.Parallel()

	textEvent := func(text string) *session.Event {
		ev := &session.Event{Author: "agent"}
		ev.LLMResponse.Content = &genai.Content{Parts: []*genai.Part{{Text: text}}}
		return ev
	}

	tests := []struct {
		name      string
		yielded   []*session.Event
		wantTexts []string
	}{
		{
			name:      "leading nil",
			yielded:   []*session.Event{nil, textEvent("a")},
			wantTexts: []string{"a"},
		},
		{
			name:      "nil between events",
			yielded:   []*session.Event{textEvent("a"), nil, textEvent("b")},
			wantTexts: []string{"a", "b"},
		},
		{
			name:      "trailing nil",
			yielded:   []*session.Event{textEvent("a"), nil},
			wantTexts: []string{"a"},
		},
		{
			name:      "consecutive nils",
			yielded:   []*session.Event{nil, nil, textEvent("a"), nil, nil},
			wantTexts: []string{"a"},
		},
		{
			name:      "only nils",
			yielded:   []*session.Event{nil, nil},
			wantTexts: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			a, err := agent.New(agent.Config{
				Name: "nil_yielder",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for _, ev := range tt.yielded {
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

			r, err := New(Config{
				AppName:           "test_app",
				Agent:             a,
				SessionService:    session.InMemoryService(),
				AutoCreateSession: true,
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			var gotTexts []string
			msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
			for ev, err := range r.Run(t.Context(), "user1", "session1", msg, agent.RunConfig{}) {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
					continue
				}
				if ev == nil {
					t.Error("runner yielded a nil event")
					continue
				}
				gotTexts = append(gotTexts, ev.LLMResponse.Content.Parts[0].Text)
			}

			if diff := cmp.Diff(tt.wantTexts, gotTexts); diff != "" {
				t.Errorf("delivered events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRunner_NilEventRunIsBounded pins that an agent yielding nil without end
// cannot trap the caller. The skip never calls yield, so the consumer's break
// is unreachable and cancelling the context from inside the range statement is
// not possible: the loop has to terminate on its own.
func TestRunner_NilEventRunIsBounded(t *testing.T) {
	t.Parallel()

	produced := 0
	a, err := agent.New(agent.Config{
		Name: "endless_nil_yielder",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for {
					produced++
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

	r, err := New(Config{
		AppName:           "test_app",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	var gotErr error
	msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
	for ev, err := range r.Run(t.Context(), "user1", "session1", msg, agent.RunConfig{}) {
		if ev != nil {
			t.Errorf("runner yielded an event, want none")
		}
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("Runner.Run() error = nil, want an error naming the agent")
	}
	if !strings.Contains(gotErr.Error(), "endless_nil_yielder") {
		t.Errorf("Runner.Run() error = %q, want it to name the offending agent", gotErr)
	}
	if produced > maxNilEventsBetweenDeliveries+1 {
		t.Errorf("agent produced %d nil events, want at most %d", produced, maxNilEventsBetweenDeliveries+1)
	}
}

// TestRunner_NilEventCounterResetsOnDelivery pins that the bound only resets
// when an event actually reaches the consumer, so a stream that keeps making
// progress is never cut off however many nils it carries in total.
func TestRunner_NilEventCounterResetsOnDelivery(t *testing.T) {
	t.Parallel()

	realEvent := func() *session.Event {
		ev := &session.Event{Author: "agent"}
		ev.LLMResponse.Content = &genai.Content{Parts: []*genai.Part{{Text: "ok"}}}
		return ev
	}

	a, err := agent.New(agent.Config{
		Name: "chatty_nil_yielder",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// Far more nils in total than the bound, but never in one run.
				for range 5 {
					for range maxNilEventsBetweenDeliveries {
						if !yield(nil, nil) {
							return
						}
					}
					if !yield(realEvent(), nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	r, err := New(Config{
		AppName:           "test_app",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	got := 0
	msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
	for ev, err := range r.Run(t.Context(), "user1", "session1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if ev == nil {
			t.Error("runner yielded a nil event")
			continue
		}
		got++
	}

	if got != 5 {
		t.Errorf("delivered %d events, want 5", got)
	}
}

// TestRunner_NilBoundIgnoresUndeliveredEvents pins that the nil bound resets on
// DELIVERY, not on merely observing a non-nil event. A plugin that rejects every
// event means nothing reaches the consumer, so an interleaved stream must still
// trip the bound instead of riding it forever.
func TestRunner_NilBoundIgnoresUndeliveredEvents(t *testing.T) {
	t.Parallel()

	realEvent := func() *session.Event {
		ev := &session.Event{Author: "agent"}
		ev.LLMResponse.Content = &genai.Content{Parts: []*genai.Part{{Text: "ok"}}}
		return ev
	}

	a, err := agent.New(agent.Config{
		Name: "interleaving_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for {
					for range maxNilEventsBetweenDeliveries {
						if !yield(nil, nil) {
							return
						}
					}
					if !yield(realEvent(), nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	// The plugin fails every event, so no event is ever delivered.
	rejectAll, err := plugin.New(plugin.Config{
		Name: "reject_all",
		OnEventCallback: func(_ agent.InvocationContext, _ *session.Event) (*session.Event, error) {
			return nil, errors.New("rejected")
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}

	r, err := New(Config{
		AppName:           "test_app",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      PluginConfig{Plugins: []*plugin.Plugin{rejectAll}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	var lastErr error
	msg := &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}
	for ev, err := range r.Run(t.Context(), "user1", "session1", msg, agent.RunConfig{}) {
		if ev != nil {
			t.Errorf("runner delivered an event, want none")
		}
		lastErr = err
	}

	if lastErr == nil || !strings.Contains(lastErr.Error(), "interleaving_agent") {
		t.Errorf("Runner.Run() last error = %v, want the nil bound to name the agent", lastErr)
	}
}
