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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
)

func TestRunner_findAgentToRun(t *testing.T) {
	t.Parallel()

	appName, userID, sessionID := "test", "userID", "sessionID"

	agentTree := agentTree(t)

	tests := []struct {
		name      string
		rootAgent agent.Agent
		session   session.Session
		wantAgent agent.Agent
		wantErr   bool
	}{
		{
			name: "last event from agent allowing transfer",
			session: createSession(t, t.Context(), appName, userID, sessionID, []*session.Event{
				{
					Author: "allows_transfer_agent",
				},
				{
					Author: "user",
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
				{
					Author: "user",
				},
			}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.root,
		},
		{
			name: "no events from agents, call root",
			session: createSession(t, t.Context(), appName, userID, sessionID, []*session.Event{
				{
					Author: "user",
				},
			}),
			rootAgent: agentTree.root,
			wantAgent: agentTree.root,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Runner{
				rootAgent: tt.rootAgent,
			}
			gotAgent, err := r.findAgentToRun(tt.session)
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

func Test_findAgent(t *testing.T) {
	agentTree := agentTree(t)

	oneAgent := must(llmagent.New(llmagent.Config{
		Name: "test",
	}))

	tests := []struct {
		name      string
		root      agent.Agent
		target    string
		wantAgent agent.Agent
	}{
		{
			name:      "ok",
			root:      agentTree.root,
			target:    agentTree.allowsTransferAgent.Name(),
			wantAgent: agentTree.allowsTransferAgent,
		},
		{
			name:      "finds in one node tree",
			root:      oneAgent,
			target:    oneAgent.Name(),
			wantAgent: oneAgent,
		},
		{
			name:      "doesn't fail if agent is missing in the tree",
			root:      agentTree.root,
			target:    "random",
			wantAgent: nil,
		},
		{
			name:      "doesn't fail on the empty tree",
			root:      nil,
			target:    "random",
			wantAgent: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotAgent := findAgent(tt.root, tt.target); gotAgent != tt.wantAgent {
				t.Errorf("Runner.findAgent() = %+v, want %+v", gotAgent.Name(), tt.wantAgent.Name())
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
