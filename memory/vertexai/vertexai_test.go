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

package vertexai

import (
	"context"
	"iter"
	"slices"
	"testing"
	"time"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestVertexAIService_AddSessionToMemory(t *testing.T) {
	fake := &fakeMemoryBankClient{generateOp: &fakeGenerateMemoriesOperation{name: "operations/generate-1"}}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	err := svc.AddSessionToMemory(t.Context(), &testSession{
		appName:   "123",
		userID:    "test-user",
		sessionID: "test-session",
	})
	if err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	want := &aiplatformpb.GenerateMemoriesRequest{
		Parent: "projects/test-project/locations/us-central1/reasoningEngines/123",
		Source: &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource_{
			VertexSessionSource: &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource{
				Session: "projects/test-project/locations/us-central1/reasoningEngines/123/sessions/test-session",
			},
		},
	}
	if diff := cmp.Diff(want, fake.generateReq, protocmp.Transform()); diff != "" {
		t.Errorf("GenerateMemoriesRequest mismatch (-want +got):\n%s", diff)
	}
	if !fake.generateOp.waitCalled {
		t.Errorf("GenerateMemories operation was not waited on")
	}
}

func TestVertexAIService_AddSessionToMemoryWithOptions(t *testing.T) {
	start := time.Date(2026, 1, 2, 3, 0, 0, 0, time.UTC)
	end := time.Date(2026, 1, 2, 4, 0, 0, 0, time.UTC)
	fake := &fakeMemoryBankClient{generateOp: &fakeGenerateMemoriesOperation{
		name: "operations/generate-1",
		resp: &aiplatformpb.GenerateMemoriesResponse{
			GeneratedMemories: []*aiplatformpb.GenerateMemoriesResponse_GeneratedMemory{
				{
					Memory: &aiplatformpb.Memory{
						Name: "projects/test-project/locations/us-central1/reasoningEngines/123/memories/memory-1",
						Fact: "I prefer concise answers.",
					},
					Action: aiplatformpb.GenerateMemoriesResponse_GeneratedMemory_CREATED,
				},
			},
		},
	}}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	got, err := svc.AddSessionToMemoryWithOptions(
		t.Context(),
		&testSession{appName: "123", userID: "test-user", sessionID: "test-session"},
		WithScope(map[string]string{"user_id": "test-user", "tenant_id": "tenant-1"}),
		WithDisableConsolidation(true),
		WithEventTimeRange(start, end),
	)
	if err != nil {
		t.Fatalf("AddSessionToMemoryWithOptions() error = %v", err)
	}

	wantReq := &aiplatformpb.GenerateMemoriesRequest{
		Parent:               "projects/test-project/locations/us-central1/reasoningEngines/123",
		DisableConsolidation: true,
		Source: &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource_{
			VertexSessionSource: &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource{
				Session:   "projects/test-project/locations/us-central1/reasoningEngines/123/sessions/test-session",
				StartTime: timestamppb.New(start),
				EndTime:   timestamppb.New(end),
			},
		},
		Scope: map[string]string{"user_id": "test-user", "tenant_id": "tenant-1"},
	}
	if diff := cmp.Diff(wantReq, fake.generateReq, protocmp.Transform()); diff != "" {
		t.Errorf("GenerateMemoriesRequest mismatch (-want +got):\n%s", diff)
	}

	want := &GenerateMemoriesResult{
		OperationName: "operations/generate-1",
		GeneratedMemories: []GeneratedMemory{
			{
				Name:   "projects/test-project/locations/us-central1/reasoningEngines/123/memories/memory-1",
				Fact:   "I prefer concise answers.",
				Action: "CREATED",
			},
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("AddSessionToMemoryWithOptions() mismatch (-want +got):\n%s", diff)
	}
}

func TestVertexAIService_GenerateMemoriesFromVertexSession(t *testing.T) {
	fake := &fakeMemoryBankClient{generateOp: &fakeGenerateMemoriesOperation{name: "operations/generate-1"}}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	got, err := svc.GenerateMemoriesFromVertexSession(
		t.Context(),
		"123",
		"test-session",
		WithWaitForCompletion(false),
	)
	if err != nil {
		t.Fatalf("GenerateMemoriesFromVertexSession() error = %v", err)
	}

	wantReq := &aiplatformpb.GenerateMemoriesRequest{
		Parent: "projects/test-project/locations/us-central1/reasoningEngines/123",
		Source: &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource_{
			VertexSessionSource: &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource{
				Session: "projects/test-project/locations/us-central1/reasoningEngines/123/sessions/test-session",
			},
		},
	}
	if diff := cmp.Diff(wantReq, fake.generateReq, protocmp.Transform()); diff != "" {
		t.Errorf("GenerateMemoriesRequest mismatch (-want +got):\n%s", diff)
	}
	if fake.generateOp.waitCalled {
		t.Errorf("GenerateMemories operation was waited on")
	}
	want := &GenerateMemoriesResult{OperationName: "operations/generate-1"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("GenerateMemoriesFromVertexSession() mismatch (-want +got):\n%s", diff)
	}
}

func TestVertexAIService_AddSessionToMemoryWithOptions_NoWait(t *testing.T) {
	fake := &fakeMemoryBankClient{generateOp: &fakeGenerateMemoriesOperation{name: "operations/generate-1"}}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	got, err := svc.AddSessionToMemoryWithOptions(
		t.Context(),
		&testSession{appName: "123", userID: "test-user", sessionID: "test-session"},
		WithWaitForCompletion(false),
	)
	if err != nil {
		t.Fatalf("AddSessionToMemoryWithOptions() error = %v", err)
	}
	if fake.generateOp.waitCalled {
		t.Errorf("GenerateMemories operation was waited on")
	}
	want := &GenerateMemoriesResult{OperationName: "operations/generate-1"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("AddSessionToMemoryWithOptions() mismatch (-want +got):\n%s", diff)
	}
}

func TestVertexAIService_GenerateMemoriesFromEvents(t *testing.T) {
	fake := &fakeMemoryBankClient{generateOp: &fakeGenerateMemoriesOperation{name: "operations/generate-1"}}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	_, err := svc.GenerateMemoriesFromEvents(t.Context(), "123", "test-user", []*session.Event{
		{
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("Please remember that I prefer concise answers.", genai.RoleUser),
			},
		},
	})
	if err != nil {
		t.Fatalf("GenerateMemoriesFromEvents() error = %v", err)
	}

	want := &aiplatformpb.GenerateMemoriesRequest{
		Parent: "projects/test-project/locations/us-central1/reasoningEngines/123",
		Source: &aiplatformpb.GenerateMemoriesRequest_DirectContentsSource_{
			DirectContentsSource: &aiplatformpb.GenerateMemoriesRequest_DirectContentsSource{
				Events: []*aiplatformpb.GenerateMemoriesRequest_DirectContentsSource_Event{
					{
						Content: &aiplatformpb.Content{
							Role: genai.RoleUser,
							Parts: []*aiplatformpb.Part{
								{Data: &aiplatformpb.Part_Text{Text: "Please remember that I prefer concise answers."}},
							},
						},
					},
				},
			},
		},
		Scope: map[string]string{"user_id": "test-user"},
	}
	if diff := cmp.Diff(want, fake.generateReq, protocmp.Transform()); diff != "" {
		t.Errorf("GenerateMemoriesRequest mismatch (-want +got):\n%s", diff)
	}
}

func TestVertexAIService_GenerateMemoriesFromContents(t *testing.T) {
	fake := &fakeMemoryBankClient{generateOp: &fakeGenerateMemoriesOperation{name: "operations/generate-1"}}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	_, err := svc.GenerateMemoriesFromContents(t.Context(), "123", "test-user", []*genai.Content{
		genai.NewContentFromText("I work in Cape Town.", genai.RoleUser),
	}, WithWaitForCompletion(false))
	if err != nil {
		t.Fatalf("GenerateMemoriesFromContents() error = %v", err)
	}
	if fake.generateOp.waitCalled {
		t.Errorf("GenerateMemories operation was waited on")
	}
}

func TestVertexAIService_GenerateMemoriesFromFacts(t *testing.T) {
	fake := &fakeMemoryBankClient{generateOp: &fakeGenerateMemoriesOperation{name: "operations/generate-1"}}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	_, err := svc.GenerateMemoriesFromFacts(t.Context(), "123", "test-user", []string{
		"The user prefers concise answers.",
	})
	if err != nil {
		t.Fatalf("GenerateMemoriesFromFacts() error = %v", err)
	}

	want := &aiplatformpb.GenerateMemoriesRequest{
		Parent: "projects/test-project/locations/us-central1/reasoningEngines/123",
		Source: &aiplatformpb.GenerateMemoriesRequest_DirectMemoriesSource_{
			DirectMemoriesSource: &aiplatformpb.GenerateMemoriesRequest_DirectMemoriesSource{
				DirectMemories: []*aiplatformpb.GenerateMemoriesRequest_DirectMemoriesSource_DirectMemory{
					{Fact: "The user prefers concise answers."},
				},
			},
		},
		Scope: map[string]string{"user_id": "test-user"},
	}
	if diff := cmp.Diff(want, fake.generateReq, protocmp.Transform()); diff != "" {
		t.Errorf("GenerateMemoriesRequest mismatch (-want +got):\n%s", diff)
	}
}

func TestVertexAIService_GenerateMemoriesFromFacts_RejectsTooManyFacts(t *testing.T) {
	svc := &vertexAIService{}
	_, err := svc.GenerateMemoriesFromFacts(t.Context(), "123", "test-user", []string{"1", "2", "3", "4", "5", "6"})
	if err == nil {
		t.Fatalf("GenerateMemoriesFromFacts() error = nil, want error")
	}
}

func TestVertexAIService_SearchMemory(t *testing.T) {
	now := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	fake := &fakeMemoryBankClient{
		retrieveResp: &aiplatformpb.RetrieveMemoriesResponse{
			RetrievedMemories: []*aiplatformpb.RetrieveMemoriesResponse_RetrievedMemory{
				{Memory: &aiplatformpb.Memory{
					Name:       "projects/test-project/locations/us-central1/reasoningEngines/123/memories/memory-1",
					Fact:       "I prefer concise answers.",
					Scope:      map[string]string{"user_id": "test-user"},
					UpdateTime: timestamppb.New(now),
				}},
			},
		},
	}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	got, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
		AppName: "123",
		UserID:  "test-user",
		Query:   "answer style",
	})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}

	wantReq := &aiplatformpb.RetrieveMemoriesRequest{
		Parent: "projects/test-project/locations/us-central1/reasoningEngines/123",
		Scope:  map[string]string{"user_id": "test-user"},
		RetrievalParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams_{
			SimilaritySearchParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams{
				SearchQuery: "answer style",
			},
		},
	}
	if diff := cmp.Diff(wantReq, fake.retrieveReq, protocmp.Transform()); diff != "" {
		t.Errorf("RetrieveMemoriesRequest mismatch (-want +got):\n%s", diff)
	}

	want := &memory.SearchResponse{Memories: []memory.Entry{
		{
			ID:             "projects/test-project/locations/us-central1/reasoningEngines/123/memories/memory-1",
			Content:        genai.NewContentFromText("I prefer concise answers.", genai.RoleModel),
			Author:         "memory",
			Timestamp:      now,
			CustomMetadata: map[string]any{"scope": map[string]string{"user_id": "test-user"}},
		},
	}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("SearchMemory() mismatch (-want +got):\n%s", diff)
	}
}

func TestVertexAIService_SearchMemoryWithScope(t *testing.T) {
	fake := &fakeMemoryBankClient{
		retrieveResp: &aiplatformpb.RetrieveMemoriesResponse{},
	}
	svc := &vertexAIService{client: &vertexAIClient{
		location:  "us-central1",
		projectID: "test-project",
		rpcClient: fake,
	}}

	_, err := svc.SearchMemoryWithScope(t.Context(), &memory.SearchRequest{
		AppName: "123",
		Query:   "answer style",
	}, map[string]string{"user_id": "test-user", "tenant_id": "tenant-1"})
	if err != nil {
		t.Fatalf("SearchMemoryWithScope() error = %v", err)
	}

	wantReq := &aiplatformpb.RetrieveMemoriesRequest{
		Parent: "projects/test-project/locations/us-central1/reasoningEngines/123",
		Scope:  map[string]string{"user_id": "test-user", "tenant_id": "tenant-1"},
		RetrievalParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams_{
			SimilaritySearchParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams{
				SearchQuery: "answer style",
			},
		},
	}
	if diff := cmp.Diff(wantReq, fake.retrieveReq, protocmp.Transform()); diff != "" {
		t.Errorf("RetrieveMemoriesRequest mismatch (-want +got):\n%s", diff)
	}
}

func TestGetReasoningEngineID(t *testing.T) {
	tests := []struct {
		name             string
		existingEngineID string
		inputAppName     string
		expectedID       string
		expectError      bool
	}{
		{
			name:             "client already has engine ID configured",
			existingEngineID: "999",
			inputAppName:     "irrelevant-input",
			expectedID:       "999",
		},
		{
			name:         "input is a direct numeric ID",
			inputAppName: "123456",
			expectedID:   "123456",
		},
		{
			name:         "input is a valid full resource path",
			inputAppName: "projects/my-project/locations/us-central1/reasoningEngines/555123",
			expectedID:   "555123",
		},
		{
			name:         "input is malformed",
			inputAppName: "some-random-app-name",
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &vertexAIClient{reasoningEngine: tt.existingEngineID}
			got, err := c.getReasoningEngineID(tt.inputAppName)
			if (err != nil) != tt.expectError {
				t.Fatalf("getReasoningEngineID() error = %v, expectError %v", err, tt.expectError)
			}
			if got != tt.expectedID {
				t.Errorf("getReasoningEngineID() got = %v, want %v", got, tt.expectedID)
			}
		})
	}
}

type fakeMemoryBankClient struct {
	generateReq  *aiplatformpb.GenerateMemoriesRequest
	generateOp   *fakeGenerateMemoriesOperation
	retrieveReq  *aiplatformpb.RetrieveMemoriesRequest
	retrieveResp *aiplatformpb.RetrieveMemoriesResponse
}

func (c *fakeMemoryBankClient) Close() error {
	return nil
}

func (c *fakeMemoryBankClient) GenerateMemories(ctx context.Context, req *aiplatformpb.GenerateMemoriesRequest) (generateMemoriesOperation, error) {
	c.generateReq = req
	return c.generateOp, nil
}

func (c *fakeMemoryBankClient) RetrieveMemories(ctx context.Context, req *aiplatformpb.RetrieveMemoriesRequest) (*aiplatformpb.RetrieveMemoriesResponse, error) {
	c.retrieveReq = req
	return c.retrieveResp, nil
}

type fakeGenerateMemoriesOperation struct {
	name       string
	resp       *aiplatformpb.GenerateMemoriesResponse
	waitCalled bool
}

func (op *fakeGenerateMemoriesOperation) Name() string {
	return op.name
}

func (op *fakeGenerateMemoriesOperation) Wait(ctx context.Context) (*aiplatformpb.GenerateMemoriesResponse, error) {
	op.waitCalled = true
	if op.resp != nil {
		return op.resp, nil
	}
	return &aiplatformpb.GenerateMemoriesResponse{}, nil
}

type testSession struct {
	appName, userID, sessionID string
	events                     []*session.Event
}

func (s *testSession) ID() string {
	return s.sessionID
}

func (s *testSession) AppName() string {
	return s.appName
}

func (s *testSession) UserID() string {
	return s.userID
}

func (s *testSession) Events() session.Events {
	return s
}

func (s *testSession) All() iter.Seq[*session.Event] {
	return slices.Values(s.events)
}

func (s *testSession) Len() int {
	return len(s.events)
}

func (s *testSession) At(i int) *session.Event {
	return s.events[i]
}

func (s *testSession) State() session.State {
	panic("not implemented")
}

func (s *testSession) LastUpdateTime() time.Time {
	panic("not implemented")
}
