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

package vertexai

import (
	"context"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	vertexaiutil "google.golang.org/adk/v2/util/vertexai"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	longrunningpb "cloud.google.com/go/longrunning/autogen/longrunningpb"
)

// fakeMemoryBank is an in-process MemoryBankServiceServer. It records every
// GenerateMemories and RetrieveMemories request it receives so tests can
// assert on what the client sent, and it can be programmed to fail either RPC.
type fakeMemoryBank struct {
	aiplatformpb.UnimplementedMemoryBankServiceServer

	mu               sync.Mutex
	generateRequests []*aiplatformpb.GenerateMemoriesRequest
	retrieveRequests []*aiplatformpb.RetrieveMemoriesRequest

	generateErr  error
	retrieveResp *aiplatformpb.RetrieveMemoriesResponse
	retrieveErr  error
}

func (f *fakeMemoryBank) GenerateMemories(_ context.Context, req *aiplatformpb.GenerateMemoriesRequest) (*longrunningpb.Operation, error) {
	f.mu.Lock()
	f.generateRequests = append(f.generateRequests, req)
	generateErr := f.generateErr
	f.mu.Unlock()
	if generateErr != nil {
		return nil, generateErr
	}
	// A done operation without a result: lro.Wait reports
	// "unsupported result type <nil>: <nil>", which addSession tolerates.
	return &longrunningpb.Operation{Name: req.GetParent() + "/operations/1", Done: true}, nil
}

func (f *fakeMemoryBank) RetrieveMemories(_ context.Context, req *aiplatformpb.RetrieveMemoriesRequest) (*aiplatformpb.RetrieveMemoriesResponse, error) {
	f.mu.Lock()
	f.retrieveRequests = append(f.retrieveRequests, req)
	retrieveErr := f.retrieveErr
	retrieveResp := f.retrieveResp
	f.mu.Unlock()
	if retrieveErr != nil {
		return nil, retrieveErr
	}
	return retrieveResp, nil
}

func (f *fakeMemoryBank) recordedGenerateRequests() []*aiplatformpb.GenerateMemoriesRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]*aiplatformpb.GenerateMemoriesRequest(nil), f.generateRequests...)
}

func (f *fakeMemoryBank) recordedRetrieveRequests() []*aiplatformpb.RetrieveMemoriesRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]*aiplatformpb.RetrieveMemoriesRequest(nil), f.retrieveRequests...)
}

var testAgentEngineData = vertexaiutil.AgentEngineData{
	Location:        "us-central1",
	ProjectID:       "test-project",
	ReasoningEngine: "123",
}

// newTestService wires a vertexAIService to fakeMemoryBank over an in-process
// bufconn connection, so no real VertexAI endpoint is ever contacted.
func newTestService(t *testing.T, stateKeySessionLastUpdateTime string, waitForCompletion bool) (*vertexAIService, *fakeMemoryBank) {
	t.Helper()

	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	fake := &fakeMemoryBank{}
	aiplatformpb.RegisterMemoryBankServiceServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	memoryBankClient, err := aiplatform.NewMemoryBankClient(t.Context(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("aiplatform.NewMemoryBankClient: %v", err)
	}
	t.Cleanup(func() { _ = memoryBankClient.Close() })

	data := testAgentEngineData
	service := &vertexAIService{
		client: &vertexAIClient{
			config: vertexAIClientConfig{
				AgentEngineData:   data,
				waitForCompletion: waitForCompletion,
			},
			client:          memoryBankClient,
			agentEngineData: &data,
			parent:          vertexaiutil.AgentEngineResource(&data),
		},
		stateKeySessionLastUpdateTime: stateKeySessionLastUpdateTime,
	}
	return service, fake
}

func createTestSession(t *testing.T, state map[string]any) session.Session {
	t.Helper()

	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName:   "test-app",
		UserID:    "user1",
		SessionID: "session1",
		State:     state,
	})
	if err != nil {
		t.Fatalf("Create session: %v", err)
	}
	return resp.Session
}

func TestAddSessionToMemory(t *testing.T) {
	lastUpdate := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

	tests := []struct {
		name                          string
		stateKeySessionLastUpdateTime string
		waitForCompletion             bool
		state                         map[string]any
		generateErr                   error
		wantErrSubstr                 string
		wantGenerateCalls             int
		wantStartTime                 *time.Time
	}{
		{
			name:              "whole session when no state key configured",
			wantGenerateCalls: 1,
		},
		{
			name:                          "missing last-update state key",
			stateKeySessionLastUpdateTime: "lastUpdate",
			wantErrSubstr:                 "state.Get(lastUpdate) failed",
		},
		{
			name:                          "non-time last-update state value",
			stateKeySessionLastUpdateTime: "lastUpdate",
			state:                         map[string]any{"lastUpdate": "not-a-time"},
			wantErrSubstr:                 "want type time.Time",
		},
		{
			name:                          "filters events by last-update time",
			stateKeySessionLastUpdateTime: "lastUpdate",
			state:                         map[string]any{"lastUpdate": lastUpdate},
			wantGenerateCalls:             1,
			wantStartTime:                 &lastUpdate,
		},
		{
			name:              "generate memories error is wrapped",
			generateErr:       status.Error(codes.Internal, "backend boom"),
			wantErrSubstr:     "addWholeSession failed",
			wantGenerateCalls: 1,
		},
		{
			name:                          "filtered generate memories error is wrapped",
			stateKeySessionLastUpdateTime: "lastUpdate",
			state:                         map[string]any{"lastUpdate": lastUpdate},
			generateErr:                   status.Error(codes.Internal, "backend boom"),
			wantErrSubstr:                 "addEventsNewerThan failed",
			wantGenerateCalls:             1,
			wantStartTime:                 &lastUpdate,
		},
		{
			name:              "wait for completion tolerates nil operation result",
			waitForCompletion: true,
			wantGenerateCalls: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service, fake := newTestService(t, tt.stateKeySessionLastUpdateTime, tt.waitForCompletion)
			fake.generateErr = tt.generateErr
			s := createTestSession(t, tt.state)

			err := service.AddSessionToMemory(t.Context(), s)

			if tt.wantErrSubstr == "" && err != nil {
				t.Fatalf("AddSessionToMemory() = %v, want nil", err)
			}
			if tt.wantErrSubstr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErrSubstr) {
					t.Fatalf("AddSessionToMemory() = %v, want error containing %q", err, tt.wantErrSubstr)
				}
			}

			requests := fake.recordedGenerateRequests()
			if len(requests) != tt.wantGenerateCalls {
				t.Fatalf("GenerateMemories calls = %d, want %d", len(requests), tt.wantGenerateCalls)
			}
			if tt.wantGenerateCalls == 0 {
				return
			}

			req := requests[0]
			wantParent := "projects/test-project/locations/us-central1/reasoningEngines/123"
			if req.GetParent() != wantParent {
				t.Errorf("request parent = %q, want %q", req.GetParent(), wantParent)
			}
			source := req.GetVertexSessionSource()
			if source == nil {
				t.Fatal("request source is not a VertexSessionSource")
			}
			if want := wantParent + "/sessions/session1"; source.GetSession() != want {
				t.Errorf("source session = %q, want %q", source.GetSession(), want)
			}
			if got := req.GetScope()["user_id"]; got != "user1" {
				t.Errorf("request scope user_id = %q, want %q", got, "user1")
			}
			if tt.wantStartTime == nil {
				if source.GetStartTime() != nil {
					t.Errorf("source start time = %v, want nil", source.GetStartTime())
				}
			} else if !source.GetStartTime().AsTime().Equal(*tt.wantStartTime) {
				t.Errorf("source start time = %v, want %v", source.GetStartTime().AsTime(), *tt.wantStartTime)
			}
		})
	}
}

func TestSearchMemory(t *testing.T) {
	t.Run("returns memory entries", func(t *testing.T) {
		service, fake := newTestService(t, "", false)
		fake.retrieveResp = &aiplatformpb.RetrieveMemoriesResponse{
			RetrievedMemories: []*aiplatformpb.RetrieveMemoriesResponse_RetrievedMemory{
				{Memory: &aiplatformpb.Memory{Fact: "likes tea"}},
				{Memory: &aiplatformpb.Memory{Fact: "lives in Paris"}},
			},
		}

		resp, err := service.SearchMemory(t.Context(), &memory.SearchRequest{Query: "preferences", UserID: "user1"})
		if err != nil {
			t.Fatalf("SearchMemory() = %v, want nil", err)
		}
		if len(resp.Memories) != 2 {
			t.Fatalf("SearchMemory() returned %d memories, want 2", len(resp.Memories))
		}
		wantFacts := []string{"likes tea", "lives in Paris"}
		for i, want := range wantFacts {
			content := resp.Memories[i].Content
			if content == nil || len(content.Parts) != 1 || content.Parts[0].Text != want {
				t.Errorf("memory %d content = %+v, want text %q", i, content, want)
			}
		}

		requests := fake.recordedRetrieveRequests()
		if len(requests) != 1 {
			t.Fatalf("RetrieveMemories calls = %d, want 1", len(requests))
		}
		if got := requests[0].GetScope()["user_id"]; got != "user1" {
			t.Errorf("request scope user_id = %q, want %q", got, "user1")
		}
	})

	t.Run("no matches returns empty slice", func(t *testing.T) {
		service, fake := newTestService(t, "", false)
		fake.retrieveResp = &aiplatformpb.RetrieveMemoriesResponse{}

		resp, err := service.SearchMemory(t.Context(), &memory.SearchRequest{Query: "nothing", UserID: "user1"})
		if err != nil {
			t.Fatalf("SearchMemory() = %v, want nil", err)
		}
		if resp == nil || len(resp.Memories) != 0 {
			t.Errorf("SearchMemory() = %+v, want empty response", resp)
		}
	})

	t.Run("retrieve error is wrapped", func(t *testing.T) {
		service, fake := newTestService(t, "", false)
		fake.retrieveErr = status.Error(codes.Unavailable, "backend down")

		_, err := service.SearchMemory(t.Context(), &memory.SearchRequest{Query: "x", UserID: "user1"})
		if err == nil || !strings.Contains(err.Error(), "RetrieveMemories failed") {
			t.Fatalf("SearchMemory() = %v, want error containing %q", err, "RetrieveMemories failed")
		}
	})
}
