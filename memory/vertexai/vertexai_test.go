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
	"testing"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/memory"
)

type fakeMemoryBank struct {
	aiplatformpb.UnimplementedMemoryBankServiceServer
	response *aiplatformpb.RetrieveMemoriesResponse
}

func (f *fakeMemoryBank) RetrieveMemories(context.Context, *aiplatformpb.RetrieveMemoriesRequest) (*aiplatformpb.RetrieveMemoriesResponse, error) {
	return f.response, nil
}

func TestSearchMemoryEntryMetadata(t *testing.T) {
	updateTime := time.Date(2026, 3, 4, 5, 6, 7, 123456789, time.UTC)
	epoch := time.Unix(0, 0).UTC()
	tests := []struct {
		name          string
		fact          string
		updateTime    *timestamppb.Timestamp
		wantTimestamp time.Time
	}{
		{"with_update_time", "likes tea", timestamppb.New(updateTime), updateTime},
		{"without_update_time", "likes coffee", nil, time.Time{}},
		{"with_unix_epoch", "likes cocoa", timestamppb.New(epoch), epoch},
	}
	response := &aiplatformpb.RetrieveMemoriesResponse{}
	for _, tt := range tests {
		response.RetrievedMemories = append(response.RetrievedMemories, &aiplatformpb.RetrieveMemoriesResponse_RetrievedMemory{
			Memory: &aiplatformpb.Memory{Fact: tt.fact, UpdateTime: tt.updateTime},
		})
	}

	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer()
	aiplatformpb.RegisterMemoryBankServiceServer(server, &fakeMemoryBank{response: response})
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return listener.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	client, err := aiplatform.NewMemoryBankClient(t.Context(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("aiplatform.NewMemoryBankClient: %v", err)
	}
	var service memory.Service = &vertexAIService{client: &vertexAIClient{
		client: client,
		parent: "projects/p/locations/us-central1/reasoningEngines/123",
	}}
	got, err := service.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: "drinks"})
	if err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if len(got.Memories) != len(tests) {
		t.Fatalf("SearchMemory returned %d memories, want %d", len(got.Memories), len(tests))
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := got.Memories[i]
			if entry.Author != "user" {
				t.Errorf("Author = %q, want %q", entry.Author, "user")
			}
			if !entry.Timestamp.Equal(tt.wantTimestamp) {
				t.Errorf("Timestamp = %s, want %s", entry.Timestamp.Format(time.RFC3339Nano), tt.wantTimestamp.Format(time.RFC3339Nano))
			}
			if c := entry.Content; c == nil || c.Role != genai.RoleUser || len(c.Parts) != 1 || c.Parts[0] == nil || c.Parts[0].Text != tt.fact {
				t.Error("Content should contain the retrieved fact in one user-role text part")
			}
		})
	}
}
