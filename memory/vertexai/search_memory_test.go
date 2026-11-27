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
	"github.com/google/go-cmp/cmp"
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
	memories []*aiplatformpb.Memory
}

func (f *fakeMemoryBank) RetrieveMemories(context.Context, *aiplatformpb.RetrieveMemoriesRequest) (*aiplatformpb.RetrieveMemoriesResponse, error) {
	resp := &aiplatformpb.RetrieveMemoriesResponse{}
	for _, m := range f.memories {
		resp.RetrievedMemories = append(resp.RetrievedMemories, &aiplatformpb.RetrieveMemoriesResponse_RetrievedMemory{Memory: m})
	}
	return resp, nil
}

// newFakeMemoryBankClient returns a client whose Memory Bank serves memories.
func newFakeMemoryBankClient(t *testing.T, memories []*aiplatformpb.Memory) *vertexAIClient {
	t.Helper()
	listener := bufconn.Listen(1024 * 1024)
	srv := grpc.NewServer()
	aiplatformpb.RegisterMemoryBankServiceServer(srv, &fakeMemoryBank{memories: memories})
	go func() { _ = srv.Serve(listener) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return listener.DialContext(ctx) }))
	if err != nil {
		t.Fatalf("grpc.NewClient() error = %v", err)
	}
	client, err := aiplatform.NewMemoryBankClient(t.Context(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewMemoryBankClient() error = %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return &vertexAIClient{client: client, parent: "projects/p/locations/l/reasoningEngines/1"}
}

func TestSearchMemory_EntryFields(t *testing.T) {
	created := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	updated := time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC)
	tests := []struct {
		name   string
		memory *aiplatformpb.Memory
		want   memory.Entry
	}{
		{
			name: "dated by the update time",
			memory: &aiplatformpb.Memory{
				Fact:       "likes tea",
				CreateTime: timestamppb.New(created),
				UpdateTime: timestamppb.New(updated),
			},
			want: memory.Entry{
				Content:   genai.NewContentFromText("likes tea", genai.RoleUser),
				Author:    "user",
				Timestamp: updated,
			},
		},
		{
			name:   "no update time leaves the timestamp zero",
			memory: &aiplatformpb.Memory{Fact: "likes tea", CreateTime: timestamppb.New(created)},
			want: memory.Entry{
				Content: genai.NewContentFromText("likes tea", genai.RoleUser),
				Author:  "user",
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := newFakeMemoryBankClient(t, []*aiplatformpb.Memory{tc.memory})

			got, err := client.searchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "u", Query: "tea"})
			if err != nil {
				t.Fatalf("searchMemory() error = %v", err)
			}
			if diff := cmp.Diff([]memory.Entry{tc.want}, got.Memories); diff != "" {
				t.Errorf("searchMemory() memories mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
