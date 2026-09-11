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

	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/emptypb"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	vertexaiutil "google.golang.org/adk/v2/util/vertexai"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	longrunningpb "cloud.google.com/go/longrunning/autogen/longrunningpb"
)

// TestCreateScope verifies the MemoryBank scope carries both app_name and
// user_id, so memories are partitioned per (app_name, user_id).
func TestCreateScope(t *testing.T) {
	scope := createScope("app-a", "user-1")

	if got, want := scope["app_name"], "app-a"; got != want {
		t.Errorf("scope[app_name] = %q, want %q", got, want)
	}
	if got, want := scope["user_id"], "user-1"; got != want {
		t.Errorf("scope[user_id] = %q, want %q", got, want)
	}
	if len(scope) != 2 {
		t.Errorf("scope has %d keys (%v), want exactly app_name and user_id", len(scope), scope)
	}
}

// testSession is a minimal session.Session. The generate/retrieve paths under
// test only read ID, AppName and UserID; the remaining methods are unused here.
type testSession struct{ appName, userID, id string }

func (s *testSession) ID() string { return s.id }

func (s *testSession) AppName() string { return s.appName }

func (s *testSession) UserID() string { return s.userID }

func (s *testSession) State() session.State { return nil }

func (s *testSession) Events() session.Events { return nil }

func (s *testSession) LastUpdateTime() time.Time { return time.Time{} }

// fakeMemoryBank is an in-process MemoryBankServiceServer that records the scope
// of every GenerateMemories / RetrieveMemories request, so a test can assert
// what each production call site puts on the wire.
type fakeMemoryBank struct {
	aiplatformpb.UnimplementedMemoryBankServiceServer
	generateScopes []map[string]string
	retrieveScopes []map[string]string
}

func (f *fakeMemoryBank) GenerateMemories(_ context.Context, req *aiplatformpb.GenerateMemoriesRequest) (*longrunningpb.Operation, error) {
	f.generateScopes = append(f.generateScopes, req.GetScope())
	done, _ := anypb.New(&emptypb.Empty{})
	return &longrunningpb.Operation{
		Name:   req.GetParent() + "/operations/1",
		Done:   true,
		Result: &longrunningpb.Operation_Response{Response: done},
	}, nil
}

func (f *fakeMemoryBank) RetrieveMemories(_ context.Context, req *aiplatformpb.RetrieveMemoriesRequest) (*aiplatformpb.RetrieveMemoriesResponse, error) {
	f.retrieveScopes = append(f.retrieveScopes, req.GetScope())
	return &aiplatformpb.RetrieveMemoriesResponse{}, nil
}

func newFakeMemoryClient(t *testing.T, fake aiplatformpb.MemoryBankServiceServer) *vertexAIClient {
	t.Helper()

	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
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

	client, err := aiplatform.NewMemoryBankClient(t.Context(), option.WithGRPCConn(conn), option.WithoutAuthentication())
	if err != nil {
		t.Fatalf("NewMemoryBankClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })

	data := &vertexaiutil.AgentEngineData{Location: "us-central1", ProjectID: "p", ReasoningEngine: "123"}
	return &vertexAIClient{
		client:          client,
		agentEngineData: data,
		parent:          vertexaiutil.AgentEngineResource(data),
	}
}

// wantScope asserts a recorded scope is exactly {app_name: app, user_id: user}.
func wantScope(t *testing.T, label string, got map[string]string, app, user string) {
	t.Helper()
	if got["app_name"] != app || got["user_id"] != user || len(got) != 2 {
		t.Errorf("%s scope = %v, want {app_name:%q, user_id:%q}", label, got, app, user)
	}
}

// TestScopeIncludesAppNameOnBothCallSites pins the two production call sites.
// The write path (GenerateMemories) must partition by app_name, so two apps for
// the same user emit different scopes; the read path (RetrieveMemories) must
// query the same (app_name, user_id) scope the writer used. Reverting either
// call site to a user_id-only scope fails this test.
func TestScopeIncludesAppNameOnBothCallSites(t *testing.T) {
	ctx := t.Context()
	fake := &fakeMemoryBank{}
	v := newFakeMemoryClient(t, fake)

	if err := v.addWholeSession(ctx, &testSession{appName: "app-a", userID: "user-1", id: "s1"}); err != nil {
		t.Fatalf("addWholeSession(app-a): %v", err)
	}
	if err := v.addWholeSession(ctx, &testSession{appName: "app-b", userID: "user-1", id: "s2"}); err != nil {
		t.Fatalf("addWholeSession(app-b): %v", err)
	}
	if len(fake.generateScopes) != 2 {
		t.Fatalf("recorded %d generate scopes, want 2", len(fake.generateScopes))
	}
	// Write path: two apps for the same user must not share a scope.
	wantScope(t, "GenerateMemories(app-a)", fake.generateScopes[0], "app-a", "user-1")
	wantScope(t, "GenerateMemories(app-b)", fake.generateScopes[1], "app-b", "user-1")
	if fake.generateScopes[0]["app_name"] == fake.generateScopes[1]["app_name"] {
		t.Errorf("write path emitted the same app_name for app-a and app-b; memories would not be partitioned")
	}

	// Read path: retrieval must query the same scope the writer used for app-a.
	if _, err := v.searchMemory(ctx, &memory.SearchRequest{AppName: "app-a", UserID: "user-1", Query: "q"}); err != nil {
		t.Fatalf("searchMemory(app-a): %v", err)
	}
	if len(fake.retrieveScopes) != 1 {
		t.Fatalf("recorded %d retrieve scopes, want 1", len(fake.retrieveScopes))
	}
	wantScope(t, "RetrieveMemories(app-a)", fake.retrieveScopes[0], "app-a", "user-1")
}
