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
	"net"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/session"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	longrunningpb "cloud.google.com/go/longrunning/autogen/longrunningpb"
)

// fakeSessions is an in-process SessionServiceServer that owns a single session,
// "owned", belonging to "user1". It records how many DeleteSession RPCs it sees
// so tests can assert that a denied delete never reaches the backend.
type fakeSessions struct {
	aiplatformpb.UnimplementedSessionServiceServer
	deletes atomic.Int32
	// appends counts AppendEvent RPCs, so a test can tell a backend error
	// apart from a local short-circuit that never reached the server.
	appends atomic.Int32
}

func (f *fakeSessions) GetSession(_ context.Context, req *aiplatformpb.GetSessionRequest) (*aiplatformpb.Session, error) {
	switch {
	case strings.HasSuffix(req.GetName(), "/sessions/owned"):
		return &aiplatformpb.Session{Name: req.GetName(), UserId: "user1", UpdateTime: timestamppb.Now()}, nil
	// "denied" stands for any backend refusal that is not a missing session.
	// It must not come back as session.ErrNotFound.
	case strings.HasSuffix(req.GetName(), "/sessions/denied"):
		return nil, status.Errorf(codes.PermissionDenied, "caller cannot read %s", req.GetName())
	default:
		return nil, status.Errorf(codes.NotFound, "Session %s not found.", req.GetName())
	}
}

func (f *fakeSessions) ListEvents(_ context.Context, req *aiplatformpb.ListEventsRequest) (*aiplatformpb.ListEventsResponse, error) {
	// Get fetches the session and its events concurrently, so this has to agree
	// with GetSession about which sessions exist. Leaving it unimplemented lets
	// a gRPC Unimplemented error win the race and mask the real not-found.
	switch {
	case strings.HasSuffix(req.GetParent(), "/sessions/owned"):
		return &aiplatformpb.ListEventsResponse{}, nil
	case strings.HasSuffix(req.GetParent(), "/sessions/denied"):
		return nil, status.Errorf(codes.PermissionDenied, "caller cannot read %s", req.GetParent())
	default:
		return nil, status.Errorf(codes.NotFound, "Session %s not found.", req.GetParent())
	}
}

func (f *fakeSessions) AppendEvent(_ context.Context, req *aiplatformpb.AppendEventRequest) (*aiplatformpb.AppendEventResponse, error) {
	f.appends.Add(1)
	switch {
	case strings.HasSuffix(req.GetName(), "/sessions/owned"):
		return &aiplatformpb.AppendEventResponse{}, nil
	case strings.HasSuffix(req.GetName(), "/sessions/denied"):
		return nil, status.Errorf(codes.PermissionDenied, "caller cannot write %s", req.GetName())
	default:
		return nil, status.Errorf(codes.NotFound, "Session %s not found.", req.GetName())
	}
}

func (f *fakeSessions) DeleteSession(_ context.Context, req *aiplatformpb.DeleteSessionRequest) (*longrunningpb.Operation, error) {
	f.deletes.Add(1)
	done, _ := anypb.New(&emptypb.Empty{})
	// Done:true with a nil Result makes lro.Wait fail with "unsupported result type".
	return &longrunningpb.Operation{
		Name:   req.GetName() + "/operations/1",
		Done:   true,
		Result: &longrunningpb.Operation_Response{Response: done},
	}, nil
}

func newFakeService(t *testing.T) (session.Service, *fakeSessions) {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	fake := &fakeSessions{}
	aiplatformpb.RegisterSessionServiceServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	s, err := NewSessionService(t.Context(), VertexAIServiceConfig{
		Location: "us-central1", ProjectID: "p", ReasoningEngine: "123",
	}, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewSessionService: %v", err)
	}
	return s, fake
}

// A delete requested by a non-owner must be rejected and must not reach the
// backend's DeleteSession.
func TestDelete_wrongUser_deniedAndNotDeleted(t *testing.T) {
	s, fake := newFakeService(t)

	err := s.Delete(t.Context(), &session.DeleteRequest{AppName: "123", UserID: "user2", SessionID: "owned"})
	if err == nil || !strings.Contains(err.Error(), "does not belong to user") {
		t.Errorf("cross-user Delete: got %v, want an ownership error", err)
	}
	if got := fake.deletes.Load(); got != 0 {
		t.Errorf("DeleteSession RPCs = %d, want 0", got)
	}
}

// The owner can still delete, so a guard that rejects everything can't pass.
func TestDelete_owner_deletes(t *testing.T) {
	s, fake := newFakeService(t)

	if err := s.Delete(t.Context(), &session.DeleteRequest{AppName: "123", UserID: "user1", SessionID: "owned"}); err != nil {
		t.Fatalf("owner Delete: got %v, want nil", err)
	}
	if got := fake.deletes.Load(); got != 1 {
		t.Errorf("DeleteSession RPCs = %d, want 1", got)
	}
}

// Deleting a session that does not exist is a no-op (no error, no delete RPC).
func TestDelete_missingSession_isNoOp(t *testing.T) {
	s, fake := newFakeService(t)

	if err := s.Delete(t.Context(), &session.DeleteRequest{AppName: "123", UserID: "user1", SessionID: "other"}); err != nil {
		t.Errorf("missing-session Delete: got %v, want nil", err)
	}
	if got := fake.deletes.Load(); got != 0 {
		t.Errorf("DeleteSession RPCs = %d, want 0", got)
	}
}
