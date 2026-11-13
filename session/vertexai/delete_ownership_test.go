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
	"errors"
	"fmt"
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
	// gets and listEvents count GetSession and ListEvents RPCs, so a test can
	// tell which reads a Get sent.
	gets       atomic.Int32
	listEvents atomic.Int32
	// deleteName is the resource name of the last DeleteSession RPC, and
	// listFilter the filter of the last ListSessions RPC.
	deleteName atomic.Pointer[string]
	listFilter atomic.Pointer[string]
}

func (f *fakeSessions) lastDeleteName() string {
	if p := f.deleteName.Load(); p != nil {
		return *p
	}
	return ""
}

func (f *fakeSessions) GetSession(_ context.Context, req *aiplatformpb.GetSessionRequest) (*aiplatformpb.Session, error) {
	f.gets.Add(1)
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
	f.listEvents.Add(1)
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

// ListSessions records the request's filter and lists no session.
func (f *fakeSessions) ListSessions(_ context.Context, req *aiplatformpb.ListSessionsRequest) (*aiplatformpb.ListSessionsResponse, error) {
	filter := req.GetFilter()
	f.listFilter.Store(&filter)
	return &aiplatformpb.ListSessionsResponse{}, nil
}

func (f *fakeSessions) DeleteSession(_ context.Context, req *aiplatformpb.DeleteSessionRequest) (*longrunningpb.Operation, error) {
	f.deletes.Add(1)
	name := req.GetName()
	f.deleteName.Store(&name)
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

// A delete requested by a non-owner must not reach the backend's DeleteSession.
// It is a no-op, the same as deleting a missing session, which is also what the
// in-memory and database services do.
func TestDelete_wrongUser_isNoOp(t *testing.T) {
	s, fake := newFakeService(t)

	if err := s.Delete(t.Context(), &session.DeleteRequest{AppName: "123", UserID: "user2", SessionID: "owned"}); err != nil {
		t.Errorf("cross-user Delete: got %v, want nil", err)
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

// Get with OmitEvents reads the session and not its events, as adk-python's
// get_session calls only sessions.get for num_recent_events 0. A missing
// session, or another user's, is still session.ErrNotFound. A negative
// NumRecentEvents fails before any RPC, with OmitEvents too.
func TestVertexAIService_Get_OmitEventsAndNegativeNumRecentEvents(t *testing.T) {
	t.Run("owned session", func(t *testing.T) {
		s, fake := newFakeService(t)
		resp, err := s.Get(t.Context(), &session.GetRequest{
			AppName: "123", UserID: "user1", SessionID: "owned",
			OmitEvents: true, NumRecentEvents: 5,
		})
		if err != nil {
			t.Fatalf("Get error = %v", err)
		}
		if got := resp.Session.ID(); got != "owned" {
			t.Errorf("Session.ID() = %q, want %q", got, "owned")
		}
		if got := resp.Session.Events().Len(); got != 0 {
			t.Errorf("Events().Len() = %d, want 0", got)
		}
		if got := fake.gets.Load(); got != 1 {
			t.Errorf("GetSession RPCs = %d, want 1", got)
		}
		if got := fake.listEvents.Load(); got != 0 {
			t.Errorf("ListEvents RPCs = %d, want 0", got)
		}
	})

	for _, tc := range []struct{ name, userID, sessionID string }{
		{"missing session", "user1", "other"},
		{"another user's session", "user2", "owned"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, fake := newFakeService(t)
			_, err := s.Get(t.Context(), &session.GetRequest{
				AppName: "123", UserID: tc.userID, SessionID: tc.sessionID,
				OmitEvents: true,
			})
			if !errors.Is(err, session.ErrNotFound) {
				t.Errorf("Get error = %v, want session.ErrNotFound", err)
			}
			if got := fake.listEvents.Load(); got != 0 {
				t.Errorf("ListEvents RPCs = %d, want 0", got)
			}
		})
	}

	for _, omit := range []bool{false, true} {
		t.Run(fmt.Sprintf("negative NumRecentEvents, OmitEvents %t", omit), func(t *testing.T) {
			s, fake := newFakeService(t)
			_, err := s.Get(t.Context(), &session.GetRequest{
				AppName: "123", UserID: "user1", SessionID: "owned",
				NumRecentEvents: -1, OmitEvents: omit,
			})
			const want = "num_recent_events must not be negative, got -1"
			if err == nil || err.Error() != want {
				t.Errorf("Get with OmitEvents %t: error = %v, want %q", omit, err, want)
			}
			if errors.Is(err, session.ErrNotFound) {
				t.Errorf("Get with OmitEvents %t: error is session.ErrNotFound, want a plain error", omit)
			}
			if got := fake.gets.Load() + fake.listEvents.Load(); got != 0 {
				t.Errorf("Get with OmitEvents %t: sent %d RPCs, want 0", omit, got)
			}
		})
	}
}

// List sends the filter userId="" for user "" when FilterEmptyUserID is set,
// as adk-python sends a filter whenever user_id is not None, and no filter
// for an empty UserID without it.
func TestVertexAIService_List_FilterEmptyUserID(t *testing.T) {
	for _, tc := range []struct {
		name       string
		req        session.ListRequest
		wantFilter string
	}{
		{"every user", session.ListRequest{AppName: "123"}, ""},
		{"user empty", session.ListRequest{AppName: "123", FilterEmptyUserID: true}, `userId=""`},
		{"user u1", session.ListRequest{AppName: "123", UserID: "u1", FilterEmptyUserID: true}, `userId="u1"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, fake := newFakeService(t)
			if _, err := s.List(t.Context(), &tc.req); err != nil {
				t.Fatalf("List error = %v", err)
			}
			got := fake.listFilter.Load()
			if got == nil {
				t.Fatal("List sent no ListSessions RPC")
			}
			if *got != tc.wantFilter {
				t.Errorf("ListSessions filter = %q, want %q", *got, tc.wantFilter)
			}
		})
	}
}
