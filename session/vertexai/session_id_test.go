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
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/session"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	longrunningpb "cloud.google.com/go/longrunning/autogen/longrunningpb"
)

// sessionIDFake serves CreateSession and GetSession as Agent Engine does: a
// CreateSession without a session id gets the id "server-gen-1". It counts
// both RPCs and keeps the session id of the last CreateSession.
type sessionIDFake struct {
	aiplatformpb.UnimplementedSessionServiceServer

	creates atomic.Int32
	gets    atomic.Int32

	mu        sync.Mutex
	created   *aiplatformpb.Session
	createdID string // the SessionId field of the last CreateSession RPC
}

func (f *sessionIDFake) CreateSession(_ context.Context, req *aiplatformpb.CreateSessionRequest) (*longrunningpb.Operation, error) {
	f.creates.Add(1)
	sess, ok := proto.Clone(req.GetSession()).(*aiplatformpb.Session)
	if !ok {
		return nil, errors.New("unexpected session type")
	}
	id := req.GetSessionId()
	if id == "" {
		id = "server-gen-1"
	}
	sess.Name = req.GetParent() + "/sessions/" + id
	sess.UpdateTime = timestamppb.Now()
	f.mu.Lock()
	f.created = sess
	f.createdID = req.GetSessionId()
	f.mu.Unlock()
	return &longrunningpb.Operation{Name: sess.Name + "/operations/1"}, nil
}

func (f *sessionIDFake) GetSession(context.Context, *aiplatformpb.GetSessionRequest) (*aiplatformpb.Session, error) {
	f.gets.Add(1)
	f.mu.Lock()
	defer f.mu.Unlock()
	sess, ok := proto.Clone(f.created).(*aiplatformpb.Session)
	if !ok {
		return nil, errors.New("unexpected session type")
	}
	return sess, nil
}

func (f *sessionIDFake) lastCreatedID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.createdID
}

// Session ids are checked as adk-python's VertexAiSessionService checks them
// (_extract_short_session_id and _validate_session_id): a resource name of the
// service's engine is reduced to its short id, which is what reaches Agent
// Engine and what the returned session carries; a resource name of another
// engine, or a short id outside ^[A-Za-z0-9_-]+$, fails before any RPC with an
// error that is not session.ErrNotFound. The services in this test are
// configured for engine "123".
func TestVertexAIService_SessionIDValidationAndExtraction(t *testing.T) {
	const ownedName = "projects/p/locations/us-central1/reasoningEngines/123/sessions/owned"
	rejected := []struct {
		name string
		id   string
	}{
		{"other engine", "projects/p/locations/us-central1/reasoningEngines/999/sessions/owned"},
		{"other engine, four segments", "reasoningEngines/999/sessions/owned"},
		{"resource name with empty id", "projects/p/locations/us-central1/reasoningEngines/123/sessions/"},
		{"path traversal", "../owned"},
		{"surrounding spaces", "  owned  "},
		{"trailing newline", "owned\n"},
		{"at sign", "invalid@id"},
		{"blank", "   "},
	}

	for _, tc := range rejected {
		t.Run("Get/"+tc.name, func(t *testing.T) {
			s, fake := newFakeService(t)
			_, err := s.Get(t.Context(), &session.GetRequest{AppName: "123", UserID: "user1", SessionID: tc.id})
			if err == nil || errors.Is(err, session.ErrNotFound) {
				t.Errorf("Get(%q) error = %v, want an error that is not session.ErrNotFound", tc.id, err)
			}
			if got := fake.gets.Load() + fake.listEvents.Load(); got != 0 {
				t.Errorf("Get(%q) sent %d RPCs, want 0", tc.id, got)
			}
		})
		t.Run("Delete/"+tc.name, func(t *testing.T) {
			s, fake := newFakeService(t)
			err := s.Delete(t.Context(), &session.DeleteRequest{AppName: "123", UserID: "user1", SessionID: tc.id})
			if err == nil || errors.Is(err, session.ErrNotFound) {
				t.Errorf("Delete(%q) error = %v, want an error that is not session.ErrNotFound", tc.id, err)
			}
			if got := fake.gets.Load() + fake.deletes.Load(); got != 0 {
				t.Errorf("Delete(%q) sent %d RPCs, want 0", tc.id, got)
			}
		})
		t.Run("Create/"+tc.name, func(t *testing.T) {
			fake := &sessionIDFake{}
			s := &vertexAiService{client: newFakeVertexAiClient(t, fake)}
			_, err := s.Create(t.Context(), &session.CreateRequest{AppName: "123", UserID: "user1", SessionID: tc.id})
			if err == nil || errors.Is(err, session.ErrNotFound) {
				t.Errorf("Create(%q) error = %v, want an error that is not session.ErrNotFound", tc.id, err)
			}
			if got := fake.creates.Load() + fake.gets.Load(); got != 0 {
				t.Errorf("Create(%q) sent %d RPCs, want 0", tc.id, got)
			}
		})
	}

	t.Run("other engine error names both engines", func(t *testing.T) {
		s, _ := newFakeService(t)
		_, err := s.Get(t.Context(), &session.GetRequest{
			AppName: "123", UserID: "user1",
			SessionID: "projects/p/locations/us-central1/reasoningEngines/999/sessions/owned",
		})
		want := `session resource name mismatch: session belongs to reasoningEngine "999", but service is configured for "123"`
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("Get error = %v, want it to contain %q", err, want)
		}
	})

	// An empty id is rejected on Get and Delete, as adk-python's
	// _validate_session_id rejects it; the existing required-field check
	// does it before any RPC.
	t.Run("Get/empty", func(t *testing.T) {
		s, fake := newFakeService(t)
		_, err := s.Get(t.Context(), &session.GetRequest{AppName: "123", UserID: "user1", SessionID: ""})
		if err == nil || errors.Is(err, session.ErrNotFound) {
			t.Errorf("Get(\"\") error = %v, want an error that is not session.ErrNotFound", err)
		}
		if got := fake.gets.Load() + fake.listEvents.Load(); got != 0 {
			t.Errorf("Get(\"\") sent %d RPCs, want 0", got)
		}
	})
	t.Run("Delete/empty", func(t *testing.T) {
		s, fake := newFakeService(t)
		err := s.Delete(t.Context(), &session.DeleteRequest{AppName: "123", UserID: "user1", SessionID: ""})
		if err == nil || errors.Is(err, session.ErrNotFound) {
			t.Errorf("Delete(\"\") error = %v, want an error that is not session.ErrNotFound", err)
		}
		if got := fake.gets.Load() + fake.deletes.Load(); got != 0 {
			t.Errorf("Delete(\"\") sent %d RPCs, want 0", got)
		}
	})

	t.Run("Get/resource name of the service's engine", func(t *testing.T) {
		for _, id := range []string{ownedName, "reasoningEngines/123/sessions/owned", "sessions/owned"} {
			s, _ := newFakeService(t)
			req := &session.GetRequest{AppName: "123", UserID: "user1", SessionID: id}
			resp, err := s.Get(t.Context(), req)
			if err != nil {
				t.Fatalf("Get(%q) error = %v", id, err)
			}
			if got := resp.Session.ID(); got != "owned" {
				t.Errorf("Get(%q).Session.ID() = %q, want %q", id, got, "owned")
			}
			if req.SessionID != id {
				t.Errorf("Get changed the request's SessionID to %q, want it left as %q", req.SessionID, id)
			}
		}
	})

	t.Run("Delete/resource name of the service's engine", func(t *testing.T) {
		s, fake := newFakeService(t)
		req := &session.DeleteRequest{AppName: "123", UserID: "user1", SessionID: ownedName}
		if err := s.Delete(t.Context(), req); err != nil {
			t.Fatalf("Delete error = %v", err)
		}
		if got := fake.deletes.Load(); got != 1 {
			t.Errorf("DeleteSession RPCs = %d, want 1", got)
		}
		if got := fake.lastDeleteName(); got != ownedName {
			t.Errorf("DeleteSession RPC name = %q, want %q", got, ownedName)
		}
		if req.SessionID != ownedName {
			t.Errorf("Delete changed the request's SessionID to %q, want it left as %q", req.SessionID, ownedName)
		}
	})

	t.Run("Create/resource name of the service's engine", func(t *testing.T) {
		const name = "projects/p/locations/us-central1/reasoningEngines/123/sessions/custom-1"
		fake := &sessionIDFake{}
		s := &vertexAiService{client: newFakeVertexAiClient(t, fake)}
		req := &session.CreateRequest{AppName: "123", UserID: "user1", SessionID: name}
		resp, err := s.Create(t.Context(), req)
		if err != nil {
			t.Fatalf("Create error = %v", err)
		}
		if got := fake.lastCreatedID(); got != "custom-1" {
			t.Errorf("CreateSession RPC session_id = %q, want %q", got, "custom-1")
		}
		if got := resp.Session.ID(); got != "custom-1" {
			t.Errorf("Create().Session.ID() = %q, want %q", got, "custom-1")
		}
		if req.SessionID != name {
			t.Errorf("Create changed the request's SessionID to %q, want it left as %q", req.SessionID, name)
		}
	})

	// adk-python checks the id on create only when one is given; without one
	// Agent Engine assigns it.
	t.Run("Create/empty id is assigned by the server", func(t *testing.T) {
		fake := &sessionIDFake{}
		s := &vertexAiService{client: newFakeVertexAiClient(t, fake)}
		resp, err := s.Create(t.Context(), &session.CreateRequest{AppName: "123", UserID: "user1"})
		if err != nil {
			t.Fatalf("Create error = %v", err)
		}
		if got := resp.Session.ID(); got != "server-gen-1" {
			t.Errorf("Create().Session.ID() = %q, want %q", got, "server-gen-1")
		}
		if got := fake.creates.Load(); got != 1 {
			t.Errorf("CreateSession RPCs = %d, want 1", got)
		}
	})
}
