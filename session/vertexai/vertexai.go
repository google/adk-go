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

// Package vertexai provides a session.Service backed by the Vertex AI
// Agent Engine runtime.
package vertexai

import (
	"context"
	"fmt"

	"golang.org/x/sync/errgroup"
	"google.golang.org/api/option"

	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/session"
)

// VertexAiSessionService
type vertexAiService struct {
	client *vertexAiClient
}

// VertexAIServiceConfig configures a Vertex AI-backed session service.
type VertexAIServiceConfig struct {
	// ProjectID with VertexAI API enabled.
	ProjectID string
	// Location where the reasoningEngine is running.
	Location string
	// ReasoningEngine is the runtime in the agent engine which will store the
	// sessions.
	// Optimal way is to create reasoningEngine per app.
	// For example, a reasoningEngine can be created via the Vertex AI REST
	// API's 'projects.locations.reasoningEngines.create' method.
	ReasoningEngine string
}

// NewSessionService returns a [session.Service] that stores sessions in the
// Vertex AI Agent Engine runtime cfg names.
//
// Create, Get and Delete require an app name and a user ID; an empty user ID
// is an error, because Agent Engine requires one. A session ID must be a
// short ID matching ^[A-Za-z0-9_-]+$, or a resource name ending in
// "/sessions/<id>", which is reduced to <id> (a name of another reasoning
// engine is an error); any other ID, the empty one included in Get and
// Delete, is an error, returned before any request is sent. Create with an
// empty session ID has Agent Engine assign one. A Create of a taken session
// ID returns Agent Engine's error, an INVALID_ARGUMENT that says the session
// already exists, and does not wrap [session.ErrAlreadyExists], as
// adk-python's Vertex AI service raises no AlreadyExistsError. Get of a
// missing session, or of another
// user's, returns an error wrapping [session.ErrNotFound], and Delete of one
// returns nil.
//
// AppendEvent accepts any [session.Session] value, as
// [session.Service.AppendEvent] describes, and departs from that contract in
// two ways. Its request names the session by app name and session ID only,
// as adk-python's does, so a value carrying another user's ID appends to that
// user's session; checking the owner would cost a read per append. And an
// append whose request fails in transport, after Agent Engine has committed
// it, returns an error with the event stored; the client cannot tell the two
// cases apart.
func NewSessionService(ctx context.Context, cfg VertexAIServiceConfig, opts ...option.ClientOption) (session.Service, error) {
	client, err := newVertexAiClient(ctx, cfg.Location, cfg.ProjectID, cfg.ReasoningEngine, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create Vertex AI client: %w", err)
	}

	return &vertexAiService{client: client}, nil
}

func (s *vertexAiService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}
	// Without an id Agent Engine assigns one, so only a given id is checked,
	// as in adk-python's create_session.
	if req.SessionID != "" {
		id, err := s.client.resolveSessionID(req.AppName, req.SessionID)
		if err != nil {
			return nil, err
		}
		r := *req
		r.SessionID = id
		req = &r
	}
	sess, err := s.client.createSession(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}
	return &session.CreateResponse{Session: sess}, nil
}

func (s *vertexAiService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, fmt.Errorf("app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}
	if err := sessionutils.CheckNumRecentEvents(req.NumRecentEvents); err != nil {
		return nil, err
	}
	id, err := s.client.resolveSessionID(req.AppName, req.SessionID)
	if err != nil {
		return nil, err
	}
	r := *req
	r.SessionID = id
	req = &r

	if req.OmitEvents {
		sess, err := s.client.getSession(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("failed to get session: %w", err)
		}
		return &session.GetResponse{Session: sess}, nil
	}

	// gCtx will be canceled if either function returns an error
	g, gCtx := errgroup.WithContext(ctx)

	var (
		sess      *localSession
		events    []*session.Event
		getErr    error
		eventsErr error
	)

	g.Go(func() error {
		sess, getErr = s.client.getSession(gCtx, req)
		if getErr != nil {
			return fmt.Errorf("failed to get session: %w", getErr)
		}
		return nil
	})

	g.Go(func() error {
		events, eventsErr = s.client.listSessionEvents(gCtx, req.AppName, req.SessionID, req.After, req.NumRecentEvents)
		if eventsErr != nil {
			return fmt.Errorf("failed to list session events: %w", eventsErr)
		}
		return nil
	})

	if err := g.Wait(); err != nil {
		// Both calls run concurrently against the same session, so a missing
		// one can surface from either, and whichever loses the race reports the
		// context cancellation instead. Report the missing session whenever
		// either call saw it, so callers can rely on
		// errors.Is(err, session.ErrNotFound).
		for _, callErr := range []error{getErr, eventsErr} {
			if callErr != nil && isNotFoundError(callErr) {
				return nil, fmt.Errorf("%w: %q: %w", session.ErrNotFound, req.SessionID, callErr)
			}
		}
		return nil, err
	}
	sess.events = events
	return &session.GetResponse{Session: sess}, nil
}

func (s *vertexAiService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	sessions, err := s.client.listSessions(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to request sessions list: %w", err)
	}
	return &session.ListResponse{Sessions: sessions}, nil
}

func (s *vertexAiService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return fmt.Errorf("app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}
	id, err := s.client.resolveSessionID(req.AppName, req.SessionID)
	if err != nil {
		return err
	}
	r := *req
	r.SessionID = id
	err = s.client.deleteSession(ctx, &r)
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}

func (s *vertexAiService) AppendEvent(ctx context.Context, sess session.Session, event *session.Event) error {
	if sess.ID() == "" || event == nil {
		return fmt.Errorf("session_id and event are required, got session_id: %q, event_id: %t", sess.ID(), event == nil)
	}
	// temp: keys are scoped to the current invocation and must not reach storage.
	// The session value below still receives the untrimmed state delta, which
	// is what keeps them readable for the rest of the invocation.
	err := s.client.appendEvent(ctx, sess.AppName(), sess.ID(), trimTempDeltaState(event))
	if err != nil {
		return fmt.Errorf("failed to append event: %w", err)
	}
	if err := commitToSession(sess, event); err != nil {
		return fmt.Errorf("event %q is stored in session %q, but the session value was not updated: %w", event.ID, sess.ID(), err)
	}
	return nil
}

// commitToSession applies a stored, non-partial event to sess, whatever its
// concrete type, as adk-python's append_event updates the Session object it is
// given (base_session_service.py:182-186). A value this service, another
// adk-go service or [session.CloneSession] made takes the event without its
// temp: keys, the whole state delta and the event's timestamp. A value the
// caller built takes only the state delta, through its State when State is
// non-nil, and the first error its Set returns is returned.
func commitToSession(sess session.Session, event *session.Event) error {
	if event.Partial {
		return nil
	}
	switch s := sess.(type) {
	case *localSession:
		return s.appendEvent(event)
	case sessionutils.MutableSession:
		s.CommitAppend(trimTempDeltaState(event), event.Actions.StateDelta, event.Timestamp)
		return nil
	}
	st := sess.State()
	if st == nil {
		return nil
	}
	for k, v := range event.Actions.StateDelta {
		if err := st.Set(k, v); err != nil {
			return err
		}
	}
	return nil
}
