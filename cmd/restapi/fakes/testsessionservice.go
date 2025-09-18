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

// package fakes contains a fake implementation of different ADK services used for testing

package fakes

import (
	"context"
	"fmt"
	"iter"
	"time"

	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
)

type TestState map[string]any

func (s TestState) Get(key string) (any, error) {
	return s[key], nil
}

func (s TestState) All() iter.Seq2[string, any] {
	return func(yield func(key string, val any) bool) {
		for k, v := range s {
			if !yield(k, v) {
				return
			}
		}
	}
}

type TestEvents []*session.Event

func (e TestEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, event := range e {
			if !yield(event) {
				return
			}
		}
	}
}

func (e TestEvents) Len() int {
	return len(e)
}

func (e TestEvents) At(i int) *session.Event {
	return e[i]
}

type TestSession struct {
	Id            session.ID
	SessionState  TestState
	SessionEvents TestEvents
	UpdatedAt     time.Time
}

func (s TestSession) ID() session.ID {
	return s.Id
}

func (s TestSession) State() session.ReadOnlyState {
	return s.SessionState
}

func (s TestSession) Events() session.Events {
	return s.SessionEvents
}

func (s TestSession) Updated() time.Time {
	return s.UpdatedAt
}

type FakeSessionService struct {
	Sessions map[session.ID]TestSession
}

func (s *FakeSessionService) Create(ctx context.Context, req *sessionservice.CreateRequest) (*sessionservice.CreateResponse, error) {
	if _, ok := s.Sessions[session.ID{AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID}]; ok {
		return nil, fmt.Errorf("session already exists")
	}

	if req.SessionID == "" {
		req.SessionID = "testID"
	}

	session := TestSession{
		Id:           session.ID{AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID},
		SessionState: req.State,
		UpdatedAt:    time.Now(),
	}
	s.Sessions[session.Id] = session
	return &sessionservice.CreateResponse{
		Session: &session,
	}, nil
}

func (s *FakeSessionService) Get(ctx context.Context, req *sessionservice.GetRequest) (*sessionservice.GetResponse, error) {
	if session, ok := s.Sessions[req.ID]; ok {
		return &sessionservice.GetResponse{
			Session: &session,
		}, nil
	}
	return nil, fmt.Errorf("not found")
}

func (s *FakeSessionService) List(ctx context.Context, req *sessionservice.ListRequest) (*sessionservice.ListResponse, error) {
	result := []sessionservice.StoredSession{}
	for _, session := range s.Sessions {
		if session.Id.AppName != req.AppName || session.Id.UserID != req.UserID {
			continue
		}
		result = append(result, session)
	}
	return &sessionservice.ListResponse{
		Sessions: result,
	}, nil
}

func (s *FakeSessionService) Delete(ctx context.Context, req *sessionservice.DeleteRequest) error {
	if _, ok := s.Sessions[req.ID]; !ok {
		return fmt.Errorf("not found")
	}
	delete(s.Sessions, req.ID)
	return nil
}

func (s *FakeSessionService) AppendEvent(ctx context.Context, session sessionservice.StoredSession, event *session.Event) error {
	TestSession, ok := session.(*TestSession)
	if !ok {
		return fmt.Errorf("invalid session type")
	}
	TestSession.SessionEvents = append(TestSession.SessionEvents, event)
	TestSession.UpdatedAt = event.Time
	s.Sessions[TestSession.Id] = *TestSession
	return nil
}

var _ sessionservice.Service = (*FakeSessionService)(nil)
