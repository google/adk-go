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

// Package fakes contains a fake implementation of different ADK services used for testing

package fakes

import (
	"context"
	"fmt"
	"iter"
	"time"

	"google.golang.org/adk/session"
)

type TestState map[string]any

func (s TestState) Get(key string) (any, error) {
	return s[key], nil
}

func (s TestState) Set(key string, val any) error {
	s[key] = val
	return nil
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
	Id            SessionKey
	SessionState  TestState
	SessionEvents TestEvents
	UpdatedAt     time.Time
}

func (s TestSession) ID() string {
	return s.Id.SessionID
}

func (s TestSession) AppName() string {
	return s.Id.AppName
}

func (s TestSession) UserID() string {
	return s.Id.UserID
}

func (s TestSession) State() session.State {
	return s.SessionState
}

func (s TestSession) Events() session.Events {
	return s.SessionEvents
}

func (s TestSession) LastUpdateTime() time.Time {
	return s.UpdatedAt
}

type FakeSessionService struct {
	Sessions map[SessionKey]TestSession
}

type SessionKey struct {
	AppName   string
	UserID    string
	SessionID string
}

func (s *FakeSessionService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if _, ok := s.Sessions[SessionKey{AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID}]; ok {
		return nil, fmt.Errorf("session already exists")
	}

	if req.SessionID == "" {
		req.SessionID = "testID"
	}

	testSession := TestSession{
		Id: SessionKey{
			AppName:   req.AppName,
			UserID:    req.UserID,
			SessionID: req.SessionID,
		},
		SessionState: req.State,
		UpdatedAt:    time.Now(),
	}
	s.Sessions[SessionKey{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: req.SessionID,
	}] = testSession
	return &session.CreateResponse{
		Session: &testSession,
	}, nil
}

func (s *FakeSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if sess, ok := s.Sessions[SessionKey{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: req.SessionID,
	}]; ok {
		return &session.GetResponse{
			Session: &sess,
		}, nil
	}
	return nil, fmt.Errorf("not found")
}

func (s *FakeSessionService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	result := []session.Session{}
	for _, session := range s.Sessions {
		if session.Id.AppName != req.AppName || session.Id.UserID != req.UserID {
			continue
		}
		result = append(result, session)
	}
	return &session.ListResponse{
		Sessions: result,
	}, nil
}

func (s *FakeSessionService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	id := SessionKey{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: req.SessionID,
	}
	if _, ok := s.Sessions[id]; !ok {
		return fmt.Errorf("not found")
	}
	delete(s.Sessions, id)
	return nil
}

func (s *FakeSessionService) AppendEvent(ctx context.Context, curSession session.Session, event *session.Event) error {
	testSession, ok := curSession.(*TestSession)
	if !ok {
		return fmt.Errorf("invalid session type")
	}
	testSession.SessionEvents = append(testSession.SessionEvents, event)
	testSession.UpdatedAt = event.Timestamp
	s.Sessions[testSession.Id] = *testSession
	return nil
}

var _ session.Service = (*FakeSessionService)(nil)
