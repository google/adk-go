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

// Package benchutil holds fixtures shared by the performance benchmarks.
package benchutil

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// BenchText is the body of every synthetic text event. ~300 bytes, so a
// 1000-event history is ~300 KiB of text, the size of a long chat.
var BenchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7)

// BenchEvents returns n synthetic conversation events authored alternately by
// "user" and agentName. Every fourth model turn is a function call followed by
// a function response, so prompt assembly exercises its call/response
// matching, and every tenth event carries a small state delta.
func BenchEvents(n int, agentName string, start time.Time) []*session.Event {
	events := make([]*session.Event, 0, n)
	for i := 0; i < n; i++ {
		ev := &session.Event{
			ID:           fmt.Sprintf("ev-%06d", i),
			InvocationID: fmt.Sprintf("inv-%06d", i/2),
			Timestamp:    start.Add(time.Duration(i) * time.Second),
		}
		switch {
		case i%2 == 0:
			ev.Author = "user"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(BenchText, genai.RoleUser)}
		case i%8 == 1:
			ev.Author = agentName
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%06d", i), Name: "lookup", Args: map[string]any{"q": "weather"}},
			}}}}
		case i%8 == 3:
			ev.Author = agentName
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: fmt.Sprintf("call-%06d", i-2), Name: "lookup", Response: map[string]any{"result": BenchText}},
			}}}}
		default:
			ev.Author = agentName
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(BenchText, genai.RoleModel)}
		}
		if i%10 == 9 {
			ev.Actions.StateDelta = map[string]any{"turn": i, "temp:scratch": i}
		}
		events = append(events, ev)
	}
	return events
}

// SeedSession creates a session holding n synthetic events in svc and returns
// it as read back from the service.
func SeedSession(ctx context.Context, svc session.Service, appName, userID, sessionID, agentName string, n int) (session.Session, error) {
	return SeedSessionEvents(ctx, svc, appName, userID, sessionID, nil, BenchEvents(n, agentName, time.Unix(1_700_000_000, 0)))
}

// SeedSessionEvents creates a session with the given initial state and
// appends events to it. Events that already carry an ID can be shared
// between in-memory sessions, because the in-memory service writes only a
// missing ID.
func SeedSessionEvents(ctx context.Context, svc session.Service, appName, userID, sessionID string, state map[string]any, events []*session.Event) (session.Session, error) {
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID, State: state})
	if err != nil {
		return nil, err
	}
	for _, ev := range events {
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			return nil, err
		}
	}
	resp, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		return nil, err
	}
	return resp.Session, nil
}

// CountingSessionService wraps a session.Service and counts calls, so a
// benchmark can report how many times a request path hits the store.
type CountingSessionService struct {
	session.Service
	Gets, Creates, Lists, Appends atomic.Int64
	// EventsLoaded sums the events returned by Get, the bytes a backend would
	// have to deserialize.
	EventsLoaded atomic.Int64
}

func (c *CountingSessionService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	c.Creates.Add(1)
	return c.Service.Create(ctx, req)
}

func (c *CountingSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	c.Gets.Add(1)
	resp, err := c.Service.Get(ctx, req)
	if err == nil && resp != nil && resp.Session != nil {
		c.EventsLoaded.Add(int64(resp.Session.Events().Len()))
	}
	return resp, err
}

func (c *CountingSessionService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	c.Lists.Add(1)
	return c.Service.List(ctx, req)
}

func (c *CountingSessionService) AppendEvent(ctx context.Context, s session.Session, ev *session.Event) error {
	c.Appends.Add(1)
	return c.Service.AppendEvent(ctx, s, ev)
}

// Reset zeroes every counter.
func (c *CountingSessionService) Reset() {
	c.Gets.Store(0)
	c.Creates.Store(0)
	c.Lists.Store(0)
	c.Appends.Store(0)
	c.EventsLoaded.Store(0)
}
