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

package llminternal

import (
	"testing"
	"time"

	"iter"

	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
)

func TestAuthPreprocessorResultStoredInContext(t *testing.T) {
	ctx := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})

	if got := authPreprocessorResultFromContext(ctx); got != nil {
		t.Fatalf("authPreprocessorResultFromContext() = %v, want nil", got)
	}

	want := &AuthPreprocessorResult{
		ToolIdsToResume: map[string]bool{"tool": true},
	}
	storeAuthPreprocessorResult(ctx, want)

	if got := authPreprocessorResultFromContext(ctx); got != want {
		t.Fatalf("authPreprocessorResultFromContext() = %v, want %v", got, want)
	}

	storeAuthPreprocessorResult(ctx, nil)
	if got := authPreprocessorResultFromContext(ctx); got != nil {
		t.Fatalf("authPreprocessorResultFromContext() after reset = %v, want nil", got)
	}
}

func TestAuthEventProcessedTracking(t *testing.T) {
	session := newFakeSession()
	ctx := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{
		Session: session,
	})

	const eventID = "event-1"

	processed, err := authEventAlreadyProcessed(ctx, eventID)
	if err != nil {
		t.Fatalf("authEventAlreadyProcessed(%q) error = %v", eventID, err)
	}
	if processed {
		t.Fatalf("authEventAlreadyProcessed(%q) = true, want false", eventID)
	}

	if err := markAuthEventProcessed(ctx, eventID); err != nil {
		t.Fatalf("markAuthEventProcessed(%q) error = %v", eventID, err)
	}

	processed, err = authEventAlreadyProcessed(ctx, eventID)
	if err != nil {
		t.Fatalf("authEventAlreadyProcessed(%q) error = %v", eventID, err)
	}
	if !processed {
		t.Fatalf("authEventAlreadyProcessed(%q) = false, want true", eventID)
	}
}

type fakeState struct {
	values map[string]any
}

func (s *fakeState) Get(key string) (any, error) {
	if v, ok := s.values[key]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}

func (s *fakeState) Set(key string, value any) error {
	if s.values == nil {
		s.values = make(map[string]any)
	}
	s.values[key] = value
	return nil
}

func (s *fakeState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.values {
			if !yield(k, v) {
				return
			}
		}
	}
}

type fakeEvents struct {
	events []*session.Event
}

func (e *fakeEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, event := range e.events {
			if !yield(event) {
				return
			}
		}
	}
}

func (e *fakeEvents) Len() int {
	return len(e.events)
}

func (e *fakeEvents) At(i int) *session.Event {
	return e.events[i]
}

type fakeSession struct {
	state  *fakeState
	events *fakeEvents
}

func newFakeSession() *fakeSession {
	return &fakeSession{
		state:  &fakeState{values: make(map[string]any)},
		events: &fakeEvents{},
	}
}

func (s *fakeSession) ID() string {
	return "session"
}

func (s *fakeSession) AppName() string {
	return "app"
}

func (s *fakeSession) UserID() string {
	return "user"
}

func (s *fakeSession) State() session.State {
	return s.state
}

func (s *fakeSession) Events() session.Events {
	return s.events
}

func (s *fakeSession) LastUpdateTime() time.Time {
	return time.Time{}
}

var _ session.Session = (*fakeSession)(nil)
