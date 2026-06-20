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

package llmagent

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// newSessionWithEvent returns an in-memory session.Session preloaded
// with a single user-authored event, plus the count of its events.
func newSessionWithEvent(t *testing.T, text string) session.Session {
	t.Helper()
	svc := session.InMemoryService()
	createResp, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	ev := session.NewEvent("inv-existing")
	ev.Author = "user"
	ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromText(text)},
	}}
	if err := svc.AppendEvent(t.Context(), createResp.Session, ev); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}
	getResp, err := svc.Get(t.Context(), &session.GetRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatalf("session.Get: %v", err)
	}
	return getResp.Session
}

func seedEvent(text string) *session.Event {
	ev := session.NewEvent("inv-seed")
	ev.Author = "user"
	ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromText(text)},
	}}
	return ev
}

// TestWrappedSession_SeedNotPersisted is the regression guard for the
// single_turn node-input contract: the seeded user event must be
// visible when building the LLM prompt (so the agent sees its input)
// but must NOT be written to the underlying session history. Before
// this behavior, the wrapper yielded the seed as a real event and the
// runner persisted it, polluting the conversation with transient node
// inputs (see the wrappedSession TODO in llm_agent_wrapper.go).
func TestWrappedSession_SeedNotPersisted(t *testing.T) {
	t.Parallel()

	base := newSessionWithEvent(t, "existing turn")
	baseLen := base.Events().Len()

	seed := seedEvent("transient node input")
	wrapped := &wrappedSession{Session: base, appended: seed}

	// The wrapped view exposes the seed appended after the originals.
	if got, want := wrapped.Events().Len(), baseLen+1; got != want {
		t.Errorf("wrapped.Events().Len() = %d, want %d", got, want)
	}
	last := wrapped.Events().At(wrapped.Events().Len() - 1)
	if last != seed {
		t.Errorf("last wrapped event = %v, want the seed event", last)
	}

	// The underlying session must be unchanged: the seed must not leak
	// into the real history.
	if got := base.Events().Len(); got != baseLen {
		t.Errorf("underlying session length changed to %d, want %d (seed must not persist)", got, baseLen)
	}
	for ev := range base.Events().All() {
		if ev == seed {
			t.Fatal("seed event leaked into the underlying session history")
		}
	}
}

// TestWrappedSession_AllIteratesOriginalsThenSeed pins iteration
// order: every original event first (in order), then the seed last.
func TestWrappedSession_AllIteratesOriginalsThenSeed(t *testing.T) {
	t.Parallel()

	base := newSessionWithEvent(t, "first")
	seed := seedEvent("seed")
	wrapped := &wrappedSession{Session: base, appended: seed}

	var got []*session.Event
	for ev := range wrapped.Events().All() {
		got = append(got, ev)
	}
	if len(got) == 0 {
		t.Fatal("wrapped.Events().All() yielded nothing")
	}
	if got[len(got)-1] != seed {
		t.Errorf("last yielded event = %v, want the seed event", got[len(got)-1])
	}
}

// TestWrappedSession_NilSeed keeps the wrapped view identical to the
// underlying session when there is no seed to append.
func TestWrappedSession_NilSeed(t *testing.T) {
	t.Parallel()

	base := newSessionWithEvent(t, "only turn")
	wrapped := &wrappedSession{Session: base, appended: nil}

	if got, want := wrapped.Events().Len(), base.Events().Len(); got != want {
		t.Errorf("wrapped.Events().Len() = %d, want %d (nil seed must not add events)", got, want)
	}
}
