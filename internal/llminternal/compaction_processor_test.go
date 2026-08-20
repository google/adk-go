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

package llminternal_test

import (
	"context"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/compactioninternal"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// seedWrappedSession stands in for the wrapper an agent installs over the real
// session when it hands a sub-agent a synthetic first turn. It decorates the
// session and is not a type any session service recognises.
type seedWrappedSession struct {
	session.Session
}

func (w *seedWrappedSession) Unwrap() session.Session { return w.Session }

// fixedSummarizer returns one canned summary.
type fixedSummarizer struct{ calls int }

func (s *fixedSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	s.calls++
	return genai.NewContentFromText("SUMMARY", "model"), nil, nil
}

// tailRetentionFixture builds a stored session holding n exchanges, the last of
// which reports a prompt token count well past any threshold a test will set.
func tailRetentionFixture(t *testing.T, n int) (session.Service, session.Session) {
	t.Helper()

	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	sess := created.Session

	base := time.Unix(1, 0)
	for i := range n {
		q := session.NewEvent(t.Context(), "inv")
		q.Author = "user"
		q.Timestamp = base.Add(time.Duration(2*i) * time.Second)
		q.LLMResponse.Content = genai.NewContentFromText("question", "user")
		if err := svc.AppendEvent(t.Context(), sess, q); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}

		a := session.NewEvent(t.Context(), "inv")
		a.Author = "assistant"
		a.Timestamp = base.Add(time.Duration(2*i+1) * time.Second)
		a.LLMResponse.Content = genai.NewContentFromText("answer", "model")
		a.LLMResponse.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5000}
		if err := svc.AppendEvent(t.Context(), sess, a); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}
	return svc, sess
}

func runCompactionProcessor(t *testing.T, svc session.Service, sess session.Session, cfg *compaction.Config) error {
	t.Helper()

	ctx := compactionctx.ToContext(t.Context(), compactionctx.New(cfg, svc))
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "assistant", Model: &testModel{}}))
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: sess,
	})

	var gotErr error
	for ev, err := range llminternal.CompactionRequestProcessor(ictx, &model.LLMRequest{}, &llminternal.Flow{}) {
		if ev != nil {
			t.Fatal("CompactionRequestProcessor yielded an event, which it must not do")
		}
		if err != nil {
			gotErr = err
		}
	}
	return gotErr
}

func storedCompactions(t *testing.T, svc session.Service) []*session.Event {
	t.Helper()

	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var out []*session.Event
	for ev := range resp.Session.Events().All() {
		if compactioninternal.HasUsableSummary(ev) {
			out = append(out, ev)
		}
	}
	return out
}

// TestCompactionProcessorAppendsThroughASessionWrapper checks that tail
// retention still works when the invocation carries a wrapped session.
//
// An agent hands a sub-agent a session wrapped to carry a synthetic first turn.
// Every session service type-asserts on its own concrete type, so appending a
// summary to the wrapper fails outright. The failure does not even surface as an
// error on the delegating path: it becomes a tool-error response and the
// coordinator answers on top of a broken delegation.
func TestCompactionProcessorAppendsThroughASessionWrapper(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)
	summarizer := &fixedSummarizer{}

	err := runCompactionProcessor(t, svc, &seedWrappedSession{Session: sess}, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	})
	if err != nil {
		t.Fatalf("compaction through a wrapped session failed: %v", err)
	}
	if summarizer.calls == 0 {
		t.Fatal("the summarizer never ran, so this test proved nothing")
	}
	if got := len(storedCompactions(t, svc)); got != 1 {
		t.Errorf("stored %d compaction events, want 1: the summary never reached the session", got)
	}
}

// TestCompactionProcessorSkipsOnCancelledContext checks that a cancelled turn
// does not spend a model call on compaction, nor write a summary the caller is
// no longer waiting for.
func TestCompactionProcessorSkipsOnCancelledContext(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)
	summarizer := &fixedSummarizer{}

	ctx, cancel := context.WithCancel(t.Context())
	ctx = compactionctx.ToContext(ctx, compactionctx.New(&compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	}, svc))
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "assistant", Model: &testModel{}}))
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: sess,
	})
	cancel()

	for range llminternal.CompactionRequestProcessor(ictx, &model.LLMRequest{}, &llminternal.Flow{}) { //nolint:revive
	}

	if summarizer.calls != 0 {
		t.Errorf("summarizer ran %d time(s) on a cancelled turn", summarizer.calls)
	}
	if got := len(storedCompactions(t, svc)); got != 0 {
		t.Errorf("stored %d compaction events on a cancelled turn", got)
	}
}

// racingSummarizer appends an event inside the range it is about to summarize,
// standing in for a concurrent invocation landing during the model call.
type racingSummarizer struct {
	svc session.Service
	t   *testing.T

	// The event this summarizer landed mid-call.
	racedInvocation string
	racedTimestamp  time.Time
}

func (s *racingSummarizer) SummarizeEvents(ctx context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	// Land a new event inside the range the summary claims, through a separate
	// handle on the same stored session. Appending through the caller's handle
	// would update that handle too, which is exactly what a concurrent
	// invocation in another goroutine does not do.
	other, err := s.svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		s.t.Fatalf("racing Get() error = %v", err)
	}
	late := session.NewEvent(ctx, "other-invocation")
	late.Author = "user"
	// Inside the range the framework will derive from the window it handed over.
	late.Timestamp = events[0].Timestamp.Add(time.Millisecond)
	late.LLMResponse.Content = genai.NewContentFromText("CONCURRENT", "user")
	if err := s.svc.AppendEvent(ctx, other.Session, late); err != nil {
		s.t.Fatalf("racing AppendEvent() error = %v", err)
	}
	s.racedInvocation, s.racedTimestamp = late.InvocationID, late.Timestamp
	return genai.NewContentFromText("SUMMARY", "model"), nil, nil
}

// TestCompactionProcessorDiscardsARacedSummary checks that a summary is thrown
// away when another invocation appended inside its range while it was being
// produced.
//
// A summary records the holes inside its range, and that list is computed from
// what the framework could see when it was built. An event that lands
// afterwards is inside the range and named by nothing, so it reads as covered
// and prompt assembly drops it, having been summarized by nothing. Discarding
// costs one wasted model call; keeping it costs a turn of conversation.
func TestCompactionProcessorDiscardsARacedSummary(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)

	summarizer := &racingSummarizer{svc: svc, t: t}
	err := runCompactionProcessor(t, svc, sess, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	})
	if err != nil {
		t.Fatalf("CompactionRequestProcessor failed: %v", err)
	}
	if summarizer.racedInvocation == "" {
		t.Fatal("the racing summarizer did not record what it appended")
	}
	if got := len(storedCompactions(t, svc)); got != 0 {
		t.Errorf("stored %d compaction events, want 0: a summary whose range was raced must be discarded", got)
	}
}
