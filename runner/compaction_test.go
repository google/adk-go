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

package runner

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// scriptedModel answers every request with a canned reply and records the
// prompts it received, so a test can assert what history the model actually saw.
type scriptedModel struct {
	mu       sync.Mutex
	prompts  [][]*genai.Content
	replyFmt string
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.prompts = append(m.prompts, req.Contents)
	n := len(m.prompts)
	m.mu.Unlock()

	reply := fmt.Sprintf(m.replyFmt, n)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(reply, "model")}, nil)
	}
}

func (m *scriptedModel) lastPrompt() []*genai.Content {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.prompts) == 0 {
		return nil
	}
	return m.prompts[len(m.prompts)-1]
}

// recordingSummarizer produces a fixed summary and records how often it ran.
type recordingSummarizer struct {
	mu      sync.Mutex
	summary string
	windows [][]string // authors of the events in each window
}

func (s *recordingSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (*session.Event, error) {
	s.mu.Lock()
	authors := make([]string, len(events))
	for i, ev := range events {
		authors[i] = ev.Author
	}
	s.windows = append(s.windows, authors)
	s.mu.Unlock()

	return compaction.NewSummaryEvent(events, genai.NewContentFromText(s.summary, "model"), nil)
}

func (s *recordingSummarizer) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.windows)
}

// drain consumes a run to completion, failing the test on any error.
func drain(t *testing.T, stream iter.Seq2[*session.Event, error]) {
	t.Helper()
	for _, err := range stream {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
	}
}

// compactionEventsIn returns the compaction events currently stored in sess.
func compactionEventsIn(sess session.Session) []*session.Event {
	var out []*session.Event
	for ev := range sess.Events().All() {
		if compaction.IsCompactionEvent(ev) {
			out = append(out, ev)
		}
	}
	return out
}

func newCompactionRunner(t *testing.T, m model.LLM, cfg *compaction.Config) (*Runner, session.Service) {
	t.Helper()

	root, err := llmagent.New(llmagent.Config{Name: "assistant", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	svc := session.InMemoryService()
	r, err := New(Config{
		AppName:                "compaction_app",
		Agent:                  root,
		SessionService:         svc,
		AutoCreateSession:      true,
		EventsCompactionConfig: cfg,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r, svc
}

func getSession(t *testing.T, svc session.Service, userID, sessionID string) session.Session {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{
		AppName: "compaction_app", UserID: userID, SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return resp.Session
}

func TestRunnerCompactsAfterInterval(t *testing.T) {
	t.Parallel()

	const userID, sessionID = "u", "s"
	m := &scriptedModel{replyFmt: "answer %d"}
	summarizer := &recordingSummarizer{summary: "Earlier the user asked some questions."}
	r, svc := newCompactionRunner(t, m, &compaction.Config{
		CompactionInterval: 2,
		OverlapSize:        1,
		Summarizer:         summarizer,
	})

	// First turn: below the interval, nothing compacts.
	drain(t, r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("q1", genai.RoleUser), agent.RunConfig{}))
	if got := summarizer.calls(); got != 0 {
		t.Fatalf("summarizer ran %d times after one invocation, want 0", got)
	}

	// Second turn: the interval is reached.
	drain(t, r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("q2", genai.RoleUser), agent.RunConfig{}))
	if got := summarizer.calls(); got != 1 {
		t.Fatalf("summarizer ran %d times after two invocations, want 1", got)
	}

	sess := getSession(t, svc, userID, sessionID)
	compactions := compactionEventsIn(sess)
	if len(compactions) != 1 {
		t.Fatalf("session holds %d compaction events, want 1", len(compactions))
	}
	stored := compactions[0]
	if stored.ID == "" {
		t.Error("stored compaction event has no ID")
	}
	if stored.InvocationID == "" {
		t.Error("stored compaction event has no InvocationID")
	}
	if stored.Timestamp.IsZero() {
		t.Error("stored compaction event has no Timestamp")
	}
	if !stored.Timestamp.After(stored.Actions.Compaction.EndTimestamp) {
		t.Errorf("compaction event timestamp %v must be after the range it covers (ends %v), or the next Apply will not see it as covering those events",
			stored.Timestamp, stored.Actions.Compaction.EndTimestamp)
	}

	// The window covered both turns: user q1, model a1, user q2, model a2.
	if got, want := len(summarizer.windows[0]), 4; got != want {
		t.Errorf("compaction window held %d events, want %d (authors: %v)", got, want, summarizer.windows[0])
	}
}

func TestRunnerCompactionShrinksTheNextPrompt(t *testing.T) {
	t.Parallel()

	const userID, sessionID = "u", "s"
	m := &scriptedModel{replyFmt: "answer %d"}
	summarizer := &recordingSummarizer{summary: "SUMMARY-OF-EARLIER-TURNS"}
	r, _ := newCompactionRunner(t, m, &compaction.Config{
		CompactionInterval: 2,
		Summarizer:         summarizer,
	})

	drain(t, r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("q1", genai.RoleUser), agent.RunConfig{}))
	drain(t, r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("q2", genai.RoleUser), agent.RunConfig{}))
	// This third turn's prompt is the first one built after a compaction landed.
	drain(t, r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("q3", genai.RoleUser), agent.RunConfig{}))

	prompt := promptText(m.lastPrompt())
	if !strings.Contains(prompt, "SUMMARY-OF-EARLIER-TURNS") {
		t.Errorf("prompt does not contain the summary:\n%s", prompt)
	}
	for _, gone := range []string{"q1", "q2", "answer 1", "answer 2"} {
		if strings.Contains(prompt, gone) {
			t.Errorf("prompt still contains compacted turn %q:\n%s", gone, prompt)
		}
	}
	if !strings.Contains(prompt, "q3") {
		t.Errorf("prompt is missing the current turn:\n%s", prompt)
	}
}

func TestRunnerWithoutCompactionConfigNeverCompacts(t *testing.T) {
	t.Parallel()

	const userID, sessionID = "u", "s"
	m := &scriptedModel{replyFmt: "answer %d"}
	r, svc := newCompactionRunner(t, m, nil)

	for range 4 {
		drain(t, r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("q", genai.RoleUser), agent.RunConfig{}))
	}

	if got := len(compactionEventsIn(getSession(t, svc, userID, sessionID))); got != 0 {
		t.Errorf("session holds %d compaction events, want 0 when compaction is not configured", got)
	}
}

func TestRunnerCompactionSummaryIsNotYielded(t *testing.T) {
	t.Parallel()

	const userID, sessionID = "u", "s"
	m := &scriptedModel{replyFmt: "answer %d"}
	summarizer := &recordingSummarizer{summary: "summary"}
	r, _ := newCompactionRunner(t, m, &compaction.Config{CompactionInterval: 1, Summarizer: summarizer})

	var yielded []*session.Event
	for ev, err := range r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("q1", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		yielded = append(yielded, ev)
	}

	// The summary is bookkeeping for the next prompt, not part of the
	// conversation, so callers must not observe it in the event stream.
	for _, ev := range yielded {
		if compaction.IsCompactionEvent(ev) {
			t.Errorf("Run yielded a compaction event, want it persisted silently")
		}
	}
	if summarizer.calls() == 0 {
		t.Error("summarizer never ran, so this test proved nothing")
	}
}

func TestNewRejectsBadCompactionConfig(t *testing.T) {
	t.Parallel()

	llmRoot, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &scriptedModel{replyFmt: "a%d"}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	plainRoot, err := agent.New(agent.Config{Name: "plain"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	tests := []struct {
		name    string
		root    agent.Agent
		cfg     *compaction.Config
		wantErr bool
	}{
		{name: "nil config is fine", root: llmRoot},
		{name: "valid sliding window", root: llmRoot, cfg: &compaction.Config{CompactionInterval: 2, OverlapSize: 1}},
		{name: "negative interval", root: llmRoot, cfg: &compaction.Config{CompactionInterval: -1}, wantErr: true},
		{name: "no strategy enabled", root: llmRoot, cfg: &compaction.Config{}, wantErr: true},
		{
			name:    "non-LLM root without an explicit summarizer",
			root:    plainRoot,
			cfg:     &compaction.Config{CompactionInterval: 2},
			wantErr: true,
		},
		{
			name: "non-LLM root with an explicit summarizer",
			root: plainRoot,
			cfg:  &compaction.Config{CompactionInterval: 2, Summarizer: &recordingSummarizer{summary: "s"}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := New(Config{
				AppName:                "app",
				Agent:                  tc.root,
				SessionService:         session.InMemoryService(),
				EventsCompactionConfig: tc.cfg,
			})
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Errorf("New() error = %v, wantErr %t", err, tc.wantErr)
			}
		})
	}
}

func TestNewDoesNotMutateCallerCompactionConfig(t *testing.T) {
	t.Parallel()

	root, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &scriptedModel{replyFmt: "a%d"}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	cfg := &compaction.Config{CompactionInterval: 2}

	if _, err := New(Config{
		AppName:                "app",
		Agent:                  root,
		SessionService:         session.InMemoryService(),
		EventsCompactionConfig: cfg,
	}); err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// A caller sharing one config across runners must not find a summarizer
	// bound to some other runner's root agent silently installed on it.
	if cfg.Summarizer != nil {
		t.Error("New() installed the default summarizer on the caller's config, want the caller's config left untouched")
	}
}

func promptText(contents []*genai.Content) string {
	var b strings.Builder
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.Text != "" {
				fmt.Fprintf(&b, "[%s] %s\n", c.Role, p.Text)
			}
		}
	}
	return b.String()
}

func (failingSummarizer) SummarizeEvents(context.Context, []*session.Event) (*session.Event, error) {
	return nil, errors.New("summarizer exploded")
}

// TestRunnerPostInvocationCompactionFailureSurfaces pins that a post-invocation
// compaction failure reaches the caller rather than being logged and dropped.
//
// Swallowing it would let a session grow unbounded, with the first visible
// symptom arriving much later as a context-limit error on some unrelated turn.
func TestRunnerPostInvocationCompactionFailureSurfaces(t *testing.T) {
	t.Parallel()

	const userID, sessionID = "u", "s"
	m := &scriptedModel{replyFmt: "answer %d"}
	r, svc := newCompactionRunner(t, m, &compaction.Config{
		CompactionInterval: 1,
		Summarizer:         failingSummarizer{},
	})

	var yielded []*session.Event
	var gotErr error
	for ev, err := range r.Run(t.Context(), userID, sessionID,
		genai.NewContentFromText("q1", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			gotErr = err
			break
		}
		yielded = append(yielded, ev)
	}

	if gotErr == nil {
		t.Fatal("run succeeded despite a failing post-invocation summarizer, want the error surfaced")
	}
	if !strings.Contains(gotErr.Error(), "compaction") {
		t.Errorf("error %q does not mention compaction, so the cause is hard to find", gotErr)
	}

	// The turn's own events are already committed, so the caller keeps
	// everything the agent produced; only the shrink failed.
	if len(yielded) == 0 {
		t.Error("no events were yielded before the compaction error; the turn's own output must be preserved")
	}
	events := sessionEventsOf(t, svc, userID, sessionID)
	if len(events) == 0 {
		t.Error("session holds no events; the turn's output must still be persisted")
	}
	if got := len(compactionEventsIn(getSession(t, svc, userID, sessionID))); got != 0 {
		t.Errorf("session holds %d compaction events after a failed summarizer, want 0", got)
	}
}

func sessionEventsOf(t *testing.T, svc session.Service, userID, sessionID string) []*session.Event {
	t.Helper()
	var events []*session.Event
	for ev := range getSession(t, svc, userID, sessionID).Events().All() {
		events = append(events, ev)
	}
	return events
}

type failingSummarizer struct{}
