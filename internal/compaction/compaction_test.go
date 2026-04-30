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

package compaction

import (
	"context"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// fakeSummarizer is a deterministic summarizer for testing.
type fakeSummarizer struct {
	calls    int
	gotInput [][]*session.Event
	output   string // "" => return nil; otherwise produce a summary event
}

func (f *fakeSummarizer) MaybeSummarize(ctx context.Context, events []*session.Event) (*session.Event, error) {
	f.calls++
	cpy := make([]*session.Event, len(events))
	copy(cpy, events)
	f.gotInput = append(f.gotInput, cpy)
	if f.output == "" {
		return nil, nil
	}
	out := session.NewEvent("comp")
	out.Author = "user"
	out.Actions.Compaction = &session.EventCompaction{
		StartTimestamp:   events[0].Timestamp,
		EndTimestamp:     events[len(events)-1].Timestamp,
		CompactedContent: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: f.output}}},
	}
	return out, nil
}

func makeSession(t *testing.T, evs []*session.Event) session.Session {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "test", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	for _, e := range evs {
		if err := svc.AppendEvent(context.Background(), resp.Session, e); err != nil {
			t.Fatalf("AppendEvent: %v", err)
		}
	}
	return resp.Session
}

func makeUserEvent(invID, text string, ts time.Time) *session.Event {
	e := session.NewEvent(invID)
	e.Author = "user"
	e.Timestamp = ts
	e.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: text}}}
	return e
}

func makeModelEvent(invID, text string, ts time.Time) *session.Event {
	e := session.NewEvent(invID)
	e.Author = "model"
	e.Timestamp = ts
	e.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{Text: text}}}
	return e
}

func TestMaybeRun_NoConfig(t *testing.T) {
	sess := makeSession(t, []*session.Event{makeUserEvent("inv-1", "hi", time.Unix(1, 0))})
	got, err := MaybeRun(context.Background(), MaybeRunInput{
		Session:        sess,
		SessionService: session.InMemoryService(),
	})
	if got || err != nil {
		t.Errorf("got %v, %v", got, err)
	}
}

func TestMaybeRun_NoSummarizer(t *testing.T) {
	sess := makeSession(t, []*session.Event{makeUserEvent("inv-1", "hi", time.Unix(1, 0))})
	thr := 1
	ret := 0
	_, err := MaybeRun(context.Background(), MaybeRunInput{
		CompactionInterval: 1, OverlapSize: 0,
		TokenThreshold:     &thr,
		EventRetentionSize: &ret,
		// Summarizer intentionally nil.
		Session:        sess,
		SessionService: session.InMemoryService(),
	})
	if err == nil {
		t.Error("expected error for nil summarizer")
	}
}

func TestMaybeRun_SlidingWindow_Triggers(t *testing.T) {
	t0 := time.Unix(100, 0)
	events := []*session.Event{
		makeUserEvent("inv-1", "u1", t0),
		makeModelEvent("inv-1", "m1", t0.Add(time.Second)),
		makeUserEvent("inv-2", "u2", t0.Add(2*time.Second)),
		makeModelEvent("inv-2", "m2", t0.Add(3*time.Second)),
	}
	srv := session.InMemoryService()
	cr, _ := srv.Create(context.Background(), &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s"})
	for _, e := range events {
		_ = srv.AppendEvent(context.Background(), cr.Session, e)
	}
	sum := &fakeSummarizer{output: "summary"}
	got, err := MaybeRun(context.Background(), MaybeRunInput{
		Summarizer:         sum,
		CompactionInterval: 2,
		OverlapSize:        0,
		Session:        cr.Session,
		SessionService: srv,
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !got {
		t.Fatal("expected sliding-window to trigger")
	}
	if sum.calls != 1 {
		t.Errorf("summarizer calls = %d, want 1", sum.calls)
	}
	// Should have summarized 4 events (inv-1 .. inv-2)
	if n := len(sum.gotInput[0]); n != 4 {
		t.Errorf("summarized %d events, want 4", n)
	}
	// Session should have an extra (compaction) event now.
	if cr.Session.Events().Len() != 5 {
		t.Errorf("session events = %d, want 5", cr.Session.Events().Len())
	}
}

func TestMaybeRun_SlidingWindow_NotEnoughInvocations(t *testing.T) {
	t0 := time.Unix(100, 0)
	events := []*session.Event{
		makeUserEvent("inv-1", "u1", t0),
		makeModelEvent("inv-1", "m1", t0.Add(time.Second)),
	}
	srv := session.InMemoryService()
	cr, _ := srv.Create(context.Background(), &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s"})
	for _, e := range events {
		_ = srv.AppendEvent(context.Background(), cr.Session, e)
	}
	sum := &fakeSummarizer{output: "summary"}
	got, _ := MaybeRun(context.Background(), MaybeRunInput{
					Summarizer:         sum,
			CompactionInterval: 5, // require 5, only have 1
			OverlapSize:        0,
		Session:        cr.Session,
		SessionService: srv,
	})
	if got {
		t.Error("expected no compaction")
	}
	if sum.calls != 0 {
		t.Errorf("summarizer should not have been called, got %d", sum.calls)
	}
}

func TestMaybeRun_TokenThreshold_FiresWhenAboveLimit(t *testing.T) {
	t0 := time.Unix(100, 0)
	events := []*session.Event{
		makeUserEvent("inv-1", "long-message-1", t0),
		makeModelEvent("inv-1", "long-response-1", t0.Add(time.Second)),
		makeUserEvent("inv-2", "long-message-2", t0.Add(2*time.Second)),
		makeModelEvent("inv-2", "long-response-2", t0.Add(3*time.Second)),
		makeUserEvent("inv-3", "current", t0.Add(4*time.Second)),
		makeModelEvent("inv-3", "now", t0.Add(5*time.Second)),
	}
	// Tag the last model event with prompt token count above threshold.
	events[5].UsageMetadata = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 1000}

	srv := session.InMemoryService()
	cr, _ := srv.Create(context.Background(), &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s"})
	for _, e := range events {
		_ = srv.AppendEvent(context.Background(), cr.Session, e)
	}
	thr := 500
	ret := 2
	sum := &fakeSummarizer{output: "summary"}
	got, err := MaybeRun(context.Background(), MaybeRunInput{
					Summarizer:         sum,
			CompactionInterval: 99, // sliding wouldn't trigger
			OverlapSize:        0,
			TokenThreshold:     &thr,
			EventRetentionSize: &ret,
		Session:        cr.Session,
		SessionService: srv,
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !got {
		t.Fatal("expected token-threshold to trigger")
	}
	// Should compact older events; retained = last 2.
	if n := len(sum.gotInput[0]); n != 4 {
		t.Errorf("summarized %d events, want 4 (6 - 2 retained)", n)
	}
}

func TestMaybeRun_TokenThreshold_BelowLimit(t *testing.T) {
	t0 := time.Unix(100, 0)
	events := []*session.Event{
		makeUserEvent("inv-1", "hi", t0),
		makeModelEvent("inv-1", "ok", t0.Add(time.Second)),
	}
	events[1].UsageMetadata = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10}

	srv := session.InMemoryService()
	cr, _ := srv.Create(context.Background(), &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s"})
	for _, e := range events {
		_ = srv.AppendEvent(context.Background(), cr.Session, e)
	}
	thr := 500
	ret := 1
	sum := &fakeSummarizer{output: "summary"}
	got, _ := MaybeRun(context.Background(), MaybeRunInput{
					Summarizer:         sum,
			CompactionInterval: 99,
			OverlapSize:        0,
			TokenThreshold:     &thr,
			EventRetentionSize: &ret,
		Session:        cr.Session,
		SessionService: srv,
	})
	if got || sum.calls != 0 {
		t.Errorf("expected no compaction, got %v with %d summarizer calls", got, sum.calls)
	}
}

func TestMaybeRun_SummarizerReturnsNil_NoAppend(t *testing.T) {
	t0 := time.Unix(100, 0)
	events := []*session.Event{
		makeUserEvent("inv-1", "u1", t0),
		makeModelEvent("inv-1", "m1", t0.Add(time.Second)),
		makeUserEvent("inv-2", "u2", t0.Add(2*time.Second)),
		makeModelEvent("inv-2", "m2", t0.Add(3*time.Second)),
	}
	srv := session.InMemoryService()
	cr, _ := srv.Create(context.Background(), &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s"})
	for _, e := range events {
		_ = srv.AppendEvent(context.Background(), cr.Session, e)
	}
	sum := &fakeSummarizer{output: ""} // return nil
	got, _ := MaybeRun(context.Background(), MaybeRunInput{
		Summarizer:         sum,
		CompactionInterval: 2,
		OverlapSize:        0,
		Session:        cr.Session,
		SessionService: srv,
	})
	if got {
		t.Error("expected no compaction (summarizer returned nil)")
	}
	if cr.Session.Events().Len() != 4 {
		t.Errorf("session events = %d, want 4 (no append)", cr.Session.Events().Len())
	}
}

func TestPendingFunctionCallIDs(t *testing.T) {
	// inv-1 has a call without a response.
	e1 := session.NewEvent("inv-1")
	e1.Content = &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "fn"}}}}
	// inv-2 has a call with response.
	e2 := session.NewEvent("inv-2")
	e2.Content = &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c2", Name: "fn"}}}}
	e3 := session.NewEvent("inv-2")
	e3.Content = &genai.Content{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "c2", Name: "fn"}}}}

	pending := pendingFunctionCallIDs([]*session.Event{e1, e2, e3})
	if _, has := pending["c1"]; !has {
		t.Errorf("expected c1 pending, got %v", pending)
	}
	if _, has := pending["c2"]; has {
		t.Errorf("c2 should not be pending, got %v", pending)
	}
}

func TestSafeTokenCompactionSplitIndex_AvoidsOrphanedResponse(t *testing.T) {
	// Sequence: call(c1), response(c1), other, retention=1
	// Initial split would be 2; that's fine because nothing past 2 references c1.
	t0 := time.Unix(0, 0)
	call := session.NewEvent("inv")
	call.Timestamp = t0
	call.Content = &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "fn"}}}}
	resp := session.NewEvent("inv")
	resp.Timestamp = t0.Add(time.Second)
	resp.Content = &genai.Content{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "fn"}}}}
	other := session.NewEvent("inv")
	other.Timestamp = t0.Add(2 * time.Second)
	other.Content = &genai.Content{Parts: []*genai.Part{{Text: "later"}}}

	// retention=2 -> initial split = 1 (only "call" in prefix). But call's response is in tail at index 1.
	// Algorithm should shift split to 0 to keep call+response together.
	got := safeTokenCompactionSplitIndex([]*session.Event{call, resp, other}, 2)
	if got != 0 {
		t.Errorf("split = %d, want 0", got)
	}
}

func TestLatestCompactionEnd_PicksLatestNonSubsumed(t *testing.T) {
	t0 := time.Unix(100, 0)
	mk := func(start, end time.Time) *session.Event {
		ev := session.NewEvent("inv")
		ev.Timestamp = end
		ev.Actions.Compaction = &session.EventCompaction{
			StartTimestamp:   start,
			EndTimestamp:     end,
			CompactedContent: &genai.Content{Parts: []*genai.Part{{Text: "x"}}},
		}
		return ev
	}
	c1 := mk(t0, t0.Add(10*time.Second))
	c2 := mk(t0, t0.Add(5*time.Second)) // subsumed by c1 (range fully contained)
	c3 := mk(t0.Add(20*time.Second), t0.Add(30*time.Second))
	got := latestCompactionEnd([]*session.Event{c1, c2, c3})
	if !got.Equal(t0.Add(30 * time.Second)) {
		t.Errorf("latest end = %v, want %v", got, t0.Add(30*time.Second))
	}
}

func TestFold_NoCompaction_PassesThrough(t *testing.T) {
	t0 := time.Unix(100, 0)
	in := []*session.Event{
		makeUserEvent("inv-1", "u1", t0),
		makeModelEvent("inv-1", "m1", t0.Add(time.Second)),
	}
	out := Fold(in)
	if len(out) != 2 {
		t.Errorf("Fold = %d events, want 2", len(out))
	}
}

func TestFold_LatestCompactionReplacesOlderEvents(t *testing.T) {
	t0 := time.Unix(100, 0)
	older := []*session.Event{
		makeUserEvent("inv-1", "u1", t0),
		makeModelEvent("inv-1", "m1", t0.Add(time.Second)),
	}
	comp := session.NewEvent("comp-1")
	comp.Author = "user"
	comp.Timestamp = t0.Add(2 * time.Second)
	comp.Actions.Compaction = &session.EventCompaction{
		StartTimestamp:   t0,
		EndTimestamp:     t0.Add(time.Second),
		CompactedContent: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "summary"}}},
	}
	newer := []*session.Event{
		makeUserEvent("inv-2", "u2", t0.Add(3*time.Second)),
		makeModelEvent("inv-2", "m2", t0.Add(4*time.Second)),
	}
	in := append(append(older, comp), newer...)
	out := Fold(in)
	// Expect: 1 synthetic seed + 2 newer events = 3
	if len(out) != 3 {
		t.Fatalf("Fold = %d events, want 3", len(out))
	}
	if out[0].Content == nil || out[0].Content.Parts[0].Text != "summary" {
		t.Errorf("first event = %+v, want summary", out[0])
	}
	if out[1].Author != "user" || out[2].Author != "model" {
		t.Errorf("newer events out of order: %v %v", out[1].Author, out[2].Author)
	}
}
