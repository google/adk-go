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

package app_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	adkapp "google.golang.org/adk/app"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

type stubLLM struct {
	name      string
	responses []*model.LLMResponse
	err       error
	gotReq    *model.LLMRequest
}

func (s *stubLLM) Name() string { return s.name }
func (s *stubLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	s.gotReq = req
	return func(yield func(*model.LLMResponse, error) bool) {
		if s.err != nil {
			yield(nil, s.err)
			return
		}
		for _, r := range s.responses {
			if !yield(r, nil) {
				return
			}
		}
	}
}

func mkEvent(author, text string, ts time.Time) *session.Event {
	e := session.NewEvent("inv-" + author)
	e.Author = author
	e.Timestamp = ts
	e.Content = &genai.Content{
		Role:  author,
		Parts: []*genai.Part{{Text: text}},
	}
	return e
}

func TestLlmEventSummarizer_BasicSummary(t *testing.T) {
	llm := &stubLLM{
		name: "stub",
		responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "summary text"}}}},
		},
	}
	s := adkapp.NewLlmEventSummarizer(llm)
	t0 := time.Unix(100, 0)
	events := []*session.Event{
		mkEvent("user", "hello", t0),
		mkEvent("model", "hi there", t0.Add(time.Second)),
	}
	got, err := s.MaybeSummarize(context.Background(), events)
	if err != nil {
		t.Fatalf("MaybeSummarize: %v", err)
	}
	if got == nil {
		t.Fatal("expected event, got nil")
	}
	if got.Author != "user" {
		t.Errorf("Author = %q, want user", got.Author)
	}
	if got.Actions.Compaction == nil {
		t.Fatal("expected Compaction set")
	}
	c := got.Actions.Compaction
	if !c.StartTimestamp.Equal(t0) {
		t.Errorf("StartTimestamp = %v, want %v", c.StartTimestamp, t0)
	}
	if !c.EndTimestamp.Equal(t0.Add(time.Second)) {
		t.Errorf("EndTimestamp = %v, want %v", c.EndTimestamp, t0.Add(time.Second))
	}
	if c.CompactedContent == nil || c.CompactedContent.Role != genai.RoleModel {
		t.Errorf("CompactedContent role = %v, want model", c.CompactedContent)
	}
	if c.CompactedContent.Parts[0].Text != "summary text" {
		t.Errorf("summary text = %q", c.CompactedContent.Parts[0].Text)
	}
	// The prompt should contain the formatted history.
	if got := llm.gotReq.Contents[0].Parts[0].Text; !strings.Contains(got, "user: hello") || !strings.Contains(got, "model: hi there") {
		t.Errorf("prompt does not include events: %q", got)
	}
}

func TestLlmEventSummarizer_EmptyEventsReturnsNil(t *testing.T) {
	llm := &stubLLM{name: "stub"}
	s := adkapp.NewLlmEventSummarizer(llm)
	got, err := s.MaybeSummarize(context.Background(), nil)
	if err != nil || got != nil {
		t.Errorf("got = %v, err = %v", got, err)
	}
}

func TestLlmEventSummarizer_NoTextEventsReturnsNil(t *testing.T) {
	llm := &stubLLM{name: "stub"}
	s := adkapp.NewLlmEventSummarizer(llm)
	// Event with no text content.
	e := session.NewEvent("inv-1")
	e.Author = "user"
	got, err := s.MaybeSummarize(context.Background(), []*session.Event{e})
	if err != nil || got != nil {
		t.Errorf("got = %v, err = %v", got, err)
	}
}

func TestLlmEventSummarizer_NoSummaryReturnsNil(t *testing.T) {
	// LLM returns no content parts.
	llm := &stubLLM{name: "stub", responses: []*model.LLMResponse{{Content: nil}}}
	s := adkapp.NewLlmEventSummarizer(llm)
	got, err := s.MaybeSummarize(context.Background(), []*session.Event{
		mkEvent("user", "hi", time.Unix(1, 0)),
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestLlmEventSummarizer_LLMError(t *testing.T) {
	wantErr := errors.New("upstream broken")
	llm := &stubLLM{name: "stub", err: wantErr}
	s := adkapp.NewLlmEventSummarizer(llm)
	_, err := s.MaybeSummarize(context.Background(), []*session.Event{
		mkEvent("user", "hi", time.Unix(1, 0)),
	})
	if !errors.Is(err, wantErr) {
		t.Errorf("err = %v, want wraps %v", err, wantErr)
	}
}

func TestLlmEventSummarizer_CustomTemplate(t *testing.T) {
	llm := &stubLLM{
		name: "stub",
		responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "ok"}}}},
		},
	}
	s := adkapp.NewLlmEventSummarizer(llm,
		adkapp.WithSummarizationPromptTemplate("CUSTOM: {conversation_history}"))
	_, _ = s.MaybeSummarize(context.Background(), []*session.Event{
		mkEvent("user", "hi", time.Unix(1, 0)),
	})
	prompt := llm.gotReq.Contents[0].Parts[0].Text
	if !strings.HasPrefix(prompt, "CUSTOM: ") {
		t.Errorf("prompt = %q, want custom template", prompt)
	}
}
