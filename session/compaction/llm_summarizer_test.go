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
	"errors"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestNewLLMSummarizer(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		cfg     LLMSummarizerConfig
		wantErr bool
	}{
		{name: "defaults", cfg: LLMSummarizerConfig{Model: &fakeModel{}}},
		{name: "missing model", cfg: LLMSummarizerConfig{}, wantErr: true},
		{
			name: "custom template with the placeholder",
			cfg:  LLMSummarizerConfig{Model: &fakeModel{}, PromptTemplate: "summarize: " + ConversationHistoryPlaceholder},
		},
		{
			name:    "custom template without the placeholder",
			cfg:     LLMSummarizerConfig{Model: &fakeModel{}, PromptTemplate: "summarize please"},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := NewLLMSummarizer(tc.cfg)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Errorf("NewLLMSummarizer() error = %v, wantErr %t", err, tc.wantErr)
			}
		})
	}
}

// promptFor runs the summarizer over events and returns the prompt text it sent
// to the model.
func promptFor(t *testing.T, cfg LLMSummarizerConfig, events []*session.Event) string {
	t.Helper()
	m, ok := cfg.Model.(*fakeModel)
	if !ok {
		t.Fatalf("promptFor requires a *fakeModel, got %T", cfg.Model)
	}
	s, err := NewLLMSummarizer(cfg)
	if err != nil {
		t.Fatalf("NewLLMSummarizer() error = %v", err)
	}
	if _, err := s.SummarizeEvents(context.Background(), events); err != nil {
		t.Fatalf("SummarizeEvents() error = %v", err)
	}
	if len(m.requests) != 1 {
		t.Fatalf("model received %d requests, want 1", len(m.requests))
	}
	return utils.TextParts(m.requests[0].Contents[0])[0]
}

func TestLLMSummarizerPromptIncludesThoughtsAndToolTraffic(t *testing.T) {
	t.Parallel()

	thought := newEvent("t", "inv1", 2, "model", &genai.Part{Text: "I should look this up", Thought: true})
	call := newEvent("c", "inv1", 3, "model", &genai.Part{
		FunctionCall: &genai.FunctionCall{ID: "c1", Name: "search", Args: map[string]any{"q": "adk"}},
	})
	resp := newEvent("r", "inv1", 4, "user", &genai.Part{
		FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "search", Response: map[string]any{"hits": 3}},
	})

	prompt := promptFor(t,
		LLMSummarizerConfig{Model: &fakeModel{responses: []*model.LLMResponse{summaryResponse("done")}}},
		[]*session.Event{
			textEvent("u", "inv1", 1, "what is adk?"),
			thought,
			call,
			resp,
			modelTextEvent("m", "inv1", 5, "ADK is a toolkit."),
		},
	)

	// Thoughts, calls and responses all carry information a text-only summary
	// would lose, so all three must reach the summarizer.
	for _, want := range []string{
		"user: what is adk?",
		"model (thought): I should look this up",
		"model called tool: search({q: adk})",
		"Tool response from search: {hits: 3}",
		"model: ADK is a toolkit.",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt is missing %q\nprompt:\n%s", want, prompt)
		}
	}
}

func TestLLMSummarizerSkipsPriorSummaryThoughts(t *testing.T) {
	t.Parallel()

	// A previous compaction's own reasoning must not be folded into the next
	// summary, or reasoning artefacts compound across compactions.
	prior := compactionEvent("s1", 1, 1, 1, "earlier summary")
	prior.LLMResponse.Content = &genai.Content{Role: "model", Parts: []*genai.Part{
		{Text: "reasoning behind the earlier summary", Thought: true},
		{Text: "earlier summary"},
	}}

	prompt := promptFor(t,
		LLMSummarizerConfig{Model: &fakeModel{responses: []*model.LLMResponse{summaryResponse("done")}}},
		[]*session.Event{prior, textEvent("u", "inv2", 2, "next question")},
	)

	if strings.Contains(prompt, "reasoning behind the earlier summary") {
		t.Errorf("prompt leaked a prior compaction's thought\nprompt:\n%s", prompt)
	}
	if !strings.Contains(prompt, "earlier summary") {
		t.Errorf("prompt dropped the prior summary text\nprompt:\n%s", prompt)
	}
}

func TestLLMSummarizerTruncatesLargeToolContent(t *testing.T) {
	t.Parallel()

	big := strings.Repeat("x", 60)
	call := newEvent("c", "inv1", 1, "model", &genai.Part{
		FunctionCall: &genai.FunctionCall{ID: "c1", Name: "search", Args: map[string]any{"q": big}},
	})

	prompt := promptFor(t,
		LLMSummarizerConfig{
			Model:               &fakeModel{responses: []*model.LLMResponse{summaryResponse("done")}},
			MaxToolContentChars: 20,
		},
		[]*session.Event{call},
	)

	if !strings.Contains(prompt, "[truncated") {
		t.Errorf("prompt was not truncated\nprompt:\n%s", prompt)
	}
	if strings.Contains(prompt, big) {
		t.Errorf("prompt contains the untruncated tool args\nprompt:\n%s", prompt)
	}
}

func TestLLMSummarizerNegativeMaxDisablesTruncation(t *testing.T) {
	t.Parallel()

	big := strings.Repeat("x", DefaultMaxToolContentChars+10)
	call := newEvent("c", "inv1", 1, "model", &genai.Part{
		FunctionCall: &genai.FunctionCall{ID: "c1", Name: "search", Args: map[string]any{"q": big}},
	})

	prompt := promptFor(t,
		LLMSummarizerConfig{
			Model:               &fakeModel{responses: []*model.LLMResponse{summaryResponse("done")}},
			MaxToolContentChars: -1,
		},
		[]*session.Event{call},
	)

	if !strings.Contains(prompt, big) {
		t.Error("a negative MaxToolContentChars should disable truncation, but the args were cut")
	}
}

func TestLLMSummarizerSummarizeEvents(t *testing.T) {
	t.Parallel()

	usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 42}
	resp := summaryResponse("the summary")
	resp.UsageMetadata = usage

	m := &fakeModel{responses: []*model.LLMResponse{resp}}
	s, err := NewLLMSummarizer(LLMSummarizerConfig{Model: m})
	if err != nil {
		t.Fatalf("NewLLMSummarizer() error = %v", err)
	}

	events := []*session.Event{textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 4, "a1")}
	got, err := s.SummarizeEvents(context.Background(), events)
	if err != nil {
		t.Fatalf("SummarizeEvents() error = %v", err)
	}
	if got == nil {
		t.Fatal("SummarizeEvents() returned nil, want a compaction event")
	}
	if got.Actions.Compaction == nil {
		t.Fatal("returned event carries no compaction")
	}
	if !got.Actions.Compaction.StartTimestamp.Equal(at(1)) || !got.Actions.Compaction.EndTimestamp.Equal(at(4)) {
		t.Errorf("compaction range = [%v, %v], want [%v, %v]",
			got.Actions.Compaction.StartTimestamp, got.Actions.Compaction.EndTimestamp, at(1), at(4))
	}
	texts := utils.TextParts(got.Actions.Compaction.CompactedContent)
	if diff := cmp.Diff([]string{"the summary"}, texts); diff != "" {
		t.Errorf("summary text mismatch (-want +got):\n%s", diff)
	}
	if got.UsageMetadata != usage {
		t.Errorf("UsageMetadata = %v, want the summarizer call's usage carried through", got.UsageMetadata)
	}
	if got.Author != "user" {
		t.Errorf("Author = %q, want %q", got.Author, "user")
	}
}

func TestLLMSummarizerEdgeCases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		model     *fakeModel
		events    []*session.Event
		wantEvent bool
		wantErr   bool
	}{
		{
			name:   "no events",
			model:  &fakeModel{},
			events: nil,
		},
		{
			name:   "model returns nothing",
			model:  &fakeModel{},
			events: []*session.Event{textEvent("a", "inv1", 1, "q1")},
		},
		{
			name:   "model returns a response with no content",
			model:  &fakeModel{responses: []*model.LLMResponse{{}}},
			events: []*session.Event{textEvent("a", "inv1", 1, "q1")},
		},
		{
			name:    "model fails",
			model:   &fakeModel{err: errors.New("boom")},
			events:  []*session.Event{textEvent("a", "inv1", 1, "q1")},
			wantErr: true,
		},
		{
			name:      "success",
			model:     &fakeModel{responses: []*model.LLMResponse{summaryResponse("ok")}},
			events:    []*session.Event{textEvent("a", "inv1", 1, "q1")},
			wantEvent: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s, err := NewLLMSummarizer(LLMSummarizerConfig{Model: tc.model})
			if err != nil {
				t.Fatalf("NewLLMSummarizer() error = %v", err)
			}
			got, err := s.SummarizeEvents(context.Background(), tc.events)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("SummarizeEvents() error = %v, wantErr %t", err, tc.wantErr)
			}
			if gotEvent := got != nil; gotEvent != tc.wantEvent {
				t.Errorf("SummarizeEvents() returned event = %t, want %t", gotEvent, tc.wantEvent)
			}
		})
	}
}

func summaryResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: text}}}}
}

// TestLLMSummarizerTruncatesByCharactersNotBytes guards the limit against Go's
// byte-oriented len and slicing.
//
// The limit is documented in characters, so a byte-based limit would cut
// non-Latin tool output several times harder than configured, and a byte slice
// can land mid-rune and produce invalid UTF-8.
func TestLLMSummarizerTruncatesByCharactersNotBytes(t *testing.T) {
	t.Parallel()

	// 2000 characters of Japanese is 6000 bytes; a byte limit of 2000 would
	// keep only ~666 of them.
	jp := strings.Repeat("検索結果", 500)
	if got, want := utf8.RuneCountInString(jp), 2000; got != want {
		t.Fatalf("fixture is %d runes, want %d", got, want)
	}

	tests := []struct {
		name      string
		text      string
		max       int
		wantRunes int  // runes kept before the "..." marker
		wantCut   bool //  whether truncation happened at all
	}{
		{name: "exactly at the limit is kept whole", text: jp, max: 2000, wantRunes: 2000},
		{name: "one over the limit is cut", text: jp, max: 1999, wantRunes: 1999, wantCut: true},
		{name: "well under the limit", text: jp, max: 5000, wantRunes: 2000},
		{name: "ascii unchanged", text: strings.Repeat("x", 100), max: 2000, wantRunes: 100},
		{name: "ascii cut", text: strings.Repeat("x", 100), max: 10, wantRunes: 10, wantCut: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s, err := NewLLMSummarizer(LLMSummarizerConfig{
				Model: &fakeModel{}, MaxToolContentChars: tc.max,
			})
			if err != nil {
				t.Fatalf("NewLLMSummarizer() error = %v", err)
			}

			got := s.truncate(tc.text)
			if !utf8.ValidString(got) {
				t.Error("truncated text is not valid UTF-8; the cut landed mid-rune")
			}

			body, marker, found := strings.Cut(got, "... [truncated ")
			if found != tc.wantCut {
				t.Fatalf("truncated = %t, want %t", found, tc.wantCut)
			}
			if gotRunes := utf8.RuneCountInString(body); gotRunes != tc.wantRunes {
				t.Errorf("kept %d runes, want %d", gotRunes, tc.wantRunes)
			}
			if !tc.wantCut {
				return
			}
			// The dropped count must be in the same unit as the limit.
			wantDropped := utf8.RuneCountInString(tc.text) - tc.wantRunes
			if want := fmt.Sprintf("%d chars]", wantDropped); marker != want {
				t.Errorf("marker = %q, want %q", marker, want)
			}
		})
	}
}

func TestLLMSummarizerTruncationIsDisabledByNegativeMax(t *testing.T) {
	t.Parallel()

	jp := strings.Repeat("検索結果", 500)
	s, err := NewLLMSummarizer(LLMSummarizerConfig{Model: &fakeModel{}, MaxToolContentChars: -1})
	if err != nil {
		t.Fatalf("NewLLMSummarizer() error = %v", err)
	}
	if got := s.truncate(jp); got != jp {
		t.Error("a negative MaxToolContentChars must disable truncation entirely")
	}
}
