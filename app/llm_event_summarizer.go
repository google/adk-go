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

package app

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// DefaultSummarizationPromptTemplate is the prompt LlmEventSummarizer uses
// when no override is provided. The placeholder "{conversation_history}" is
// replaced with a formatted transcript of the events being summarized.
const DefaultSummarizationPromptTemplate = "The following is a conversation history between a user and an AI agent. Please summarize the conversation, focusing on key information and decisions made, as well as any unresolved questions or tasks. The summary should be concise and capture the essence of the interaction.\n\n{conversation_history}"

// LlmEventSummarizer summarizes a range of events into a single compacted
// event using an LLM. Mirrors adk-python's apps/llm_event_summarizer.py.
//
// The summarizer formats text-bearing events as "<author>: <text>" lines,
// substitutes them into the prompt template, calls the LLM with stream=false,
// and packages the first non-empty content into an EventCompaction.
type LlmEventSummarizer struct {
	llm            model.LLM
	promptTemplate string
}

// LlmEventSummarizerOption customizes an LlmEventSummarizer.
type LlmEventSummarizerOption func(*LlmEventSummarizer)

// WithSummarizationPromptTemplate overrides the default summarization
// prompt template. The template should contain "{conversation_history}".
func WithSummarizationPromptTemplate(tpl string) LlmEventSummarizerOption {
	return func(s *LlmEventSummarizer) { s.promptTemplate = tpl }
}

// NewLlmEventSummarizer returns an EventsSummarizer that uses llm to produce
// summaries.
func NewLlmEventSummarizer(llm model.LLM, opts ...LlmEventSummarizerOption) *LlmEventSummarizer {
	s := &LlmEventSummarizer{
		llm:            llm,
		promptTemplate: DefaultSummarizationPromptTemplate,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// MaybeSummarize implements EventsSummarizer.
func (s *LlmEventSummarizer) MaybeSummarize(ctx context.Context, events []*session.Event) (*session.Event, error) {
	if len(events) == 0 {
		return nil, nil
	}
	conv := formatEventsForSummary(events)
	if conv == "" {
		return nil, nil
	}
	prompt := strings.ReplaceAll(s.promptTemplate, "{conversation_history}", conv)

	req := &model.LLMRequest{
		Model: s.llm.Name(),
		Contents: []*genai.Content{{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: prompt}},
		}},
	}
	var summary *genai.Content
	for resp, err := range s.llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("summarizer: LLM error: %w", err)
		}
		if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
			summary = resp.Content
			break
		}
	}
	if summary == nil {
		return nil, nil
	}
	// Force the role to "model" so the compacted content is treated as a
	// model turn when the contents-builder folds it back in.
	summary.Role = genai.RoleModel

	startTs := events[0].Timestamp
	endTs := events[len(events)-1].Timestamp

	out := session.NewEvent(uuid.NewString())
	out.Author = "user"
	out.Actions.Compaction = &session.EventCompaction{
		StartTimestamp:   startTs,
		EndTimestamp:     endTs,
		CompactedContent: summary,
	}
	return out, nil
}

// formatEventsForSummary mirrors LlmEventSummarizer._format_events_for_prompt.
// Only text parts are included; binary parts and function calls / responses
// are skipped.
func formatEventsForSummary(events []*session.Event) string {
	var lines []string
	for _, ev := range events {
		if ev == nil || ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil {
				continue
			}
			if t := part.Text; t != "" {
				author := ev.Author
				if author == "" {
					author = "unknown"
				}
				lines = append(lines, fmt.Sprintf("%s: %s", author, t))
			}
		}
	}
	return strings.Join(lines, "\n")
}
