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
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// ConversationHistoryPlaceholder is the token an [LLMSummarizer] prompt
// template must contain. It is replaced with the rendered event transcript.
const ConversationHistoryPlaceholder = "{conversation_history}"

// DefaultPromptTemplate is the prompt [LLMSummarizer] uses when none is given.
const DefaultPromptTemplate = "The following is a conversation history between a user and an AI agent." +
	" It may or may not start from a compacted history. Please identify and" +
	" reiterate the user request, summarize the context so far, focusing on" +
	" key decisions made and information obtained, as well as any unresolved" +
	" questions or tasks. " +
	"CRITICAL INSTRUCTIONS: " +
	"1. Explicitly identify and state the primary language used by the user " +
	`at the top of your summary (e.g., "Conversation Language: English"). ` +
	"2. If the agent called any tools, accurately list the exact tool names " +
	"used to maintain tool grounding. " +
	"The rest of the summary should be concise and capture the" +
	" essence of the interaction.\n\n" + ConversationHistoryPlaceholder

// DefaultMaxToolContentChars caps how much of a single tool call's arguments or
// response is rendered into the summarizer prompt.
const DefaultMaxToolContentChars = 2000

// LLMSummarizerConfig configures [NewLLMSummarizer].
type LLMSummarizerConfig struct {
	// Model summarizes the conversation. Required.
	Model model.LLM

	// PromptTemplate is the instruction wrapped around the rendered
	// conversation. It must contain [ConversationHistoryPlaceholder]. Defaults
	// to [DefaultPromptTemplate].
	PromptTemplate string

	// MaxToolContentChars caps the rendered length of a single tool call's
	// arguments or response. Defaults to [DefaultMaxToolContentChars]; a
	// negative value disables truncation.
	MaxToolContentChars int
}

// LLMSummarizer is the default [Summarizer]. It renders the events as a
// labelled transcript and asks a model to summarize them.
//
// The transcript carries text, agent thoughts, function calls and function
// responses. Thoughts and tool traffic are included because they hold the
// reasoning and the evidence gathered so far, which a text-only summary would
// silently lose. Tool arguments and responses are truncated so compaction does
// not inflate the very context it exists to shrink, and thoughts belonging to
// an earlier compaction event are skipped so a previous summary's reasoning
// does not leak into the next one.
type LLMSummarizer struct {
	model               model.LLM
	promptTemplate      string
	maxToolContentChars int
}

var _ Summarizer = (*LLMSummarizer)(nil)

// NewLLMSummarizer creates an [LLMSummarizer].
func NewLLMSummarizer(cfg LLMSummarizerConfig) (*LLMSummarizer, error) {
	if cfg.Model == nil {
		return nil, fmt.Errorf("LLMSummarizerConfig.Model is required")
	}
	template := cfg.PromptTemplate
	if template == "" {
		template = DefaultPromptTemplate
	}
	if !strings.Contains(template, ConversationHistoryPlaceholder) {
		return nil, fmt.Errorf("PromptTemplate must contain the placeholder %q", ConversationHistoryPlaceholder)
	}
	maxChars := cfg.MaxToolContentChars
	if maxChars == 0 {
		maxChars = DefaultMaxToolContentChars
	}
	return &LLMSummarizer{
		model:               cfg.Model,
		promptTemplate:      template,
		maxToolContentChars: maxChars,
	}, nil
}

// SummarizeEvents implements [Summarizer].
func (s *LLMSummarizer) SummarizeEvents(ctx context.Context, events []*session.Event) (*session.Event, error) {
	if len(events) == 0 {
		return nil, nil
	}

	prompt := strings.Replace(s.promptTemplate, ConversationHistoryPlaceholder, s.formatEvents(events), 1)
	req := &model.LLMRequest{
		Model:    s.model.Name(),
		Contents: []*genai.Content{genai.NewContentFromText(prompt, genai.RoleUser)},
	}

	for resp, err := range s.model.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("summarizer model call failed: %w", err)
		}
		if resp == nil || resp.Content == nil {
			continue
		}
		summary, err := NewSummaryEvent(events, resp.Content, resp.UsageMetadata)
		if err != nil {
			return nil, err
		}
		return summary, nil
	}
	// No content came back. Treat it as "nothing to compact this time" rather
	// than an error: the caller skips the compaction and retries on the next
	// trigger, leaving history untouched.
	return nil, nil
}

// formatEvents renders events as one labelled line per part.
func (s *LLMSummarizer) formatEvents(events []*session.Event) string {
	var lines []string
	for _, ev := range events {
		content := utils.Content(ev)
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		isCompaction := ev.Actions.Compaction != nil
		for _, p := range content.Parts {
			switch {
			case p.Thought && p.Text != "":
				if !isCompaction {
					lines = append(lines, fmt.Sprintf("%s (thought): %s", ev.Author, p.Text))
				}
			case p.Text != "":
				lines = append(lines, fmt.Sprintf("%s: %s", ev.Author, p.Text))
			}
			if p.FunctionCall != nil {
				lines = append(lines, fmt.Sprintf("%s called tool: %s(%s)",
					ev.Author, p.FunctionCall.Name, s.truncate(stringify(p.FunctionCall.Args))))
			}
			if p.FunctionResponse != nil {
				lines = append(lines, fmt.Sprintf("Tool response from %s: %s",
					p.FunctionResponse.Name, s.truncate(stringify(p.FunctionResponse.Response))))
			}
		}
	}
	return strings.Join(lines, "\n")
}

// truncate caps text at the configured limit, noting how much was dropped.
//
// The limit counts characters, not bytes, as the field name says. Go's len and
// slice operators work on bytes, so using them here would cut non-Latin tool
// output far harder than the configured limit implies, since 2000 "chars" of
// Japanese is about 666 actual characters of UTF-8. A byte slice can also land
// mid-rune and emit invalid UTF-8 into the prompt.
func (s *LLMSummarizer) truncate(text string) string {
	if s.maxToolContentChars < 0 {
		return text
	}
	// A string never holds more runes than bytes, so text already within the
	// limit by byte length needs no counting. This is the ASCII fast path.
	if len(text) <= s.maxToolContentChars {
		return text
	}
	if utf8.RuneCountInString(text) <= s.maxToolContentChars {
		return text
	}
	runes := []rune(text)
	return fmt.Sprintf("%s... [truncated %d chars]",
		string(runes[:s.maxToolContentChars]), len(runes)-s.maxToolContentChars)
}

// stringify renders tool arguments and responses for the transcript.
func stringify(v map[string]any) string {
	if len(v) == 0 {
		return ""
	}
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	// Deterministic ordering keeps summarizer prompts stable across runs, which
	// matters for record/replay tests and for prompt caching.
	slices.Sort(keys)
	var b strings.Builder
	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s: %v", k, v[k])
	}
	b.WriteByte('}')
	return b.String()
}
