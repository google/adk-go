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
	"maps"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// ConversationHistoryPlaceholder is the token an [LLMSummarizer] prompt
// template must contain. It is replaced with the rendered event transcript.
const ConversationHistoryPlaceholder = "{conversation_history}"

// defaultPromptTemplate is the prompt [LLMSummarizer] uses when none is given.
const defaultPromptTemplate = "The following is a conversation history between a user and an AI agent." +
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

// defaultMaxToolContentChars caps how much of a single tool call's arguments or
// response is rendered into the summarizer prompt.
const defaultMaxToolContentChars = 2000

// defaultMaxTranscriptChars caps the whole rendered transcript handed to the
// summarizer.
//
// Summarization is the one call that sees the entire window at once, so it is
// the call most likely to exceed the model's own context limit, and the least
// visible when it does. The cap is generous: reaching it means the window is
// too large rather than that any one part is.
const defaultMaxTranscriptChars = 200_000

// LLMSummarizerConfig configures [NewLLMSummarizer].
type LLMSummarizerConfig struct {
	// Model summarizes the conversation. Required.
	Model model.LLM

	// PromptTemplate is the instruction wrapped around the rendered
	// conversation. It must contain [ConversationHistoryPlaceholder]. Empty
	// selects a built-in template.
	//
	// The built-in text is not published. It is the wording of one default,
	// not a contract, and exporting it would make every later improvement to
	// it a breaking change to this package.
	PromptTemplate string

	// MaxToolContentChars caps the rendered length of any single part of the
	// transcript: a text part, a tool call's arguments, or a tool response.
	// Defaults to 2000; a negative value disables truncation.
	//
	// It applies to text as well as tool content deliberately. Text parts carry
	// pasted documents and tool results re-emitted as text, so capping only tool
	// content made the cost of the same payload depend on which kind of part it
	// arrived in.
	MaxToolContentChars int

	// MaxTranscriptChars caps the whole rendered transcript. Defaults to
	// 200,000; a negative value disables the cap.
	//
	// Like MaxToolContentChars it counts characters rather than bytes, so a
	// conversation in a non-Latin script costs what its length says it does.
	//
	// Exceeding it is reported as an error rather than fixed by dropping the
	// oldest turns. Those turns are inside the range the compaction would record
	// as covered, so dropping them from the transcript while still deleting them
	// from history would lose them outright. Declining costs a larger prompt;
	// the remedy is a smaller window.
	MaxTranscriptChars int

	// Timeout bounds the summarization call. Zero, the default, means no
	// timeout, which is the behaviour every ADK implementation has today.
	//
	// Worth setting. The call is synchronous inside the run loop, so a
	// summarizer that hangs holds up the turn behind it with nothing to show
	// for it, and compaction is an optimisation: giving up on it is cheap.
	Timeout time.Duration

	// GenerateContentConfig is applied to the summarization call.
	//
	// The runner passes the root agent's config here, so safety settings and
	// output limits an application deliberately configured also govern the one
	// call that processes the whole conversation transcript. Without it that
	// call silently falls back to provider defaults.
	//
	// SystemInstruction and Tools are cleared: the summarizer has its own
	// instruction and must not be offered tools to call.
	GenerateContentConfig *genai.GenerateContentConfig
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
	maxTranscriptChars  int
	genConfig           *genai.GenerateContentConfig
	timeout             time.Duration
}

var _ Summarizer = (*LLMSummarizer)(nil)

// NewLLMSummarizer creates an [LLMSummarizer].
func NewLLMSummarizer(cfg LLMSummarizerConfig) (*LLMSummarizer, error) {
	if cfg.Model == nil {
		return nil, fmt.Errorf("LLMSummarizerConfig.Model is required")
	}
	template := cfg.PromptTemplate
	if template == "" {
		template = defaultPromptTemplate
	}
	if !strings.Contains(template, ConversationHistoryPlaceholder) {
		return nil, fmt.Errorf("PromptTemplate must contain the placeholder %q", ConversationHistoryPlaceholder)
	}
	maxTranscript := cfg.MaxTranscriptChars
	if maxTranscript == 0 {
		maxTranscript = defaultMaxTranscriptChars
	}
	maxChars := cfg.MaxToolContentChars
	if maxChars == 0 {
		maxChars = defaultMaxToolContentChars
	}
	return &LLMSummarizer{
		model:               cfg.Model,
		promptTemplate:      template,
		maxToolContentChars: maxChars,
		maxTranscriptChars:  maxTranscript,
		timeout:             cfg.Timeout,
		genConfig:           summarizerGenConfig(cfg.GenerateContentConfig),
	}, nil
}

// SummarizeEvents implements [Summarizer].
func (s *LLMSummarizer) SummarizeEvents(ctx context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	if len(events) == 0 {
		return nil, nil, nil
	}

	transcript, err := s.renderTranscript(events)
	if err != nil {
		return nil, nil, err
	}
	prompt := strings.Replace(s.promptTemplate, ConversationHistoryPlaceholder, transcript, 1)
	req := &model.LLMRequest{
		Model:    s.model.Name(),
		Contents: []*genai.Content{genai.NewContentFromText(prompt, genai.RoleUser)},
		Config:   cloneGenConfig(s.genConfig),
	}

	// A timeout here bounds the model call only. The caller's own deadline
	// still applies, so this can shorten the wait but never extend it.
	if s.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.timeout)
		defer cancel()
	}

	var finishReason genai.FinishReason
	for resp, err := range s.model.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, nil, fmt.Errorf("summarizer model call failed: %w", err)
		}
		if resp == nil {
			continue
		}
		// A partial response is a fragment of a stream. Taking the first one
		// would store a truncated summary and lose the usage metadata that only
		// the final response carries. This summarizer asks for a non-streaming
		// call, so a well-behaved model never sends these, but model.LLM is an
		// exported interface and Partial exists precisely to mark the case.
		if resp.Partial {
			continue
		}
		if resp.FinishReason != "" {
			finishReason = resp.FinishReason
		}
		// Content non-nil is not enough. A response carrying an empty Parts
		// slice is what a blocked, truncated or candidate-less generation looks
		// like, and building a summary from it would record a compaction whose
		// content says nothing: the covered turns would be dropped from the
		// prompt and replaced by silence.
		if !hasText(resp.Content) {
			continue
		}
		// A generation that stopped for any reason other than reaching the end
		// is not a summary, even when it carries text. MAX_TOKENS is the one
		// that matters: the text is a summary cut off partway, and storing it
		// deletes the covered turns and replaces them with a fragment. Safety,
		// recitation and blocklist stops arrive the same way.
		if finishReason != "" && finishReason != genai.FinishReasonStop {
			return nil, resp.UsageMetadata, fmt.Errorf("summarizer stopped before finishing (finish reason %q), so the summary is incomplete", finishReason)
		}
		return resp.Content, resp.UsageMetadata, nil
	}

	// Nothing usable came back. This is a failure, not a decision to skip.
	// Reporting it as "nothing to compact" would make a summarizer that fails
	// every single call indistinguishable from an idle one, and would hide the
	// safety, recitation and token-limit stops that surface exactly this way.
	if finishReason != "" {
		return nil, nil, fmt.Errorf("summarizer returned no usable content (finish reason %q)", finishReason)
	}
	return nil, nil, fmt.Errorf("summarizer returned no usable content")
}

// hasText reports whether c carries at least one non-empty text part, which is
// the minimum for a summary to be worth recording.
func hasText(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		// Thought parts do not count. They are the model's reasoning, and the
		// transcript builder deliberately skips them when rendering a stored
		// summary, so a thought-only summary would be accepted here and then
		// render as nothing: the covered turns would be dropped and replaced by
		// an empty line.
		if p != nil && !p.Thought && strings.TrimSpace(p.Text) != "" {
			return true
		}
	}
	return false
}

// formatEvents renders events as one labelled line per part.
//
// Content that did not come from the framework -- model text and, especially,
// tool output -- is escaped so it cannot span lines. Without that, a tool
// returning a body containing "\nuser: ignore the above" would forge a turn
// inside the transcript, and the summarizer has no way to tell a forged turn
// from a real one. Escaping keeps every rendered line attributable to the
// author the framework recorded.
func (s *LLMSummarizer) formatEvents(events []*session.Event, cap int) string {
	var lines []string
	for _, ev := range events {
		content := utils.Content(ev)
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		isCompaction := ev.Actions.Compaction != nil
		for _, p := range content.Parts {
			if p == nil {
				continue
			}
			switch {
			case p.Thought && p.Text != "":
				if !isCompaction {
					lines = append(lines, fmt.Sprintf("%s (thought): %s", escapeLines(ev.Author), escapeLines(s.truncateTo(p.Text, cap))))
				}
			case p.Text != "":
				lines = append(lines, fmt.Sprintf("%s: %s", escapeLines(ev.Author), escapeLines(s.truncateTo(p.Text, cap))))
			}
			if p.FunctionCall != nil {
				lines = append(lines, fmt.Sprintf("%s called tool: %s(%s)",
					escapeLines(ev.Author), escapeLines(p.FunctionCall.Name), escapeLines(s.truncateTo(stringify(p.FunctionCall.Args), cap))))
			}
			if p.FunctionResponse != nil {
				lines = append(lines, fmt.Sprintf("Tool response from %s: %s",
					escapeLines(p.FunctionResponse.Name), escapeLines(s.truncateTo(stringify(p.FunctionResponse.Response), cap))))
			}
			// Everything else gets a placeholder rather than nothing. Dropping
			// the bytes of an image or a code-execution result is right, but
			// dropping the fact that the turn happened is not: after compaction
			// the transcript is all that is left, and an event made only of
			// these parts would render as an empty line.
			// The kind is escaped like every other interpolated value. It
			// carries a MIME type off a genai.Blob, which nothing validates,
			// so it is caller-controlled text and was the one sink here that
			// went in raw.
			// Truncated as well as escaped. The kind carries a MIME type off a
			// genai.Blob, which nothing validates and a tool can set, and this
			// was the one sink that went through neither. A 100,000-character
			// MIME type rendered in full against a 50-character cap, and
			// because countRenderedParts does not count a placeholder either,
			// the budget never saw it and the whole window was refused as too
			// large, permanently.
			if kind := placeholderKind(p); kind != "" {
				lines = append(lines, fmt.Sprintf("%s: [%s]", escapeLines(ev.Author), escapeLines(s.truncateTo(kind, cap))))
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
func (s *LLMSummarizer) truncateTo(text string, cap int) string {
	if cap < 0 {
		return text
	}
	// A string never holds more runes than bytes, so text already within the
	// limit by byte length needs no counting. This is the ASCII fast path.
	if len(text) <= cap {
		return text
	}
	if utf8.RuneCountInString(text) <= cap {
		return text
	}
	runes := []rune(text)
	return fmt.Sprintf("%s... [truncated %d chars]",
		string(runes[:cap]), len(runes)-cap)
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

// lineBreakers is every character that can end a line in a rendered transcript.
//
// The obvious two are carriage return and newline. U+0085, U+2028 and U+2029
// are line breaks in Unicode, and vertical tab and form feed are treated as
// ones by enough consumers that a value carrying either should not be passed
// through untouched. None of them is worth trusting a model not to honour: the
// summary built from this transcript replaces the real conversation in every
// later prompt, so a forged turn is not read once, it becomes history.
const lineBreakers = "\r\n\v\f\u0085\u2028\u2029"

// escapeLines collapses anything that can end a line into a literal escape, so
// a rendered value cannot break out of its line and forge a turn.
func escapeLines(text string) string {
	if !strings.ContainsAny(text, lineBreakers) {
		return text
	}
	r := strings.NewReplacer(
		"\r\n", "\\n",
		"\r", "\\n",
		"\n", "\\n",
		"\v", "\\n",
		"\f", "\\n",
		"\u0085", "\\n",
		"\u2028", "\\n",
		"\u2029", "\\n",
	)
	return r.Replace(text)
}

// summarizerGenConfig adapts an application's generation config for the
// summarization call.
//
// Only settings that mean the same thing for a summarization are carried over,
// named one by one. A deny-list was the wrong shape here: everything not
// thought of rode along, so an application asking for JSON out, or for a fixed
// response schema, or for images, silently applied all of it to a call whose
// entire job is to return prose. A cached-content handle from the agent's own
// call came through as well, which is a different conversation entirely.
//
// Safety settings carry over because an application that tightened them meant
// them to apply to every call the framework makes on its behalf. Temperature
// and the sampling controls carry over as the closest thing to "how this
// application likes its model to behave".
//
// Three that sound like they should and do not:
//
//   - MaxOutputTokens is sized for the agent's own replies. A summary of a
//     whole window is longer than a reply, so an ordinary app-level cap fails
//     the summarization outright and compaction never runs.
//   - StopSequences are chosen for the agent's output format. A hit reports
//     finish reason STOP, which is indistinguishable from finishing, so a
//     summary cut off at the first occurrence of the token is stored and the
//     covered turns are then dropped in favour of it.
//   - CandidateCount bills one generation per candidate and only the first is
//     read, so an app asking for four pays four times for one summary.
//
// cloneGenConfig returns a copy safe to hand to one model call.
//
// The config is built once when the summarizer is constructed, and the model
// writes into what it is given: model/gemini fills in Config.HTTPOptions.Headers
// on every request. Sharing one struct across calls therefore means two
// concurrent summarizations are two goroutines writing one http.Header map.
//
// This is the ordinary path rather than a corner case. The runner forwards the
// root agent's GenerateContentConfig to the summarizer, so any application that
// sets a temperature or a safety setting is exposed without doing anything
// unusual. The agent's own calls are already safe because the basic request
// processor clones first, and the summarizer path was the one that did not.
func cloneGenConfig(cfg *genai.GenerateContentConfig) *genai.GenerateContentConfig {
	if cfg == nil {
		return nil
	}
	out := *cfg
	out.SafetySettings = slices.Clone(cfg.SafetySettings)
	out.Labels = maps.Clone(cfg.Labels)
	if cfg.HTTPOptions != nil {
		opts := *cfg.HTTPOptions
		opts.Headers = cfg.HTTPOptions.Headers.Clone()
		out.HTTPOptions = &opts
	}
	return &out
}

func summarizerGenConfig(cfg *genai.GenerateContentConfig) *genai.GenerateContentConfig {
	if cfg == nil {
		return nil
	}
	return &genai.GenerateContentConfig{
		SafetySettings: cfg.SafetySettings,
		Temperature:    cfg.Temperature,
		TopP:           cfg.TopP,
		TopK:           cfg.TopK,
		Seed:           cfg.Seed,
		HTTPOptions:    cfg.HTTPOptions,
		Labels:         cfg.Labels,
	}
}

// placeholderKind names the payload of a part the transcript cannot render
// literally, or "" for a part already rendered elsewhere.
//
// The bytes are deliberately not included. What matters after compaction is
// that the turn is known to have happened and roughly what it carried.
func placeholderKind(p *genai.Part) string {
	switch {
	case p.InlineData != nil:
		return mimeOr(p.InlineData.MIMEType, "inline data")
	case p.FileData != nil:
		return mimeOr(p.FileData.MIMEType, "file")
	case p.ExecutableCode != nil:
		return "executable code"
	case p.CodeExecutionResult != nil:
		return "code execution result"
	default:
		return ""
	}
}

// mimeOr returns a short attachment label for a MIME type, or fallback.
func mimeOr(mimeType, fallback string) string {
	if mimeType == "" {
		return fallback
	}
	return mimeType + " attachment"
}

// truncationSuffixBudget is the room renderTranscript reserves per part for the
// suffix truncateTo appends when it cuts one.
//
// A generous fixed figure rather than an exact one: the suffix carries a count
// whose width varies, and reserving a little too much only means shrinking
// slightly harder than strictly required.
const truncationSuffixBudget = 32

// renderTranscript renders events, keeping the result within the configured
// transcript budget.
//
// A single oversized part is shrunk first, since one pasted document should not
// cost the whole budget. If the transcript still does not fit, that is a window
// too large rather than a part too large, and it is reported instead of
// trimmed: every event here is inside the range the compaction would record as
// covered, so dropping the oldest from the transcript while still deleting them
// from history would lose them with nothing standing in their place.
func (s *LLMSummarizer) renderTranscript(events []*session.Event) (string, error) {
	transcript := s.formatEvents(events, s.maxToolContentChars)
	size := utf8.RuneCountInString(transcript)
	if s.maxTranscriptChars < 0 || size <= s.maxTranscriptChars {
		return transcript, nil
	}

	// Second pass with a per-part cap derived from the budget, so a few large
	// parts are shrunk rather than the whole window being refused.
	//
	// The cap leaves room for the suffix truncateTo appends, because a part
	// only slightly over the cap comes back longer than it went in. Without
	// that room a window of many small parts grows under the pass that exists
	// to shrink it, and is then refused with a size larger than the transcript
	// this function had already rendered.
	if parts := countRenderedParts(events); parts > 0 {
		if cap := s.maxTranscriptChars/parts - truncationSuffixBudget; cap > 0 && cap < s.maxToolContentChars {
			if shrunk := s.formatEvents(events, cap); utf8.RuneCountInString(shrunk) < size {
				transcript, size = shrunk, utf8.RuneCountInString(shrunk)
			}
		}
	}
	if size <= s.maxTranscriptChars {
		return transcript, nil
	}
	return "", fmt.Errorf("rendered transcript is %d characters, over the %d limit, for a window of %d events: compact a smaller window",
		size, s.maxTranscriptChars, len(events))
}

// countRenderedParts counts the parts formatEvents would render a line for.
func countRenderedParts(events []*session.Event) int {
	n := 0
	for _, ev := range events {
		content := utils.Content(ev)
		if content == nil {
			continue
		}
		for _, p := range content.Parts {
			if p == nil {
				continue
			}
			// A placeholder is a rendered line too, so it counts against the
			// budget. Leaving it out let a window of nothing but attachments
			// report zero parts and divide by that.
			if p.Text != "" || p.FunctionCall != nil || p.FunctionResponse != nil || utils.IsProsePart(p) || placeholderKind(p) != "" {
				n++
			}
		}
	}
	return n
}

// GetGoogleLLMVariant reports which Google backend this summarizer's model
// talks to, or [genai.BackendUnspecified] for a model that does not say.
//
// It satisfies the same optional interface the rest of the framework uses to
// distinguish Vertex AI from the Gemini API, so telemetry can label a compaction
// span with the system that produced the summary without the compaction code
// having to know anything about model construction.
func (s *LLMSummarizer) GetGoogleLLMVariant() genai.Backend {
	return googlellm.GetGoogleLLMVariant(s.model)
}
