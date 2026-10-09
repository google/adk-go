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

package llmagent_test

import (
	"context"
	"iter"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// recordingModel forwards to a real model and keeps the prompts it was sent.
//
// The summarizer holds its own model rather than the agent's, so an agent
// BeforeModelCallback never sees a summarization call. Wrapping is the only way
// to observe what the summarizer was actually asked to summarize, which is the
// one thing this test is about.
type recordingModel struct {
	inner model.LLM

	mu      sync.Mutex
	prompts []string
}

func (m *recordingModel) Name() string { return m.inner.Name() }

// GetGoogleLLMVariant forwards to the wrapped model, so the summarizer's
// telemetry reports the real backend rather than an unspecified one.
func (m *recordingModel) GetGoogleLLMVariant() genai.Backend {
	return googlellm.GetGoogleLLMVariant(m.inner)
}

func (m *recordingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	var sb strings.Builder
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.Text != "" {
				sb.WriteString(p.Text)
			}
		}
	}
	m.mu.Lock()
	m.prompts = append(m.prompts, sb.String())
	m.mu.Unlock()
	return m.inner.GenerateContent(ctx, req, stream)
}

func (m *recordingModel) seen() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.prompts...)
}

// TestTailRetentionE2E drives the rolling-summary path through the real
// summarizer and a real model.
//
// The selection logic and the runner already have offline coverage: the seed
// in internal/compactioninternal (TestSelectTailRetentionWindowSeedsPreviousSummary)
// and supersession and boundedness in runner (TestTailRetentionKeepsThePromptBounded).
// Those use fake summarizers. What only this test does is run
// compaction.NewLLMSummarizer, with its default prompt template and transcript
// rendering, around a rolling seed, so the assertions are aimed at what that
// summarizer was actually sent:
//
//   - Every summarization after the first was handed the complete previous
//     summary, as the transcript renders it. That is the seed path, and it is
//     the whole difference from the sliding window TestCompactionE2E covers.
//   - Successive records supersede rather than accumulate: each covers a range
//     starting where the first one did, so exactly one summary materializes
//     into a prompt no matter how many passes have run.
//   - Each agent prompt that follows a new summary holds only that summary,
//     the retained tail and the new message.
//
// Replay keys on exact request bytes, so most defects in these change a request
// and arrive as a replay miss. The checks on captured prompts run before that
// miss is reported, as in TestCompactionE2E, so the failure names the property
// that broke rather than only saying the bytes moved.
//
// It deliberately does not assert that any particular fact survived. Recall is
// a property of the model and the prompt, it varies run to run, and pinning it
// to one recording would make a green cassette look like a guarantee the
// framework cannot give. The measurement for that lives in
// examples/compactionrecall.
//
// Recording: this test needs a cassette. With credentials available, run
//
//	GOOGLE_API_KEY=... go test ./agent/llmagent/ \
//	    -run '^TestTailRetentionE2E$' -httprecord='TestTailRetentionE2E\.httprr$' -count=1 -v
//
// The two regexes differ on purpose, as in TestCompactionE2E: -run matches test
// names and is anchored, -httprecord matches the cassette file path and must not
// be. A failed recording still leaves a plausibly sized file behind, so delete
// it before retrying.
//
// The cassette is sensitive to anything that changes prompt bytes, including the
// summarizer prompt template, the transcript line format and the thresholds
// below. Any of those changes requires re-recording.
//
//go:generate go test -httprecord=^testdata[/\\]TestTailRetentionE2E\.httprr$

func TestTailRetentionE2E(t *testing.T) {
	requireCassette(t)

	// The summarizer's own model, wrapped so its prompts can be inspected.
	//
	// No timeout, for the same reason as TestCompactionE2E: a deadline on the
	// summarization call travels to the wire as an X-Server-Timeout header and
	// would make the cassette depend on that number.
	summarizerModel := &recordingModel{inner: compactionModel(t)}
	summarizer, err := compaction.NewLLMSummarizer(compaction.LLMSummarizerConfig{
		Model: summarizerModel,
	})
	if err != nil {
		t.Fatalf("NewLLMSummarizer() error = %v", err)
	}

	// Each agent prompt is captured with the number of summarizations run by
	// then, so the boundedness check can tell which prompts follow a new
	// summary. Compaction runs inside the turn, before the agent's model call.
	type agentPrompt struct {
		contents       []*genai.Content
		summarizations int
	}
	var (
		mu      sync.Mutex
		prompts []agentPrompt
		capture = func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			n := len(summarizerModel.seen())
			mu.Lock()
			defer mu.Unlock()
			prompts = append(prompts, agentPrompt{contents: req.Contents, summarizations: n})
			return nil, nil
		}
	)

	a, err := llmagent.New(llmagent.Config{
		Name:                     "tail_retention_agent",
		Description:              "agent used to exercise tail-retention compaction",
		Model:                    compactionModel(t),
		Instruction:              "You are a concise assistant. Answer in one short sentence.",
		BeforeModelCallbacks:     []llmagent.BeforeModelCallback{capture},
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	// A low threshold with a short retained tail, so a handful of turns
	// produces several passes. What degrades a rolling summary is the number of
	// times it is re-summarized, not the size at which that starts, so a small
	// threshold buys the behaviour under test in a short recording.
	//
	// Deliberately well below the point where compaction merely starts firing.
	// At 700 it fired once in one recording and twice in another, depending on
	// how verbose the model happened to be, which would make a re-record a coin
	// flip on whether the test exercises anything.
	cfg := &compaction.Config{
		TokenThreshold:     400,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	}
	r := testutil.NewTestAgentRunnerWithCompaction(t, a, cfg)

	const sessionID = "tail_retention_session"
	turns := []string{
		"The deployment region for this project is europe-west4.",
		"Explain in one sentence why connection pooling matters.",
		"In one sentence, when is a circuit breaker worth adding?",
		"One sentence: what is the point of a canary deploy?",
		"Briefly, why do idempotency keys matter for retries?",
		"One sentence on structured logging versus plain text.",
		"Why might p99 latency matter more than the mean? One sentence.",
		"One sentence: what does backpressure mean in a queue?",
	}
	var runErr error
	failedTurn := -1
	for i, turn := range turns {
		if _, err := testutil.CollectTextParts(r.Run(t, sessionID, turn)); err != nil {
			runErr, failedTurn = err, i
			break
		}
	}

	// Boundedness. Tail retention fires on prompt tokens, not on content
	// count, so how far a prompt grows between summaries depends on how much
	// the model writes and thinks; in this recording thinking tokens carry most
	// of each turn's margin over the threshold. What the trigger does
	// guarantee is that the prompt right after a new summary holds only that
	// summary, the retained tail and the new message, so only those prompts
	// are held to the bound. A re-record with a terser model must not fail
	// correct code.
	mu.Lock()
	captured := append([]agentPrompt(nil), prompts...)
	mu.Unlock()
	maxContents := cfg.EventRetentionSize + 2
	var afterSummary, ran int
	for i, p := range captured {
		if p.summarizations == ran {
			continue
		}
		ran = p.summarizations
		afterSummary++
		if len(p.contents) > maxContents {
			t.Errorf("prompt %d, the first after summarization %d, carries %d contents, want at most %d (summary, retained tail of %d, new message): history is not being replaced by a summary",
				i+1, ran, len(p.contents), maxContents, cfg.EventRetentionSize)
		}
	}
	if afterSummary == 0 {
		t.Errorf("none of %d agent prompts followed a summarization, so compaction never ran", len(captured))
	}

	// The seed. The summarizer is called once per stored record, so call k
	// produced summaries[k] and must have been handed summaries[k-1]. It is
	// matched whole, with newlines escaped as the transcript renders them so a
	// tool response cannot forge a turn. A summary longer than the
	// transcript's per-part cap would arrive truncated and fail here.
	events := sessionEventsFor(t, r, sessionID)
	var summaries []*session.Event
	for _, ev := range events {
		if compactioninternal.HasUsableSummary(ev) {
			summaries = append(summaries, ev)
		}
	}
	calls := summarizerModel.seen()
	for k := 1; k < len(calls) && k <= len(summaries); k++ {
		prev := strings.TrimSpace(textOf(summaries[k-1].Actions.Compaction.CompactedContent))
		if prev == "" {
			t.Errorf("summary %d is empty", k)
			continue
		}
		if rendered := strings.ReplaceAll(prev, "\n", `\n`); !strings.Contains(calls[k], rendered) {
			t.Errorf("summarization %d was not handed summary %d (%d chars as rendered), so it was not built on its predecessor", k+1, k, len(rendered))
		}
	}

	if runErr != nil {
		t.Fatalf("turn %d (%q) failed: %v", failedTurn+1, turns[failedTurn], runErr)
	}
	if len(summaries) < 2 {
		t.Fatalf("got %d compaction records over %d turns, need at least 2 for one summary to be built on another; this test exercised nothing", len(summaries), len(turns))
	}
	if len(calls) != len(summaries) {
		t.Errorf("summarizer called %d times for %d stored records, so calls cannot be paired with the records they produced", len(calls), len(summaries))
	}

	// Supersession. Every record rolls forward from where the first one began,
	// so one summary replaces its predecessor instead of stacking beside it.
	base := summaries[0].Actions.Compaction.StartTimestamp
	for i, s := range summaries[1:] {
		c := s.Actions.Compaction
		if !c.StartTimestamp.Equal(base) {
			t.Errorf("summary %d starts at %v, want %v: records are accumulating rather than rolling", i+1, c.StartTimestamp, base)
		}
		if prev := summaries[i].Actions.Compaction; !c.EndTimestamp.After(prev.EndTimestamp) {
			t.Errorf("summary %d ends at %v, not after its predecessor's %v: the window did not advance", i+1, c.EndTimestamp, prev.EndTimestamp)
		}
	}
}
