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
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/internal/httprr"
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

// TestTailRetentionE2E drives the rolling-summary path against a real model.
//
// TestCompactionE2E covers the sliding window, which summarizes each group of
// turns once and never revisits it. Tail retention is the other trigger and the
// one with a failure mode of its own: it seeds every new summary with the
// previous one, so a value has to survive being recompressed on every pass.
// Nothing exercised that end to end, which is why the default summarizer prompt
// could lose the early conversation without a single test noticing.
//
// What it asserts, in order of what would go unnoticed without it:
//
//   - A later summarization was handed an earlier summary. That is the seed
//     path, and it is the whole difference from the sliding window.
//   - Successive records supersede rather than accumulate: each covers a range
//     starting where the first one did, so exactly one summary materializes
//     into a prompt no matter how many passes have run.
//   - The prompt stays bounded while the conversation grows, which is the
//     property tail retention exists to provide.
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
	trace := filepath.Join("testdata", t.Name()+".httprr")
	if recording, _ := httprr.Recording(trace); !recording {
		const reRecord = "Re-record with: GOOGLE_API_KEY=... go test ./agent/llmagent/ " +
			"-run '^TestTailRetentionE2E$' -httprecord='TestTailRetentionE2E\\.httprr$' -count=1 -v"
		info, err := os.Stat(trace)
		if err != nil {
			t.Fatalf("no cassette at %s: %v. It is committed, so this means it was lost or renamed. %s", trace, err, reRecord)
		}
		if info.Size() < minCassetteBytes {
			t.Fatalf("the cassette at %s is %d bytes, too small to hold a conversation. "+
				"A re-record that failed partway leaves a header-only stub. %s", trace, info.Size(), reRecord)
		}
	}

	var (
		mu      sync.Mutex
		prompts [][]*genai.Content
		capture = func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			mu.Lock()
			defer mu.Unlock()
			prompts = append(prompts, req.Contents)
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

	// A low threshold with a short retained tail, so a handful of turns
	// produces several passes. What degrades a rolling summary is the number of
	// times it is re-summarized, not the size at which that starts, so a small
	// threshold buys the behaviour under test in a short recording.
	//
	// Deliberately well below the point where compaction merely starts firing.
	// At 700 it fired once in one recording and twice in another, depending on
	// how verbose the model happened to be, which would make a re-record a coin
	// flip on whether the test exercises anything.
	r := testutil.NewTestAgentRunnerWithCompaction(t, a, &compaction.Config{
		TokenThreshold:     400,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	})

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
	if runErr != nil {
		t.Fatalf("turn %d (%q) failed: %v", failedTurn+1, turns[failedTurn], runErr)
	}

	events := sessionEventsFor(t, r, sessionID)
	var summaries []*session.Event
	for _, ev := range events {
		if compactioninternal.HasUsableSummary(ev) {
			summaries = append(summaries, ev)
		}
	}
	if len(summaries) < 2 {
		t.Fatalf("got %d compaction records over %d turns, need at least 2 for one summary to be built on another; this test exercised nothing", len(summaries), len(turns))
	}

	// The seed. A later summarization must have been handed the text of an
	// earlier summary, or the rolling path never ran and this is a sliding
	// window with extra steps.
	first := strings.TrimSpace(textOf(summaries[0].Actions.Compaction.CompactedContent))
	if first == "" {
		t.Fatal("the first stored summary is empty")
	}
	// Matched on the longest single line rather than the whole summary. The
	// transcript escapes newlines so a tool response cannot forge a turn, so a
	// multi-line summary never appears in a prompt in the form it was stored.
	var needle string
	for _, line := range strings.Split(first, "\n") {
		if line = strings.TrimSpace(line); len(line) > len(needle) {
			needle = line
		}
	}
	if len(needle) < 40 {
		t.Fatalf("the first summary has no line long enough to identify it, so this assertion cannot distinguish anything.\nsummary:\n%s", first)
	}
	var reSummarized bool
	for _, p := range summarizerModel.seen() {
		if strings.Contains(p, needle) {
			reSummarized = true
			break
		}
	}
	if !reSummarized {
		t.Errorf("no summarization prompt contained a line from the first summary, so no summary was built on a previous one.\nlooked for:\n%s", needle)
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

	// Boundedness, the property tail retention is chosen for. The last prompt
	// must not be the whole conversation: with a retained tail of 2 it carries
	// one summary plus a few recent turns however long the session runs.
	mu.Lock()
	captured := append([][]*genai.Content(nil), prompts...)
	mu.Unlock()
	if len(captured) < len(turns) {
		t.Fatalf("captured %d prompts for %d turns", len(captured), len(turns))
	}
	last := captured[len(captured)-1]
	if len(last) >= len(turns)*2 {
		t.Errorf("the final prompt carries %d contents for %d turns, so history is not being replaced by a summary", len(last), len(turns))
	}
}
