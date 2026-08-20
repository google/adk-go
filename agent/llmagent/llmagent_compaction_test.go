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
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// minCassetteBytes is the floor below which a cassette cannot hold a recorded
// conversation. The header alone is a few dozen bytes and a real recording here
// is tens of kilobytes, so anything under this is a stub from a failed record.
const minCassetteBytes = 1024

// TestCompactionE2E drives a real model through enough turns to trigger a
// sliding-window compaction, then checks that the next prompt carries the
// summary in place of the turns it covers, and is still accepted by the model.
//
// Not that the prompt is smaller. On a three-turn recording it is not: one
// model-written summary of two one-line turns is longer than the two lines.
// Compaction pays off against real history, and this fixture is not that.
//
// What a passing run establishes, stated narrowly because replay keys on exact
// request bytes and it is easy to read more into a green test than it holds:
//
//   - A real model accepted a compacted prompt, at the time the cassette was
//     recorded. Not since.
//   - A summary and live tool traffic coexisted in one accepted prompt.
//   - Offline, over the prompts the agent actually sent: the summary replaced
//     the turns it covers, the covered turns are still in the session, and no
//     prompt carries a function response without its call.
//
// What it does not establish is that anything still works against a live API. A
// structural defect introduced later changes the prompt bytes, so it arrives as
// a cassette miss rather than an API rejection. The offline assertions run
// before that miss is reported, so the failure says which property broke rather
// than only that the bytes moved.
//
// The fourth turn calls a tool after the compaction point on purpose, so the
// prompt it produces carries a summary and function traffic together. Without
// it the recording never exercises call pairing across a summary, which is the
// one thing this test is best placed to check.
//
// Deliberately not asserted: any particular wording for the summary. It is model
// output, and pinning it would fail on any model or prompt revision without
// indicating a real problem. What is asserted is that whatever the summarizer
// produced is what reaches the next prompt.
//
// Recording: this test needs a cassette. With credentials available, run
//
//	GOOGLE_API_KEY=... go test ./agent/llmagent/ \
//	    -run '^TestCompactionE2E$' -httprecord='TestCompactionE2E\.httprr$' -count=1 -v
//
// Note the two regexes differ on purpose. -run matches test names, so it is
// anchored. -httprecord matches the cassette FILE PATH, so anchoring it the same
// way would never match "testdata/TestCompactionE2E.httprr", so nothing would
// be recorded.
//
// Commit the resulting testdata/TestCompactionE2E.httprr. The cassette is
// committed, so a missing one is a lost or renamed file and fails the test
// rather than skipping it.
//
// This test deliberately has no //go:generate directive of its own. The
// package-level one already carries -httprecord=Test, which matches every
// cassette here, so adding a third changed nothing except the number of ways to
// re-record all of them by accident. For the same reason, do not record with
// "go generate ./agent/llmagent/...". Note also that a failed
// recording still leaves a cassette behind, and it can look plausibly sized
// because the failing exchange is recorded too. Delete it, or the next run
// replays the recorded failure.
//
// The cassette is sensitive to anything that changes prompt bytes, including the
// summarizer prompt template, the transcript line format, tool-argument
// rendering and truncation behaviour. Any of those changes requires re-recording.
func TestCompactionE2E(t *testing.T) {
	// Matches llmagent_delegation_test.go, which is the most recently recorded
	// suite in this package. Change it only alongside a re-record, since the
	// model name is part of the request URL the cassette keys on.

	// The cassette is committed, so its absence is a lost or renamed file rather
	// than an unrecorded checkout. Skipping would turn that into a silent pass,
	// which is how a test quietly stops running for months. Every other cassette
	// test in this package fails instead, and so does this one. Recording mode
	// goes ahead regardless, since that is the run that creates the file.
	trace := filepath.Join("testdata", t.Name()+".httprr")
	if recording, _ := httprr.Recording(trace); !recording {
		const reRecord = "Re-record with: GOOGLE_API_KEY=... go test ./agent/llmagent/ " +
			"-run '^TestCompactionE2E$' -httprecord='TestCompactionE2E\\.httprr$' -count=1 -v"
		info, err := os.Stat(trace)
		if err != nil {
			t.Fatalf("no cassette at %s: %v. It is committed, so this means it was lost or renamed. %s", trace, err, reRecord)
		}
		// A failed or interrupted re-record leaves the header and nothing else.
		// Accepting it meant dying eighty lines later on a replay miss, with a
		// message that never mentioned the cassette.
		if info.Size() < minCassetteBytes {
			t.Fatalf("the cassette at %s is %d bytes, too small to hold a conversation. "+
				"A re-record that failed partway leaves a header-only stub. %s", trace, info.Size(), reRecord)
		}
	}

	// Captured before each model call, so the assertions can look at the exact
	// history the agent sent rather than inferring it from the session.
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

	// A tool gives the transcript function calls and responses to render, which
	// is the part of the summarizer prompt most likely to break.
	type cityArgs struct {
		City string `json:"city" jsonschema:"the city to look up"`
	}
	type weatherResult struct {
		Weather string `json:"weather"`
	}
	weather, err := functiontool.New[cityArgs, weatherResult](
		functiontool.Config{Name: "get_weather", Description: "Returns the weather in a city."},
		func(_ agent.Context, args cityArgs) (weatherResult, error) {
			return weatherResult{Weather: "sunny in " + args.City}, nil
		},
	)
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:                     "compaction_agent",
		Description:              "agent used to exercise context compaction",
		Model:                    compactionModel(t),
		Instruction:              "You are a concise assistant. Answer in one short sentence.",
		Tools:                    []tool.Tool{weather},
		BeforeModelCallbacks:     []llmagent.BeforeModelCallback{capture},
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	// Interval 2 keeps the recording short: three turns produce exactly one
	// compaction, which is all this test needs.
	//
	// No OverlapSize. It would be inert here and claiming otherwise was wrong:
	// overlap only reaches back past an earlier compaction, and the first
	// compaction of a session starts at the first invocation whatever the
	// overlap is. Exercising the seam needs a second window, so it belongs in
	// the offline tests where windows are cheap.
	// An explicit summarizer with no timeout, because a deadline on the
	// summarization call travels to the wire as an X-Server-Timeout header and
	// so becomes part of what the recording has to match. The runner installs
	// one with a timeout by default, which is right in production and would
	// make every cassette holding a summarizer call depend on that number.
	summarizer, err := compaction.NewLLMSummarizer(compaction.LLMSummarizerConfig{
		Model: compactionModel(t),
	})
	if err != nil {
		t.Fatalf("NewLLMSummarizer() error = %v", err)
	}
	r := testutil.NewTestAgentRunnerWithCompaction(t, a, &compaction.Config{
		CompactionInterval: 2,
		Summarizer:         summarizer,
	})

	const sessionID = "compaction_session"
	turns := []string{
		"What is the weather in Zurich?",
		"My favourite colour is teal, remember that.",
		"What was my favourite colour again?",
		// A tool call after the compaction point, so the prompt that follows it
		// carries a function call and its response alongside a summary. That is
		// the arrangement the call-pairing and call-recovery paths exist for,
		// and without this turn the recording never produces one.
		"Now check the weather in Oslo.",
	}
	answers := make([][]string, len(turns))
	var runErr error
	failedTurn := -1
	for i, turn := range turns {
		answer, err := testutil.CollectTextParts(r.Run(t, sessionID, turn))
		if err != nil {
			runErr, failedTurn = err, i
			break
		}
		answers[i] = answer
	}

	// The offline checks run on whatever was captured, before any run failure
	// is reported. A compaction defect changes the bytes of the prompt, so it
	// reaches this test as a replay miss, and failing on that first put every
	// assertion in this file behind an error that says only "cached HTTP
	// response not found". The checks below are what say which defect it was.
	//
	// This one is weaker than it looks: the framework's own pairing guard
	// rejects an orphaned response before a prompt is ever sent, so it catches
	// a defect only in the window where the prompt is assembled but not yet
	// validated. Running it is still the difference between a diagnosis and a
	// byte mismatch.
	mu.Lock()
	captured := append([][]*genai.Content(nil), prompts...)
	mu.Unlock()
	for i, p := range captured {
		assertNoOrphanFunctionResponses(t, i, p)
	}

	if runErr != nil {
		t.Fatalf("turn %d (%q) failed: %v", failedTurn+1, turns[failedTurn], runErr)
	}

	// A compaction must have landed. Without this the rest proves nothing.
	events := sessionEventsFor(t, r, sessionID)
	summaries := make([]*session.Event, 0, 1)
	for _, ev := range events {
		if compactioninternal.HasUsableSummary(ev) {
			summaries = append(summaries, ev)
		}
	}
	if len(summaries) == 0 {
		t.Fatalf("no compaction event after %d turns, so this test exercised nothing", len(turns))
	}

	// The first summary, not the last. With four turns the last compaction is
	// written after the final model call, so it cannot appear in any recorded
	// prompt; the first one covers the opening turns and is what the later
	// prompts stand on.
	summaryText := textOf(summaries[0].Actions.Compaction.CompactedContent)
	if strings.TrimSpace(summaryText) == "" {
		t.Error("the stored summary is empty")
	}

	// The final prompt is the interesting one: it is the first assembled after a
	// summary existed. It must carry the summary instead of the turns it covers,
	// and the model must have accepted it, which the absence of an error above
	// already establishes.
	mu.Lock()
	defer mu.Unlock()
	if len(prompts) < len(turns) {
		t.Fatalf("captured %d prompts, want at least %d", len(prompts), len(turns))
	}
	final := promptTextOf(prompts[len(prompts)-1])

	if !strings.Contains(final, strings.TrimSpace(summaryText)) {
		t.Errorf("final prompt does not contain the stored summary.\nsummary:\n%s\n\nprompt:\n%s", summaryText, final)
	}
	if strings.Contains(final, turns[0]) {
		t.Errorf("final prompt still contains the compacted first turn %q:\n%s", turns[0], final)
	}
	if !strings.Contains(final, turns[len(turns)-1]) {
		t.Errorf("final prompt is missing the current turn %q:\n%s", turns[len(turns)-1], final)
	}
	// Turn 2 is inside the compacted range as well, so it must be gone for the
	// same reason turn 1 is. Asserting only turn 1 left half the range unchecked.
	if strings.Contains(final, turns[1]) {
		t.Errorf("final prompt still contains the compacted second turn %q:\n%s", turns[1], final)
	}

	// The prompt is measured rather than assumed, and what it establishes is
	// that the covered turns stopped being sent, not that the total shrank.
	//
	// On this recording the total grows slightly, 782 characters to 835, and
	// that is correct rather than a defect: three short turns are replaced by
	// one model-written summary, and a summary of two one-line turns is longer
	// than the two lines. Compaction pays off against real history, not against
	// a three-turn fixture. The doc comment used to claim the prompt was
	// smaller here, which the recording contradicts.
	//
	// Growth well beyond that would mean something is riding along inside the
	// summary that is not prose, so the bound is loose but not absent.
	beforeCompaction := len(promptTextOf(prompts[len(prompts)-2]))
	if afterCompaction := len(final); afterCompaction > beforeCompaction*2 {
		t.Errorf("the prompt more than doubled across the compaction, %d characters to %d, so the summary is carrying something other than prose",
			beforeCompaction, afterCompaction)
	}

	// Compacting rather than truncating, asserted against the store rather than
	// inferred from the prompt. Nothing here distinguished "absent from the
	// prompt" from "absent from the session" before: a compaction patched to
	// drop the covered events outright took the session from 14 events to 2,
	// lost every raw turn, and this test still passed.
	//
	// The session is the audit record. A summary standing in for turns inside a
	// prompt is the feature, and those same turns disappearing from storage is
	// data loss.
	if len(events) < len(turns) {
		t.Errorf("the session holds %d events for %d turns, so history was deleted rather than compacted",
			len(events), len(turns))
	}
	for _, want := range turns {
		if !sessionHoldsText(events, want) {
			t.Errorf("turn %q is no longer in the session: compaction must leave history intact", want)
		}
	}

	// The point of compacting rather than truncating: the fact survives into the
	// summary and the model can still answer from it. This is turn 3
	// specifically, the one that asks, rather than whichever turn happens to be
	// last. Without it the test proved the prompt shrank, not that it kept
	// working.
	recall := strings.ToLower(strings.Join(answers[2], " "))
	if !strings.Contains(recall, "teal") {
		t.Errorf("the model could not recall the colour from the summary alone; answer was %q", recall)
	}

	// Structural checks, asserted here rather than inferred from the fact that a
	// recorded model once accepted the bytes. Every prompt is checked, not only
	// the final one.
	// The summary must actually stand for the range it claims. Every event the
	// first compaction covers has to be absent from the final prompt: the two
	// turns asserted above are the visible part of that, but the range is the
	// contract, so it is checked directly.
	covered := summaries[0].Actions.Compaction

	// Searched with the summary removed. The summary is in the prompt on
	// purpose and it paraphrases the turns it covers, so any overlap of wording
	// reads as a covered turn surviving. The recorded summary says "The current
	// weather in Zurich is sunny." against a covered "The weather in Zurich is
	// currently sunny.", which is one word's ordering away from failing this
	// test for no reason on the next re-record.
	outsideSummary := strings.ReplaceAll(final, strings.TrimSpace(summaryText), "")

	// hasCompaction, not HasUsableSummary: the latter answers "is there a
	// usable summary here", which its own doc says is a different question from
	// "is this bookkeeping". Filtering on it left the compacted tool traffic
	// unexamined, which is exactly the pair the range is most likely to break.
	checked := 0
	for _, ev := range events {
		if ev.Actions.Compaction != nil ||
			ev.Timestamp.Before(covered.StartTimestamp) || ev.Timestamp.After(covered.EndTimestamp) {
			continue
		}
		content := utils.Content(ev)
		if content == nil {
			// A covered event with no content used to take the package binary
			// down with a nil dereference here.
			continue
		}
		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			if text := strings.TrimSpace(part.Text); text != "" {
				checked++
				if strings.Contains(outsideSummary, text) {
					t.Errorf("event %q is covered by the summary but its text is still in the final prompt: %q", ev.ID, part.Text)
				}
			}
			// The tool traffic, matched by call ID rather than by text. Two of
			// the six covered events are a call and its response, and neither
			// carries text, so a text-only sweep never looked at them.
			if fc := part.FunctionCall; fc != nil && fc.ID != "" {
				checked++
				if promptMentionsCallID(prompts[len(prompts)-1], fc.ID) {
					t.Errorf("event %q is covered but its function call %q is still in the final prompt", ev.ID, fc.ID)
				}
			}
			if fr := part.FunctionResponse; fr != nil && fr.ID != "" {
				checked++
				if promptMentionsCallID(prompts[len(prompts)-1], fr.ID) {
					t.Errorf("event %q is covered but its function response %q is still in the final prompt", ev.ID, fr.ID)
				}
			}
		}
	}
	// Without a floor this loop goes quiet rather than failing: blanking the
	// covered events' parts left it making zero comparisons and the test still
	// passed. Six events are covered in the recording and four of them carry
	// something to compare, so anything below that means the sweep stopped
	// looking rather than stopped finding.
	if checked < 4 {
		t.Errorf("the covered-range sweep made %d comparisons, want at least 4: it is not examining what it claims to", checked)
	}

	withSummaryAndTools := 0
	for _, p := range prompts {
		if promptHasFunctionTraffic(p) && strings.Contains(promptTextOf(p), strings.TrimSpace(summaryText)) {
			withSummaryAndTools++
		}
	}
	// At least one prompt must carry a summary and function traffic together.
	// That is the arrangement call pairing has to survive, and the only one this
	// test is better placed to check than an offline test is. A re-record whose
	// conversation stops calling tools after the compaction point would lose it
	// silently, so it is asserted rather than assumed.
	if withSummaryAndTools == 0 {
		t.Error("no recorded prompt carries both a summary and function traffic, so pairing across a summary was never exercised")
	}
}

// promptHasFunctionTraffic reports whether contents carry a function call or
// response.
func promptHasFunctionTraffic(contents []*genai.Content) bool {
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, part := range c.Parts {
			if part != nil && (part.FunctionCall != nil || part.FunctionResponse != nil) {
				return true
			}
		}
	}
	return false
}

// assertNoOrphanFunctionResponses checks that every function response in a
// prompt is preceded by the call it answers.
//
// This is the property compaction is most likely to break: the summary replaces
// a span of history, and a response whose call fell inside that span while the
// response itself did not would reach the model unpaired, which real backends
// reject.
func assertNoOrphanFunctionResponses(t *testing.T, promptIdx int, contents []*genai.Content) {
	t.Helper()

	seenCalls := make(map[string]bool)
	responses := 0
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, part := range c.Parts {
			if part == nil {
				continue
			}
			if fc := part.FunctionCall; fc != nil {
				seenCalls[callKey(fc.ID, fc.Name)] = true
			}
			if fr := part.FunctionResponse; fr != nil {
				responses++
				if !seenCalls[callKey(fr.ID, fr.Name)] {
					t.Errorf("prompt %d carries a function response %q with no preceding call", promptIdx, fr.Name)
				}
			}
		}
	}
	if responses == 0 {
		// Not a failure. It is the documented gap: this recording has no tool
		// traffic after the compaction point, so for the final prompt there is
		// nothing here to check.
		t.Logf("prompt %d carries no function responses, so pairing was not exercised in it", promptIdx)
	}
}

// callKey identifies a call by ID when the model supplies one, and by name
// otherwise, which is what happens with models that omit call IDs.
func callKey(id, name string) string {
	if id != "" {
		return "id:" + id
	}
	return "name:" + name
}

func sessionEventsFor(t *testing.T, r *testutil.TestAgentRunner, sessionID string) []*session.Event {
	t.Helper()
	resp, err := r.SessionService().Get(context.Background(), &session.GetRequest{
		AppName: "test_app", UserID: "test_user", SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("session Get() error = %v", err)
	}
	var events []*session.Event
	for ev := range resp.Session.Events().All() {
		events = append(events, ev)
	}
	return events
}

func textOf(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func promptTextOf(contents []*genai.Content) string {
	var b strings.Builder
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			switch {
			case p == nil:
			case p.Text != "":
				b.WriteString("[" + c.Role + "] " + p.Text + "\n")
			case p.FunctionCall != nil:
				b.WriteString("[" + c.Role + "] CALL " + p.FunctionCall.Name + "\n")
			case p.FunctionResponse != nil:
				b.WriteString("[" + c.Role + "] RESPONSE " + p.FunctionResponse.Name + "\n")
			}
		}
	}
	return b.String()
}

// sessionHoldsText reports whether any stored event still carries want.
//
// Read against the session rather than the prompt, so it answers "is the
// history still there" rather than "was it shown to the model", which is the
// distinction between compacting and truncating.
func sessionHoldsText(events []*session.Event, want string) bool {
	for _, ev := range events {
		content := utils.Content(ev)
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part != nil && strings.Contains(part.Text, want) {
				return true
			}
		}
	}
	return false
}

// promptMentionsCallID reports whether any part of the prompt carries a
// function call or response with the given ID.
func promptMentionsCallID(contents []*genai.Content, id string) bool {
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, part := range c.Parts {
			if part == nil {
				continue
			}
			if fc := part.FunctionCall; fc != nil && fc.ID == id {
				return true
			}
			if fr := part.FunctionResponse; fr != nil && fr.ID == id {
				return true
			}
		}
	}
	return false
}

// compactionModelName is the model this end-to-end test records against.
const compactionModelName = "gemini-3.5-flash"

// compactionModel returns the one Gemini model this test uses, for both the
// agent and the summarizer.
//
// It must be one instance. newGeminiModel opens a recorder per call, and in
// record mode that is an os.Create on a path derived from the test name:
// truncating, with its own write offset. Two of them on one trace overwrite
// each other, and the result stops parsing after one record while still
// clearing the size floor this file checks, so the documented re-record command
// produced a cassette that looked fine and was not. Replay is unaffected, which
// is why every assertion passed. The delegation tests memoise for exactly this
// reason.
func compactionModel(t *testing.T) model.LLM {
	t.Helper()
	if m, ok := compactionModels.Load(t.Name()); ok {
		return m.(model.LLM)
	}
	m := newGeminiModel(t, compactionModelName, nil)
	compactionModels.Store(t.Name(), m)
	t.Cleanup(func() { compactionModels.Delete(t.Name()) })
	return m
}

// compactionModels caches one model instance per test name, cleared in
// t.Cleanup so a -count=N run does not reuse a stale one.
var compactionModels sync.Map
