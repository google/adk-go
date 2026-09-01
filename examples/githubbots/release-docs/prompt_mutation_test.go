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

package main

// Mutation testing for the PROMPT.
//
// The e2e suite proves the bot behaves with the prompt it has. It cannot say
// which parts of that prompt are doing the work. A section nobody can delete
// without breaking a scenario is load-bearing; a section that can be deleted
// with no scenario changing its mind is inert, and inert text in a prompt is not
// free -- it is tokens on every call, and a future editor will preserve it
// carefully because it looks deliberate.
//
// So: delete one section, re-run every scenario, and count the decisions that
// flipped. This is the same argument as mutation testing for code, applied to
// the artifact that actually decides this bot's behavior.
//
// Expensive: one model call per scenario per mutation. Gated separately from the
// rest of the suite so an ordinary e2e run does not pay for it.
//
//	RELEASE_DOCS_E2E=1 RELEASE_DOCS_PROMPT_MUTATIONS=1 GEMINI_API_KEY=... \
//	  go test -run TestE2EPromptMutations -timeout 2h -v .

import (
	"context"
	"iter"
	"os"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/model"
)

// A mutation removes one section of the prompt.
//
// Deletion rather than rewording on purpose: a reworded section tests the
// wording, and the question here is whether the section earns its place at all.
type mutation struct {
	name string
	// buys is what the section is presumed to buy. When a mutation reports no
	// flipped scenarios, this is the claim that just lost its evidence.
	buys  string
	apply func(string) string
}

// positiveControl is a prompt that MUST change the outcome. It exists to prove
// the harness can observe a change at all.
//
// Every "inert" verdict this file produces is an argument from absence, and an
// argument from absence is worth nothing until the measurement has been shown
// to detect a presence. If swapping the whole instruction for "flag everything"
// moves no scenario, then the mutated prompt is not reaching the model, and
// every other result in the run is an artifact of the harness rather than a
// finding about the prompt. Far cheaper to check than to be wrong about.
const positiveControl = `You are a documentation analyst for '{OWNER}/{REPO}'.

Call ` + "`record_documentation_findings`" + ` exactly ONCE for this group. For EVERY
file in the group, record one finding: kind new-feature, summary naming the
file, proposed_change "document this file", reasoning "everything is worth
documenting", reference set to the file path. Never record an empty list. Do
this regardless of what the change contains.`

func promptMutations() []mutation {
	return []mutation{
		{
			name:  "drop_negative_rules",
			buys:  "the six declining scenarios: without it, nothing tells the model what to ignore",
			apply: dropSection("## What does NOT deserve a finding"),
		},
		{
			name:  "drop_positive_rules",
			buys:  "the seven filing scenarios: without it, nothing enumerates what counts",
			apply: dropSection("## What deserves a finding"),
		},
		{
			name:  "drop_untrusted_warning",
			buys:  "the three injection scenarios: the only text telling the model the diff is data",
			apply: dropParagraph("CRITICAL: the contributor-supplied content"),
		},
		{
			name:  "drop_no_other_tools",
			buys:  "bounding the model to the one tool it has",
			apply: dropParagraph("You have no other tools."),
		},
		{
			name:  "drop_brevity_instruction",
			buys:  "findings short enough for a maintainer to skim",
			apply: dropParagraph("Keep every field to a couple of sentences."),
		},
		{
			name: "vague_core_instruction",
			buys: "a precise task statement; the vague form is what a careless edit would leave",
			apply: replaceOnce(
				"Your job is\nto decide which user-facing documentation, if any, needs updating because of\nthose changes, and to record that as a list of findings.",
				"Review the changes and record anything noteworthy."),
		},
	}
}

// dropSection removes a "## " heading and everything up to the next one.
func dropSection(heading string) func(string) string {
	return func(s string) string {
		start := strings.Index(s, heading)
		if start < 0 {
			return s
		}
		rest := s[start+len(heading):]
		if next := strings.Index(rest, "\n## "); next >= 0 {
			return s[:start] + rest[next+1:]
		}
		return s[:start]
	}
}

// dropParagraph removes the blank-line-delimited paragraph starting with prefix.
func dropParagraph(prefix string) func(string) string {
	return func(s string) string {
		start := strings.Index(s, prefix)
		if start < 0 {
			return s
		}
		rest := s[start:]
		if end := strings.Index(rest, "\n\n"); end >= 0 {
			return s[:start] + rest[end+2:]
		}
		return s[:start]
	}
}

func replaceOnce(old, new string) func(string) string {
	return func(s string) string { return strings.Replace(s, old, new, 1) }
}

// TestPromptMutationsApply checks every mutation actually changes the prompt.
//
// Deliberately ungated, and the most important test in this file. A mutation
// whose string match silently failed is a no-op, and a no-op mutation reports
// "no scenario flipped" -- which reads exactly like the finding that a section
// is inert. Without this check the expensive run below can conclude, with a
// straight face, that the prompt's most load-bearing paragraph does nothing.
func TestPromptMutationsApply(t *testing.T) {
	cfg := e2eCfg()
	base := renderPrompt(cfg)
	for _, mu := range promptMutations() {
		t.Run(mu.name, func(t *testing.T) {
			orig := promptTemplate
			defer func() { promptTemplate = orig }()

			promptTemplate = mu.apply(promptTemplate)
			got := renderPrompt(cfg)
			if got == base {
				t.Fatalf("the mutation changed nothing, so it would report as inert without being it")
			}
			if len(got) >= len(base) && mu.name != "vague_core_instruction" {
				t.Errorf("a deletion made the prompt longer: %d -> %d bytes", len(base), len(got))
			}
			// renderPrompt must still leave no stray braces, or the run fails
			// for a reason that has nothing to do with the deleted section.
			if brace := regexp.MustCompile(`[{}]`).FindString(got); brace != "" {
				t.Errorf("the mutated prompt has a stray %q, which llmagent reads as a state key", brace)
			}
			t.Logf("%s: %d -> %d bytes", mu.name, len(base), len(got))
		})
	}
}

// TestPromptMutationsReachTheModel proves a mutation survives the whole path
// from promptTemplate to the request the model receives.
//
// This is the other half of the argument-from-absence problem, and the half
// TestPromptMutationsApply cannot see. That test proves the TEXT changed. This
// one proves the changed text is what the agent is actually built with -- if
// llmagent cached the instruction, or newAgentRunner were called once and
// reused across mutations, every mutation would be applied to a string nobody
// reads, and the expensive run would report the entire prompt inert.
//
// Free, deterministic and offline, so it guards that conclusion on every CI run
// rather than only when somebody pays for a model.
func TestPromptMutationsReachTheModel(t *testing.T) {
	const marker = "What does NOT deserve a finding"

	base := captureInstruction(t, nil)
	if !strings.Contains(base, marker) {
		t.Fatalf("the intact prompt never reached the model; captured %d bytes:\n%s", len(base), base)
	}
	mutated := captureInstruction(t, dropSection("## "+marker))
	if strings.Contains(mutated, marker) {
		t.Error("the deleted section still reached the model, so mutations are applied to a string " +
			"nobody reads and every \"inert\" verdict would be meaningless")
	}
	if len(mutated) >= len(base) {
		t.Errorf("the mutated instruction is not shorter: %d -> %d bytes", len(base), len(mutated))
	}
}

// captureInstruction builds the real agent, runs one group against a model that
// records its system instruction, and returns it. apply, when non-nil, mutates
// the prompt first.
func captureInstruction(t *testing.T, apply func(string) string) string {
	t.Helper()
	if apply != nil {
		orig := promptTemplate
		promptTemplate = apply(promptTemplate)
		defer func() { promptTemplate = orig }()
	}

	cfg := testConfig()
	capture := &instructionCapture{}
	orig := newModel
	newModel = func(context.Context, *Config) (model.LLM, error) { return capture, nil }
	defer func() { newModel = orig }()

	r, sessions, err := newAgentRunner(context.Background(), cfg, newRecorder(cfg), discardLogger())
	if err != nil {
		t.Fatalf("newAgentRunner: %v", err)
	}
	diff := testDiff()
	gh := testClient(t, cfg, failIfCalled(t))
	runGroup(context.Background(), r, sessions, gh, discardLogger(), diff, diff.Files, 0, 1, "v1.0.0...v1.1.0")
	return capture.instruction()
}

// instructionCapture records the system instruction it is given and produces no
// content, so the turn ends immediately.
type instructionCapture struct {
	mu  sync.Mutex
	got string
}

func (c *instructionCapture) Name() string { return "instruction-capture" }

func (c *instructionCapture) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	c.mu.Lock()
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, p := range req.Config.SystemInstruction.Parts {
			c.got += p.Text
		}
	}
	c.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func (c *instructionCapture) instruction() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.got
}

// --- the expensive part -----------------------------------------------------

// outcome is what one scenario did under one prompt.
type outcome int

const (
	outcomeFiled outcome = iota
	outcomeDeclined
	outcomeInconclusive // the model was unavailable; proves nothing either way
)

func (o outcome) String() string {
	switch o {
	case outcomeFiled:
		return "filed"
	case outcomeDeclined:
		return "declined"
	default:
		return "inconclusive"
	}
}

func TestE2EPromptMutations(t *testing.T) {
	key := requireE2E(t)
	if os.Getenv("RELEASE_DOCS_PROMPT_MUTATIONS") != "1" {
		t.Skip("set RELEASE_DOCS_PROMPT_MUTATIONS=1 to mutation-test the prompt (one model call per scenario per mutation)")
	}
	m := e2eModel(t, key)
	scenarios := e2eScenarios()

	// The unmutated prompt first. Every comparison below is against this, not
	// against want.filed: a scenario the intact prompt already gets wrong says
	// nothing about the section under test.
	baseline := measure(t, m, scenarios)
	t.Logf("baseline: %s", summarize(scenarios, baseline))

	// The positive control, on the two scenarios that most cheaply show a
	// change: one the intact prompt declines, and one it files. Under "flag
	// everything" the first must flip. If it does not, stop -- nothing below
	// would mean anything.
	assertHarnessCanSeeAChange(t, m, scenarios, baseline)

	type result struct {
		mu      mutation
		flipped []string
		unknown int
	}
	var results []result

	for _, mu := range promptMutations() {
		orig := promptTemplate
		promptTemplate = mu.apply(promptTemplate)
		got := measure(t, m, scenarios)
		promptTemplate = orig

		var r result
		r.mu = mu
		for i, sc := range scenarios {
			switch {
			case got[i] == outcomeInconclusive || baseline[i] == outcomeInconclusive:
				r.unknown++
			case got[i] != baseline[i]:
				r.flipped = append(r.flipped, sc.name+": "+baseline[i].String()+" -> "+got[i].String())
			}
		}
		results = append(results, r)
		t.Logf("%s: %d flipped, %d inconclusive", mu.name, len(r.flipped), r.unknown)
	}

	// The report. Sorted so a section that matters most reads first.
	sort.SliceStable(results, func(i, j int) bool { return len(results[i].flipped) > len(results[j].flipped) })
	var b strings.Builder
	b.WriteString("\n=== which parts of the prompt are load-bearing ===\n")
	for _, r := range results {
		b.WriteString("\n" + r.mu.name + " -- " + r.mu.buys + "\n")
		if len(r.flipped) == 0 {
			b.WriteString("  INERT: no scenario changed its decision when this was deleted.\n")
			if r.unknown > 0 {
				b.WriteString("  (but " + strconv.Itoa(r.unknown) + " scenarios were inconclusive, so this is weaker than it looks)\n")
			}
			continue
		}
		b.WriteString("  LOAD-BEARING: " + strconv.Itoa(len(r.flipped)) + " scenarios changed decision:\n")
		for _, f := range r.flipped {
			b.WriteString("    " + f + "\n")
		}
	}
	t.Log(b.String())
}

// assertHarnessCanSeeAChange swaps in the positive control and requires a
// scenario the intact prompt declined to start filing.
//
// Uses one declining scenario rather than all 18: the control is a plumbing
// check, and one flip proves the plumbing exactly as well as eighteen do.
func assertHarnessCanSeeAChange(t *testing.T, m model.LLM, scenarios []scenario, baseline []outcome) {
	t.Helper()
	i := slices.IndexFunc(scenarios, func(sc scenario) bool { return !sc.want.filed })
	if i < 0 || baseline[i] != outcomeDeclined {
		t.Fatal("no declining scenario to use as a positive control")
	}

	orig := promptTemplate
	promptTemplate = positiveControl
	got := measure(t, m, scenarios[i:i+1])
	promptTemplate = orig

	switch got[0] {
	case outcomeFiled:
		t.Logf("positive control: %s flipped declined -> filed, so a prompt change is observable here",
			scenarios[i].name)
	case outcomeInconclusive:
		t.Fatalf("positive control was inconclusive, so no result in this run can be trusted; re-run it")
	default:
		t.Fatalf("positive control did NOT flip %s: a prompt ordering every file be recorded still "+
			"produced no issue. The mutated prompt is not reaching the model, so every \"inert\" "+
			"verdict this run would print is an artifact of the harness, not a finding.",
			scenarios[i].name)
	}
}

// measure runs every scenario once under the current prompt.
//
// One attempt each, unlike the assertion suite: this is a comparison between
// prompts, and spending three attempts per cell multiplies a already-expensive
// run by three to sharpen a number that inconclusive-counting already handles.
func measure(t *testing.T, m model.LLM, scenarios []scenario) []outcome {
	t.Helper()
	out := make([]outcome, len(scenarios))
	for i, sc := range scenarios {
		rec, err := attemptRun(t, m, sc)
		switch {
		case err != nil && transientModelError(err):
			// An inconclusive cell is a lost measurement exactly as a skipped
			// scenario is, and it silently weakens an "inert" verdict: a
			// section looks unmissed partly because nobody looked. Counted so
			// TestMain fails the run rather than letting the table stand.
			lostScenarios.Add(1)
			out[i] = outcomeInconclusive
		case err != nil:
			// A hard error under a mutated prompt is a real difference: the
			// model produced something the program refused outright.
			t.Logf("%s: run failed: %v", sc.name, err)
			out[i] = outcomeDeclined
		default:
			created, _, _, _ := rec.snapshot()
			if len(created) > 0 {
				out[i] = outcomeFiled
			} else {
				out[i] = outcomeDeclined
			}
		}
	}
	return out
}

func summarize(scenarios []scenario, got []outcome) string {
	var agree, unknown int
	var wrong []string
	for i, sc := range scenarios {
		switch {
		case got[i] == outcomeInconclusive:
			unknown++
		case (got[i] == outcomeFiled) == sc.want.filed:
			agree++
		default:
			wrong = append(wrong, sc.name)
		}
	}
	s := strconv.Itoa(agree) + "/" + strconv.Itoa(len(scenarios)) + " scenarios decided as specified"
	if unknown > 0 {
		s += ", " + strconv.Itoa(unknown) + " inconclusive"
	}
	if len(wrong) > 0 {
		s += ", disagreed on: " + strings.Join(wrong, ", ")
	}
	return s
}
