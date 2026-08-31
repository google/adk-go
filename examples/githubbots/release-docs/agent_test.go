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

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// stubModel is a model.LLM that records the request the agent built and replies
// with a scripted turn.
//
// It exists because everything between "the program builds an agent" and "a
// finding is recorded" was otherwise driven by no test: the tool-level tests all
// hand-build a scoped context and call recordFindings directly, so they cannot
// see whether ADK carries the context through functiontool at all, nor which
// tools the agent was actually constructed with.
type stubModel struct {
	// reply is called with the turn number (0-based) and returns the parts the
	// model emits. Returning nil ends the turn with no content.
	reply func(turn int) []*genai.Part
	// errorOnTurn, when set, makes that turn yield a model error instead of
	// content, so a test can drive a group that records and THEN fails.
	errorOnTurn func(turn int) bool

	mu sync.Mutex
	// toolNames is the tool inventory the agent offered, taken from the first
	// request.
	toolNames []string
	// declaredArgs are the top-level argument names of the first function
	// declaration the agent offered.
	declaredArgs []string
	// requiredFindingFields are the per-finding fields the schema marks required.
	requiredFindingFields []string
	// schemaRead records that the required-fields probe found a schema to read,
	// so an assertion over it cannot pass on an empty result.
	schemaRead bool
	turns      int
}

func (m *stubModel) Name() string { return "stub" }

func (m *stubModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	turn := m.turns
	m.turns++
	if turn == 0 {
		for name := range req.Tools {
			m.toolNames = append(m.toolNames, name)
		}
		slices.Sort(m.toolNames)
		if req.Config != nil {
			for _, t := range req.Config.Tools {
				for _, d := range t.FunctionDeclarations {
					m.declaredArgs = append(m.declaredArgs, schemaProperties(d)...)
					if req, ok := findingRequiredFields(d); ok {
						m.schemaRead = true
						m.requiredFindingFields = append(m.requiredFindingFields, req...)
					}
				}
			}
			slices.Sort(m.declaredArgs)
		}
	}
	m.mu.Unlock()

	if m.errorOnTurn != nil && m.errorOnTurn(turn) {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(&model.LLMResponse{ErrorCode: "TEST_ERROR", ErrorMessage: "forced"}, nil)
		}
	}
	parts := m.reply(turn)
	return func(yield func(*model.LLMResponse, error) bool) {
		if parts == nil {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
			return
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: string(genai.RoleModel), Parts: parts}}, nil)
	}
}

func (m *stubModel) inventory() ([]string, []string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.toolNames), slices.Clone(m.declaredArgs)
}

// schemaProperties returns the top-level argument names of a function
// declaration, from whichever of the two schema fields the SDK populated.
func schemaProperties(d *genai.FunctionDeclaration) []string {
	var out []string
	if d.Parameters != nil {
		for arg := range d.Parameters.Properties {
			out = append(out, arg)
		}
		return out
	}
	raw, err := json.Marshal(d.ParametersJsonSchema)
	if err != nil {
		return nil
	}
	var schema struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil
	}
	for arg := range schema.Properties {
		out = append(out, arg)
	}
	return out
}

// findingRequiredFields returns the fields the tool's schema marks required on a
// single finding, so a test can assert none of them are.
//
// It reports whether it could read the schema at all. Returning a bare nil would
// make the assertion pass vacuously if the SDK populated Parameters rather than
// ParametersJsonSchema, which is exactly how a test comes to certify a property
// it never looked at.
func findingRequiredFields(d *genai.FunctionDeclaration) (fields []string, ok bool) {
	if d.Parameters != nil {
		p, found := d.Parameters.Properties["findings"]
		if !found || p.Items == nil {
			return nil, false
		}
		return p.Items.Required, true
	}
	if d.ParametersJsonSchema == nil {
		return nil, false
	}
	raw, err := json.Marshal(d.ParametersJsonSchema)
	if err != nil {
		return nil, false
	}
	var schema struct {
		Properties struct {
			Findings struct {
				Items struct {
					Required []string `json:"required"`
				} `json:"items"`
			} `json:"findings"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil, false
	}
	return schema.Properties.Findings.Items.Required, true
}

// recordCall builds the function call the model would emit to record findings.
func recordCall(release string, group int, findings []any) *genai.Part {
	return &genai.Part{FunctionCall: &genai.FunctionCall{
		Name: "record_documentation_findings",
		Args: map[string]any{
			"release":     release,
			"group_index": group,
			"findings":    findings,
		},
	}}
}

// withStubModel installs a stub model for the duration of the test.
func withStubModel(t *testing.T, m *stubModel) {
	t.Helper()
	orig := newModel
	newModel = func(context.Context, *Config) (model.LLM, error) { return m, nil }
	t.Cleanup(func() { newModel = orig })
}

// testDiff is a one-file release diff.
func testDiff() *ReleaseDiff {
	return &ReleaseDiff{
		BaseTag: "v1.0.0", HeadTag: "v1.1.0", TotalFiles: 1,
		Files: []ChangedFile{{Path: "agent/agent.go", Status: "modified", Patch: "+// New exported API."}},
	}
}

// This is the test the whole authority model rests on, and nothing else drives
// it: invariant 3's enforcement (authorizeGroup) and the entire happy path both
// depend on ADK carrying the context passed to runner.Run through to the
// agent.Context the tool closure receives. Every other tool test hand-builds a
// scoped context, so if ADK stopped propagating, authorizeGroup would refuse
// every call, the bot would file nothing forever, and the suite would stay
// green.
//
// Mutation that must fail this test: delete `gctx = withAuditedGroup(...)` in
// analyzeGroup, or make authorizeGroup return false unconditionally.
func TestAgentRecordsThroughTheRealRunner(t *testing.T) {
	cfg := testConfig()
	m := &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{
			map[string]any{
				"kind":            "new-feature",
				"doc_file":        "docs/agents.md",
				"summary":         "a new exported API",
				"proposed_change": "document it",
			},
		})}
	}}
	withStubModel(t, m)

	rec := newRecorder(cfg)
	gh := testClient(t, cfg, failIfCalled(t))
	r, sessions, err := newAgentRunner(context.Background(), cfg, rec, discardLogger())
	if err != nil {
		t.Fatalf("newAgentRunner: %v", err)
	}

	diff := testDiff()
	ok := analyzeGroup(context.Background(), cfg, "v1.0.0...v1.1.0", 0, func(gctx context.Context, i int) bool {
		return runGroup(gctx, r, sessions, gh, discardLogger(), diff, diff.Files, i, 1, "v1.0.0...v1.1.0")
	})
	if !ok {
		t.Fatal("runGroup reported failure")
	}

	got := rec.findings()
	if len(got) != 1 {
		t.Fatalf("recorded %d findings, want 1: the tool call did not reach the recorder "+
			"(if authorizeGroup refused it, ADK is not propagating the session scope)", len(got))
	}
	if got[0].DocFile != "docs/agents.md" || got[0].Kind != "new-feature" {
		t.Errorf("recorded finding = %+v, want the model's values after sanitization", got[0])
	}
	if n := rec.unreported(1); n != 0 {
		t.Errorf("unreported(1) = %d after a successful record, want 0", n)
	}
}

// The tool inventory pinned at the AGENT, not one call frame earlier. A tool
// added at the llmagent.Config site rather than in recorder.tools() would leave
// TestToolInventoryIsPinned green while the model gained an ungated capability.
// This asserts what the model was actually offered.
//
// Mutation that must fail this test: add a second tool to the Tools field in
// newAgentRunner.
func TestAgentOffersExactlyTheOneTool(t *testing.T) {
	cfg := testConfig()
	m := &stubModel{reply: func(int) []*genai.Part { return nil }}
	withStubModel(t, m)

	gh := testClient(t, cfg, failIfCalled(t))
	r, sessions, err := newAgentRunner(context.Background(), cfg, newRecorder(cfg), discardLogger())
	if err != nil {
		t.Fatalf("newAgentRunner: %v", err)
	}
	diff := testDiff()
	analyzeGroup(context.Background(), cfg, "v1.0.0...v1.1.0", 0, func(gctx context.Context, i int) bool {
		return runGroup(gctx, r, sessions, gh, discardLogger(), diff, diff.Files, i, 1, "v1.0.0...v1.1.0")
	})

	names, args := m.inventory()
	want := []string{"record_documentation_findings"}
	if !slices.Equal(names, want) {
		t.Errorf("the agent offered the model %v, want exactly %v", names, want)
	}
	// The schema the model fills. If the reflector stopped exposing these, the
	// model would fill nothing, every finding would be dropped as empty, and the
	// bot would report "no suggestions" on every release with a green suite.
	wantArgs := []string{"findings", "group_index", "release"}
	if !slices.Equal(args, wantArgs) {
		t.Errorf("the tool's argument schema = %v, want %v", args, wantArgs)
	}

	// No per-finding field may be required. The reflector marks every struct
	// field required unless the json tag says omitempty, and a required field the
	// model omits makes ADK reject the tool call outright -- losing the whole
	// group's findings, silently, on a run that otherwise looks healthy. This was
	// live until the omitempty tags were added.
	//
	// Mutation that must fail this test: drop ",omitempty" from any Finding tag.
	m.mu.Lock()
	required, read := slices.Clone(m.requiredFindingFields), m.schemaRead
	m.mu.Unlock()
	if !read {
		t.Fatal("could not read the finding schema from either declaration field; " +
			"the assertion below would pass without looking at anything")
	}
	if len(required) != 0 {
		t.Errorf("the finding schema marks %v as required; a model omitting any of them "+
			"has its whole tool call rejected", required)
	}
}

// The same defect, driven end to end: a model that fills only the fields it has
// must still be recorded.
//
// Mutation that must fail this test: drop ",omitempty" from the Finding json tags.
func TestAgentRecordsAFindingWithOptionalFieldsOmitted(t *testing.T) {
	cfg := testConfig()
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		// Only two of the six fields.
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{
			map[string]any{"kind": "behavior-change", "summary": "the default changed"},
		})}
	}})

	rec := newRecorder(cfg)
	gh := testClient(t, cfg, failIfCalled(t))
	r, sessions, err := newAgentRunner(context.Background(), cfg, rec, discardLogger())
	if err != nil {
		t.Fatalf("newAgentRunner: %v", err)
	}
	diff := testDiff()
	analyzeGroup(context.Background(), cfg, "v1.0.0...v1.1.0", 0, func(gctx context.Context, i int) bool {
		return runGroup(gctx, r, sessions, gh, discardLogger(), diff, diff.Files, i, 1, "v1.0.0...v1.1.0")
	})

	got := rec.findings()
	if len(got) != 1 {
		t.Fatalf("recorded %d findings, want 1: a partly filled finding was rejected by the schema", len(got))
	}
	if got[0].Kind != "behavior-change" || got[0].Summary != "the default changed" {
		t.Errorf("recorded finding = %+v", got[0])
	}
}

// A model steered by contributor text into naming another group must be refused
// end-to-end, not merely by a directly-called Go function.
func TestAgentCannotRecordForAnotherGroup(t *testing.T) {
	cfg := testConfig()
	m := &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 7, []any{
			map[string]any{"kind": "new-feature", "summary": "cross-group"},
		})}
	}}
	withStubModel(t, m)

	rec := newRecorder(cfg)
	gh := testClient(t, cfg, failIfCalled(t))
	r, sessions, err := newAgentRunner(context.Background(), cfg, rec, discardLogger())
	if err != nil {
		t.Fatalf("newAgentRunner: %v", err)
	}
	diff := testDiff()
	analyzeGroup(context.Background(), cfg, "v1.0.0...v1.1.0", 0, func(gctx context.Context, i int) bool {
		return runGroup(gctx, r, sessions, gh, discardLogger(), diff, diff.Files, i, 1, "v1.0.0...v1.1.0")
	})

	if got := rec.findings(); len(got) != 0 {
		t.Errorf("a call naming group 7 from a session scoped to group 0 recorded %d findings, want 0", len(got))
	}
	if n := rec.unreported(1); n != 1 {
		t.Errorf("unreported(1) = %d, want 1: a refused call must leave the group unreported", n)
	}
}

// A nonce draw failure must abort the group before any model call, not fall back
// to a guessable marker: a predictable fence lets a contributor pre-write the
// closing marker in a code comment and escape it.
//
// The previous version of this test asserted that its own stub returned an
// error and that renderGroupPrompt("") produces "[UNTRUSTED:]" — both values the
// test itself supplied. runGroup was never called, so deleting its fail-closed
// return left the whole suite green.
//
// Mutation that must fail this test: delete the `return false` after the
// newNonce error check in runGroup.
func TestRunGroupAbortsWhenTheNonceCannotBeDrawn(t *testing.T) {
	cfg := testConfig()
	m := &stubModel{reply: func(int) []*genai.Part {
		t.Error("the model was invoked despite the fence being unbuildable")
		return nil
	}}
	withStubModel(t, m)

	orig := newNonce
	newNonce = func() (string, error) { return "", errors.New("no entropy") }
	t.Cleanup(func() { newNonce = orig })

	rec := newRecorder(cfg)
	gh := testClient(t, cfg, failIfCalled(t))
	r, sessions, err := newAgentRunner(context.Background(), cfg, rec, discardLogger())
	if err != nil {
		t.Fatalf("newAgentRunner: %v", err)
	}
	diff := testDiff()
	ok := analyzeGroup(context.Background(), cfg, "v1.0.0...v1.1.0", 0, func(gctx context.Context, i int) bool {
		return runGroup(gctx, r, sessions, gh, discardLogger(), diff, diff.Files, i, 1, "v1.0.0...v1.1.0")
	})

	if ok {
		t.Error("runGroup reported success on an unbuildable fence")
	}
	if !gh.hadError() {
		t.Error("the failure was not recorded, so the run would exit 0")
	}
	if turns := m.turns; turns != 0 {
		t.Errorf("the model was called %d times, want 0", turns)
	}
	if len(rec.findings()) != 0 {
		t.Error("findings were recorded despite the abort")
	}
}

// The end-to-end filing path, which no test reached: everything that produces
// the issue GitHub actually receives.
//
// It pins the coupling that duplicate detection rests on — the title the create
// call sends must be the title findBySearch queries for, and the body's first
// line must be the marker findByList matches. Mutating issueTitle(head) to
// issueTitle(base) at the create site left the whole suite green while silently
// breaking the search probe.
//
// Mutation that must fail this test: change `issueTitle(head)` to
// `issueTitle(base)` in runWith, or drop the marker line from buildIssueBody.
func TestRunWithFilesAnIssueTheDuplicateProbesCanFind(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.RunBudget, cfg.GroupTimeout = time.Minute, time.Minute
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{
			map[string]any{"kind": "new-feature", "summary": "a new exported API", "doc_file": "docs/agents.md"},
		})}
	}})

	h := &filingHandler{}
	gh := testClient(t, cfg, h)
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith: %v", err)
	}

	if h.creates != 1 {
		t.Fatalf("created %d issues, want 1", h.creates)
	}
	// The title the create sends must be the one the search probe looks for.
	if want := issueTitle("v1.1.0"); h.title != want {
		t.Errorf("filed title = %q, want %q (findBySearch queries this exact string)", h.title, want)
	}
	if h.searchQuery != "" && !strings.Contains(h.searchQuery, h.title) {
		t.Errorf("the search probe queried %q, which does not contain the title it would file (%q)", h.searchQuery, h.title)
	}
	// The body's first line must be the marker the list probe matches.
	if want := bodyMarker("v1.0.0", "v1.1.0"); !hasBodyMarker(h.body, want) {
		t.Errorf("the filed body does not start with %q:\n%s", want, firstLine(h.body))
	}
	if !strings.Contains(h.body, "a new exported API") {
		t.Error("the filed body does not carry the model's finding")
	}
	if strings.Contains(h.body, "The analysis is partial") {
		t.Error("a complete analysis was filed as partial")
	}
}

// Dry run over the WHOLE program, not just the one mutation helper: the
// chokepoint invariant is "no write escapes dry-run", and only driving every
// step proves it. A second mutating call added anywhere would show up here.
//
// Mutation that must fail this test: remove the shouldSkip guard from
// FileReleaseIssue.
func TestRunWithUnderDryRunMakesNoWriteRequest(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.RunBudget, cfg.GroupTimeout = time.Minute, time.Minute
	cfg.DryRun = true
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{
			map[string]any{"kind": "new-feature", "summary": "a new exported API"},
		})}
	}})

	h := &filingHandler{}
	gh := testClient(t, cfg, h)
	var rendered strings.Builder
	gh.out = &rendered
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if h.writes != 0 {
		t.Errorf("dry run made %d write requests, want 0", h.writes)
	}
	if !strings.Contains(rendered.String(), "a new exported API") {
		t.Error("dry run did not render the issue it would have filed")
	}
}
