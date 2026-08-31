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
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// scriptedModel replays a fixed sequence of responses, one per turn, ignoring
// the request. It is enough to drive the real runner, which executes any
// FunctionCall it is handed -- which is the point: these tests exercise the
// framework path the bot actually runs on, not a stand-in for it.
type scriptedModel struct {
	mu    sync.Mutex
	turns []*genai.Content
	seen  int
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	turn := m.seen
	m.seen++
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		if turn >= len(m.turns) {
			return
		}
		yield(&model.LLMResponse{Content: m.turns[turn]}, nil)
	}
}

func toolCall(name string, args map[string]any) *genai.Content {
	return &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: name, Args: args}}},
	}
}

func finalText(s string) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: s}}}
}

// githubStub answers the three requests one triage run makes: the search that
// picks the work set, the by-number read the write revalidates against, and the
// mutation itself. It records every mutation so a test can assert what the bot
// actually did to the repository.
type githubStub struct {
	mu sync.Mutex
	// patched records issue number -> type for every PATCH that arrived, and
	// labelled records issue number -> label for every label POST.
	patched  map[int]string
	labelled map[int]string
	// mutationStatus, when non-zero, is returned for every mutating request.
	mutationStatus int
	// untriaged is the issue number the search returns. notFound makes the
	// by-number read report the issue as gone.
	untriaged int
	notFound  bool
	// reads counts the by-number revalidation reads.
	reads int
}

func newStub(untriaged int) *githubStub {
	return &githubStub{patched: map[int]string{}, labelled: map[int]string{}, untriaged: untriaged}
}

func (g *githubStub) handler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/graphql" {
			var req struct {
				Variables map[string]any `json:"variables"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode graphql request: %v", err)
			}
			// The search carries "q"; the by-number read carries "number".
			if _, isSearch := req.Variables["q"]; isSearch {
				_, _ = w.Write([]byte(`{"data":{"search":{"pageInfo":{"hasNextPage":false,"endCursor":""},
					"nodes":[{"number":` + strconv.Itoa(g.untriaged) + `,"title":"crash on nil","body":"boom",
					"issueType":null,"labels":{"nodes":[]}}]}}}`))
				return
			}
			g.mu.Lock()
			g.reads++
			g.mu.Unlock()
			if g.notFound {
				_, _ = w.Write([]byte(`{"data":{"repository":{"issue":null}},
					"errors":[{"type":"NOT_FOUND","message":"Could not resolve to an Issue"}]}`))
				return
			}
			num, _ := req.Variables["number"].(float64)
			_, _ = w.Write([]byte(`{"data":{"repository":{"issue":{"number":` + strconv.Itoa(int(num)) +
				`,"title":"crash on nil","body":"boom","issueType":null,"labels":{"nodes":[]}}}}}`))
			return
		}

		if g.mutationStatus != 0 {
			w.WriteHeader(g.mutationStatus)
			_, _ = w.Write([]byte(`{"message":"boom"}`))
			return
		}
		number := issueNumberFromPath(r.URL.Path)
		if strings.HasSuffix(r.URL.Path, "/labels") {
			// go-github posts the label names as a bare JSON array, not an
			// object with a "labels" key.
			var names []string
			_ = json.NewDecoder(r.Body).Decode(&names)
			label := ""
			if len(names) > 0 {
				label = names[0]
			}
			g.mu.Lock()
			g.labelled[number] = label
			g.mu.Unlock()
			_, _ = w.Write([]byte(`[{"name":"` + label + `"}]`))
			return
		}
		var body struct {
			Type string `json:"type"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		g.mu.Lock()
		g.patched[number] = body.Type
		g.mu.Unlock()
		_, _ = w.Write([]byte(`{"number":` + strconv.Itoa(number) + `,"type":{"name":"` + body.Type + `"}}`))
	})
}

// writes returns every mutation the run performed, type and label alike, so a
// test asserting an ABSENCE cannot miss one because it looked at the wrong map.
func (g *githubStub) writes() map[int]string {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make(map[int]string, len(g.patched)+len(g.labelled))
	for k, v := range g.patched {
		out[k] = v
	}
	for k, v := range g.labelled {
		out[k] = v
	}
	return out
}

// issueNumberFromPath pulls N out of /repos/owner/repo/issues/N[/labels].
func issueNumberFromPath(p string) int {
	parts := strings.Split(strings.Trim(p, "/"), "/")
	for i, seg := range parts {
		if seg != "issues" || i+1 >= len(parts) {
			continue
		}
		n := 0
		for _, c := range parts[i+1] {
			if c < '0' || c > '9' {
				return 0
			}
			n = n*10 + int(c-'0')
		}
		return n
	}
	return 0
}

// runSpec configures one end-to-end drive of the real run function.
type runSpec struct {
	stub *githubStub
	// turns is the model's script, one response per turn.
	turns []*genai.Content
	// args is passed to run as the command line, so the -issue path is
	// reachable and not only the sweep.
	args []string
	// dryRun sets DRY_RUN for this run.
	dryRun bool
	// preauthorize is authorized on the client before the run starts, so a test
	// can isolate the session-scope gate from the need-claim gate: without it
	// both refuse an out-of-scope issue and no single mutation separates them.
	preauthorize int
}

// driveRun runs the REAL run function against a scripted model and a local
// GitHub. It returns what was written, how many model turns were consumed, and
// run's error. The turn count matters: a test asserting that nothing was
// written must be able to tell "the bot refused" from "the bot never ran".
func driveRun(t *testing.T, spec runSpec) (writes map[int]string, turnsUsed int, err error) {
	t.Helper()
	stub := spec.stub
	setRequired(t)
	t.Setenv("ISSUE_COUNT", "1")
	t.Setenv("ISSUE_TIMEOUT", "30s")
	t.Setenv("SWEEP_TIMEOUT", "30s")
	if spec.dryRun {
		t.Setenv("DRY_RUN", "true")
	}

	srv := httptest.NewServer(stub.handler(t))
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse base url: %v", err)
	}

	scripted := &scriptedModel{turns: spec.turns}
	origClient, origModel := newClientFn, newModelFn
	newClientFn = func(cfg *Config, log *slog.Logger) *Client {
		c := NewClient(cfg, log)
		c.rest.BaseURL = base
		if spec.preauthorize > 0 {
			c.authorize(spec.preauthorize, need{typ: true, label: true})
		}
		return c
	}
	newModelFn = func(context.Context, *Config) (model.LLM, error) { return scripted, nil }
	t.Cleanup(func() { newClientFn, newModelFn = origClient, origModel })

	err = run(context.Background(), discardLogger(), spec.args)

	scripted.mu.Lock()
	turnsUsed = scripted.seen
	scripted.mu.Unlock()
	return stub.writes(), turnsUsed, err
}

// The happy path, driven end to end through the real ADK runner.
//
// Nothing else in the suite executes this chain. Every other test calls
// doChangeType directly with a context it built, which leaves the load-bearing
// assumption unchecked: that a value put on the context handed to runner.Run is
// visible to a tool through agent.Context. Reading the framework source says it
// is. This runs it.
//
// Killing mutation: `_ = withAuditedIssue(ictx, iss.Number)` in triageOne, so
// the scope is computed but never reaches Run. Every direct-call test still
// passes, because each builds its own scoped context.
func TestRunSetsTheTypeThroughTheRealRunner(t *testing.T) {
	stub := newStub(42)
	writes, turns, err := driveRun(t, runSpec{stub: stub, turns: []*genai.Content{
		toolCall("change_issue_type", map[string]any{"issue_number": 42, "issue_type": "Bug"}),
		finalText("triaged #42 as a Bug"),
	}})
	if err != nil {
		t.Fatalf("run() = %v, want nil", err)
	}
	if turns != 2 {
		t.Errorf("the model was asked for %d turns, want 2: the session did not run to completion", turns)
	}
	if got := writes[42]; got != "Bug" {
		t.Errorf("issue #42 type = %q, want %q (writes: %v)", got, "Bug", writes)
	}
}

// The label tool, on the same path. doAddLabel duplicates doChangeType's
// peek/re-read/claim block, which is exactly where the two would drift.
func TestRunAddsTheLabelThroughTheRealRunner(t *testing.T) {
	stub := newStub(42)
	writes, turns, err := driveRun(t, runSpec{stub: stub, turns: []*genai.Content{
		toolCall("add_label_to_issue", map[string]any{"issue_number": 42, "label": "bug"}),
		finalText("labelled #42"),
	}})
	if err != nil {
		t.Fatalf("run() = %v, want nil", err)
	}
	if turns != 2 {
		t.Errorf("the model was asked for %d turns, want 2", turns)
	}
	if got := writes[42]; got != "bug" {
		t.Errorf("issue #42 label = %q, want %q (writes: %v)", got, "bug", writes)
	}
}

// The prompt-injection case, end to end and ISOLATED to the session scope.
//
// #99 is authorized on the client before the run, so the need-claim gate would
// let this through. Only the per-session context scope stands in the way, and
// the run is scoped to #42. Without the preauthorization all three gates refuse
// the call and no single mutation separates them, which is what made the first
// version of this test unkillable.
//
// Killing mutation: `authorizeIssue` returning ("", true) unconditionally.
func TestRunRefusesAnIssueOutsideTheSessionScope(t *testing.T) {
	stub := newStub(42)
	writes, turns, err := driveRun(t, runSpec{
		stub:         stub,
		preauthorize: 99,
		turns: []*genai.Content{
			toolCall("change_issue_type", map[string]any{"issue_number": 99, "issue_type": "Bug"}),
			finalText("done"),
		},
	})
	if err != nil {
		t.Fatalf("run() = %v, want nil: a refused tool call is data, not an infrastructure failure", err)
	}
	// Without this the test also passes when the model was never consulted, the
	// search returned nothing, or the tool never ran -- an absence assertion
	// alone cannot tell a refusal from a session that did not happen.
	if turns != 2 {
		t.Fatalf("the model was asked for %d turns, want 2: the session did not run, so the refusal proves nothing", turns)
	}
	if len(writes) != 0 {
		t.Errorf("the bot wrote %v; a session scoped to #42 must not touch #99 even when #99 is authorized", writes)
	}
}

// A mutation the framework hands back to the model as data would otherwise let
// the process exit 0 with nothing done. run must join it and fail.
//
// Killing mutation: delete the hadToolError join from run.
func TestRunFailsWhenAToolCallFailed(t *testing.T) {
	stub := newStub(42)
	stub.mutationStatus = http.StatusInternalServerError
	_, turns, err := driveRun(t, runSpec{stub: stub, turns: []*genai.Content{
		toolCall("change_issue_type", map[string]any{"issue_number": 42, "issue_type": "Bug"}),
		finalText("done"),
	}})
	if turns != 2 {
		t.Fatalf("the model was asked for %d turns, want 2", turns)
	}
	if err == nil {
		t.Fatal("run() = nil, want the failed tool call surfaced so CI goes red")
	}
	if !strings.Contains(err.Error(), "tool calls failed") {
		t.Errorf("run() = %v, want it to name the failed tool call", err)
	}
}

// The -issue path, which is what every `issues: [opened]` run takes. The sweep
// path is what every other end-to-end test exercises, so a defect confined to
// this branch would reach production unseen.
//
// Killing mutation: `cfg.SingleIssue >= 0` in selectIssues, which turns an
// -issue run into a fetch of issue #0.
func TestRunTriagesASingleRequestedIssue(t *testing.T) {
	stub := newStub(0) // no search result: the run must not sweep
	writes, turns, err := driveRun(t, runSpec{
		stub: stub,
		args: []string{"-issue", "42"},
		turns: []*genai.Content{
			toolCall("change_issue_type", map[string]any{"issue_number": 42, "issue_type": "Bug"}),
			finalText("done"),
		},
	})
	if err != nil {
		t.Fatalf("run(-issue 42) = %v, want nil", err)
	}
	if turns != 2 {
		t.Fatalf("the model was asked for %d turns, want 2", turns)
	}
	if got := writes[42]; got != "Bug" {
		t.Errorf("issue #42 type = %q, want %q (writes: %v)", got, "Bug", writes)
	}
}

// A requested issue that does not exist, or is a pull request, is logged and
// skipped. It is not an infrastructure failure, so the run must stay green --
// otherwise every `issues: opened` event for a converted issue reds the job.
//
// Killing mutation: `return nil, err` instead of `return nil, nil` on the
// ErrIssueNotFound branch of selectIssues.
func TestRunSkipsAnIssueThatDoesNotExist(t *testing.T) {
	stub := newStub(0)
	stub.notFound = true
	writes, turns, err := driveRun(t, runSpec{
		stub:  stub,
		args:  []string{"-issue", "42"},
		turns: []*genai.Content{finalText("unused")},
	})
	if err != nil {
		t.Fatalf("run(-issue 42) on a missing issue = %v, want nil", err)
	}
	if turns != 0 {
		t.Errorf("the model was consulted %d times, want 0: there was nothing to triage", turns)
	}
	if len(writes) != 0 {
		t.Errorf("the bot wrote %v for an issue that does not exist", writes)
	}
}

// Dry-run must suppress every mutation at the run level, not only when SetType
// and AddLabel are called directly. This is the end-to-end guard on the
// chokepoint: a future mutating method that forgot shouldSkip fails here.
//
// Killing mutation: remove the shouldSkip call from SetType.
func TestRunInDryRunWritesNothing(t *testing.T) {
	stub := newStub(42)
	writes, turns, err := driveRun(t, runSpec{
		stub:   stub,
		dryRun: true,
		turns: []*genai.Content{
			toolCall("change_issue_type", map[string]any{"issue_number": 42, "issue_type": "Bug"}),
			toolCall("add_label_to_issue", map[string]any{"issue_number": 42, "label": "bug"}),
			finalText("would have triaged #42"),
		},
	})
	if err != nil {
		t.Fatalf("run() in dry-run = %v, want nil", err)
	}
	// The work must actually have been attempted, or "nothing was written" is
	// true for the wrong reason.
	if turns < 2 {
		t.Fatalf("the model was asked for %d turns, want at least 2: no tool call was attempted", turns)
	}
	if stub.reads == 0 {
		t.Error("no revalidation read happened, so no tool reached the write it should have suppressed")
	}
	if len(writes) != 0 {
		t.Errorf("dry-run wrote %v, want nothing", writes)
	}
}
