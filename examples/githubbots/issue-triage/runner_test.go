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
	"strings"
	"sync"
	"testing"

	"github.com/google/go-github/v66/github"
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
	// patched records issue number -> type for every PATCH that arrived.
	patched map[int]string
	// mutationStatus, when non-zero, is returned for every mutating request.
	mutationStatus int
	// untriaged is the issue number the search returns.
	untriaged int
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
					"nodes":[{"number":` + itoa(g.untriaged) + `,"title":"crash on nil","body":"boom",
					"issueType":null,"labels":{"nodes":[{"name":"bug"}]}}]}}}`))
				return
			}
			num, _ := req.Variables["number"].(float64)
			_, _ = w.Write([]byte(`{"data":{"repository":{"issue":{"number":` + itoa(int(num)) +
				`,"title":"crash on nil","body":"boom","issueType":null,"labels":{"nodes":[{"name":"bug"}]}}}}}`))
			return
		}

		if g.mutationStatus != 0 {
			w.WriteHeader(g.mutationStatus)
			_, _ = w.Write([]byte(`{"message":"boom"}`))
			return
		}
		var body struct {
			Type string `json:"type"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		g.mu.Lock()
		g.patched[issueNumberFromPath(r.URL.Path)] = body.Type
		g.mu.Unlock()
		_, _ = w.Write([]byte(`{"number":1,"type":{"name":"` + body.Type + `"}}`))
	})
}

func (g *githubStub) writes() map[int]string {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make(map[int]string, len(g.patched))
	for k, v := range g.patched {
		out[k] = v
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for ; n > 0; n /= 10 {
		d = append([]byte{byte('0' + n%10)}, d...)
	}
	return string(d)
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

// driveRun runs the REAL run function against a scripted model and a local
// GitHub, and returns what was written together with run's error.
func driveRun(t *testing.T, stub *githubStub, turns ...*genai.Content) (map[int]string, error) {
	t.Helper()
	setRequired(t)
	t.Setenv("ISSUE_COUNT", "1")
	t.Setenv("ISSUE_TIMEOUT", "30s")
	t.Setenv("SWEEP_TIMEOUT", "30s")

	srv := httptest.NewServer(stub.handler(t))
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse base url: %v", err)
	}

	origClient, origModel := newClientFn, newModelFn
	newClientFn = func(cfg *Config, log *slog.Logger) *Client {
		c := NewClient(cfg, log)
		c.rest.BaseURL = base
		return c
	}
	newModelFn = func(context.Context, *Config) (model.LLM, error) {
		return &scriptedModel{turns: turns}, nil
	}
	t.Cleanup(func() { newClientFn, newModelFn = origClient, origModel })

	err = run(context.Background(), discardLogger(), nil)
	return stub.writes(), err
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
	stub := &githubStub{patched: map[int]string{}, untriaged: 42}
	writes, err := driveRun(t, stub,
		toolCall("change_issue_type", map[string]any{"issue_number": 42, "issue_type": "Bug"}),
		finalText("triaged #42 as a Bug"),
	)
	if err != nil {
		t.Fatalf("run() = %v, want nil", err)
	}
	if got := writes[42]; got != "Bug" {
		t.Errorf("issue #42 type = %q, want %q (writes: %v)", got, "Bug", writes)
	}
}

// The prompt-injection case, end to end. The model is entirely
// attacker-controlled by assumption, so here it asks to act on an issue the
// session was never scoped to, and no write may reach the repository.
//
// This pins the two gates TOGETHER rather than isolating either: in a real run
// only the current issue is ever in the authorization map, so the session scope
// and the need-claim refuse the same call and no single mutation separates
// them. Isolating the scope gate is what TestMutatingToolsRefuseOutOfScopeIssue
// does. What is unique here is that the refusal is observed through the real
// framework, against what actually reached GitHub.
func TestRunRefusesAnIssueTheSessionWasNotScopedTo(t *testing.T) {
	stub := &githubStub{patched: map[int]string{}, untriaged: 42}
	writes, err := driveRun(t, stub,
		// The work set holds only #42. The model asks for #99.
		toolCall("change_issue_type", map[string]any{"issue_number": 99, "issue_type": "Bug"}),
		finalText("done"),
	)
	if err != nil {
		t.Fatalf("run() = %v, want nil: a refused tool call is data, not an infrastructure failure", err)
	}
	if len(writes) != 0 {
		t.Errorf("the bot wrote %v; a session scoped to #42 must not touch another issue", writes)
	}
}

// A mutation the framework hands back to the model as data would otherwise let
// the process exit 0 with nothing done. run must join it and fail.
//
// Killing mutation: delete the hadToolError join from run.
func TestRunFailsWhenAToolCallFailed(t *testing.T) {
	stub := &githubStub{patched: map[int]string{}, untriaged: 42, mutationStatus: http.StatusInternalServerError}
	_, err := driveRun(t, stub,
		toolCall("change_issue_type", map[string]any{"issue_number": 42, "issue_type": "Bug"}),
		finalText("done"),
	)
	if err == nil {
		t.Fatal("run() = nil, want the failed tool call surfaced so CI goes red")
	}
	if !strings.Contains(err.Error(), "tool calls failed") {
		t.Errorf("run() = %v, want it to name the failed tool call", err)
	}
}

var _ = github.Client{}
