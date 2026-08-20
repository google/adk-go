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

package adkrest_test

import (
	"context"
	"fmt"
	"iter"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

const (
	compactionApp  = "compaction_app"
	compactionUser = "u"
)

// echoModel answers every request with a canned reply and records the prompts
// it was given, so a test can inspect the history the server assembled.
type echoModel struct {
	mu      sync.Mutex
	prompts [][]*genai.Content
}

func (m *echoModel) Name() string { return "echo" }

func (m *echoModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.prompts = append(m.prompts, req.Contents)
	n := len(m.prompts)
	m.mu.Unlock()

	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(fmt.Sprintf("answer %d", n), "model")}, nil)
	}
}

func (m *echoModel) lastPrompt() []*genai.Content {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.prompts) == 0 {
		return nil
	}
	return m.prompts[len(m.prompts)-1]
}

// stubSummarizer returns a fixed summary, so the test does not depend on a real
// model's wording.
type stubSummarizer struct{ text string }

func (s stubSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	return genai.NewContentFromText(s.text, "model"), nil, nil
}

// TestRESTCompaction_EnabledViaServerConfig is the guard that context
// compaction is actually reachable from the REST server, not just from a direct
// runner.New. It exercises the whole chain: ServerConfig.EventsCompactionConfig
// → NewRuntimeAPIController option → runner.Config → compaction.
func TestRESTCompaction_EnabledViaServerConfig(t *testing.T) {
	m := &echoModel{}
	sessionService := session.InMemoryService()
	srv := httptest.NewServer(newCompactionServer(t, m, sessionService, &compaction.Config{
		CompactionInterval: 2,
		Summarizer:         stubSummarizer{text: "SUMMARY-OF-EARLIER-TURNS"},
	}))
	defer srv.Close()

	sid := createCompactionSession(t, srv.URL)
	runCompactionTurn(t, srv.URL, sid, "q1")
	runCompactionTurn(t, srv.URL, sid, "q2")

	events := sessionEvents(t, sessionService, sid)
	if got := countCompactions(events); got != 1 {
		t.Fatalf("session holds %d compaction events after 2 turns, want 1; compaction is not reaching the REST runner", got)
	}

	// A third turn must be prompted with the summary rather than the raw turns.
	runCompactionTurn(t, srv.URL, sid, "q3")
	prompt := compactionPromptText(m.lastPrompt())
	if !strings.Contains(prompt, "SUMMARY-OF-EARLIER-TURNS") {
		t.Errorf("prompt does not contain the summary:\n%s", prompt)
	}
	for _, gone := range []string{"q1", "q2"} {
		if strings.Contains(prompt, gone) {
			t.Errorf("prompt still contains compacted turn %q:\n%s", gone, prompt)
		}
	}
}

// TestRESTCompaction_DisabledByDefault pins that leaving the field unset keeps
// the previous behaviour exactly.
func TestRESTCompaction_DisabledByDefault(t *testing.T) {
	m := &echoModel{}
	sessionService := session.InMemoryService()
	srv := httptest.NewServer(newCompactionServer(t, m, sessionService, nil))
	defer srv.Close()

	sid := createCompactionSession(t, srv.URL)
	for i := range 4 {
		runCompactionTurn(t, srv.URL, sid, fmt.Sprintf("q%d", i))
	}

	if got := countCompactions(sessionEvents(t, sessionService, sid)); got != 0 {
		t.Errorf("session holds %d compaction events with no config set, want 0", got)
	}
}

func newCompactionServer(t *testing.T, m model.LLM, sessionService session.Service, cfg *compaction.Config) *adkrest.Server {
	t.Helper()
	root, err := llmagent.New(llmagent.Config{
		Name:        compactionApp,
		Model:       m,
		Instruction: "You are a helpful assistant.",
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:         sessionService,
		AgentLoader:            agent.NewSingleLoader(root),
		EventsCompactionConfig: cfg,
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() error = %v", err)
	}
	return srv
}

func createCompactionSession(t *testing.T, baseURL string) string {
	t.Helper()
	var resp struct {
		ID string `json:"id"`
	}
	postJSON(t, fmt.Sprintf("%s/apps/%s/users/%s/sessions", baseURL, compactionApp, compactionUser),
		map[string]any{}, &resp)
	if resp.ID == "" {
		t.Fatal("create session returned an empty ID")
	}
	return resp.ID
}

func runCompactionTurn(t *testing.T, baseURL, sid, text string) {
	t.Helper()
	var events []restEvent
	postJSON(t, baseURL+"/run", map[string]any{
		"appName":    compactionApp,
		"userId":     compactionUser,
		"sessionId":  sid,
		"newMessage": genai.NewContentFromText(text, genai.RoleUser),
	}, &events)
}

func sessionEvents(t *testing.T, svc session.Service, sid string) []*session.Event {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{
		AppName: compactionApp, UserID: compactionUser, SessionID: sid,
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

func countCompactions(events []*session.Event) int {
	n := 0
	for _, ev := range events {
		if compactioninternal.HasUsableSummary(ev) {
			n++
		}
	}
	return n
}

func compactionPromptText(contents []*genai.Content) string {
	var b strings.Builder
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.Text != "" {
				fmt.Fprintf(&b, "[%s] %s\n", c.Role, p.Text)
			}
		}
	}
	return b.String()
}

// TestNewServerRejectsInvalidCompactionConfig checks that an unusable
// compaction config stops the server starting.
//
// runner.New validates the config, and the server builds a runner per request,
// so without a check at construction an invalid config produces a server that
// starts cleanly and then fails every request with a 500. The operator sees a
// broken deployment rather than a refused start naming the field.
func TestNewServerRejectsInvalidCompactionConfig(t *testing.T) {
	t.Parallel()

	m := &echoModel{}
	root, err := llmagent.New(llmagent.Config{Name: "assistant", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	_, err = adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(root),
		// Overlap without an interval: sliding-window compaction can never run.
		EventsCompactionConfig: &compaction.Config{OverlapSize: 2},
	})
	if err == nil {
		t.Fatal("NewServer() accepted an invalid EventsCompactionConfig, want it refused at startup")
	}
	if !strings.Contains(err.Error(), "EventsCompactionConfig") {
		t.Errorf("error %q does not name the offending field", err)
	}
}

// TestRESTCompaction_StartupRejectsAConfigItCannotServe pins that a compaction
// config the server cannot actually run is refused at construction.
//
// NewServer's own comment says it validates here so a bad config cannot "start
// cleanly and then fail every request with a 500 that names nothing the
// operator can act on", and that is exactly what happened: Validate() checks
// the config's shape on its own, and a config with no Summarizer is
// well-shaped. Over a root agent that is not an LLM agent there is no model to
// build the default summarizer from, so every request 500'd.
func TestRESTCompaction_StartupRejectsAConfigItCannotServe(t *testing.T) {
	// A workflow agent has no model of its own.
	root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "wf_app"}})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}

	_, err = adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(root),
		// Well-shaped, and unserveable: no Summarizer and no model to make one.
		EventsCompactionConfig: &compaction.Config{CompactionInterval: 2},
	})
	if err == nil {
		t.Fatal("NewServer() accepted a compaction config that cannot serve its only app")
	}
	if !strings.Contains(err.Error(), "wf_app") {
		t.Errorf("error %q does not name the app the operator has to fix", err)
	}
}
