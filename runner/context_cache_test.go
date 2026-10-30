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

package runner

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/session"
)

// cachingGeminiAPI stands in for the Gemini API: it creates caches on request
// and answers every generation with a fixed reply reporting a large prompt.
type cachingGeminiAPI struct {
	mu        sync.Mutex
	creates   int
	generates []map[string]any
}

// snapshot returns what the server has seen so far, under the lock the handler
// goroutines write with.
func (f *cachingGeminiAPI) snapshot() (creates int, generates []map[string]any) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.creates, append([]map[string]any(nil), f.generates...)
}

func (f *cachingGeminiAPI) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var decoded map[string]any
	_ = json.Unmarshal(body, &decoded)

	f.mu.Lock()
	defer f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/cachedContents"):
		f.creates++
		_, _ = fmt.Fprintf(w, `{"name": "cachedContents/%d", "expireTime": %q}`, f.creates, time.Now().Add(time.Hour).Format(time.RFC3339))
	case r.Method == http.MethodDelete:
		_, _ = fmt.Fprint(w, `{}`)
	case strings.HasSuffix(r.URL.Path, ":generateContent"):
		f.generates = append(f.generates, decoded)
		_, _ = fmt.Fprintf(w, `{"candidates": [{"content": {"role": "model", "parts": [{"text": "answer %d"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 5000}}`, len(f.generates))
	default:
		w.WriteHeader(http.StatusNotFound)
		_, _ = fmt.Fprint(w, `{"error": {"code": 404, "message": "unexpected request", "status": "NOT_FOUND"}}`)
	}
}

// cachingAgent returns an LLM agent named assistant on a Gemini model that
// talks to api.
func cachingAgent(t *testing.T, api *cachingGeminiAPI, cfg llmagent.Config) agent.Agent {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(api.serve))
	t.Cleanup(srv.Close)
	llm, err := gemini.NewModel(t.Context(), "test-model", &genai.ClientConfig{
		APIKey:      "key",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatalf("gemini.NewModel() error = %v", err)
	}
	cfg.Name = "assistant"
	cfg.Model = llm
	cfg.Instruction = "You are a helpful assistant."
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

func newCachingRunner(t *testing.T, root agent.Agent, cacheCfg *model.ContextCacheConfig) (*Runner, session.Service) {
	t.Helper()
	sessionService := session.InMemoryService()
	r, err := New(Config{
		AppName:           "app",
		Agent:             root,
		SessionService:    sessionService,
		AutoCreateSession: true,
		ContextCache:      cacheCfg,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r, sessionService
}

// runTurns runs n turns of session s.
func runTurns(t *testing.T, r *Runner, n int) {
	t.Helper()
	for i := range n {
		msg := genai.NewContentFromText(fmt.Sprintf("question %d", i+1), genai.RoleUser)
		drain(t, r.Run(t.Context(), "u", "s", msg, agent.RunConfig{}))
	}
}

// storedCacheMetadata returns the cache state stored on each of the
// assistant's replies, in order.
func storedCacheMetadata(t *testing.T, sessionService session.Service) []*model.CacheMetadata {
	t.Helper()
	resp, err := sessionService.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var stored []*model.CacheMetadata
	for ev := range resp.Session.Events().All() {
		if ev.Author == "assistant" {
			stored = append(stored, ev.CacheMetadata)
		}
	}
	return stored
}

// TestRunner_ContextCacheAcrossTurns drives three turns of one session. The
// first only fingerprints its prefix, the second finds the prefix unchanged
// and creates a cache, and the third reuses that cache. A workflow root takes
// a different path through the runner than an LLM agent root, so both run.
func TestRunner_ContextCacheAcrossTurns(t *testing.T) {
	tests := []struct {
		name string
		root func(leaf agent.Agent) (agent.Agent, error)
	}{
		{
			name: "LLM agent root",
			root: func(leaf agent.Agent) (agent.Agent, error) { return leaf, nil },
		},
		{
			name: "parallel agent root",
			root: func(leaf agent.Agent) (agent.Agent, error) {
				return parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "fanout", SubAgents: []agent.Agent{leaf}}})
			},
		},
		{
			name: "sequential agent root",
			root: func(leaf agent.Agent) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{leaf}}})
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &cachingGeminiAPI{}
			root, err := tt.root(cachingAgent(t, api, llmagent.Config{}))
			if err != nil {
				t.Fatalf("building the root agent: %v", err)
			}
			r, sessionService := newCachingRunner(t, root, &model.ContextCacheConfig{})

			runTurns(t, r, 3)

			creates, generates := api.snapshot()
			if creates != 1 {
				t.Errorf("caches created = %d, want 1", creates)
			}
			if len(generates) != 3 {
				t.Fatalf("generateContent calls = %d, want 3", len(generates))
			}
			wantCache := []string{"", "cachedContents/1", "cachedContents/1"}
			// Turn 2 holds its question past the cached first exchange, and
			// turn 3 the second exchange and its question.
			wantContents := []int{1, 1, 3}
			for i, sent := range generates {
				got, _ := sent["cachedContent"].(string)
				if got != wantCache[i] {
					t.Errorf("turn %d cachedContent = %q, want %q", i+1, got, wantCache[i])
				}
				_, hasInstruction := sent["systemInstruction"]
				if hasInstruction != (wantCache[i] == "") {
					t.Errorf("turn %d sent a systemInstruction = %v, want %v", i+1, hasInstruction, wantCache[i] == "")
				}
				contents, _ := sent["contents"].([]any)
				if len(contents) != wantContents[i] {
					t.Errorf("turn %d sent %d contents, want %d", i+1, len(contents), wantContents[i])
				}
			}

			stored := storedCacheMetadata(t, sessionService)
			if len(stored) != 3 {
				t.Fatalf("assistant events = %d, want 3", len(stored))
			}
			if stored[0] == nil || stored[0].Active() {
				t.Errorf("turn 1 CacheMetadata = %+v, want fingerprint-only", stored[0])
			}
			for _, want := range []struct{ turn, used int }{{turn: 2, used: 1}, {turn: 3, used: 2}} {
				if got := stored[want.turn-1]; !got.Active() || got.InvocationsUsed != want.used {
					t.Errorf("turn %d CacheMetadata = %+v, want an active cache used by %d invocations", want.turn, got, want.used)
				}
			}
		})
	}
}

// TestRunner_ContextCacheSurvivesRewrittenReply checks that an after-model
// callback building a new reply does not lose the cache state. Without it the
// third turn finds no record of the cache the second created, and creates
// another. The callback keeps the usage, as one that knows about it would; a
// reply without usage gives no prompt size to create a cache from.
func TestRunner_ContextCacheSurvivesRewrittenReply(t *testing.T) {
	api := &cachingGeminiAPI{}
	rewrite := func(ctx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
		if err != nil || resp == nil {
			return nil, err
		}
		return &model.LLMResponse{
			Content:       genai.NewContentFromText("[redacted]", genai.RoleModel),
			UsageMetadata: resp.UsageMetadata,
		}, nil
	}
	root := cachingAgent(t, api, llmagent.Config{AfterModelCallbacks: []llmagent.AfterModelCallback{rewrite}})
	r, sessionService := newCachingRunner(t, root, &model.ContextCacheConfig{})

	runTurns(t, r, 3)

	if creates, _ := api.snapshot(); creates != 1 {
		t.Errorf("caches created = %d, want 1", creates)
	}
	stored := storedCacheMetadata(t, sessionService)
	if len(stored) != 3 || !stored[2].Active() {
		t.Errorf("stored CacheMetadata = %+v, want the third reply to record the active cache", stored)
	}
}

// TestRunner_ContextCacheOff checks that a runner without a ContextCache config
// sends every request whole and records no cache state.
func TestRunner_ContextCacheOff(t *testing.T) {
	api := &cachingGeminiAPI{}
	r, _ := newCachingRunner(t, cachingAgent(t, api, llmagent.Config{}), nil)

	for i := range 3 {
		msg := genai.NewContentFromText(fmt.Sprintf("question %d", i+1), genai.RoleUser)
		for ev, err := range r.Run(t.Context(), "u", "s", msg, agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if ev.CacheMetadata != nil {
				t.Errorf("event carries CacheMetadata %+v with caching off", ev.CacheMetadata)
			}
		}
	}
	creates, generates := api.snapshot()
	if creates != 0 {
		t.Errorf("caches created = %d, want 0", creates)
	}
	for i, sent := range generates {
		if _, ok := sent["cachedContent"]; ok {
			t.Errorf("turn %d named a cache with caching off", i+1)
		}
	}
}

func TestNew_InvalidContextCache(t *testing.T) {
	a, err := llmagent.New(llmagent.Config{Name: "assistant"})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	_, err = New(Config{
		AppName:        "app",
		Agent:          a,
		SessionService: session.InMemoryService(),
		ContextCache:   &model.ContextCacheConfig{CacheIntervals: 101},
	})
	if err == nil || !strings.Contains(err.Error(), "invalid ContextCache") {
		t.Errorf("New() error = %v, want an invalid ContextCache error", err)
	}
}

// TestNew_CopiesContextCache checks that the runner keeps its own copy, so a
// caller changing its config after New does not change the requests the
// runner makes.
func TestNew_CopiesContextCache(t *testing.T) {
	api := &cachingGeminiAPI{}
	cfg := &model.ContextCacheConfig{}
	r, _ := newCachingRunner(t, cachingAgent(t, api, llmagent.Config{}), cfg)

	// Shared with the runner, this would stop every cache being created.
	cfg.MinTokens = 1 << 30
	runTurns(t, r, 2)

	if creates, _ := api.snapshot(); creates != 1 {
		t.Errorf("caches created = %d after the caller raised its MinTokens, want 1 from the runner's own copy", creates)
	}
}
