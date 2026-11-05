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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

// cachingGeminiAPI stands in for the Gemini API. It creates caches on request
// and tracks which still exist, rejects a generation naming one that does not
// as the Gemini API does, and answers every other generation with a fixed
// reply reporting a large prompt.
type cachingGeminiAPI struct {
	mu            sync.Mutex
	creates       int
	createHeaders []http.Header
	generates     []map[string]any
	live          map[string]bool
	// emptyReplies holds the generations, numbered from 1, answered with no
	// content.
	emptyReplies map[int]bool
}

// snapshot returns what the server has seen so far, under the lock the handler
// goroutines write with.
func (f *cachingGeminiAPI) snapshot() (creates int, generates []map[string]any) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.creates, append([]map[string]any(nil), f.generates...)
}

// liveCaches returns the caches that exist on the server, sorted.
func (f *cachingGeminiAPI) liveCaches() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Sorted(maps.Keys(f.live))
}

// forget deletes a cache behind the runner's back, as an expiry or another
// client would.
func (f *cachingGeminiAPI) forget(name string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.live, name)
}

func (f *cachingGeminiAPI) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var decoded map[string]any
	_ = json.Unmarshal(body, &decoded)

	f.mu.Lock()
	defer f.mu.Unlock()
	if f.live == nil {
		f.live = make(map[string]bool)
	}
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/cachedContents"):
		f.creates++
		f.createHeaders = append(f.createHeaders, r.Header.Clone())
		name := fmt.Sprintf("cachedContents/%d", f.creates)
		f.live[name] = true
		_, _ = fmt.Fprintf(w, `{"name": %q, "expireTime": %q}`, name, time.Now().Add(time.Hour).Format(time.RFC3339))
	case r.Method == http.MethodDelete:
		delete(f.live, r.URL.Path[strings.Index(r.URL.Path, "cachedContents/"):])
		_, _ = fmt.Fprint(w, `{}`)
	case strings.HasSuffix(r.URL.Path, ":generateContent"):
		f.generates = append(f.generates, decoded)
		if name, _ := decoded["cachedContent"].(string); name != "" && !f.live[name] {
			w.WriteHeader(http.StatusNotFound)
			_, _ = fmt.Fprint(w, `{"error": {"code": 404, "message": "CachedContent not found (or permission denied)", "status": "NOT_FOUND"}}`)
			return
		}
		if f.emptyReplies[len(f.generates)] {
			_, _ = fmt.Fprint(w, `{"candidates": [{"finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 5000}}`)
			return
		}
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
	return newCachingRunnerWith(t, Config{Agent: root, SessionService: session.InMemoryService(), ContextCache: cacheCfg})
}

// newCachingRunnerWith returns a runner for app "app" built from cfg.
func newCachingRunnerWith(t *testing.T, cfg Config) (*Runner, session.Service) {
	t.Helper()
	cfg.AppName = "app"
	cfg.AutoCreateSession = true
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r, cfg.SessionService
}

// runTurn runs one turn of session s, numbered from 1, and returns the first
// error it yields.
func runTurn(t *testing.T, r *Runner, turn int) error {
	t.Helper()
	msg := genai.NewContentFromText(fmt.Sprintf("question %d", turn), genai.RoleUser)
	var firstErr error
	for _, err := range r.Run(t.Context(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
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
		if ev.Author == "assistant" && ev.CacheMetadata != nil {
			stored = append(stored, ev.CacheMetadata)
		}
	}
	return stored
}

// checkCachesMatchSession checks the two properties cache cleanup keeps after
// a turn: every cache that exists is the one the agent's latest stored record
// names, and the cache that record names exists.
func checkCachesMatchSession(t *testing.T, api *cachingGeminiAPI, sessionService session.Service, turn int) {
	t.Helper()
	var named []string
	if stored := storedCacheMetadata(t, sessionService); len(stored) > 0 && stored[len(stored)-1].Active() {
		named = []string{stored[len(stored)-1].CacheName}
	}
	if diff := cmp.Diff(named, api.liveCaches()); diff != "" {
		t.Errorf("after turn %d, the caches on the provider do not match the one the session names (-named +live):\n%s", turn, diff)
	}
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
				t.Fatalf("assistant events with cache state = %d, want 3", len(stored))
			}
			if stored[0].Active() {
				t.Errorf("turn 1 CacheMetadata = %+v, want fingerprint-only", stored[0])
			}
			for _, want := range []struct{ turn, used int }{{turn: 2, used: 1}, {turn: 3, used: 2}} {
				if got := stored[want.turn-1]; !got.Active() || got.InvocationsUsed != want.used {
					t.Errorf("turn %d CacheMetadata = %+v, want an active cache used by %d invocations", want.turn, got, want.used)
				}
			}
			checkCachesMatchSession(t, api, sessionService, 3)
		})
	}
}

// TestRunner_ContextCacheReplaced checks that a cache the agent stops using is
// deleted once the replacing turn is stored, under every kind of root: an LLM
// agent root and a workflow root settle on different paths through the
// runner. With CacheIntervals 1, turn 3 replaces turn 2's cache.
func TestRunner_ContextCacheReplaced(t *testing.T) {
	tests := []struct {
		name string
		root func(leaf agent.Agent) (agent.Agent, error)
	}{
		{
			name: "LLM agent root",
			root: func(leaf agent.Agent) (agent.Agent, error) { return leaf, nil },
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
			r, sessionService := newCachingRunner(t, root, &model.ContextCacheConfig{CacheIntervals: 1})

			for turn := 1; turn <= 4; turn++ {
				if err := runTurn(t, r, turn); err != nil {
					t.Fatalf("turn %d failed: %v", turn, err)
				}
				checkCachesMatchSession(t, api, sessionService, turn)
			}
			if creates, _ := api.snapshot(); creates < 2 {
				t.Errorf("caches created = %d, want a replacement, so this test checks nothing", creates)
			}
		})
	}
}

// TestRunner_ContextCacheSurvivesRewrittenReply checks that an after-model
// callback building a new reply, with neither usage nor cache state, still
// lets the agent cache. The flow carries both onto the replacement. Without
// the cache state, no turn records a fingerprint, so no cache is ever
// created; without the usage, no turn has a prompt size to create one from.
func TestRunner_ContextCacheSurvivesRewrittenReply(t *testing.T) {
	api := &cachingGeminiAPI{}
	rewrite := func(ctx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
		if err != nil || resp == nil {
			return nil, err
		}
		return &model.LLMResponse{Content: genai.NewContentFromText("[redacted]", genai.RoleModel)}, nil
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

// TestRunner_ContextCacheEmptyReplyFilledByCallback reproduces a review
// finding. On the turn that creates a cache, the model answers with no
// content and an after-model callback fills the reply in. The session stores
// that reply, naming the new cache, so the cache must survive the turn; were
// it deleted, every later turn would send a name the provider rejects.
func TestRunner_ContextCacheEmptyReplyFilledByCallback(t *testing.T) {
	api := &cachingGeminiAPI{emptyReplies: map[int]bool{2: true}}
	fill := func(ctx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
		if err != nil || resp == nil || resp.Content != nil {
			return nil, err
		}
		return &model.LLMResponse{Content: genai.NewContentFromText("Sorry, could you rephrase that?", genai.RoleModel)}, nil
	}
	root := cachingAgent(t, api, llmagent.Config{AfterModelCallbacks: []llmagent.AfterModelCallback{fill}})
	r, sessionService := newCachingRunner(t, root, &model.ContextCacheConfig{})

	for turn := 1; turn <= 5; turn++ {
		if err := runTurn(t, r, turn); err != nil {
			t.Fatalf("turn %d failed: %v", turn, err)
		}
		checkCachesMatchSession(t, api, sessionService, turn)
	}
	if creates, _ := api.snapshot(); creates != 1 {
		t.Errorf("caches created = %d, want 1 reused from turn 2 on", creates)
	}
}

// failingAppendService refuses to store the assistant's reply on the turn
// failTurn names, and stores everything else.
type failingAppendService struct {
	session.Service
	failTurn *int
	turn     *int
}

func (s failingAppendService) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	if ev.Author == "assistant" && *s.turn == *s.failTurn {
		return errors.New("session store unavailable")
	}
	return s.Service.AppendEvent(ctx, sess, ev)
}

// TestRunner_ContextCacheReplyNotStored reproduces three review findings with
// one shape. With CacheIntervals 1, turn 3 replaces turn 2's cache with a new
// one, and then nothing is stored for turn 3. The session still names the old
// cache, so the old one has to stay, and the new one, recorded nowhere, has to
// go.
func TestRunner_ContextCacheReplyNotStored(t *testing.T) {
	const failTurn = 3
	tests := []struct {
		name  string
		build func(api *cachingGeminiAPI, turn *int) (agent.Agent, Config)
	}{
		{
			// The flow drops a reply with no content.
			name: "a callback empties the reply",
			build: func(api *cachingGeminiAPI, turn *int) (agent.Agent, Config) {
				empty := func(ctx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
					if err != nil || *turn != failTurn {
						return nil, err
					}
					return &model.LLMResponse{}, nil
				}
				return cachingAgent(t, api, llmagent.Config{AfterModelCallbacks: []llmagent.AfterModelCallback{empty}}), Config{}
			},
		},
		{
			name: "the session store refuses the reply",
			build: func(api *cachingGeminiAPI, turn *int) (agent.Agent, Config) {
				ft := failTurn
				return cachingAgent(t, api, llmagent.Config{}), Config{
					SessionService: failingAppendService{Service: session.InMemoryService(), failTurn: &ft, turn: turn},
				}
			},
		},
		{
			// The plugin's event is stored in place of the model's, without
			// its cache state.
			name: "a plugin replaces the reply",
			build: func(api *cachingGeminiAPI, turn *int) (agent.Agent, Config) {
				p, err := plugin.New(plugin.Config{
					Name: "replacer",
					OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
						if ev.Author != "assistant" || *turn != failTurn {
							return nil, nil
						}
						return &session.Event{Author: ev.Author, LLMResponse: model.LLMResponse{Content: ev.Content}}, nil
					},
				})
				if err != nil {
					t.Fatalf("plugin.New() error = %v", err)
				}
				return cachingAgent(t, api, llmagent.Config{}), Config{PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &cachingGeminiAPI{}
			turn := 0
			root, cfg := tt.build(api, &turn)
			cfg.Agent = root
			cfg.ContextCache = &model.ContextCacheConfig{CacheIntervals: 1}
			if cfg.SessionService == nil {
				cfg.SessionService = session.InMemoryService()
			}
			r, sessionService := newCachingRunnerWith(t, cfg)

			for turn = 1; turn <= 4; turn++ {
				_ = runTurn(t, r, turn)
				checkCachesMatchSession(t, api, sessionService, turn)
				if turn == failTurn {
					if got, want := api.liveCaches(), []string{"cachedContents/1"}; !cmp.Equal(got, want) {
						t.Errorf("after the unstored turn, live caches = %v, want only turn 2's %v", got, want)
					}
				}
			}
		})
	}
}

// TestRunner_ContextCacheMissingCache checks that a session naming a cache the
// provider no longer holds recovers on the next turn: the request is sent
// again without the cache, and the turn after creates a new one.
func TestRunner_ContextCacheMissingCache(t *testing.T) {
	api := &cachingGeminiAPI{}
	r, sessionService := newCachingRunner(t, cachingAgent(t, api, llmagent.Config{}), &model.ContextCacheConfig{})

	for turn := 1; turn <= 4; turn++ {
		if turn == 3 {
			api.forget("cachedContents/1")
		}
		if err := runTurn(t, r, turn); err != nil {
			t.Fatalf("turn %d failed: %v", turn, err)
		}
		checkCachesMatchSession(t, api, sessionService, turn)
	}
	if creates, _ := api.snapshot(); creates != 2 {
		t.Errorf("caches created = %d, want 2: the first, and its replacement after it went missing", creates)
	}
}

// TestRunner_ContextCacheHTTPOptionsNotShared checks that the runner's copy of
// CreateHTTPOptions shares nothing with the caller's or with any request's. A
// header the caller adds after New, and one a callback adds to an earlier
// request's config, must not reach a cache creation, and the callback's must
// not reach the caller's map either.
func TestRunner_ContextCacheHTTPOptionsNotShared(t *testing.T) {
	api := &cachingGeminiAPI{}
	turn := 0
	tamper := func(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		if turn == 1 && req.CacheConfig != nil {
			req.CacheConfig.CreateHTTPOptions.Headers.Set("X-Callback", "turn 1")
		}
		return nil, nil
	}
	callerHeaders := http.Header{"X-Caller": []string{"yes"}}
	cfg := &model.ContextCacheConfig{CreateHTTPOptions: &genai.HTTPOptions{Headers: callerHeaders}}
	r, _ := newCachingRunner(t, cachingAgent(t, api, llmagent.Config{BeforeModelCallbacks: []llmagent.BeforeModelCallback{tamper}}), cfg)
	callerHeaders.Set("X-After-New", "yes")

	// A caller writing its map while the runner creates caches is a data
	// race the race detector reports, unless nothing is shared.
	done := make(chan struct{})
	var writer sync.WaitGroup
	writer.Go(func() {
		for {
			select {
			case <-done:
				return
			default:
				callerHeaders.Set("X-Writer", time.Now().String())
			}
		}
	})
	for turn = 1; turn <= 2; turn++ {
		if err := runTurn(t, r, turn); err != nil {
			t.Fatalf("turn %d failed: %v", turn, err)
		}
	}
	close(done)
	writer.Wait()

	api.mu.Lock()
	createHeaders := api.createHeaders
	api.mu.Unlock()
	if len(createHeaders) != 1 {
		t.Fatalf("caches created = %d, want 1", len(createHeaders))
	}
	h := createHeaders[0]
	if h.Get("X-Caller") != "yes" {
		t.Error("the caller's header from before New did not reach the cache creation")
	}
	for _, leaked := range []string{"X-After-New", "X-Callback", "X-Writer"} {
		if got := h.Get(leaked); got != "" {
			t.Errorf("cache creation carried %s = %q, want it absent", leaked, got)
		}
	}
	if got := callerHeaders.Get("X-Callback"); got != "" {
		t.Errorf("the caller's map gained X-Callback = %q from a request's config", got)
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
