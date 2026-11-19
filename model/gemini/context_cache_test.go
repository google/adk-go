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

package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// cacheTestNow is the clock every context cache test runs at.
var cacheTestNow = time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)

// fakeCacheAPI stands in for the Gemini API's cachedContents and
// generateContent endpoints, and records what each received, in order.
type fakeCacheAPI struct {
	// createStatus and generateStatus, when non-zero, are the error status
	// every cache creation or generation gets.
	createStatus   int
	generateStatus int
	// emptyReply makes generations answer with a STOP and no content, which
	// the flow drops without storing an event.
	emptyReply bool
	// expireTime is the expiry a created cache reports. Zero leaves it out.
	expireTime time.Time
	// gone names caches the provider no longer holds. A generation naming one
	// fails with goneStatus, 404 when zero, and the message the Gemini API
	// gives for a cache it cannot find.
	gone       map[string]bool
	goneStatus int
	// goneMessage replaces the message, so a 403 can be one about something
	// other than the cache.
	goneMessage string
	// goneAfterChunk makes a streamed generation that names a gone cache send
	// one chunk of text before its error, as a stream that fails part-way
	// does, instead of failing before any output.
	goneAfterChunk bool

	mu            sync.Mutex
	calls         []string
	creates       []map[string]any
	createHeaders []http.Header
	deleteHeaders []http.Header
	generates     []map[string]any
}

func (f *fakeCacheAPI) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var decoded map[string]any
	_ = json.Unmarshal(body, &decoded)
	path := r.URL.Path

	f.mu.Lock()
	defer f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	fail := func(status int) {
		w.WriteHeader(status)
		_, _ = fmt.Fprintf(w, `{"error": {"code": %d, "message": "refused", "status": "INVALID_ARGUMENT"}}`, status)
	}
	// failGone answers a generation that names a cache the provider no longer
	// holds, and reports whether it did.
	failGone := func() bool {
		name, _ := decoded["cachedContent"].(string)
		if !f.gone[name] {
			return false
		}
		status := f.goneStatus
		if status == 0 {
			status = http.StatusNotFound
		}
		message := f.goneMessage
		if message == "" {
			message = "CachedContent not found (or permission denied)"
		}
		w.WriteHeader(status)
		_, _ = fmt.Fprintf(w, `{"error": {"code": %d, "message": %q, "status": "NOT_FOUND"}}`, status, message)
		return true
	}
	switch {
	case r.Method == http.MethodPost && strings.HasSuffix(path, "/cachedContents"):
		f.calls = append(f.calls, "create")
		f.creates = append(f.creates, decoded)
		f.createHeaders = append(f.createHeaders, r.Header.Clone())
		if f.createStatus != 0 {
			fail(f.createStatus)
			return
		}
		resp := map[string]any{"name": fmt.Sprintf("cachedContents/%d", len(f.creates))}
		if !f.expireTime.IsZero() {
			resp["expireTime"] = f.expireTime.Format(time.RFC3339)
		}
		_ = json.NewEncoder(w).Encode(resp)
	case r.Method == http.MethodDelete:
		f.calls = append(f.calls, "delete "+path[strings.Index(path, "cachedContents/"):])
		f.deleteHeaders = append(f.deleteHeaders, r.Header.Clone())
		_, _ = fmt.Fprint(w, `{}`)
	case strings.HasSuffix(path, ":streamGenerateContent"):
		f.calls = append(f.calls, "generate")
		f.generates = append(f.generates, decoded)
		if f.generateStatus != 0 {
			fail(f.generateStatus)
			return
		}
		if name, _ := decoded["cachedContent"].(string); f.goneAfterChunk && f.gone[name] {
			w.Header().Set("Content-Type", "text/event-stream")
			chunk, _ := json.Marshal(map[string]any{"candidates": []any{map[string]any{
				"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": "Hel"}}},
			}}})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			_, _ = fmt.Fprint(w, `{"error": {"code": 404, "message": "CachedContent not found (or permission denied)", "status": "NOT_FOUND"}}`+"\n\n")
			return
		}
		if failGone() {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range []string{"Hel", "lo"} {
			data, _ := json.Marshal(map[string]any{"candidates": []any{map[string]any{
				"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": chunk}}},
			}}})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		}
		data, _ := json.Marshal(map[string]any{
			"candidates":    []any{map[string]any{"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": ""}}}, "finishReason": "STOP"}},
			"usageMetadata": map[string]any{"promptTokenCount": 10},
		})
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
	case strings.HasSuffix(path, ":generateContent"):
		f.calls = append(f.calls, "generate")
		f.generates = append(f.generates, decoded)
		if f.generateStatus != 0 {
			fail(f.generateStatus)
			return
		}
		if failGone() {
			return
		}
		if f.emptyReply {
			_, _ = fmt.Fprint(w, `{"candidates": [{"finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10}}`)
			return
		}
		_, _ = fmt.Fprint(w, `{"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10}}`)
	default:
		w.WriteHeader(http.StatusNotFound)
		_, _ = fmt.Fprintf(w, `{"error": {"code": 404, "message": "unexpected %s %s", "status": "NOT_FOUND"}}`, r.Method, path)
	}
}

// newCacheTestModel returns a model named modelName that talks to api.
func newCacheTestModel(t *testing.T, api *fakeCacheAPI, modelName string) *geminiModel {
	t.Helper()
	return newCacheTestModelWithProvider(t, api, modelName, nil)
}

// newCacheTestModelWithProvider returns a model named modelName that talks to
// api, whose client sets provider as its extras request provider.
func newCacheTestModelWithProvider(t *testing.T, api *fakeCacheAPI, modelName string, provider genai.ExtrasRequestProvider) *geminiModel {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(api.serve))
	t.Cleanup(srv.Close)
	m, err := NewModel(t.Context(), modelName, &genai.ClientConfig{
		APIKey:      "key",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL, ExtrasRequestProvider: provider},
	})
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return m.(*geminiModel)
}

// fingerprintOf returns the fingerprint m gives the first count contents of
// req.
func fingerprintOf(m *geminiModel, req *model.LLMRequest, count int) string {
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	return newFingerprinter(req, m.modelName(req), scopeFor(m.client.ClientConfig())).of(count)
}

func cacheTestContext(t *testing.T) context.Context {
	return platform.WithTimeProvider(t.Context(), func() time.Time { return cacheTestNow })
}

// cacheTestRequest returns a request with a system instruction, a tool, and
// three contents: a finished exchange, which is the cacheable prefix, and the
// question being asked.
func cacheTestRequest(cfg *model.ContextCacheConfig) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("first question", genai.RoleUser),
			genai.NewContentFromText("first answer", genai.RoleModel),
			genai.NewContentFromText("second question", genai.RoleUser),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are a helpful assistant.", genai.RoleUser),
			Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: "lookup", Description: "Looks something up."},
			}}},
		},
		CacheConfig: cfg,
	}
}

// generate calls m with req and returns the last response and error.
func generate(ctx context.Context, m *geminiModel, req *model.LLMRequest) (*model.LLMResponse, error) {
	var got *model.LLMResponse
	var gotErr error
	for resp, err := range m.GenerateContent(ctx, req, false) {
		got, gotErr = resp, err
	}
	return got, gotErr
}

// generateOnce calls m with req and returns the one response.
func generateOnce(t *testing.T, ctx context.Context, m *geminiModel, req *model.LLMRequest) *model.LLMResponse {
	t.Helper()
	got, err := generate(ctx, m, req)
	if err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}
	if got == nil {
		t.Fatal("GenerateContent() returned no response")
	}
	return got
}

func TestCountContentsToCache(t *testing.T) {
	user := genai.NewContentFromText("u", genai.RoleUser)
	modelTurn := genai.NewContentFromText("m", genai.RoleModel)
	tests := []struct {
		name     string
		contents []*genai.Content
		want     int
	}{
		{name: "empty", contents: nil, want: 0},
		{name: "only user contents", contents: []*genai.Content{user, user}, want: 0},
		{name: "exchange then a question", contents: []*genai.Content{user, modelTurn, user}, want: 2},
		{name: "a run of trailing user contents", contents: []*genai.Content{user, modelTurn, user, user}, want: 2},
		{name: "ends on the model", contents: []*genai.Content{user, modelTurn}, want: 2},
		{name: "earlier user contents stay in the prefix", contents: []*genai.Content{user, user, modelTurn, user}, want: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := countContentsToCache(tt.contents); got != tt.want {
				t.Errorf("countContentsToCache() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestContextCacheFingerprint(t *testing.T) {
	m := newCacheTestModel(t, &fakeCacheAPI{}, "test-model")
	base := fingerprintOf(m, cacheTestRequest(nil), 2)
	if base == "" {
		t.Fatal("fingerprint is empty, want a fingerprint")
	}

	tests := []struct {
		name     string
		model    *geminiModel
		edit     func(*model.LLMRequest)
		count    int
		wantSame bool
	}{
		{
			name:     "identical request",
			edit:     func(*model.LLMRequest) {},
			count:    2,
			wantSame: true,
		},
		{
			name: "contents past the prefix differ",
			edit: func(r *model.LLMRequest) {
				r.Contents[2] = genai.NewContentFromText("another question", genai.RoleUser)
			},
			count:    2,
			wantSame: true,
		},
		{
			name: "system instruction changed",
			edit: func(r *model.LLMRequest) {
				r.Config.SystemInstruction = genai.NewContentFromText("You are terse.", genai.RoleUser)
			},
			count: 2,
		},
		{
			name: "tool added",
			edit: func(r *model.LLMRequest) {
				r.Config.Tools = append(r.Config.Tools, &genai.Tool{GoogleSearch: &genai.GoogleSearch{}})
			},
			count: 2,
		},
		{
			name: "tool config added",
			edit: func(r *model.LLMRequest) {
				r.Config.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny}}
			},
			count: 2,
		},
		{
			name: "a cached content changed",
			edit: func(r *model.LLMRequest) {
				r.Contents[1] = genai.NewContentFromText("a different answer", genai.RoleModel)
			},
			count: 2,
		},
		{
			name:  "longer prefix",
			edit:  func(*model.LLMRequest) {},
			count: 3,
		},
		{
			name:  "different model",
			edit:  func(r *model.LLMRequest) { r.Model = "other-model" },
			count: 2,
		},
		{
			name:  "different endpoint",
			model: newCacheTestModel(t, &fakeCacheAPI{}, "test-model"),
			edit:  func(*model.LLMRequest) {},
			count: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fm := tt.model
			if fm == nil {
				fm = m
			}
			req := cacheTestRequest(nil)
			tt.edit(req)
			if got := fingerprintOf(fm, req, tt.count); (got == base) != tt.wantSame {
				t.Errorf("fingerprint equals the base request's = %v, want %v", got == base, tt.wantSame)
			}
		})
	}

	t.Run("reordering does not change it", func(t *testing.T) {
		ordered := cacheTestRequest(nil)
		ordered.Config.Tools = []*genai.Tool{
			{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "a"}, {Name: "b"}}},
			{GoogleSearch: &genai.GoogleSearch{}},
		}
		reordered := cacheTestRequest(nil)
		reordered.Config.Tools = []*genai.Tool{
			{GoogleSearch: &genai.GoogleSearch{}},
			{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "b"}, {Name: "a"}}},
		}
		if a, b := fingerprintOf(m, ordered, 2), fingerprintOf(m, reordered, 2); a != b {
			t.Errorf("fingerprints of reordered tools differ: %q and %q", a, b)
		}
		if got := reordered.Config.Tools[1].FunctionDeclarations[0].Name; got != "b" {
			t.Errorf("fingerprinting reordered the request's own declarations, first is now %q", got)
		}
	})
}

func TestScopeFor(t *testing.T) {
	tests := []struct {
		name string
		cfg  genai.ClientConfig
		want cacheScope
	}{
		{
			name: "gemini API",
			cfg:  genai.ClientConfig{Backend: genai.BackendGeminiAPI, HTTPOptions: genai.HTTPOptions{BaseURL: "https://example.test"}},
			want: cacheScope{Backend: "gemini", BaseURL: "https://example.test"},
		},
		{
			name: "gemini API ignores a project",
			cfg:  genai.ClientConfig{Backend: genai.BackendGeminiAPI, Project: "p", Location: "l"},
			want: cacheScope{Backend: "gemini"},
		},
		{
			name: "vertex AI names its project and location",
			cfg:  genai.ClientConfig{Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1"},
			want: cacheScope{Backend: "vertex", Project: "p", Location: "us-central1"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, scopeFor(tt.cfg)); diff != "" {
				t.Errorf("scopeFor() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGenerateContent_ContextCache(t *testing.T) {
	expire := cacheTestNow.Add(30 * time.Minute)
	defaultCfg := &model.ContextCacheConfig{CacheIntervals: 3, TTL: time.Hour, MinTokens: 1000}

	// The metadata of each case is written against a placeholder fingerprint,
	// replaced with the real one of the case's server, whose URL is part of
	// it.
	const prefix = "PREFIX"
	fingerprintOnly := &model.CacheMetadata{Fingerprint: prefix, ContentsCount: 2}
	active := func(used int, expireTime time.Time) *model.CacheMetadata {
		return &model.CacheMetadata{
			CacheName:       "cachedContents/old",
			ExpireTime:      expireTime,
			InvocationsUsed: used,
			CreatedAt:       cacheTestNow.Add(-time.Minute),
			Fingerprint:     prefix,
			ContentsCount:   2,
		}
	}
	created := &model.CacheMetadata{
		CacheName:       "cachedContents/1",
		ExpireTime:      expire,
		InvocationsUsed: 1,
		CreatedAt:       cacheTestNow,
		Fingerprint:     prefix,
		ContentsCount:   2,
	}
	changedInstruction := func(r *model.LLMRequest) {
		r.Config.SystemInstruction = genai.NewContentFromText("You are terse.", genai.RoleUser)
	}

	tests := []struct {
		name         string
		cfg          *model.ContextCacheConfig
		createStatus int
		metadata     *model.CacheMetadata
		promptTokens int32
		edit         func(*model.LLMRequest)
		wantCalls    []string
		wantCached   bool
		// wantMetadata nil means fingerprint-only metadata of the request
		// as sent.
		wantMetadata *model.CacheMetadata
	}{
		{
			name:         "first request only fingerprints",
			promptTokens: 5000,
			wantCalls:    []string{"generate"},
			wantMetadata: fingerprintOnly,
		},
		{
			name:         "matching prefix creates a cache",
			metadata:     fingerprintOnly,
			promptTokens: 5000,
			wantCalls:    []string{"create", "generate"},
			wantCached:   true,
			wantMetadata: created,
		},
		{
			name:         "valid cache is reused",
			metadata:     active(2, expire),
			promptTokens: 5000,
			wantCalls:    []string{"generate"},
			wantCached:   true,
			wantMetadata: active(2, expire),
		},
		{
			name:         "cache at its last interval is reused",
			metadata:     active(3, expire),
			promptTokens: 5000,
			wantCalls:    []string{"generate"},
			wantCached:   true,
			wantMetadata: active(3, expire),
		},
		{
			name:         "expired cache is replaced, and deleted after the response",
			metadata:     active(2, cacheTestNow),
			promptTokens: 5000,
			wantCalls:    []string{"create", "generate", "delete cachedContents/old"},
			wantCached:   true,
			wantMetadata: created,
		},
		{
			name:         "cache past its intervals is replaced",
			metadata:     active(4, expire),
			promptTokens: 5000,
			wantCalls:    []string{"create", "generate", "delete cachedContents/old"},
			wantCached:   true,
			wantMetadata: created,
		},
		{
			name:         "changed prefix drops the cache and starts over",
			metadata:     active(2, expire),
			promptTokens: 5000,
			edit:         changedInstruction,
			wantCalls:    []string{"generate", "delete cachedContents/old"},
		},
		{
			name:         "changed prefix after fingerprint only starts over",
			metadata:     &model.CacheMetadata{Fingerprint: "stale", ContentsCount: 2},
			promptTokens: 5000,
			wantCalls:    []string{"generate"},
			wantMetadata: fingerprintOnly,
		},
		{
			name:         "unknown earlier prompt size creates nothing",
			cfg:          &model.ContextCacheConfig{},
			metadata:     fingerprintOnly,
			wantCalls:    []string{"generate"},
			wantMetadata: fingerprintOnly,
		},
		{
			name:         "earlier prompt below MinTokens creates nothing",
			metadata:     fingerprintOnly,
			promptTokens: 999,
			wantCalls:    []string{"generate"},
			wantMetadata: fingerprintOnly,
		},
		{
			name:         "earlier prompt at MinTokens creates a cache",
			metadata:     fingerprintOnly,
			promptTokens: 1000,
			wantCalls:    []string{"create", "generate"},
			wantCached:   true,
			wantMetadata: created,
		},
		{
			name:         "refused creation leaves the request uncached",
			createStatus: http.StatusBadRequest,
			metadata:     fingerprintOnly,
			promptTokens: 5000,
			wantCalls:    []string{"create", "generate"},
			wantMetadata: fingerprintOnly,
		},
		{
			name:         "refused replacement still drops the old cache",
			createStatus: http.StatusBadRequest,
			metadata:     active(2, cacheTestNow),
			promptTokens: 5000,
			wantCalls:    []string{"create", "generate", "delete cachedContents/old"},
			wantMetadata: fingerprintOnly,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &fakeCacheAPI{createStatus: tt.createStatus, expireTime: expire}
			m := newCacheTestModel(t, api, "test-model")
			real := fingerprintOf(m, cacheTestRequest(nil), 2)
			withFingerprint := func(md *model.CacheMetadata) *model.CacheMetadata {
				out := *md
				if out.Fingerprint == prefix {
					out.Fingerprint = real
				}
				return &out
			}

			cfg := tt.cfg
			if cfg == nil {
				cfg = defaultCfg
			}
			req := cacheTestRequest(cfg)
			req.PreviousPromptTokenCount = tt.promptTokens
			if tt.metadata != nil {
				req.CacheMetadata = withFingerprint(tt.metadata)
			}
			if tt.edit != nil {
				tt.edit(req)
			}

			resp := generateOnce(t, cacheTestContext(t), m, req)

			if diff := cmp.Diff(tt.wantCalls, api.calls); diff != "" {
				t.Errorf("API calls mismatch (-want +got):\n%s", diff)
			}
			_, gotCached := api.generates[0]["cachedContent"]
			if gotCached != tt.wantCached {
				t.Errorf("request named a cache = %v, want %v", gotCached, tt.wantCached)
			}
			want := &model.CacheMetadata{Fingerprint: fingerprintOf(m, req, 2), ContentsCount: 2}
			if tt.wantMetadata != nil {
				want = withFingerprint(tt.wantMetadata)
			}
			if diff := cmp.Diff(want, resp.CacheMetadata); diff != "" {
				t.Errorf("response CacheMetadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestGenerateContent_ContextCacheGrowsToEarlierPrefix checks that a cache
// covers the prefix the earlier request recorded when this request's own
// prefix is shorter, which happens when it ends on more user contents.
func TestGenerateContent_ContextCacheGrowsToEarlierPrefix(t *testing.T) {
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.Contents = append(req.Contents, genai.NewContentFromText("one more thing", genai.RoleUser))
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, req, 3), ContentsCount: 3}

	resp := generateOnce(t, cacheTestContext(t), m, req)

	if len(api.creates) != 1 {
		t.Fatalf("caches created = %d, want 1", len(api.creates))
	}
	if got := contentTexts(api.creates[0]["contents"]); len(got) != 3 {
		t.Errorf("cached contents = %q, want the earlier request's 3", got)
	}
	if got := contentTexts(api.generates[0]["contents"]); !cmp.Equal(got, []string{"one more thing"}) {
		t.Errorf("request contents = %q, want only the content past the cache", got)
	}
	if resp.CacheMetadata.ContentsCount != 3 {
		t.Errorf("response ContentsCount = %d, want 3", resp.CacheMetadata.ContentsCount)
	}
}

// TestGenerateContent_ContextCacheRequestShape checks what the provider
// receives when a cache is created and then used: the cache holds the prefix,
// the instruction and the tools, and the request that uses it carries none of
// them, only the question.
func TestGenerateContent_ContextCacheRequestShape(t *testing.T) {
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{
		TTL:               45 * time.Minute,
		CreateHTTPOptions: &genai.HTTPOptions{Headers: http.Header{"X-Create-Test": []string{"yes"}}},
	})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

	generateOnce(t, cacheTestContext(t), m, req)

	if len(api.creates) != 1 {
		t.Fatalf("caches created = %d, want 1", len(api.creates))
	}
	create := api.creates[0]
	if got, want := create["ttl"], "2700s"; got != want {
		t.Errorf("cache ttl = %v, want %v", got, want)
	}
	if got, want := create["model"], "models/test-model"; got != want {
		t.Errorf("cache model = %v, want %v", got, want)
	}
	if got := contentTexts(create["contents"]); !cmp.Equal(got, []string{"first question", "first answer"}) {
		t.Errorf("cached contents = %q, want the finished exchange", got)
	}
	if _, ok := create["systemInstruction"]; !ok {
		t.Error("cache has no systemInstruction, want the request's")
	}
	if _, ok := create["tools"]; !ok {
		t.Error("cache has no tools, want the request's")
	}
	headers := api.createHeaders[0]
	if got := headers.Get("X-Create-Test"); got != "yes" {
		t.Errorf("CreateHTTPOptions header on the create call = %q, want %q", got, "yes")
	}
	// genai appends its own identifier after ADK's, as on model calls.
	if got := headers.Get("x-goog-api-client"); !strings.HasPrefix(got, m.versionHeaderValue) {
		t.Errorf("x-goog-api-client on the create call = %q, want it to start with ADK's %q", got, m.versionHeaderValue)
	}

	sent := api.generates[0]
	if got, want := sent["cachedContent"], "cachedContents/1"; got != want {
		t.Errorf("request cachedContent = %v, want %v", got, want)
	}
	for _, field := range []string{"systemInstruction", "tools", "toolConfig"} {
		if _, ok := sent[field]; ok {
			t.Errorf("request carries %s alongside a cache, which the API rejects", field)
		}
	}
	if got := contentTexts(sent["contents"]); !cmp.Equal(got, []string{"second question"}) {
		t.Errorf("request contents = %q, want only the uncached question", got)
	}
}

// TestGenerateContent_ContextCacheDefaults checks the TTL a zero config
// creates a cache with, and the expiry recorded when the provider reports
// none.
func TestGenerateContent_ContextCacheDefaults(t *testing.T) {
	api := &fakeCacheAPI{}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

	resp := generateOnce(t, cacheTestContext(t), m, req)

	if len(api.creates) != 1 {
		t.Fatalf("caches created = %d, want 1", len(api.creates))
	}
	if got, want := api.creates[0]["ttl"], "1800s"; got != want {
		t.Errorf("cache ttl = %v, want the default %v", got, want)
	}
	if got, want := resp.CacheMetadata.ExpireTime, cacheTestNow.Add(30*time.Minute); !got.Equal(want) {
		t.Errorf("ExpireTime = %v, want the creation time plus the TTL, %v", got, want)
	}
}

// TestGenerateContent_ContextCacheLeavesSharedOptions checks that the options
// a request's config shares with the caller are not written to. genai adds a
// header map to options that have none, and ADK sets its tracking headers in
// the map the options have.
func TestGenerateContent_ContextCacheLeavesSharedOptions(t *testing.T) {
	tests := []struct {
		name    string
		headers http.Header
	}{
		{name: "no header map"},
		{name: "a header map of the caller's", headers: http.Header{"X-Caller": []string{"yes"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
			m := newCacheTestModel(t, api, "test-model")
			timeout := 10 * time.Second
			shared := &genai.HTTPOptions{Timeout: &timeout, Headers: tt.headers}
			want := tt.headers.Clone()
			req := cacheTestRequest(&model.ContextCacheConfig{CreateHTTPOptions: shared})
			req.PreviousPromptTokenCount = 5000
			req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

			generateOnce(t, cacheTestContext(t), m, req)

			if len(api.creates) != 1 {
				t.Fatal("no cache was created, so this test checks nothing")
			}
			if diff := cmp.Diff(want, shared.Headers); diff != "" {
				t.Errorf("shared CreateHTTPOptions.Headers changed by a create (-want +got):\n%s", diff)
			}
			if got := api.createHeaders[0].Get("X-Caller"); tt.headers != nil && got != "yes" {
				t.Errorf("caller's header on the create call = %q, want it sent", got)
			}
		})
	}
}

// TestGenerateContent_ContextCacheFailedCall checks the caches a failed model
// call leaves behind. A cache it created is recorded nowhere, so it is
// deleted. A cache it was replacing is still named by the session, so it is
// kept.
func TestGenerateContent_ContextCacheFailedCall(t *testing.T) {
	tests := []struct {
		name      string
		old       func(fingerprint string) *model.CacheMetadata
		edit      func(*model.LLMRequest)
		wantCalls []string
	}{
		{
			name: "new cache is deleted",
			old: func(fp string) *model.CacheMetadata {
				return &model.CacheMetadata{Fingerprint: fp, ContentsCount: 2}
			},
			wantCalls: []string{"create", "generate", "delete cachedContents/1"},
		},
		{
			name: "replaced cache is kept and its replacement deleted",
			old: func(fp string) *model.CacheMetadata {
				return &model.CacheMetadata{CacheName: "cachedContents/old", ExpireTime: cacheTestNow, InvocationsUsed: 1, Fingerprint: fp, ContentsCount: 2}
			},
			wantCalls: []string{"create", "generate", "delete cachedContents/1"},
		},
		{
			name: "cache of a changed prefix is kept",
			old: func(fp string) *model.CacheMetadata {
				return &model.CacheMetadata{CacheName: "cachedContents/old", ExpireTime: cacheTestNow.Add(time.Hour), InvocationsUsed: 1, Fingerprint: fp, ContentsCount: 2}
			},
			edit: func(r *model.LLMRequest) {
				r.Config.SystemInstruction = genai.NewContentFromText("You are terse.", genai.RoleUser)
			},
			wantCalls: []string{"generate"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &fakeCacheAPI{generateStatus: http.StatusInternalServerError, expireTime: cacheTestNow.Add(time.Hour)}
			m := newCacheTestModel(t, api, "test-model")
			req := cacheTestRequest(&model.ContextCacheConfig{})
			req.PreviousPromptTokenCount = 5000
			req.CacheMetadata = tt.old(fingerprintOf(m, cacheTestRequest(nil), 2))
			if tt.edit != nil {
				tt.edit(req)
			}

			if _, err := generate(cacheTestContext(t), m, req); err == nil {
				t.Fatal("GenerateContent() error = nil, want the model's error")
			}
			if diff := cmp.Diff(tt.wantCalls, api.calls); diff != "" {
				t.Errorf("API calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestGenerateContent_ContextCacheConsumerStops checks that a consumer that
// stops reading before the complete response leaves no cache behind: it never
// received the metadata that names the new cache.
func TestGenerateContent_ContextCacheConsumerStops(t *testing.T) {
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

	for resp, err := range m.GenerateContent(cacheTestContext(t), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if !resp.Partial {
			t.Fatal("the first response was complete, so the consumer did not stop early")
		}
		break
	}

	if diff := cmp.Diff([]string{"create", "generate", "delete cachedContents/1"}, api.calls); diff != "" {
		t.Errorf("API calls mismatch (-want +got):\n%s", diff)
	}
	if len(api.deleteHeaders) == 1 {
		if got := api.deleteHeaders[0].Get("x-goog-api-client"); !strings.HasPrefix(got, m.versionHeaderValue) {
			t.Errorf("x-goog-api-client on the delete call = %q, want it to start with ADK's %q", got, m.versionHeaderValue)
		}
	}
}

// cacheEvents is a session's events for settling a ledger.
type cacheEvents []*session.Event

func (e cacheEvents) All() iter.Seq[*session.Event] { return slices.Values(e) }
func (e cacheEvents) Len() int                      { return len(e) }
func (e cacheEvents) At(i int) *session.Event       { return e[i] }

// loadEvents returns a loader that yields events, as a session store would.
func loadEvents(events session.Events) func(context.Context) (session.Events, error) {
	return func(context.Context) (session.Events, error) { return events, nil }
}

// storedRecord returns a stored model event of one agent recording metadata.
func storedRecord(metadata *model.CacheMetadata) *session.Event {
	return &session.Event{Author: "assistant", LLMResponse: model.LLMResponse{CacheMetadata: metadata}}
}

// TestGenerateContent_ContextCacheLedger checks that under a runner the model
// deletes nothing itself: the caches a call created or stopped using go into
// the ledger, and settling it against what the session stored keeps those a
// stored record names and deletes the rest.
func TestGenerateContent_ContextCacheLedger(t *testing.T) {
	old := func(fp string) *model.CacheMetadata {
		return &model.CacheMetadata{CacheName: "cachedContents/old", ExpireTime: cacheTestNow, InvocationsUsed: 1, Fingerprint: fp, ContentsCount: 2}
	}
	tests := []struct {
		name string
		// metadata is the agent's latest record before the call.
		metadata func(fp string) *model.CacheMetadata
		// emptyReply makes the model answer with no content.
		emptyReply bool
		// stored is what the session stored for the call, given the
		// response's metadata. Nil stores nothing.
		stored      func(resp, before *model.CacheMetadata) []*session.Event
		wantDeleted []string
	}{
		{
			name:     "a created cache the session stored is kept",
			metadata: func(fp string) *model.CacheMetadata { return &model.CacheMetadata{Fingerprint: fp, ContentsCount: 2} },
			stored: func(resp, _ *model.CacheMetadata) []*session.Event {
				return []*session.Event{storedRecord(resp)}
			},
		},
		{
			name:        "a created cache the session did not store is deleted",
			metadata:    func(fp string) *model.CacheMetadata { return &model.CacheMetadata{Fingerprint: fp, ContentsCount: 2} },
			wantDeleted: []string{"cachedContents/1"},
		},
		{
			name:     "a replaced cache is deleted once its replacement is stored",
			metadata: old,
			stored: func(resp, before *model.CacheMetadata) []*session.Event {
				return []*session.Event{storedRecord(before), storedRecord(resp)}
			},
			wantDeleted: []string{"cachedContents/old"},
		},
		{
			name:     "a replaced cache is kept while the session still names it",
			metadata: old,
			stored: func(_, before *model.CacheMetadata) []*session.Event {
				return []*session.Event{storedRecord(before)}
			},
			wantDeleted: []string{"cachedContents/1"},
		},
		{
			// The flow drops an empty reply, and a callback may fill it in:
			// either way the stored events alone decide.
			name:       "an empty reply a callback filled in is stored and kept",
			metadata:   old,
			emptyReply: true,
			stored: func(resp, before *model.CacheMetadata) []*session.Event {
				return []*session.Event{storedRecord(before), storedRecord(resp)}
			},
			wantDeleted: []string{"cachedContents/old"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &fakeCacheAPI{emptyReply: tt.emptyReply, expireTime: cacheTestNow.Add(time.Hour)}
			m := newCacheTestModel(t, api, "test-model")
			req := cacheTestRequest(&model.ContextCacheConfig{})
			req.PreviousPromptTokenCount = 5000
			before := tt.metadata(fingerprintOf(m, cacheTestRequest(nil), 2))
			req.CacheMetadata = before
			ledger := llminternal.NewContextCacheLedger()
			ctx := llminternal.WithContextCacheLedger(cacheTestContext(t), ledger)

			resp := generateOnce(t, ctx, m, req)

			if diff := cmp.Diff([]string{"create", "generate"}, api.calls); diff != "" {
				t.Fatalf("API calls before settling mismatch (-want +got):\n%s", diff)
			}
			var events cacheEvents
			if tt.stored != nil {
				events = tt.stored(resp.CacheMetadata, before)
			}
			ledger.Settle(ctx, loadEvents(events))
			var deleted []string
			for _, call := range api.calls[2:] {
				deleted = append(deleted, strings.TrimPrefix(call, "delete "))
			}
			if diff := cmp.Diff(tt.wantDeleted, deleted); diff != "" {
				t.Errorf("caches deleted on settling mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestGenerateContent_ContextCacheMissingCache checks that a request naming a
// cache the provider no longer holds is sent again without the cache, and that
// its fingerprint is kept so the next request can create a cache at once.
func TestGenerateContent_ContextCacheMissingCache(t *testing.T) {
	tests := []struct {
		name        string
		stream      bool
		goneStatus  int
		goneMessage string
		wantRetry   bool
	}{
		{name: "not found", goneStatus: http.StatusNotFound, wantRetry: true},
		{name: "not found while streaming", stream: true, goneStatus: http.StatusNotFound, wantRetry: true},
		{name: "forbidden naming the cache", goneStatus: http.StatusForbidden, wantRetry: true},
		// A key without access to the model fails the same way without the
		// cache, so sending the request again would only fail twice.
		{name: "forbidden for another reason", goneStatus: http.StatusForbidden, goneMessage: "Permission denied on resource project", wantRetry: false},
		{name: "another error", goneStatus: http.StatusBadRequest, wantRetry: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &fakeCacheAPI{gone: map[string]bool{"cachedContents/gone": true}, goneStatus: tt.goneStatus, goneMessage: tt.goneMessage}
			m := newCacheTestModel(t, api, "test-model")
			fp := fingerprintOf(m, cacheTestRequest(nil), 2)
			req := cacheTestRequest(&model.ContextCacheConfig{})
			req.PreviousPromptTokenCount = 5000
			req.CacheMetadata = &model.CacheMetadata{CacheName: "cachedContents/gone", ExpireTime: cacheTestNow.Add(time.Hour), InvocationsUsed: 1, Fingerprint: fp, ContentsCount: 2}

			var last *model.LLMResponse
			var lastErr error
			for resp, err := range m.GenerateContent(cacheTestContext(t), req, tt.stream) {
				if resp != nil && !resp.Partial {
					last = resp
				}
				lastErr = err
			}

			if !tt.wantRetry {
				if lastErr == nil {
					t.Fatal("GenerateContent() error = nil, want the provider's error")
				}
				if len(api.generates) != 1 {
					t.Errorf("generations = %d, want 1 with no retry", len(api.generates))
				}
				return
			}
			if lastErr != nil {
				t.Fatalf("GenerateContent() error = %v, want the retry to succeed", lastErr)
			}
			if len(api.generates) != 2 {
				t.Fatalf("generations = %d, want 2", len(api.generates))
			}
			if _, ok := api.generates[1]["cachedContent"]; ok {
				t.Error("the retry named a cache, want it sent whole")
			}
			if _, ok := api.generates[1]["systemInstruction"]; !ok {
				t.Error("the retry has no systemInstruction, want the full request")
			}
			if diff := cmp.Diff(&model.CacheMetadata{Fingerprint: fp, ContentsCount: 2}, last.CacheMetadata); diff != "" {
				t.Errorf("response CacheMetadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestGenerateContent_ContextCacheMissingCacheAfterOutput checks that a stream
// that fails with a missing-cache error after it has already produced output is
// not sent again: the consumer has seen the first attempt's text, so a second
// attempt would repeat it. The error goes to the caller.
func TestGenerateContent_ContextCacheMissingCacheAfterOutput(t *testing.T) {
	api := &fakeCacheAPI{gone: map[string]bool{"cachedContents/gone": true}, goneAfterChunk: true}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{CacheName: "cachedContents/gone", ExpireTime: cacheTestNow.Add(time.Hour), InvocationsUsed: 1, Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

	var chunks int
	var lastErr error
	for resp, err := range m.GenerateContent(cacheTestContext(t), req, true) {
		if err != nil {
			lastErr = err
			continue
		}
		if resp.Partial {
			chunks++
		}
	}

	var apiErr genai.APIError
	if !errors.As(lastErr, &apiErr) || apiErr.Code != http.StatusNotFound {
		t.Errorf("error = %v, want the provider's 404 passed to the caller", lastErr)
	}
	if chunks != 1 {
		t.Errorf("partial chunks yielded = %d, want the 1 sent before the failure", chunks)
	}
	if diff := cmp.Diff([]string{"generate"}, api.calls); diff != "" {
		t.Errorf("API calls mismatch (-want +got):\n%s", diff)
	}
}

// TestGenerateContent_ContextCacheMissingCacheCleanup checks that the cache the
// provider reported missing is deleted, since the provider may only have
// reported it unreachable: straight away without a ledger, and once the
// invocation settles under one.
func TestGenerateContent_ContextCacheMissingCacheCleanup(t *testing.T) {
	newRequest := func(t *testing.T, api *fakeCacheAPI) (*geminiModel, *model.LLMRequest) {
		m := newCacheTestModel(t, api, "test-model")
		req := cacheTestRequest(&model.ContextCacheConfig{})
		req.PreviousPromptTokenCount = 5000
		req.CacheMetadata = &model.CacheMetadata{CacheName: "cachedContents/gone", ExpireTime: cacheTestNow.Add(time.Hour), InvocationsUsed: 1, Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}
		return m, req
	}

	t.Run("without a ledger", func(t *testing.T) {
		api := &fakeCacheAPI{gone: map[string]bool{"cachedContents/gone": true}}
		m, req := newRequest(t, api)

		generateOnce(t, cacheTestContext(t), m, req)

		if diff := cmp.Diff([]string{"generate", "delete cachedContents/gone", "generate"}, api.calls); diff != "" {
			t.Errorf("API calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("under a ledger", func(t *testing.T) {
		api := &fakeCacheAPI{gone: map[string]bool{"cachedContents/gone": true}}
		m, req := newRequest(t, api)
		ledger := llminternal.NewContextCacheLedger()
		ctx := llminternal.WithContextCacheLedger(cacheTestContext(t), ledger)

		resp := generateOnce(t, ctx, m, req)

		if diff := cmp.Diff([]string{"generate", "generate"}, api.calls); diff != "" {
			t.Fatalf("API calls before settling mismatch (-want +got):\n%s", diff)
		}
		// The retry recorded fingerprint-only metadata, so nothing stored names
		// the missing cache.
		ledger.Settle(ctx, loadEvents(cacheEvents{storedRecord(resp.CacheMetadata)}))
		if diff := cmp.Diff([]string{"generate", "generate", "delete cachedContents/gone"}, api.calls); diff != "" {
			t.Errorf("API calls after settling mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestGenerateContent_ContextCacheDefaultIntervals checks the number of
// invocations a zero config lets one cache serve.
func TestGenerateContent_ContextCacheDefaultIntervals(t *testing.T) {
	tests := []struct {
		name            string
		invocationsUsed int
		wantCreates     int
	}{
		{name: "the tenth invocation reuses the cache", invocationsUsed: 10, wantCreates: 0},
		{name: "the eleventh replaces it", invocationsUsed: 11, wantCreates: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
			m := newCacheTestModel(t, api, "test-model")
			req := cacheTestRequest(&model.ContextCacheConfig{})
			req.PreviousPromptTokenCount = 5000
			req.CacheMetadata = &model.CacheMetadata{
				CacheName:       "cachedContents/old",
				ExpireTime:      cacheTestNow.Add(time.Hour),
				InvocationsUsed: tt.invocationsUsed,
				Fingerprint:     fingerprintOf(m, cacheTestRequest(nil), 2),
				ContentsCount:   2,
			}

			generateOnce(t, cacheTestContext(t), m, req)

			if len(api.creates) != tt.wantCreates {
				t.Errorf("caches created = %d, want %d", len(api.creates), tt.wantCreates)
			}
		})
	}
}

// TestGenerateContent_ContextCacheCreateBody checks that a cache creation gets
// the request-body handling a generation does: the client's extras provider,
// which genai drops when it merges per-call options, and the go-genai#931
// workaround for parts that carry only a thought signature.
func TestGenerateContent_ContextCacheCreateBody(t *testing.T) {
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModelWithProvider(t, api, "test-model", func(body map[string]any) map[string]any {
		body["adkTestMarker"] = "client"
		return body
	})
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.Contents[1] = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "", ThoughtSignature: []byte("signature")}}}
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, req, 2), ContentsCount: 2}

	generateOnce(t, cacheTestContext(t), m, req)

	if len(api.creates) != 1 {
		t.Fatalf("caches created = %d, want 1", len(api.creates))
	}
	create := api.creates[0]
	if got := create["adkTestMarker"]; got != "client" {
		t.Errorf("create body adkTestMarker = %v, want the client provider's %q", got, "client")
	}
	contents, _ := create["contents"].([]any)
	if len(contents) != 2 {
		t.Fatalf("cached contents = %d, want 2", len(contents))
	}
	part, _ := contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if _, ok := part["text"]; !ok {
		t.Errorf("cached signature-only part = %v, want an empty text field", part)
	}
}

// TestGenerateContent_ContextCacheCleanupSurvivesCancellation checks that a
// cache is still deleted when the request's context is canceled, which is
// when a consumer abandons a stream.
func TestGenerateContent_ContextCacheCleanupSurvivesCancellation(t *testing.T) {
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

	ctx, cancel := context.WithCancel(cacheTestContext(t))
	defer cancel()
	for resp, err := range m.GenerateContent(ctx, req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if !resp.Partial {
			t.Fatal("the first response was complete, so the consumer did not stop early")
		}
		cancel()
		break
	}

	if diff := cmp.Diff([]string{"create", "generate", "delete cachedContents/1"}, api.calls); diff != "" {
		t.Errorf("API calls mismatch (-want +got):\n%s", diff)
	}
}

// TestGenerateContent_ContextCacheKeepsExplicitCache checks that a request
// already naming a cache is sent as it is, rather than having that cache
// replaced.
func TestGenerateContent_ContextCacheKeepsExplicitCache(t *testing.T) {
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.Config.CachedContent = "cachedContents/mine"
	req.Config.SystemInstruction = nil
	req.Config.Tools = nil
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{CacheName: "cachedContents/old", ExpireTime: cacheTestNow, InvocationsUsed: 1, Fingerprint: "x", ContentsCount: 2}

	resp := generateOnce(t, cacheTestContext(t), m, req)

	if diff := cmp.Diff([]string{"generate"}, api.calls); diff != "" {
		t.Errorf("API calls mismatch (-want +got):\n%s", diff)
	}
	if got, want := api.generates[0]["cachedContent"], "cachedContents/mine"; got != want {
		t.Errorf("request cachedContent = %v, want the caller's %v", got, want)
	}
	if resp.CacheMetadata != nil {
		t.Errorf("CacheMetadata = %+v, want none for a cache ADK does not manage", resp.CacheMetadata)
	}
}

// TestGenerateContent_ContextCacheKeepsCallerRequest checks that the caller's
// request still holds everything after a cached call, since the flow and its
// telemetry read it afterwards.
func TestGenerateContent_ContextCacheKeepsCallerRequest(t *testing.T) {
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

	generateOnce(t, cacheTestContext(t), m, req)

	if _, ok := api.generates[0]["cachedContent"]; !ok {
		t.Fatal("request did not use a cache, so this test checks nothing")
	}
	if req.Config.SystemInstruction == nil || len(req.Config.Tools) != 1 || req.Config.CachedContent != "" {
		t.Errorf("caller's config changed: SystemInstruction=%v Tools=%d CachedContent=%q", req.Config.SystemInstruction, len(req.Config.Tools), req.Config.CachedContent)
	}
	if len(req.Contents) != 3 {
		t.Errorf("caller's contents = %d, want 3", len(req.Contents))
	}
}

func TestGenerateContent_ContextCacheModelMinimum(t *testing.T) {
	tests := []struct {
		name         string
		modelName    string
		promptTokens int32
		wantCreates  int
	}{
		{name: "gemini 2.5 one under its minimum", modelName: "gemini-2.5-flash", promptTokens: 2047, wantCreates: 0},
		{name: "gemini 2.5 at its minimum", modelName: "gemini-2.5-flash", promptTokens: 2048, wantCreates: 1},
		{name: "gemini 3 one under its minimum", modelName: "gemini-3-pro-preview", promptTokens: 4095, wantCreates: 0},
		{name: "gemini 3 at its minimum", modelName: "gemini-3-pro-preview", promptTokens: 4096, wantCreates: 1},
		{name: "gemini 3.5 one under the gemini 3 minimum", modelName: "gemini-3.5-flash", promptTokens: 4095, wantCreates: 0},
		{name: "resource name of a gemini 2.5 model", modelName: "projects/p/locations/l/publishers/google/models/gemini-2.5-pro", promptTokens: 2047, wantCreates: 0},
		{name: "unknown family is left to the server", modelName: "tuned-model", promptTokens: 1, wantCreates: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
			m := newCacheTestModel(t, api, tt.modelName)
			// Only the question follows the prefix, and it has no text, so
			// the prefix's estimated share of the earlier prompt is all of
			// it and the estimate equals promptTokens.
			req := cacheTestRequest(&model.ContextCacheConfig{})
			req.Contents = []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("x")}}}}}
			req.PreviousPromptTokenCount = tt.promptTokens
			req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, req, 0)}

			generateOnce(t, cacheTestContext(t), m, req)

			if len(api.creates) != tt.wantCreates {
				t.Errorf("caches created = %d, want %d", len(api.creates), tt.wantCreates)
			}
		})
	}
}

// TestEstimatePrefixTokens checks that the earlier prompt's count is scaled by
// the prefix's share of the text, so a long conversation whose prefix is small
// is not taken to clear the model's minimum.
func TestEstimatePrefixTokens(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText(strings.Repeat("a", 100), genai.RoleUser),
			genai.NewContentFromText(strings.Repeat("b", 100), genai.RoleModel),
			genai.NewContentFromText(strings.Repeat("c", 200), genai.RoleUser),
		},
		Config:                   &genai.GenerateContentConfig{},
		PreviousPromptTokenCount: 4000,
	}
	noText := &model.LLMRequest{
		Contents:                 []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{InlineData: &genai.Blob{Data: []byte("x")}}}}},
		Config:                   &genai.GenerateContentConfig{},
		PreviousPromptTokenCount: 4000,
	}
	tests := []struct {
		name  string
		req   *model.LLMRequest
		count int
		want  int32
	}{
		{name: "no prefix", req: req, count: 0, want: 0},
		{name: "a quarter of the text", req: req, count: 1, want: 1000},
		{name: "half of the text", req: req, count: 2, want: 2000},
		{name: "all of the text", req: req, count: 3, want: 4000},
		{name: "no text keeps the real count", req: noText, count: 0, want: 4000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := estimatePrefixTokens(tt.req, tt.count); got != tt.want {
				t.Errorf("estimatePrefixTokens(count=%d) = %d, want %d", tt.count, got, tt.want)
			}
		})
	}
}

func TestApplyCache(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("a", genai.RoleUser),
		genai.NewContentFromText("b", genai.RoleModel),
	}
	tests := []struct {
		name  string
		count int
		want  int
	}{
		{name: "nothing cached", count: 0, want: 2},
		{name: "prefix cached", count: 1, want: 1},
		{name: "every content cached keeps the last", count: 2, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := applyCache(&model.LLMRequest{Contents: contents, Config: &genai.GenerateContentConfig{}}, "cachedContents/1", tt.count)
			if len(got.Contents) != tt.want {
				t.Errorf("contents sent = %d, want %d", len(got.Contents), tt.want)
			}
			if got.Config.CachedContent != "cachedContents/1" {
				t.Errorf("Config.CachedContent = %q, want cachedContents/1", got.Config.CachedContent)
			}
		})
	}
}

// TestGenerateContent_ContextCacheStreaming checks that only the complete
// response of a stream carries cache metadata, since partial chunks are not
// stored.
func TestGenerateContent_ContextCacheStreaming(t *testing.T) {
	api := &fakeCacheAPI{}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})

	var partial, complete int
	for resp, err := range m.GenerateContent(cacheTestContext(t), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if resp.Partial {
			partial++
			if resp.CacheMetadata != nil {
				t.Errorf("partial response carries CacheMetadata %v, want none", resp.CacheMetadata)
			}
			continue
		}
		complete++
		if resp.CacheMetadata == nil || resp.CacheMetadata.ContentsCount != 2 {
			t.Errorf("complete response CacheMetadata = %v, want fingerprint-only for 2 contents", resp.CacheMetadata)
		}
	}
	if partial == 0 || complete == 0 {
		t.Fatalf("stream had %d partial and %d complete responses, want some of each", partial, complete)
	}
}

// TestGenerateContent_NoContextCache checks that a request without a cache
// config touches no cache and carries no metadata.
func TestGenerateContent_NoContextCache(t *testing.T) {
	api := &fakeCacheAPI{}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(nil)
	req.PreviousPromptTokenCount = 5000

	resp := generateOnce(t, cacheTestContext(t), m, req)

	if diff := cmp.Diff([]string{"generate"}, api.calls); diff != "" {
		t.Errorf("API calls mismatch (-want +got):\n%s", diff)
	}
	if resp.CacheMetadata != nil {
		t.Errorf("CacheMetadata = %v, want nil", resp.CacheMetadata)
	}
}

// contentTexts returns the text of each content in a decoded request body.
func contentTexts(v any) []string {
	contents, _ := v.([]any)
	var out []string
	for _, c := range contents {
		cm, _ := c.(map[string]any)
		parts, _ := cm["parts"].([]any)
		var text strings.Builder
		for _, p := range parts {
			pm, _ := p.(map[string]any)
			s, _ := pm["text"].(string)
			text.WriteString(s)
		}
		out = append(out, text.String())
	}
	return out
}
