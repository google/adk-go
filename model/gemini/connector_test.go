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

package gemini_test

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
)

type clientProvider interface {
	Client() *genai.Client
}

type variantProvider interface {
	GetGoogleLLMVariant() genai.Backend
}

// clearGenAIEnv removes every environment variable genai reads to pick a
// backend and its credentials, so a test sees a machine without credentials.
func clearGenAIEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"GOOGLE_API_KEY",
		"GEMINI_API_KEY",
		"GOOGLE_GENAI_USE_VERTEXAI",
		"GOOGLE_GENAI_USE_ENTERPRISE",
		"GOOGLE_CLOUD_PROJECT",
		"GOOGLE_CLOUD_LOCATION",
		"GOOGLE_CLOUD_REGION",
	} {
		t.Setenv(name, "")
	}
}

func TestRegisteredGemini_ResolvesWithoutCredentials(t *testing.T) {
	clearGenAIEnv(t)
	ctx := t.Context()

	m, err := model.NewLLM(ctx, "gemini-3.5-flash")
	if err != nil {
		t.Fatalf(`model.NewLLM("gemini-3.5-flash") error = %v, want nil`, err)
	}

	if !gemini.IsGeminiConnector(m) {
		t.Errorf("IsGeminiConnector(%T) = false, want true", m)
	}
	if got, want := m.Name(), "gemini-3.5-flash"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
	req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)}}
	next, stop := iter.Pull2(m.GenerateContent(ctx, req, false))
	defer stop()
	resp, genErr, ok := next()
	if !ok {
		t.Fatal("GenerateContent() yielded nothing, want one error")
	}
	if resp != nil || genErr == nil {
		t.Errorf("GenerateContent() = (%v, %v), want (nil, an error)", resp, genErr)
	}
	nextStream, stopStream := iter.Pull2(m.GenerateContent(ctx, req, true))
	defer stopStream()
	resp, genErr, ok = nextStream()
	if !ok {
		t.Fatal("GenerateContent(stream) yielded nothing, want one error")
	}
	if resp != nil || genErr == nil {
		t.Errorf("GenerateContent(stream) = (%v, %v), want (nil, an error)", resp, genErr)
	}
	cachedReq := &model.LLMRequest{
		Contents:    []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
		CacheConfig: &model.ContextCacheConfig{},
	}
	nextCached, stopCached := iter.Pull2(m.GenerateContent(ctx, cachedReq, false))
	defer stopCached()
	resp, genErr, ok = nextCached()
	if !ok {
		t.Fatal("GenerateContent(cached) yielded nothing, want one error")
	}
	if resp != nil || genErr == nil {
		t.Errorf("GenerateContent(cached) = (%v, %v), want (nil, an error)", resp, genErr)
	}
	cp, ok := m.(clientProvider)
	if !ok {
		t.Fatalf("%T has no Client method", m)
	}
	if c := cp.Client(); c != nil {
		t.Errorf("Client() = %v, want nil when the client cannot be created", c)
	}
	if got := m.(variantProvider).GetGoogleLLMVariant(); got != genai.BackendUnspecified {
		t.Errorf("GetGoogleLLMVariant() = %v, want BackendUnspecified when the client cannot be created", got)
	}
}

func TestRegisteredGemini_ClientFollowsEnvironmentOnFirstUse(t *testing.T) {
	clearGenAIEnv(t)
	ctx := t.Context()

	// Each accessor is the first use of its own model.
	forVariant, err := model.NewLLM(ctx, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf(`model.NewLLM("gemini-2.5-flash") error = %v, want nil`, err)
	}
	forClient, err := model.NewLLM(ctx, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf(`model.NewLLM("gemini-2.5-flash") error = %v, want nil`, err)
	}
	// The key appears after resolution and before first use.
	t.Setenv("GOOGLE_API_KEY", "k")

	if got := forVariant.(variantProvider).GetGoogleLLMVariant(); got != genai.BackendGeminiAPI {
		t.Errorf("GetGoogleLLMVariant() = %v, want BackendGeminiAPI", got)
	}
	if c := forClient.(clientProvider).Client(); c == nil {
		t.Error("Client() = nil, want the client created from the environment")
	}
}

func TestRegisteredGemini_ContextCacheCreatesClientOnFirstUse(t *testing.T) {
	clearGenAIEnv(t)
	ctx := t.Context()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10}}`)
	}))
	t.Cleanup(srv.Close)

	m, err := model.NewLLM(ctx, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf(`model.NewLLM("gemini-2.5-flash") error = %v, want nil`, err)
	}
	t.Setenv("GOOGLE_API_KEY", "k")
	t.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("q1", genai.RoleUser),
			genai.NewContentFromText("a1", genai.RoleModel),
			genai.NewContentFromText("q2", genai.RoleUser),
		},
		CacheConfig: &model.ContextCacheConfig{},
	}
	next, stop := iter.Pull2(m.GenerateContent(ctx, req, false))
	defer stop()
	resp, genErr, ok := next()
	if !ok || genErr != nil {
		t.Fatalf("GenerateContent(cached) = (%v, %v, %v), want response and nil error", resp, genErr, ok)
	}
	if resp == nil || resp.CacheMetadata == nil || resp.CacheMetadata.Fingerprint == "" {
		t.Fatalf("GenerateContent(cached) CacheMetadata = %+v, want non-empty Fingerprint", resp.CacheMetadata)
	}
}

func TestRegisteredGemini_FailedClientCreationIsRetried(t *testing.T) {
	clearGenAIEnv(t)
	ctx := t.Context()
	m, err := model.NewLLM(ctx, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf(`model.NewLLM("gemini-2.5-flash") error = %v, want nil`, err)
	}
	req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)}}
	next, stop := iter.Pull2(m.GenerateContent(ctx, req, false))
	defer stop()
	if _, genErr, _ := next(); genErr == nil {
		t.Fatal("GenerateContent() without credentials = nil error, want an error")
	}
	if c := m.(clientProvider).Client(); c != nil {
		t.Errorf("Client() without credentials = %v, want nil", c)
	}
	if got := m.(variantProvider).GetGoogleLLMVariant(); got != genai.BackendUnspecified {
		t.Errorf("GetGoogleLLMVariant() without credentials = %v, want BackendUnspecified", got)
	}

	t.Setenv("GOOGLE_API_KEY", "k")

	if c := m.(clientProvider).Client(); c == nil {
		t.Error("Client() after the key is set = nil, want the client created from the environment")
	}
	if got := m.(variantProvider).GetGoogleLLMVariant(); got != genai.BackendGeminiAPI {
		t.Errorf("GetGoogleLLMVariant() after the key is set = %v, want BackendGeminiAPI", got)
	}
}

// Run with -race: the callers race to create the client, and every one must
// get the single client that the model keeps.
func TestRegisteredGemini_ConcurrentFirstUseSharesOneClient(t *testing.T) {
	clearGenAIEnv(t)
	t.Setenv("GOOGLE_API_KEY", "k")
	m, err := model.NewLLM(t.Context(), "gemini-2.5-flash")
	if err != nil {
		t.Fatalf(`model.NewLLM("gemini-2.5-flash") error = %v, want nil`, err)
	}
	cp := m.(clientProvider)

	const callers = 8
	clients := make([]*genai.Client, callers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range clients {
		wg.Go(func() {
			<-start
			clients[i] = cp.Client()
		})
	}
	close(start)
	wg.Wait()

	if clients[0] == nil {
		t.Fatal("Client() = nil, want the client created from the environment")
	}
	for i, c := range clients {
		if c != clients[0] {
			t.Errorf("Client() in caller %d = %p, want %p, the client caller 0 got", i, c, clients[0])
		}
	}
	if c := cp.Client(); c != clients[0] {
		t.Errorf("Client() after the first use = %p, want %p, the client the first use created", c, clients[0])
	}
}

func TestGeminiPatterns(t *testing.T) {
	clearGenAIEnv(t)
	tests := []struct {
		name    string
		resolve bool
	}{
		{name: "gemini-2.5-flash", resolve: true},
		{name: "gemma-4-31b-it", resolve: true},
		{name: "model-optimizer-x", resolve: true},
		{name: "projects/p/locations/l/endpoints/e", resolve: true},
		{name: "projects/p/locations/l/publishers/google/models/gemini-2.5-flash", resolve: true},
		{name: "GEMINI-2.5-FLASH", resolve: false},
		{name: "my-gemini-x", resolve: false},
		{name: "gemma-3-1b", resolve: false},
		{name: "projects/p/locations/l/publishers/google/models/claude-x", resolve: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := model.NewLLM(t.Context(), tt.name)

			if !tt.resolve {
				if err == nil || !strings.Contains(err.Error(), "model: no registered LLM matches") {
					t.Errorf("model.NewLLM(%q) error = %v, want no registered LLM matches", tt.name, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("model.NewLLM(%q) error = %v, want nil", tt.name, err)
			}
			if !gemini.IsGeminiConnector(m) {
				t.Errorf("IsGeminiConnector(model.NewLLM(%q)) = false, want true", tt.name)
			}
		})
	}
}

type otherLLM struct{}

func (otherLLM) Name() string { return "gemini-2.5-flash" }

func (otherLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(func(*model.LLMResponse, error) bool) {}
}

func TestIsGeminiConnector(t *testing.T) {
	clearGenAIEnv(t)
	ctx := t.Context()
	built, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{Backend: genai.BackendGeminiAPI, APIKey: "k"})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}
	resolved, err := model.NewLLM(ctx, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf("model.NewLLM() error = %v", err)
	}

	tests := []struct {
		name string
		m    model.LLM
		want bool
	}{
		{name: "built by NewModel", m: built, want: true},
		{name: "resolved through the registry", m: resolved, want: true},
		{name: "nil", m: nil, want: false},
		{name: "another model with a Gemini name", m: otherLLM{}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := gemini.IsGeminiConnector(tt.m); got != tt.want {
				t.Errorf("IsGeminiConnector() = %v, want %v", got, tt.want)
			}
		})
	}
}
