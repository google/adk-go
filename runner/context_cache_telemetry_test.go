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
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/model"
)

// cacheSpanSummary renders each context cache span the exporter holds as one
// line: its name, cache_action or cache_name when set, and the status code of
// the provider's error when the span failed.
func cacheSpanSummary(exp *tracetest.InMemoryExporter) []string {
	var out []string
	for _, s := range exp.GetSpans() {
		if s.Name != "handle_context_caching" && s.Name != "create_cache" && s.Name != "delete_cache" {
			continue
		}
		line := s.Name
		for _, a := range s.Attributes {
			if a.Key == "cache_action" || (a.Key == "cache_name" && s.Name != "handle_context_caching") {
				line += " " + a.Value.AsString()
			}
		}
		if s.Status.Code == codes.Error {
			line += " failed: " + strings.SplitN(s.Status.Description, ",", 2)[0]
		}
		out = append(out, line)
	}
	return out
}

// TestRunner_ContextCacheSpans checks that, with caching on, the traces say
// which cache action each turn took and that a cache creation or deletion
// failed, while every turn still succeeds.
func TestRunner_ContextCacheSpans(t *testing.T) {
	tests := []struct {
		name string
		api  *cachingGeminiAPI
		cfg  *model.ContextCacheConfig
		want []string
	}{
		{
			name: "caches created and reused",
			api:  &cachingGeminiAPI{},
			cfg:  &model.ContextCacheConfig{},
			want: []string{
				"handle_context_caching fingerprint_only",
				"create_cache cachedContents/1",
				"handle_context_caching active_cache",
				"handle_context_caching active_cache",
			},
		},
		{
			// A free-tier key: the provider refuses every cache.
			name: "every creation refused",
			api:  &cachingGeminiAPI{createStatus: http.StatusTooManyRequests},
			cfg:  &model.ContextCacheConfig{},
			want: []string{
				"handle_context_caching fingerprint_only",
				"create_cache failed: Error 429",
				"handle_context_caching fingerprint_only",
				"create_cache failed: Error 429",
				"handle_context_caching fingerprint_only",
			},
		},
		{
			name: "deletion of a replaced cache refused",
			api:  &cachingGeminiAPI{deleteStatus: http.StatusInternalServerError},
			cfg:  &model.ContextCacheConfig{CacheIntervals: 1},
			want: []string{
				"handle_context_caching fingerprint_only",
				"create_cache cachedContents/1",
				"handle_context_caching active_cache",
				"create_cache cachedContents/2",
				"handle_context_caching active_cache",
				"delete_cache cachedContents/1 failed: Error 500",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			exp := tracetest.NewInMemoryExporter()
			telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
			r, _ := newCachingRunner(t, cachingAgent(t, tt.api, llmagent.Config{}), tt.cfg)

			for turn := 1; turn <= 3; turn++ {
				if err := runTurn(t, r, turn); err != nil {
					t.Fatalf("turn %d failed: %v", turn, err)
				}
			}

			if diff := cmp.Diff(tt.want, cacheSpanSummary(exp)); diff != "" {
				t.Errorf("context cache spans mismatch (-want +got):\n%s", diff)
			}
			// Each decision belongs to the model call it was made for.
			calls := map[string]bool{}
			for _, s := range exp.GetSpans() {
				if strings.HasPrefix(s.Name, "generate_content") {
					calls[s.SpanContext.SpanID().String()] = true
				}
			}
			for _, s := range exp.GetSpans() {
				if s.Name == "handle_context_caching" && !calls[s.Parent.SpanID().String()] {
					t.Errorf("handle_context_caching span parent %v is not a generate_content span", s.Parent.SpanID())
				}
			}
		})
	}
}

// TestRunner_ContextCacheOffEmitsNoSpans checks that a runner without a
// ContextCache config emits no context cache span.
func TestRunner_ContextCacheOffEmitsNoSpans(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	r, _ := newCachingRunner(t, cachingAgent(t, &cachingGeminiAPI{}, llmagent.Config{}), nil)

	for turn := 1; turn <= 3; turn++ {
		if err := runTurn(t, r, turn); err != nil {
			t.Fatalf("turn %d failed: %v", turn, err)
		}
	}

	if got := cacheSpanSummary(exp); len(got) != 0 {
		t.Errorf("context cache spans = %q, want none", got)
	}
	if len(exp.GetSpans()) == 0 {
		t.Fatal("no spans recorded at all, so this test checks nothing")
	}
}
