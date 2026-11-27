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
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/model"
)

// cacheSpan is what a context cache span recorded, reduced to what the tests
// compare.
type cacheSpan struct {
	Name  string
	Attrs map[string]any
	// Failed reports an error status. Error is the text of the span's first
	// error event, whether or not the span failed.
	Failed bool
	Error  string
}

// recordCacheSpans makes tracing record into an in-memory exporter until the
// test ends, and returns a func reading the context cache spans recorded so
// far, in the order they ended.
func recordCacheSpans(t *testing.T) func() []cacheSpan {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	return func() []cacheSpan {
		var out []cacheSpan
		for _, s := range exp.GetSpans() {
			switch s.Name {
			case "handle_context_caching", "create_cache", "delete_cache":
			default:
				continue
			}
			got := cacheSpan{Name: s.Name, Attrs: map[string]any{}, Failed: s.Status.Code == codes.Error}
			for _, a := range s.Attributes {
				// error.type comes from RecordError and is checked through
				// Failed.
				if a.Key != "error.type" {
					got.Attrs[string(a.Key)] = a.Value.AsInterface()
				}
			}
			for _, e := range s.Events {
				if e.Name == "exception" && got.Error == "" {
					for _, a := range e.Attributes {
						if a.Key == "exception.message" {
							got.Error = a.Value.AsString()
						}
					}
				}
			}
			out = append(out, got)
		}
		return out
	}
}

// TestGenerateContent_ContextCacheSpans checks the spans that say which cache
// action a request took and whether a cache creation or deletion failed.
func TestGenerateContent_ContextCacheSpans(t *testing.T) {
	expire := cacheTestNow.Add(30 * time.Minute)
	cfg := &model.ContextCacheConfig{TTL: time.Hour}
	active := func(fp string, expireTime time.Time) *model.CacheMetadata {
		return &model.CacheMetadata{CacheName: "cachedContents/old", ExpireTime: expireTime, InvocationsUsed: 1, CreatedAt: cacheTestNow.Add(-time.Minute), Fingerprint: fp, ContentsCount: 2}
	}
	fingerprintOnly := func(fp string) *model.CacheMetadata {
		return &model.CacheMetadata{Fingerprint: fp, ContentsCount: 2}
	}
	created := cacheSpan{Name: "create_cache", Attrs: map[string]any{"model": "test-model", "cache_contents_count": int64(2), "ttl_seconds": int64(3600), "cache_name": "cachedContents/1"}}
	refused := cacheSpan{Name: "create_cache", Attrs: map[string]any{"model": "test-model", "cache_contents_count": int64(2), "ttl_seconds": int64(3600)}, Failed: true, Error: "Error 400, Message: refused"}
	handle := func(action, name string) cacheSpan {
		s := cacheSpan{Name: "handle_context_caching", Attrs: map[string]any{"cache_action": action}}
		if name != "" {
			s.Attrs["cache_name"] = name
		}
		return s
	}

	tests := []struct {
		name         string
		createStatus int
		deleteStatus int
		createNoName bool
		// gone names caches the provider no longer holds.
		gone map[string]bool
		// off sends the request with caching off.
		off bool
		// ownCache is a cache the request names in its own config.
		ownCache string
		metadata func(fp string) *model.CacheMetadata
		want     []cacheSpan
	}{
		{
			name: "caching off emits no span",
			off:  true,
		},
		{
			name: "first request is sent without a cache",
			want: []cacheSpan{handle("fingerprint_only", "")},
		},
		{
			name:     "a request naming its own cache",
			ownCache: "cachedContents/mine",
			want:     []cacheSpan{handle("active_cache", "cachedContents/mine")},
		},
		{
			name:     "a created cache",
			metadata: fingerprintOnly,
			want:     []cacheSpan{created, handle("active_cache", "cachedContents/1")},
		},
		{
			name:     "a reused cache",
			metadata: func(fp string) *model.CacheMetadata { return active(fp, expire) },
			want:     []cacheSpan{handle("active_cache", "cachedContents/old")},
		},
		{
			name:         "a refused creation records the provider's error",
			createStatus: http.StatusTooManyRequests,
			metadata:     fingerprintOnly,
			want: []cacheSpan{
				{Name: "create_cache", Attrs: refused.Attrs, Failed: true, Error: "Error 429, Message: refused"},
				handle("fingerprint_only", ""),
			},
		},
		{
			name:         "a creation answered without a cache name",
			createNoName: true,
			metadata:     fingerprintOnly,
			want: []cacheSpan{
				{Name: "create_cache", Attrs: refused.Attrs, Failed: true, Error: "the cache service returned no cache name"},
				handle("fingerprint_only", ""),
			},
		},
		{
			name:     "a replaced cache is deleted",
			metadata: func(fp string) *model.CacheMetadata { return active(fp, cacheTestNow) },
			want: []cacheSpan{
				created,
				handle("active_cache", "cachedContents/1"),
				{Name: "delete_cache", Attrs: map[string]any{"cache_name": "cachedContents/old"}},
			},
		},
		{
			// Typical of a cache that expired: the provider dropped it first.
			name:     "deleting a cache the provider already dropped is not a failure",
			gone:     map[string]bool{"cachedContents/old": true},
			metadata: func(fp string) *model.CacheMetadata { return active(fp, cacheTestNow) },
			want: []cacheSpan{
				created,
				handle("active_cache", "cachedContents/1"),
				{Name: "delete_cache", Attrs: map[string]any{"cache_name": "cachedContents/old"}},
			},
		},
		{
			name:         "a refused deletion records the provider's error",
			deleteStatus: http.StatusInternalServerError,
			metadata:     func(fp string) *model.CacheMetadata { return active(fp, cacheTestNow) },
			want: []cacheSpan{
				created,
				handle("active_cache", "cachedContents/1"),
				{Name: "delete_cache", Attrs: map[string]any{"cache_name": "cachedContents/old"}, Failed: true, Error: "Error 500, Message: refused"},
			},
		},
		{
			name:         "a refused replacement and a refused deletion",
			createStatus: http.StatusBadRequest,
			deleteStatus: http.StatusInternalServerError,
			metadata:     func(fp string) *model.CacheMetadata { return active(fp, cacheTestNow) },
			want: []cacheSpan{
				refused,
				handle("fingerprint_only", ""),
				{Name: "delete_cache", Attrs: map[string]any{"cache_name": "cachedContents/old"}, Failed: true, Error: "Error 500, Message: refused"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spans := recordCacheSpans(t)
			api := &fakeCacheAPI{createStatus: tt.createStatus, deleteStatus: tt.deleteStatus, createNoName: tt.createNoName, gone: tt.gone, expireTime: expire}
			m := newCacheTestModel(t, api, "test-model")
			req := cacheTestRequest(cfg)
			if tt.off {
				req.CacheConfig = nil
			}
			req.Config.CachedContent = tt.ownCache
			req.PreviousPromptTokenCount = 5000
			if tt.metadata != nil {
				req.CacheMetadata = tt.metadata(fingerprintOf(m, cacheTestRequest(nil), 2))
			}

			if _, err := generate(cacheTestContext(t), m, req); err != nil {
				t.Fatalf("GenerateContent() error = %v, want the cache failure kept from the caller", err)
			}

			if diff := cmp.Diff(tt.want, trimErrors(spans())); diff != "" {
				t.Errorf("context cache spans mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// trimErrors cuts each span's error text to its status code and message, the
// part the provider's error is identified by.
func trimErrors(spans []cacheSpan) []cacheSpan {
	for i := range spans {
		if j := strings.Index(spans[i].Error, ", Status:"); j >= 0 {
			spans[i].Error = spans[i].Error[:j]
		}
	}
	return spans
}

// TestGenerateContent_ContextCacheSpanNesting checks that a cache creation is
// traced inside the decision that made it, and a deletion outside it.
func TestGenerateContent_ContextCacheSpanNesting(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	api := &fakeCacheAPI{expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{CacheName: "cachedContents/old", ExpireTime: cacheTestNow, InvocationsUsed: 1, Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}

	generateOnce(t, cacheTestContext(t), m, req)

	byName := map[string]tracetest.SpanStub{}
	for _, s := range exp.GetSpans() {
		byName[s.Name] = s
	}
	handle, create, del := byName["handle_context_caching"], byName["create_cache"], byName["delete_cache"]
	if !handle.SpanContext.IsValid() || !create.SpanContext.IsValid() || !del.SpanContext.IsValid() {
		t.Fatalf("spans recorded = %v, want handle_context_caching, create_cache and delete_cache", exp.GetSpans())
	}
	if create.Parent.SpanID() != handle.SpanContext.SpanID() {
		t.Errorf("create_cache parent = %v, want handle_context_caching %v", create.Parent.SpanID(), handle.SpanContext.SpanID())
	}
	if del.Parent.SpanID() == handle.SpanContext.SpanID() {
		t.Error("delete_cache is a child of handle_context_caching, which ended before the request was sent")
	}
}

// TestGenerateContent_ContextCacheSpanLedgerDeletion checks that a deletion
// the runner's ledger runs after the invocation is traced, failure included.
func TestGenerateContent_ContextCacheSpanLedgerDeletion(t *testing.T) {
	spans := recordCacheSpans(t)
	api := &fakeCacheAPI{deleteStatus: http.StatusForbidden, expireTime: cacheTestNow.Add(time.Hour)}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.PreviousPromptTokenCount = 5000
	req.CacheMetadata = &model.CacheMetadata{Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}
	ledger := llminternal.NewContextCacheLedger()
	ctx := llminternal.WithContextCacheLedger(cacheTestContext(t), ledger)

	generateOnce(t, ctx, m, req)
	// Nothing stored names the created cache, so settling deletes it.
	ledger.Settle(ctx, loadEvents(cacheEvents{}))

	got := trimErrors(spans())
	want := cacheSpan{Name: "delete_cache", Attrs: map[string]any{"cache_name": "cachedContents/1"}, Failed: true, Error: "Error 403, Message: refused"}
	if len(got) == 0 || !cmp.Equal(want, got[len(got)-1]) {
		t.Errorf("last context cache span = %+v, want %+v", got, want)
	}
}

// TestGenerateContent_ContextCacheSpanMissingCache checks that a request
// resent without a cache the provider no longer holds keeps one cache action,
// adds an event naming the missing cache to the span the call was made under,
// and does not count deleting that cache as a failure.
func TestGenerateContent_ContextCacheSpanMissingCache(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	telemetry.OverrideTracerForTesting(t, tp)
	api := &fakeCacheAPI{gone: map[string]bool{"cachedContents/gone": true}}
	m := newCacheTestModel(t, api, "test-model")
	req := cacheTestRequest(&model.ContextCacheConfig{})
	req.CacheMetadata = &model.CacheMetadata{CacheName: "cachedContents/gone", ExpireTime: cacheTestNow.Add(time.Hour), InvocationsUsed: 1, Fingerprint: fingerprintOf(m, cacheTestRequest(nil), 2), ContentsCount: 2}
	// Stands in for the generate_content span the flow opens.
	ctx, call := tp.Tracer("test").Start(cacheTestContext(t), "call")

	generateOnce(t, ctx, m, req)
	call.End()

	var got []string
	var events []map[string]string
	for _, s := range exp.GetSpans() {
		switch s.Name {
		case "call":
			for _, e := range s.Events {
				attrs := map[string]string{"name": e.Name}
				for _, a := range e.Attributes {
					attrs[string(a.Key)] = a.Value.AsString()
				}
				events = append(events, attrs)
			}
		case "handle_context_caching", "delete_cache":
			line := s.Name
			for _, a := range s.Attributes {
				line += " " + string(a.Key) + "=" + a.Value.AsString()
			}
			if s.Status.Code == codes.Error {
				line += " failed"
			}
			got = append(got, line)
		}
	}
	wantSpans := []string{
		"handle_context_caching cache_action=active_cache cache_name=cachedContents/gone",
		"delete_cache cache_name=cachedContents/gone",
	}
	if diff := cmp.Diff(wantSpans, got); diff != "" {
		t.Errorf("context cache spans mismatch (-want +got):\n%s", diff)
	}
	wantEvents := []map[string]string{{
		"name":              "context_cache_missing",
		"cache_name":        "cachedContents/gone",
		"exception.message": "failed to call model: Error 404, Message: CachedContent not found (or permission denied), Status: NOT_FOUND, Details: []",
	}}
	if diff := cmp.Diff(wantEvents, events); diff != "" {
		t.Errorf("events on the calling span mismatch (-want +got):\n%s", diff)
	}
}
