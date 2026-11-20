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

package telemetry

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/model"
)

// Span names and attribute keys of context caching. The first two spans and
// every key match adk-python (models/google_llm.py and
// models/gemini_context_cache_manager.py), so treat them as fixed. adk-python
// has no delete_cache span: it logs a failed deletion instead, and ADK Go
// library code has no logger.
const (
	handleContextCachingName = "handle_context_caching"
	createCacheName          = "create_cache"
	deleteCacheName          = "delete_cache"
)

// Values of the cache_action attribute.
const (
	// ContextCacheActionActive marks a request sent through a cache, reused
	// or created for it. A create_cache child span tells the two apart.
	ContextCacheActionActive = "active_cache"
	// ContextCacheActionFingerprintOnly marks a request sent without a cache.
	ContextCacheActionFingerprintOnly = "fingerprint_only"
)

var (
	contextCacheAction        = attribute.Key("cache_action")
	contextCacheName          = attribute.Key("cache_name")
	contextCacheModel         = attribute.Key("model")
	contextCacheContentsCount = attribute.Key("cache_contents_count")
	contextCacheTTLSeconds    = attribute.Key("ttl_seconds")
)

// StartHandleContextCachingSpan starts the span covering the decision of which
// context cache one model request uses, including any cache it creates.
func StartHandleContextCachingSpan(ctx context.Context) (context.Context, trace.Span) {
	return tracer().Start(ctx, handleContextCachingName)
}

// TraceContextCacheAction records on span which cache action m describes: an
// active cache and its name, or a request sent without a cache. A nil m, a
// request caching does not apply to, records nothing, as in adk-python.
func TraceContextCacheAction(span trace.Span, m *model.CacheMetadata) {
	switch {
	case m == nil:
	case m.Active():
		span.SetAttributes(contextCacheAction.String(ContextCacheActionActive), contextCacheName.String(m.CacheName))
	default:
		span.SetAttributes(contextCacheAction.String(ContextCacheActionFingerprintOnly))
	}
}

// StartCreateCacheSpanParams contains parameters for [StartCreateCacheSpan].
type StartCreateCacheSpanParams struct {
	// Model is the model the cache is created for.
	Model string
	// ContentsCount is how many leading contents the cache holds.
	ContentsCount int
	// TTL is the cache's requested lifetime. The span records it in whole
	// seconds, the type adk-python records.
	TTL time.Duration
}

// StartCreateCacheSpan starts the span covering one call that creates a
// context cache on the provider.
func StartCreateCacheSpan(ctx context.Context, params StartCreateCacheSpanParams) (context.Context, trace.Span) {
	return tracer().Start(ctx, createCacheName, trace.WithAttributes(
		contextCacheContentsCount.Int(params.ContentsCount),
		contextCacheModel.String(params.Model),
		contextCacheTTLSeconds.Int64(int64(params.TTL/time.Second)),
	))
}

// TraceCreateCacheResult records on span the cache that was created, or the
// error that prevented it.
func TraceCreateCacheResult(span trace.Span, name string, err error) {
	if err != nil {
		RecordError(span, err)
		return
	}
	span.SetAttributes(contextCacheName.String(name))
}

// StartDeleteCacheSpan starts the span covering one call that deletes the
// named context cache. Record its outcome with [RecordError].
func StartDeleteCacheSpan(ctx context.Context, name string) (context.Context, trace.Span) {
	return tracer().Start(ctx, deleteCacheName, trace.WithAttributes(contextCacheName.String(name)))
}

// TraceContextCacheRetry records a request that named the cache name, which
// the provider no longer holds, and is being sent again without it. It opens
// a second handle_context_caching span, marked fingerprint_only and carrying
// err as an event, so the request's last cache action is the one that stands.
// The span is not marked failed: the request goes on.
func TraceContextCacheRetry(ctx context.Context, name string, err error) {
	_, span := StartHandleContextCachingSpan(ctx)
	defer span.End()
	span.SetAttributes(contextCacheAction.String(ContextCacheActionFingerprintOnly))
	span.RecordError(err, trace.WithAttributes(contextCacheName.String(name)))
}
