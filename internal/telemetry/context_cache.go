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
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
)

// Span names and attribute keys of context caching. The first two spans and
// every key match adk-python (models/google_llm.py and
// models/gemini_context_cache_manager.py), so treat them as fixed. adk-python
// has no delete_cache span or context_cache_missing event: it logs a failed
// deletion, and fails a request naming a missing cache.
const (
	handleContextCachingName = "handle_context_caching"
	createCacheName          = "create_cache"
	deleteCacheName          = "delete_cache"
	contextCacheMissingName  = "context_cache_missing"
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

// TraceContextCacheAction records on span the cache action of a request that
// is sent naming the cache name: active_cache and the name, or
// fingerprint_only when name is empty. A request that names its own cache is
// active_cache too, as adk-python records a pre-created cache.
func TraceContextCacheAction(span trace.Span, name string) {
	if name == "" {
		span.SetAttributes(contextCacheAction.String(ContextCacheActionFingerprintOnly))
		return
	}
	span.SetAttributes(contextCacheAction.String(ContextCacheActionActive), contextCacheName.String(name))
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
// named context cache. Record a failure with [RecordError].
func StartDeleteCacheSpan(ctx context.Context, name string) (context.Context, trace.Span) {
	return tracer().Start(ctx, deleteCacheName, trace.WithAttributes(contextCacheName.String(name)))
}

// TraceContextCacheMissing records, on the span in ctx, that a request named
// the cache name, which the provider no longer holds, and is being sent again
// without it. Under an agent that span is the request's generate_content span.
// It is an event rather than a second handle_context_caching span, so a
// request has one cache action, as in adk-python, and it leaves the span's
// status alone: the request goes on.
func TraceContextCacheMissing(ctx context.Context, name string, err error) {
	trace.SpanFromContext(ctx).AddEvent(contextCacheMissingName, trace.WithAttributes(
		contextCacheName.String(name),
		semconv.ExceptionMessage(err.Error()),
	))
}
