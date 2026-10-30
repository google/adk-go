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

package llminternal

import (
	"context"
	"iter"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

type contextCacheConfigKey struct{}

// WithContextCacheConfig returns a copy of ctx carrying cfg, which
// ContextCacheRequestProcessor puts on every model request made under it. A
// nil cfg turns caching off, including for a run nested inside one that has it
// on, so each runner decides for the agents it drives.
func WithContextCacheConfig(ctx context.Context, cfg *model.ContextCacheConfig) context.Context {
	return context.WithValue(ctx, contextCacheConfigKey{}, cfg)
}

// contextCacheConfigFromContext returns the config [WithContextCacheConfig]
// put on ctx, or nil when caching is off.
func contextCacheConfigFromContext(ctx context.Context) *model.ContextCacheConfig {
	cfg, _ := ctx.Value(contextCacheConfigKey{}).(*model.ContextCacheConfig)
	return cfg
}

// ContextCacheRequestProcessor prepares a request for context caching when the
// runner has it on. It sets the cache config, and copies from the session the
// agent's latest cache metadata and the prompt token count of its previous
// request. The model decides what to do with them.
//
// It follows adk-python's ContextCacheRequestProcessor
// (flows/llm_flows/context/_cache.py), and yields no events.
func ContextCacheRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		cfg := contextCacheConfigFromContext(ctx)
		if cfg == nil {
			return
		}
		// A copy, because the runner's config is shared by every request.
		cfgCopy := *cfg
		req.CacheConfig = &cfgCopy

		if ctx.Session() == nil {
			return
		}
		scope := cacheScope{
			agentName:      ctx.Agent().Name(),
			branch:         ctx.Branch(),
			isolationScope: ctx.IsolationScope(),
		}
		req.CacheMetadata, req.PreviousPromptTokenCount = findCacheInfo(ctx.Session().Events(), scope, ctx.InvocationID())
	}
}

// cacheScope names whose cache state a request may use: the agent's own model
// responses on its own branch and isolation scope.
//
// adk-python matches on the author alone. That lets the copies of one agent
// that a parallel worker runs side by side, each on its own branch and with
// its own history, find one another's caches: each sees the other's prefix as
// changed and deletes a cache the other is still using. The branch and scope
// are the same boundary the contents processor draws around the history the
// cache holds.
type cacheScope struct {
	agentName      string
	branch         string
	isolationScope string
}

func (s cacheScope) contains(ev *session.Event) bool {
	return ev.Author == s.agentName && ev.Branch == s.branch && ev.IsolationScope == s.isolationScope
}

// findCacheInfo walks events newest first for those in scope, and returns a
// copy of the latest cache metadata and the latest reported prompt token
// count.
//
// Metadata of an active cache that an earlier invocation recorded comes back
// with InvocationsUsed raised by one, counting the invocation now running.
// Steps within one invocation share a count, so a long tool loop does not use
// up the cache's intervals.
func findCacheInfo(events session.Events, scope cacheScope, invocationID string) (*model.CacheMetadata, int32) {
	var metadata *model.CacheMetadata
	var promptTokens int32
	for i := events.Len() - 1; i >= 0; i-- {
		ev := events.At(i)
		if !scope.contains(ev) {
			continue
		}
		if metadata == nil && ev.CacheMetadata != nil {
			m := *ev.CacheMetadata
			if ev.InvocationID != "" && ev.InvocationID != invocationID && m.Active() {
				m.InvocationsUsed++
			}
			metadata = &m
		}
		if promptTokens == 0 && ev.UsageMetadata != nil && ev.UsageMetadata.PromptTokenCount > 0 {
			promptTokens = ev.UsageMetadata.PromptTokenCount
		}
		if metadata != nil && promptTokens != 0 {
			break
		}
	}
	return metadata, promptTokens
}
