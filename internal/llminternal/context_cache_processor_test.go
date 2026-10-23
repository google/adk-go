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

package llminternal_test

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

const (
	cacheTestAgent  = "assistant"
	cacheTestBranch = "root.assistant"
)

// cacheEvent is one stored event for a context cache processor test. Unset
// author and branch default to the agent under test and its branch.
type cacheEvent struct {
	author         string
	branch         string
	isolationScope string
	invocationID   string
	metadata       *model.CacheMetadata
	promptTokens   int32
}

// runContextCacheProcessor stores events in a fresh session, then runs the
// processor for cacheTestAgent on cacheTestBranch with isolationScope, in
// invocation "current" under cfg, and returns the request it prepared.
func runContextCacheProcessor(t *testing.T, cfg *model.ContextCacheConfig, isolationScope string, events []cacheEvent) *model.LLMRequest {
	t.Helper()

	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	sess := created.Session
	base := time.Unix(1, 0)
	for i, e := range events {
		ev := session.NewEvent(t.Context(), e.invocationID)
		ev.Author = orDefault(e.author, cacheTestAgent)
		ev.Branch = orDefault(e.branch, cacheTestBranch)
		ev.IsolationScope = e.isolationScope
		ev.Timestamp = base.Add(time.Duration(i) * time.Second)
		ev.Content = genai.NewContentFromText("text", genai.RoleModel)
		ev.CacheMetadata = e.metadata
		if e.promptTokens > 0 {
			ev.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: e.promptTokens}
		}
		if err := svc.AppendEvent(t.Context(), sess, ev); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}

	ctx := llminternal.WithContextCacheConfig(t.Context(), cfg)
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:          utils.Must(llmagent.New(llmagent.Config{Name: cacheTestAgent, Model: &testModel{}})),
		Session:        sess,
		Branch:         cacheTestBranch,
		IsolationScope: isolationScope,
		InvocationID:   "current",
	})
	req := &model.LLMRequest{}
	for ev, err := range llminternal.ContextCacheRequestProcessor(ictx, req, &llminternal.Flow{}) {
		if ev != nil || err != nil {
			t.Fatalf("ContextCacheRequestProcessor yielded (%v, %v), want nothing", ev, err)
		}
	}
	return req
}

func orDefault(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

func activeCache(invocationsUsed int) *model.CacheMetadata {
	return &model.CacheMetadata{
		CacheName:       "cachedContents/1",
		ExpireTime:      time.Unix(3600, 0),
		InvocationsUsed: invocationsUsed,
		Fingerprint:     "fp",
		ContentsCount:   2,
	}
}

func TestContextCacheRequestProcessor(t *testing.T) {
	cfg := &model.ContextCacheConfig{CacheIntervals: 3, TTL: time.Hour}
	fingerprintOnly := &model.CacheMetadata{Fingerprint: "fp", ContentsCount: 2}

	tests := []struct {
		name           string
		isolationScope string
		events         []cacheEvent
		wantMetadata   *model.CacheMetadata
		wantTokens     int32
	}{
		{
			name: "empty session",
		},
		{
			name: "active cache from an earlier invocation counts this one",
			events: []cacheEvent{
				{invocationID: "earlier", metadata: activeCache(1), promptTokens: 5000},
			},
			wantMetadata: activeCache(2),
			wantTokens:   5000,
		},
		{
			name: "active cache from this invocation is not counted again",
			events: []cacheEvent{
				{invocationID: "current", metadata: activeCache(1), promptTokens: 5000},
			},
			wantMetadata: activeCache(1),
			wantTokens:   5000,
		},
		{
			name: "active cache on an event with no invocation is not counted",
			events: []cacheEvent{
				{metadata: activeCache(1), promptTokens: 5000},
			},
			wantMetadata: activeCache(1),
			wantTokens:   5000,
		},
		{
			name: "fingerprint-only metadata from an earlier invocation is not counted",
			events: []cacheEvent{
				{invocationID: "earlier", metadata: fingerprintOnly, promptTokens: 700},
			},
			wantMetadata: fingerprintOnly,
			wantTokens:   700,
		},
		{
			name: "latest metadata and latest token count win",
			events: []cacheEvent{
				{invocationID: "first", metadata: &model.CacheMetadata{Fingerprint: "old"}, promptTokens: 100},
				{invocationID: "second", metadata: fingerprintOnly, promptTokens: 200},
			},
			wantMetadata: fingerprintOnly,
			wantTokens:   200,
		},
		{
			name: "token count can come from a later event than the metadata",
			events: []cacheEvent{
				{invocationID: "earlier", metadata: fingerprintOnly},
				{invocationID: "earlier", promptTokens: 900},
			},
			wantMetadata: fingerprintOnly,
			wantTokens:   900,
		},
		{
			name: "other authors are ignored",
			events: []cacheEvent{
				{invocationID: "earlier", metadata: fingerprintOnly, promptTokens: 300},
				{author: "other_agent", invocationID: "earlier", metadata: activeCache(1), promptTokens: 9000},
				{author: "user", invocationID: "current"},
			},
			wantMetadata: fingerprintOnly,
			wantTokens:   300,
		},
		{
			// A parallel worker runs copies of one agent on sibling branches,
			// each with its own history and cache.
			name: "the same agent on another branch is ignored",
			events: []cacheEvent{
				{invocationID: "earlier", metadata: fingerprintOnly, promptTokens: 300},
				{branch: cacheTestBranch + "@2", invocationID: "current", metadata: activeCache(1), promptTokens: 9000},
			},
			wantMetadata: fingerprintOnly,
			wantTokens:   300,
		},
		{
			name: "the parent branch is ignored",
			events: []cacheEvent{
				{branch: "root", invocationID: "earlier", metadata: activeCache(1), promptTokens: 9000},
			},
		},
		{
			name:           "another isolation scope is ignored",
			isolationScope: "task-1",
			events: []cacheEvent{
				{isolationScope: "task-1", invocationID: "earlier", metadata: fingerprintOnly, promptTokens: 300},
				{isolationScope: "task-2", invocationID: "current", metadata: activeCache(1), promptTokens: 9000},
				{invocationID: "current", metadata: activeCache(1), promptTokens: 9000},
			},
			wantMetadata: fingerprintOnly,
			wantTokens:   300,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := runContextCacheProcessor(t, cfg, tt.isolationScope, tt.events)

			if diff := cmp.Diff(cfg, req.CacheConfig); diff != "" {
				t.Errorf("CacheConfig mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantMetadata, req.CacheMetadata); diff != "" {
				t.Errorf("CacheMetadata mismatch (-want +got):\n%s", diff)
			}
			if req.PreviousPromptTokenCount != tt.wantTokens {
				t.Errorf("PreviousPromptTokenCount = %d, want %d", req.PreviousPromptTokenCount, tt.wantTokens)
			}
		})
	}
}

func TestContextCacheRequestProcessor_Disabled(t *testing.T) {
	events := []cacheEvent{{invocationID: "earlier", metadata: activeCache(1), promptTokens: 5000}}
	req := runContextCacheProcessor(t, nil, "", events)
	if req.CacheConfig != nil || req.CacheMetadata != nil || req.PreviousPromptTokenCount != 0 {
		t.Errorf("with caching off the request got CacheConfig=%v CacheMetadata=%v PreviousPromptTokenCount=%d, want none set",
			req.CacheConfig, req.CacheMetadata, req.PreviousPromptTokenCount)
	}
}

// TestContextCacheRequestProcessor_DoesNotShareState checks that the request
// gets copies: a model editing what it was handed must change neither the
// runner's config nor the metadata stored on the session's event.
func TestContextCacheRequestProcessor_DoesNotShareState(t *testing.T) {
	cfg := &model.ContextCacheConfig{CacheIntervals: 3}
	stored := activeCache(1)
	req := runContextCacheProcessor(t, cfg, "", []cacheEvent{
		{invocationID: "current", metadata: stored, promptTokens: 5000},
	})

	req.CacheConfig.CacheIntervals = 99
	req.CacheMetadata.InvocationsUsed = 99
	if cfg.CacheIntervals != 3 {
		t.Errorf("runner config CacheIntervals = %d after editing the request's copy, want 3", cfg.CacheIntervals)
	}
	if stored.InvocationsUsed != 1 {
		t.Errorf("stored metadata InvocationsUsed = %d after editing the request's copy, want 1", stored.InvocationsUsed)
	}
}

// TestWithContextCacheConfig_NilOverrides checks that a nested run with
// caching off is not handed the config of the run around it.
func TestWithContextCacheConfig_NilOverrides(t *testing.T) {
	outer := llminternal.WithContextCacheConfig(context.Background(), &model.ContextCacheConfig{})
	inner := llminternal.WithContextCacheConfig(outer, nil)

	ictx := icontext.NewInvocationContext(inner, icontext.InvocationContextParams{
		Agent: utils.Must(llmagent.New(llmagent.Config{Name: cacheTestAgent, Model: &testModel{}})),
	})
	req := &model.LLMRequest{}
	for ev, err := range llminternal.ContextCacheRequestProcessor(ictx, req, &llminternal.Flow{}) {
		if ev != nil || err != nil {
			t.Fatalf("ContextCacheRequestProcessor yielded (%v, %v), want nothing", ev, err)
		}
	}
	if req.CacheConfig != nil {
		t.Errorf("CacheConfig = %v under a nil override, want nil", req.CacheConfig)
	}
}
