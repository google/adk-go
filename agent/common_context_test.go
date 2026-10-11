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

package agent

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/session"
)

func TestCommonContext_ContextFallbackDelegation(t *testing.T) {
	t.Parallel()

	baseIC := &invocationContext{
		Context: t.Context(),
	}

	wantPath := "wf/child@123"
	wantAncestors := []string{"wf/root", "wf/parent"}
	runID := "123"
	var subScheduler DynamicSubScheduler = nil
	// Create a dynamic node context that explicitly populates path and outputForAncestors.
	delta := &CommonContextDelta{
		Path:               &wantPath,
		OutputForAncestors: &wantAncestors,
		RunID:              &runID,
		SubScheduler:       &subScheduler,
	}

	dynCtx := PromoteWithDelta(baseIC, delta)

	tests := []struct {
		name         string
		buildWrapped func(parent Context) Context
	}{
		{
			name: "Direct dynamic node context (fast path baseline)",
			buildWrapped: func(parent Context) Context {
				return parent
			},
		},
		{
			name: "NewToolContext wrapping branchOverride adapter (delegates fallback to c.Context)",
			buildWrapped: func(parent Context) Context {
				tc := NewToolContext(parent, "call-id-1", nil, nil)
				branch := "parallel-branch"
				return tc.WithDelta(&CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: &branch}})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotCtx := tc.buildWrapped(dynCtx)

			if gotPath := gotCtx.Path(); gotPath != wantPath {
				t.Errorf("Path() = %q, want %q", gotPath, wantPath)
			}

			gotAncestors := gotCtx.OutputForAncestors()
			if diff := cmp.Diff(wantAncestors, gotAncestors); diff != "" {
				t.Errorf("OutputForAncestors() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

type saveOnlyArtifacts struct {
	Artifacts
	save func(context.Context, string, *genai.Part) (*artifact.SaveResponse, error)
}

func (a saveOnlyArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	return a.save(ctx, name, data)
}

func TestTrackedArtifacts_ConcurrentSave(t *testing.T) {
	for _, tc := range []struct {
		name string
		wrap func(InvocationContext, *session.EventActions) Context
	}{
		{"tool", func(ic InvocationContext, actions *session.EventActions) Context {
			return NewToolContext(ic, "call", actions, nil)
		}},
		{"callback", NewCallbackContextWithArtifactTracking},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			const saves = 16
			entered := make(chan struct{}, saves)
			release := make(chan struct{})
			backend := saveOnlyArtifacts{save: func(ctx context.Context, _ string, _ *genai.Part) (*artifact.SaveResponse, error) {
				entered <- struct{}{}
				select {
				case <-release:
					return &artifact.SaveResponse{Version: 1}, nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}}
			actions := &session.EventActions{}
			toolCtx := tc.wrap(&invocationContext{Context: ctx, artifacts: backend}, actions)
			var wg sync.WaitGroup
			t.Cleanup(wg.Wait)
			for i := range saves {
				wg.Go(func() {
					_, err := toolCtx.Artifacts().Save(toolCtx, fmt.Sprintf("report-%d.txt", i), genai.NewPartFromText("synthetic"))
					if err != nil {
						t.Error("artifact save failed")
					}
				})
			}
			// Every backend call must start before any is allowed to finish.
			for range saves {
				select {
				case <-entered:
				case <-ctx.Done():
					t.Fatal("backend saves did not run concurrently")
				}
			}
			close(release)
			wg.Wait()
			delta := actions.ArtifactDelta
			if len(delta) != saves {
				t.Fatalf("delta has %d entries, want %d", len(delta), saves)
			}
			for i := range saves {
				if delta[fmt.Sprintf("report-%d.txt", i)] != 1 {
					t.Errorf("save %d is missing its version in the delta", i)
				}
			}
		})
	}
}

func TestTrackedArtifacts_OutOfOrderSave(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	entered := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int64
	backend := saveOnlyArtifacts{save: func(ctx context.Context, _ string, _ *genai.Part) (*artifact.SaveResponse, error) {
		version := calls.Add(1)
		if version == 1 {
			close(entered)
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		return &artifact.SaveResponse{Version: version}, nil
	}}
	toolCtx := NewToolContext(&invocationContext{Context: ctx, artifacts: backend}, "call", nil, nil)
	var wg sync.WaitGroup
	t.Cleanup(wg.Wait)
	wg.Go(func() {
		if _, err := toolCtx.Artifacts().Save(toolCtx, "report.txt", genai.NewPartFromText("first")); err != nil {
			t.Error("first save failed")
		}
	})
	<-entered
	if _, err := toolCtx.Artifacts().Save(toolCtx, "report.txt", genai.NewPartFromText("second")); err != nil {
		t.Fatal("second save failed")
	}
	close(release)
	wg.Wait()
	if got := toolCtx.Actions().ArtifactDelta["report.txt"]; got != 2 {
		t.Fatalf("recorded version = %d, want 2", got)
	}
}

func TestTrackedArtifacts_SaveDelta(t *testing.T) {
	for _, tc := range []struct {
		name    string
		actions *session.EventActions
		version int64
		want    map[string]int64
	}{
		{"zero version", &session.EventActions{}, 0, map[string]int64{"report.txt": 0}},
		{"newer version", &session.EventActions{ArtifactDelta: map[string]int64{"report.txt": 1}}, 2, map[string]int64{"report.txt": 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := &artifact.SaveResponse{Version: tc.version}
			backend := saveOnlyArtifacts{save: func(context.Context, string, *genai.Part) (*artifact.SaveResponse, error) {
				return response, nil
			}}
			tracked := newTrackedArtifacts(backend, tc.actions)
			got, err := tracked.Save(t.Context(), "report.txt", genai.NewPartFromText("synthetic"))
			if got != response || err != nil {
				t.Fatal("save did not return the backend response successfully")
			}
			if !cmp.Equal(tc.actions.ArtifactDelta, tc.want) {
				t.Error("artifact delta does not contain the expected versions")
			}
		})
	}
}
