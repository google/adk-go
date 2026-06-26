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
	"testing"

	"github.com/google/go-cmp/cmp"
)

// customContextDecorator is a non-commonContext struct that wraps a Context.
// When passed to ADK constructors like NewNodeContext or NewContext, the
// constructor does not short-circuit (since it is not *commonContext) and
// re-wraps it in a new commonContext envelope.
type customContextDecorator struct {
	Context
}

func TestCommonContext_ContextFallbackDelegation(t *testing.T) {
	t.Parallel()

	baseIC := &invocationContext{
		Context: t.Context(),
	}
	baseCtx := NewContext(baseIC)

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

	dynCtx := baseCtx.WithDelta(delta)

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
			name: "NewNodeContext wrapping custom decorator (delegates fallback to c.Context)",
			buildWrapped: func(parent Context) Context {
				decorator := &customContextDecorator{Context: parent}
				return NewContext(decorator)
			},
		},
		{
			name: "NewContext wrapping custom decorator (delegates fallback to c.Context)",
			buildWrapped: func(parent Context) Context {
				decorator := &customContextDecorator{Context: parent}
				return NewContext(decorator)
			},
		},
		{
			name: "NewToolContext wrapping branchOverride adapter (delegates fallback to c.Context)",
			buildWrapped: func(parent Context) Context {
				branched := &branchOverride{Context: parent, branch: "parallel-branch"}
				return NewToolContext(branched, "call-id-1", nil, nil)
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
