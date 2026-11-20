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
	"iter"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

// nodeAttemptContext builds the context of a node attempt numbered n, the way
// a workflow scheduler derives one from the invocation.
func nodeAttemptContext(ic InvocationContext, n int) Context {
	return NewContext(ic).WithDelta(&CommonContextDelta{AttemptCount: &n})
}

// embeddingContext is a Context implemented outside the attempt machinery: it
// embeds a node's Context and forwards nothing else.
type embeddingContext struct {
	Context
}

func TestAttemptCount_BelowOneReadsOne(t *testing.T) {
	invocation := func(ic InvocationContext) Context { return NewContext(ic) }
	nodeThree := func(ic InvocationContext) Context { return nodeAttemptContext(ic, 3) }
	tests := []struct {
		name   string
		parent func(ic InvocationContext) Context
		n      int
	}{
		{name: "zero", parent: invocation, n: 0},
		{name: "negative", parent: invocation, n: -4},
		{name: "zero over node(3)", parent: nodeThree, n: 0},
		{name: "negative over node(3)", parent: nodeThree, n: -4},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.parent(&invocationContext{Context: t.Context()}).WithDelta(&CommonContextDelta{AttemptCount: &tc.n})
			if got := AttemptCount(ctx); got != 1 {
				t.Errorf("AttemptCount(%s) = %d, want 1", tc.name, got)
			}
		})
	}
}

func TestAttemptCount_FollowsContextCopies(t *testing.T) {
	tests := []struct {
		name   string
		derive func(t *testing.T, ctx Context) Context
	}{
		{
			name: "WithAgentCancel",
			derive: func(t *testing.T, ctx Context) Context {
				derived, cancel := ctx.WithAgentCancel()
				t.Cleanup(cancel)
				return derived
			},
		},
		{
			name: "WithAgentTimeout",
			derive: func(t *testing.T, ctx Context) Context {
				derived, cancel := ctx.WithAgentTimeout(time.Hour)
				t.Cleanup(cancel)
				return derived
			},
		},
		{
			name: "WithAgentContext",
			derive: func(t *testing.T, ctx Context) Context {
				return ctx.WithAgentContext(context.Background())
			},
		},
		{
			name: "WithDelta without AttemptCount",
			derive: func(t *testing.T, ctx Context) Context {
				p := "parent/child@1"
				return ctx.WithDelta(&CommonContextDelta{Path: &p})
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.derive(t, nodeAttemptContext(&invocationContext{Context: t.Context()}, 3))
			if got := AttemptCount(ctx); got != 3 {
				t.Errorf("AttemptCount(%s of node(3)) = %d, want 3", tc.name, got)
			}
		})
	}
}

func TestWithDelta_AttemptCount_ChildIsIndependent(t *testing.T) {
	parent := nodeAttemptContext(&invocationContext{Context: t.Context()}, 3)
	two := 2

	child := parent.WithDelta(&CommonContextDelta{AttemptCount: &two})
	two = 9

	if got := AttemptCount(child); got != 2 {
		t.Errorf("AttemptCount(child) = %d, want 2", got)
	}
	if got := AttemptCount(parent); got != 3 {
		t.Errorf("AttemptCount(parent) = %d, want 3", got)
	}
	if got := AttemptCount(parent.WithDelta(&CommonContextDelta{})); got != 3 {
		t.Errorf("AttemptCount(parent.WithDelta(empty)) = %d, want 3", got)
	}
}

func TestAttemptCount_NotANodeAttemptReadsOne(t *testing.T) {
	tests := []struct {
		name  string
		build func(t *testing.T) Context
		want  int
	}{
		{
			name:  "nil",
			build: func(t *testing.T) Context { return nil },
			want:  1,
		},
		{
			name:  "NewContext of an invocation",
			build: func(t *testing.T) Context { return NewContext(&invocationContext{Context: t.Context()}) },
			want:  1,
		},
		{
			name: "NewContext of a node attempt",
			build: func(t *testing.T) Context {
				return NewContext(nodeAttemptContext(&invocationContext{Context: t.Context()}, 3))
			},
			want: 1,
		},
		{
			name: "NewCallbackContext of a node attempt",
			build: func(t *testing.T) Context {
				return NewCallbackContext(nodeAttemptContext(&invocationContext{Context: t.Context()}, 3), nil)
			},
			want: 1,
		},
		{
			name:  "ContextMock",
			build: func(t *testing.T) Context { return &ContextMock{} },
			want:  1,
		},
		{
			name:  "StrictContextMock",
			build: func(t *testing.T) Context { return &StrictContextMock{Ctx: t.Context()} },
			want:  1,
		},
		{
			name: "outside type embedding a node attempt",
			build: func(t *testing.T) Context {
				return embeddingContext{Context: nodeAttemptContext(&invocationContext{Context: t.Context()}, 3)}
			},
			want: 1,
		},
		{
			name: "NewToolContext of a node attempt",
			build: func(t *testing.T) Context {
				return NewToolContext(nodeAttemptContext(&invocationContext{Context: t.Context()}, 3), "", nil, nil)
			},
			want: 3,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := AttemptCount(tc.build(t)); got != tc.want {
				t.Errorf("AttemptCount(%s) = %d, want %d", tc.name, got, tc.want)
			}
		})
	}
}

func TestAttemptCount_ToolContextAndDerivedReadNodeNumber(t *testing.T) {
	tc := NewToolContext(nodeAttemptContext(&invocationContext{Context: t.Context()}, 3), "id", nil, nil)
	if got := AttemptCount(tc); got != 3 {
		t.Errorf("AttemptCount(tool context of node(3)) = %d, want 3", got)
	}

	cancelled, cancel := tc.WithAgentCancel()
	t.Cleanup(cancel)
	if got := AttemptCount(cancelled); got != 3 {
		t.Errorf("AttemptCount(tool context WithAgentCancel) = %d, want 3", got)
	}

	timed, cancelTimeout := tc.WithAgentTimeout(time.Hour)
	t.Cleanup(cancelTimeout)
	if got := AttemptCount(timed); got != 3 {
		t.Errorf("AttemptCount(tool context WithAgentTimeout) = %d, want 3", got)
	}

	if got := AttemptCount(tc.WithDelta(&CommonContextDelta{})); got != 3 {
		t.Errorf("AttemptCount(tool context WithDelta(empty)) = %d, want 3", got)
	}

	if got := AttemptCount(NewToolContext(tc.WithICDelta(&InvocationContextDelta{}), "", nil, nil)); got != 3 {
		t.Errorf("AttemptCount(NewToolContext(tool context WithICDelta(empty))) = %d, want 3", got)
	}
}

func TestRun_AttemptCountInsideAgentReadsOne(t *testing.T) {
	var got []int
	testAgent, err := New(Config{
		Name: "records_attempt",
		Run: func(ctx InvocationContext) iter.Seq2[*session.Event, error] {
			got = append(got, AttemptCount(NewToolContext(ctx, "", nil, nil)))
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ic := &invocationContext{
		Context: t.Context(),
		agent:   testAgent,
		session: &mockSession{sessionID: "test-session"},
	}

	for _, err := range testAgent.Run(nodeAttemptContext(ic, 3)) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	if diff := cmp.Diff([]int{1}, got); diff != "" {
		t.Errorf("AttemptCount inside the agent's run mismatch (-want +got):\n%s", diff)
	}
}
