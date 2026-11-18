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
	"cmp"
	"slices"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

// stateAllPair is one key and value yielded by a state's All.
type stateAllPair struct {
	Key   string
	Value any
}

// newStateAllInvocation returns an invocation over a new in-memory session
// whose state is seed.
func newStateAllInvocation(t *testing.T, seed map[string]any) *invocationContext {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   seed,
	})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	return &invocationContext{
		Context: t.Context(),
		agent:   &agent{name: "agent"},
		session: created.Session,
	}
}

// collectSortedPairs ranges all and returns every yielded pair, sorted by key.
// A key yielded twice appears twice.
func collectSortedPairs(state session.State) []stateAllPair {
	var pairs []stateAllPair
	for k, v := range state.All() {
		pairs = append(pairs, stateAllPair{Key: k, Value: v})
	}
	slices.SortStableFunc(pairs, func(a, b stateAllPair) int { return cmp.Compare(a.Key, b.Key) })
	return pairs
}

func TestCallbackContextState_All_OverlaysPendingDelta(t *testing.T) {
	ic := newStateAllInvocation(t, map[string]any{"seed": 1, "a": 1})
	actions := &session.EventActions{StateDelta: map[string]any{"a": 2, "b": 3}}
	ctx := NewToolContext(ic, "fc-1", actions, nil)

	got := collectSortedPairs(ctx.State())

	want := []stateAllPair{{Key: "a", Value: 2}, {Key: "b", Value: 3}, {Key: "seed", Value: 1}}
	if diff := gocmp.Diff(want, got); diff != "" {
		t.Errorf("State().All() mismatch (-want +got):\n%s", diff)
	}
}

func TestCallbackContextState_All_IsASnapshot(t *testing.T) {
	// A live range over a map visits a key added in its body in most runs, so
	// 50 fresh contexts make a non-snapshot iterator fail reliably.
	for range 50 {
		ic := newStateAllInvocation(t, map[string]any{"seed": 1})
		actions := &session.EventActions{StateDelta: map[string]any{"a": 2}}
		ctx := NewToolContext(ic, "fc-1", actions, nil)

		var visited []string
		for k, v := range ctx.State().All() {
			visited = append(visited, k)
			if err := ctx.State().Set(k+"_copy", v); err != nil {
				t.Fatalf("State().Set(%q) error = %v", k+"_copy", err)
			}
		}

		slices.Sort(visited)
		if diff := gocmp.Diff([]string{"a", "seed"}, visited); diff != "" {
			t.Fatalf("keys visited by State().All() mismatch (-want +got):\n%s", diff)
		}
	}
}

func TestCallbackContextState_All_NilActionsYieldsSessionState(t *testing.T) {
	ic := newStateAllInvocation(t, map[string]any{"seed": 1})
	ctx := NewContext(ic)

	got := collectSortedPairs(ctx.State())

	want := []stateAllPair{{Key: "seed", Value: 1}}
	if diff := gocmp.Diff(want, got); diff != "" {
		t.Errorf("State().All() mismatch (-want +got):\n%s", diff)
	}
}
