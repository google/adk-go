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

package workflow

import (
	"testing"

	"google.golang.org/adk/agent"
)

func TestNewNodeContext_TriggeredByRoundTrip(t *testing.T) {
	tests := []struct {
		name        string
		triggeredBy string
	}{
		{name: "empty (initial START activation)", triggeredBy: ""},
		{name: "named upstream node", triggeredBy: "upstream"},
		{name: "node name with dots (branch path)", triggeredBy: "agent_1.agent_2"},
	}

	parent := newMockCtx(t)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newNodeContext(parent, tt.triggeredBy, nil, nil)
			if got := c.TriggeredBy(); got != tt.triggeredBy {
				t.Errorf("TriggeredBy() = %q, want %q", got, tt.triggeredBy)
			}
		})
	}
}

func TestNewNodeContext_InNodes(t *testing.T) {
	parent := newMockCtx(t)

	t.Run("nil predecessor set returns nil", func(t *testing.T) {
		c := newNodeContext(parent, "", nil, nil)
		if got := c.InNodes(); got != nil {
			t.Errorf("InNodes() = %v, want nil for empty predecessor set", got)
		}
	})

	t.Run("empty predecessor set returns nil", func(t *testing.T) {
		c := newNodeContext(parent, "", map[string]struct{}{}, nil)
		if got := c.InNodes(); got != nil {
			t.Errorf("InNodes() = %v, want nil for empty predecessor set", got)
		}
	})

	t.Run("populated set returns names", func(t *testing.T) {
		c := newNodeContext(parent, "A", map[string]struct{}{
			"A": {},
			"B": {},
			"C": {},
		}, nil)
		got := c.InNodes()
		if len(got) != 3 {
			t.Fatalf("InNodes() length = %d, want 3 (got %v)", len(got), got)
		}
		want := map[string]bool{"A": true, "B": true, "C": true}
		for _, name := range got {
			if !want[name] {
				t.Errorf("unexpected predecessor name %q in InNodes() = %v", name, got)
			}
		}
	})

	t.Run("returned slice is independent of internal set", func(t *testing.T) {
		// Mutating the returned slice must not affect a subsequent call.
		c := newNodeContext(parent, "", map[string]struct{}{"A": {}, "B": {}}, nil)
		first := c.InNodes()
		first[0] = "MUTATED"
		second := c.InNodes()
		for _, name := range second {
			if name == "MUTATED" {
				t.Errorf("InNodes() returned slice must be a defensive copy; got mutation leaked: %v", second)
			}
		}
	})
}

// Compile-time assertion: *nodeContext satisfies agent.InvocationContext.
var _ agent.InvocationContext = (*nodeContext)(nil)
