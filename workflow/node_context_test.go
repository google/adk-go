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
			c := newNodeContext(parent, tt.triggeredBy)
			if got := c.TriggeredBy(); got != tt.triggeredBy {
				t.Errorf("TriggeredBy() = %q, want %q", got, tt.triggeredBy)
			}
		})
	}
}

// Compile-time assertion: *nodeContext satisfies agent.InvocationContext.
var _ agent.InvocationContext = (*nodeContext)(nil)
