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

package agentstate_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agentstate"
	"google.golang.org/adk/v2/internal/utils"
)

func TestInt(t *testing.T) {
	tests := []struct {
		name  string
		state map[string]any
		want  int
	}{
		{"int", map[string]any{"n": 2}, 2},
		{"int64", map[string]any{"n": int64(2)}, 2},
		// A session store that keeps JSON reads every number back as float64.
		{"float64", map[string]any{"n": float64(2)}, 2},
		{"missing", map[string]any{}, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := agentstate.Int(tc.state, "n"); got != tc.want {
				t.Errorf("Int() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestIndex(t *testing.T) {
	subAgents := []agent.Agent{
		utils.Must(agent.New(agent.Config{Name: "a"})),
		utils.Must(agent.New(agent.Config{Name: "b"})),
	}
	if got := agentstate.Index(subAgents, map[string]any{"k": "b"}, "k"); got != 1 {
		t.Errorf("Index() = %d, want 1", got)
	}
	// A sub-agent removed since the checkpoint restarts from the first, as in
	// adk-python.
	if got := agentstate.Index(subAgents, map[string]any{"k": "gone"}, "k"); got != 0 {
		t.Errorf("Index() for a removed sub-agent = %d, want 0", got)
	}
}
