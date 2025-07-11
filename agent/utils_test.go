// Copyright 2025 Google LLC
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

	"github.com/google/adk-go"
)

func TestRootAgent(t *testing.T) {
	root := &LLMAgent{AgentName: "root"}
	a := &LLMAgent{AgentName: "a"}
	b := &LLMAgent{AgentName: "b"}
	nonLLM := mockAgent("mock")

	root.AddSubAgents(a)
	a.AddSubAgents(b)
	b.AddSubAgents(nonLLM)

	agentName := func(a adk.Agent) string {
		if a == nil {
			return "nil"
		}
		return a.Name()
	}

	for _, tc := range []struct {
		agent adk.Agent
		want  adk.Agent
	}{
		{root, root},
		{a, root},
		{b, root},
		{nonLLM, nonLLM}, // TODO: nonLLM agent should be able to have root.
		{nil, nil},
	} {
		t.Run("agent="+agentName(tc.agent), func(t *testing.T) {
			gotRoot := rootAgent(tc.agent)
			if got, want := agentName(gotRoot), agentName(tc.want); got != want {
				t.Errorf("rootAgent(%q) = %q, want %q", agentName(tc.agent), got, want)
			}
		})
	}
}
