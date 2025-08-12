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

package utils_test

import (
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/llm"
)

func TestRootAgent(t *testing.T) {
	model := struct {
		llm.Model
	}{}

	nonLLM := utils.Must(agent.New(agent.Config{
		Name: "mock",
	}))
	b := utils.Must(llmagent.New(llmagent.Config{
		Name:      "b",
		Model:     model,
		SubAgents: []agent.Agent{nonLLM},
	}))
	a := utils.Must(llmagent.New(llmagent.Config{
		Name:      "a",
		Model:     model,
		SubAgents: []agent.Agent{b},
	}))
	root := utils.Must(llmagent.New(llmagent.Config{
		Name:      "root",
		Model:     model,
		SubAgents: []agent.Agent{a},
	}))

	agentName := func(a agent.Agent) string {
		if a == nil {
			return "nil"
		}
		return a.Name()
	}

	for _, tc := range []struct {
		agent agent.Agent
		want  agent.Agent
	}{
		{root, root},
		{a, root},
		{b, root},
		{nonLLM, root},
		{nil, nil},
	} {
		t.Run("agent="+agentName(tc.agent), func(t *testing.T) {
			gotRoot := utils.RootAgent(tc.agent)
			if got, want := agentName(gotRoot), agentName(tc.want); got != want {
				t.Errorf("rootAgent(%q) = %q, want %q", agentName(tc.agent), got, want)
			}
		})
	}
}
