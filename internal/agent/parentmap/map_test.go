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

package parentmap_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
)

func TestNew(t *testing.T) {
	child1_1 := utils.Must(agent.New(agent.Config{
		Name: "child1_1",
	}))

	child1 := utils.Must(agent.New(agent.Config{
		Name:      "child1",
		SubAgents: []agent.Agent{child1_1},
	}))

	child2 := utils.Must(agent.New(agent.Config{
		Name: "child2",
	}))

	root := utils.Must(agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{child1, child2},
	}))

	got, err := parentmap.New(root)
	if err != nil {
		t.Fatal(err)
	}
	want := parentmap.Map{
		child1_1.Name(): child1,
		child1.Name():   root,
		child2.Name():   root,
	}

	agentNames := cmp.Transformer("agentNames", func(m parentmap.Map) map[string]string {
		if m == nil {
			return nil
		}
		res := make(map[string]string)
		for k, v := range m {
			res[k] = v.Name()
		}
		return res
	})

	if diff := cmp.Diff(want, got, agentNames); diff != "" {
		t.Errorf("New() = %v, got %v diff (-want/+got): %v", got, want, diff)
	}
}

func TestMap_RootAgent(t *testing.T) {
	model := struct {
		model.LLM
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

	parents, err := parentmap.New(root)
	if err != nil {
		t.Fatal(err)
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
			gotRoot := parents.RootAgent(tc.agent)
			if got, want := agentName(gotRoot), agentName(tc.want); got != want {
				t.Errorf("rootAgent(%q) = %q, want %q", agentName(tc.agent), got, want)
			}
		})
	}
}
