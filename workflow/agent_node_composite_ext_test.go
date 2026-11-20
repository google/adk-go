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

package workflow_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/workflow"
)

func TestNewAgentNode_CompositeRerunsOnResume(t *testing.T) {
	leaf := func(name string) agent.Agent {
		a, err := agent.New(agent.Config{Name: name})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	composite, err := agent.New(agent.Config{Name: "comp", SubAgents: []agent.Agent{leaf("sub")}})
	if err != nil {
		t.Fatal(err)
	}
	coordinator, err := llmagent.New(llmagent.Config{Name: "coord", SubAgents: []agent.Agent{leaf("helper")}})
	if err != nil {
		t.Fatal(err)
	}
	f := false

	tests := []struct {
		name string
		a    agent.Agent
		cfg  workflow.NodeConfig
		want *bool
	}{
		{name: "composite", a: composite, want: new(true)},
		{name: "composite with handoff set", a: composite, cfg: workflow.NodeConfig{RerunOnResume: &f}, want: &f},
		// An LLM agent handles the answers to its own requests, so it keeps
		// the engine's default even when it can transfer to sub-agents.
		{name: "LLM agent with sub-agents", a: coordinator},
		{name: "agent without sub-agents", a: leaf("plain")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n, err := workflow.NewAgentNode(tc.a, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.want, n.Config().RerunOnResume); diff != "" {
				t.Errorf("RerunOnResume mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
