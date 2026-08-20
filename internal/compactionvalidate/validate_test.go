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

package compactionvalidate_test

import (
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/compactionvalidate"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TestAgainstAgentsRefusesAConfigNoAgentCanServe pins the case operators
// actually hit: a config that passes its own shape check and still cannot run.
//
// compaction.Config.Validate cannot see the agent, so it accepts a config with
// no Summarizer. Resolving the default summarizer needs the root agent's model,
// and a workflow agent has none. Without a dry run the process starts, reports
// healthy, and fails every request instead.
func TestAgainstAgentsRefusesAConfigNoAgentCanServe(t *testing.T) {
	t.Parallel()

	leaf, err := llmagent.New(llmagent.Config{Name: "leaf"})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	wf, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "wfapp", SubAgents: []agent.Agent{leaf}}})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}

	cfg := &compaction.Config{CompactionInterval: 2}
	base := runner.Config{SessionService: session.InMemoryService()}

	if err := compactionvalidate.AgainstAgents(cfg, agent.NewSingleLoader(wf), base); err == nil {
		t.Error("AgainstAgents() = nil, want a refusal: no Summarizer and no model to default to")
	} else if !strings.Contains(err.Error(), "wfapp") {
		t.Errorf("error %q does not name the app that cannot be served", err)
	}

	// A nil config is not a problem, and neither is a nil loader.
	if err := compactionvalidate.AgainstAgents(nil, agent.NewSingleLoader(wf), base); err != nil {
		t.Errorf("AgainstAgents(nil) = %v, want nil", err)
	}
	if err := compactionvalidate.AgainstAgents(cfg, nil, base); err != nil {
		t.Errorf("AgainstAgents(nil loader) = %v, want nil", err)
	}
}
