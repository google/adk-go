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

	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/workflow"
)

func TestNewAgentNode_NameInheritedFromAgent(t *testing.T) {
	t.Parallel()
	const agentName = "my_inner_agent"
	a, err := llmagent.New(llmagent.Config{Name: agentName, Mode: llmagent.ModeChat})
	if err != nil {
		t.Fatal(err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := node.Name(), agentName; got != want {
		t.Errorf("node.Name() = %q, want %q (must inherit from wrapped agent)", got, want)
	}
}
