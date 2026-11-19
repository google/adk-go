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

package context_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
)

func TestReadonlyContextAgentName(t *testing.T) {
	root, err := agent.New(agent.Config{Name: "root"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	inv := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: root})

	tests := []struct {
		name string
		ctx  agent.InvocationContext
	}{
		{
			// Plugin callbacks hand this context to instruction providers.
			name: "callback context",
			ctx:  agent.NewCallbackContext(inv, nil),
		},
		{
			// Answers AgentName but returns nil from Agent.
			name: "invocation context without an agent that names one",
			ctx:  namedWithoutAgent{InvocationContext: inv, name: "root"},
		},
		{
			// The bare invocation context has no AgentName method.
			name: "invocation context",
			ctx:  inv,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := icontext.NewReadonlyContext(tt.ctx).AgentName(); got != "root" {
				t.Errorf("AgentName() = %q, want %q", got, "root")
			}
		})
	}
}

// namedWithoutAgent is an invocation context, such as one decorated outside
// ADK, that answers AgentName and returns nil from Agent.
type namedWithoutAgent struct {
	agent.InvocationContext
	name string
}

func (c namedWithoutAgent) AgentName() string { return c.name }

func (namedWithoutAgent) Agent() agent.Agent { return nil }
