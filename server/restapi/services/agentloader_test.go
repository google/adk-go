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

package services

import (
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
)

func TestDuplicateName(t *testing.T) {
	agent1, err := llmagent.New(llmagent.Config{
		Name: "weather_time_agent",
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}
	// duplicate name
	agent2, err := llmagent.New(llmagent.Config{
		Name: "weather_time_agent",
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}
	agent3, err := llmagent.New(llmagent.Config{
		Name: "unique",
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	tests := []struct {
		name    string
		root    agent.Agent
		agents  []agent.Agent
		wantErr bool
	}{
		{
			name:    "root only",
			root:    agent1,
			agents:  []agent.Agent{},
			wantErr: false,
		},
		{
			name:    "root duplicate object",
			root:    agent1,
			agents:  []agent.Agent{agent1},
			wantErr: true,
		},
		{
			name:    "root duplicate name",
			root:    agent1,
			agents:  []agent.Agent{agent2},
			wantErr: true,
		},
		{
			name:    "non-root duplicate name",
			root:    agent3,
			agents:  []agent.Agent{agent1, agent2},
			wantErr: true,
		},
		{
			name:    "non-root duplicate object",
			root:    agent3,
			agents:  []agent.Agent{agent1, agent1},
			wantErr: true,
		},
		{
			name:    "no duplicates",
			root:    agent1,
			agents:  []agent.Agent{agent3},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		_, err := NewMultiAgentLoader(tt.root, tt.agents...)
		if (err != nil) != tt.wantErr {
			t.Errorf("NewMultiAgentLoader() name=%v, error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}
	}

}
