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
	"iter"
	"testing"

	"google.golang.org/adk/session"
)

var _ Agent = (*testAgent)(nil)

type testAgent struct {
	name string
}

func (a *testAgent) Name() string {
	return a.name
}

func (a *testAgent) Description() string {
	panic("not implemented")
}

func (a *testAgent) Run(InvocationContext) iter.Seq2[*session.Event, error] {
	panic("not implemented")
}

func (a *testAgent) SubAgents() []Agent {
	panic("not implemented")
}

func (a *testAgent) internal() *agent {
	panic("not implemented")
}

func TestDuplicateName(t *testing.T) {
	agent1 := &testAgent{name: "weather_time_agent"}
	// duplicate name
	agent2 := &testAgent{name: "weather_time_agent"}
	agent3 := &testAgent{name: "unique"}

	tests := []struct {
		name    string
		root    Agent
		agents  []Agent
		wantErr bool
	}{
		{
			name:    "root only",
			root:    agent1,
			agents:  []Agent{},
			wantErr: false,
		},
		{
			name:    "root duplicate object",
			root:    agent1,
			agents:  []Agent{agent1},
			wantErr: true,
		},
		{
			name:    "root duplicate name",
			root:    agent1,
			agents:  []Agent{agent2},
			wantErr: true,
		},
		{
			name:    "non-root duplicate name",
			root:    agent3,
			agents:  []Agent{agent1, agent2},
			wantErr: true,
		},
		{
			name:    "non-root duplicate object",
			root:    agent3,
			agents:  []Agent{agent1, agent1},
			wantErr: true,
		},
		{
			name:    "no duplicates",
			root:    agent1,
			agents:  []Agent{agent3},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		_, err := NewMultiLoader(tt.root, tt.agents...)
		if (err != nil) != tt.wantErr {
			t.Errorf("NewMultiLoader() name=%v, error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}
	}
}
