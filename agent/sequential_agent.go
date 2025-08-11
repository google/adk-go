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
	"google.golang.org/adk/types"
)

// SequentialAgent executes its sub-agents in the order they are listed.
//
// Use the SequentialAgent when you want the execution to occur in a fixed,
// strict order.
type SequentialAgent struct {
	// sequential agent is a LoopAgent with maxIterations = 1
	*loopAgent
}

// NewSequentialAgent creates a new SequentialAgent.
func NewSequentialAgent(name string, opts ...AgentOption) (*SequentialAgent, error) {
	a, err := NewLoopAgent(name, 1, opts...)
	if err != nil {
		return nil, err
	}

	return &SequentialAgent{
		loopAgent: a,
	}, nil
}

type loopAgent = LoopAgent

var _ types.Agent = (*SequentialAgent)(nil)
