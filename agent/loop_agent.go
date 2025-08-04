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
	"context"
	"fmt"
	"iter"

	"google.golang.org/adk/types"
)

// LoopAgent repeatedly runs a sequence of agents for a specified number of
// iterations or until a termination condition is met.
//
// Use the LoopAgent when your workflow involves repetition or iterative
// refinement, such as like revising code.
type LoopAgent struct {
	agentSpec *types.AgentSpec

	maxIterations uint
}

// NewLoopAgent creates a LoopAgent.
// If maxIterations == 0, it runs indefinitely or until any subagent escalates.
func NewLoopAgent(name string, maxIterations uint, opts ...AgentOption) (*LoopAgent, error) {
	agentSpec := &types.AgentSpec{Name: name}

	a := &LoopAgent{
		maxIterations: maxIterations,
		agentSpec:     agentSpec,
	}

	for _, o := range opts {
		if _, ok := o.(LLMAgentOption); ok {
			continue
		}
		if err := o.apply2AgentSpec(a); err != nil {
			return nil, err
		}
	}

	if err := agentSpec.Init(a); err != nil {
		return nil, fmt.Errorf("failed to init loop agent spec: %v", err)
	}

	return a, nil
}

func (a *LoopAgent) Spec() *types.AgentSpec {
	return a.agentSpec
}

func (a *LoopAgent) Run(ctx context.Context, ictx *types.InvocationContext) iter.Seq2[*types.Event, error] {
	count := a.maxIterations

	return func(yield func(*types.Event, error) bool) {
		for {
			for _, subAgent := range ictx.Agent.Spec().SubAgents {
				for event, err := range subAgent.Run(ctx, ictx) {
					if !yield(event, err) {
						return
					}

					if event.Actions != nil && event.Actions.Escalate {
						return
					}
				}
			}

			if count > 0 {
				count--
				if count == 0 {
					return
				}
			}
		}
	}
}

var _ types.Agent = (*LoopAgent)(nil)
