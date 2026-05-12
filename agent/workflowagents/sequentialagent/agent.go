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

// Package sequentialagent provides an agent that runs its sub-agents in a sequence.
package sequentialagent

import (
	"fmt"
	"iter"
	"sync"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// New creates a SequentialAgent.
//
// SequentialAgent executes its sub-agents once, in the order they are listed.
type seqAgent struct {
	agent.Agent
	*agentinternal.State
	impl *sequentialAgent
}

func (s *seqAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return s.impl.RunLive(ctx)
}

// Use the SequentialAgent when you want the execution to occur in a fixed,
// strict order.
func New(cfg Config) (agent.Agent, error) {
	if cfg.AgentConfig.Run != nil {
		return nil, fmt.Errorf("LoopAgent doesn't allow custom Run implementations")
	}

	sequentialAgentImpl := &sequentialAgent{}
	cfg.AgentConfig.Run = sequentialAgentImpl.Run

	sequentialAgent, err := agent.New(cfg.AgentConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base agent: %w", err)
	}

	internalAgent, ok := sequentialAgent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	state := agentinternal.Reveal(internalAgent)
	state.AgentType = agentinternal.TypeSequentialAgent
	state.Config = cfg

	return &seqAgent{Agent: sequentialAgent, State: state, impl: sequentialAgentImpl}, nil
}

// Config defines the configuration for a SequentialAgent.
type Config struct {
	// Basic agent setup.
	AgentConfig agent.Config
}

type sequentialAgent struct{}

func (a *sequentialAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for _, subAgent := range ctx.Agent().SubAgents() {
			for event, err := range subAgent.Run(ctx) {
				// TODO: ensure consistency -- if there's an error, return and close iterator, verify everywhere in ADK.
				if !yield(event, err) {
					return
				}
			}
		}
	}
}

type sequentialLiveSession struct {
	mu         sync.Mutex
	activeSess agent.LiveSession
	closed     bool
}

func (s *sequentialLiveSession) Send(req agent.LiveRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("session is closed")
	}
	if s.activeSess == nil {
		return fmt.Errorf("no active sub-agent live session")
	}
	return s.activeSess.Send(req)
}

func (s *sequentialLiveSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	if s.activeSess != nil {
		return s.activeSess.Close()
	}
	return nil
}

func (s *sequentialLiveSession) setActiveSession(sess agent.LiveSession) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeSess = sess
}

func (a *sequentialAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	subAgents := ctx.Agent().SubAgents()
	if len(subAgents) == 0 {
		return nil, nil, fmt.Errorf("sequential agent has no sub-agents")
	}

	// Inject task_completed tool into sub LLM agents
	type taskCompletedArgs struct{}
	type taskCompletedResults struct {
		Result string `json:"result"`
	}

	taskCompletedTool, err := functiontool.New(functiontool.Config{
		Name:        "task_completed",
		Description: "Signals that the agent has successfully completed the user's question or task.",
	}, func(ctx tool.Context, args taskCompletedArgs) (taskCompletedResults, error) {
		return taskCompletedResults{Result: "Task completion signaled."}, nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create task_completed tool: %w", err)
	}

	for _, subAgent := range subAgents {
		if llmAgent, ok := subAgent.(llminternal.Agent); ok {
			state := llminternal.Reveal(llmAgent)
			hasTaskCompleted := false
			for _, t := range state.Tools {
				if t.Name() == "task_completed" {
					hasTaskCompleted = true
					break
				}
			}
			if !hasTaskCompleted {
				state.Tools = append(state.Tools, taskCompletedTool)
				instructionSuffix := "\nIf you finished the user's request according to its description, call the task_completed function to exit so the next agents can take over. When calling this function, do not generate any text other than the function call."
				state.Instruction += instructionSuffix
			}
		}
	}

	seqSess := &sequentialLiveSession{}

	wrappedIter := func(yield func(*session.Event, error) bool) {
		for _, subAgent := range subAgents {
			liveAgent, ok := subAgent.(interface {
				RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
			})
			if !ok {
				if !yield(nil, fmt.Errorf("sub-agent %s does not support Live Run", subAgent.Name())) {
					return
				}
				return
			}

			subSess, innerIter, err := liveAgent.RunLive(ctx)
			if err != nil {
				if !yield(nil, fmt.Errorf("sub-agent %s RunLive failed: %w", subAgent.Name(), err)) {
					return
				}
				return
			}

			seqSess.setActiveSession(subSess)

			for ev, err := range innerIter {
				if !yield(ev, err) {
					subSess.Close()
					return
				}
			}
			subSess.Close()
		}
	}

	return seqSess, wrappedIter, nil
}
