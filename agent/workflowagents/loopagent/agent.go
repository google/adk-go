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

// Package loopagent provides an agent that repeatedly runs its sub-agents for a
// specified number of iterations or until termination condition is met.
package loopagent

import (
	"fmt"
	"iter"
	"slices"
	"sync"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// Config defines the configuration for a LoopAgent.
type Config struct {
	// Basic agent setup.
	AgentConfig agent.Config

	// If MaxIterations == 0, then LoopAgent runs indefinitely or until any
	// sub-agent escalates. A sub-agent error ends the loop regardless of
	// MaxIterations. The context is checked before each pass, and if it is
	// done the loop ends with the context's error. A LoopAgent with no
	// sub-agents returns immediately without yielding any event or error.
	MaxIterations uint
}

// New creates a LoopAgent.
//
// LoopAgent repeatedly runs its sub-agents in sequence for a specified number
// of iterations or until a termination condition is met.
//
// Use the LoopAgent when your workflow involves repetition or iterative
// refinement, such as like revising code.
func New(cfg Config) (agent.Agent, error) {
	if cfg.AgentConfig.Run != nil {
		return nil, fmt.Errorf("LoopAgent doesn't allow custom Run implementations")
	}

	loopAgentImpl := &loopAgent{
		maxIterations: cfg.MaxIterations,
	}
	cfg.AgentConfig.Run = loopAgentImpl.Run

	loopAgent, err := agent.New(cfg.AgentConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base agent: %w", err)
	}

	internalAgent, ok := loopAgent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	state := agentinternal.Reveal(internalAgent)
	state.AgentType = agentinternal.TypeLoopAgent
	state.Config = cfg

	return loopAgent, nil
}

const maxTrackedPausedInvocations = 256

type loopAgent struct {
	maxIterations uint

	mu           sync.Mutex
	pausedPasses map[string]uint
	pausedOrder  []string
}

func pauseKey(ctx agent.InvocationContext) string {
	if ctx.InvocationID() == "" {
		return ""
	}
	sessionID := ""
	if ctx.Session() != nil {
		sessionID = ctx.Session().ID()
	}
	return sessionID + "\x00" + ctx.InvocationID() + "\x00" + ctx.Branch()
}

func (a *loopAgent) loadPausedPasses(ctx agent.InvocationContext) uint {
	key := pauseKey(ctx)
	if key != "" {
		a.mu.Lock()
		passes, ok := a.pausedPasses[key]
		a.mu.Unlock()
		if ok {
			return passes
		}
	}
	return utils.CompletedLoopPasses(ctx)
}

func (a *loopAgent) savePausedPasses(ctx agent.InvocationContext, passes uint) {
	key := pauseKey(ctx)
	if key == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.pausedPasses == nil {
		a.pausedPasses = make(map[string]uint)
	}
	if _, exists := a.pausedPasses[key]; !exists {
		if len(a.pausedPasses) >= maxTrackedPausedInvocations && len(a.pausedOrder) > 0 {
			oldest := a.pausedOrder[0]
			a.pausedOrder = a.pausedOrder[1:]
			delete(a.pausedPasses, oldest)
		}
		a.pausedOrder = append(a.pausedOrder, key)
	}
	a.pausedPasses[key] = passes
}

func (a *loopAgent) clearPausedPasses(ctx agent.InvocationContext) {
	key := pauseKey(ctx)
	if key == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, exists := a.pausedPasses[key]; !exists {
		return
	}
	delete(a.pausedPasses, key)
	a.pausedOrder = slices.DeleteFunc(a.pausedOrder, func(k string) bool { return k == key })
}

func (a *loopAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// Without sub-agents there is nothing to run, escalate or fail. With
		// MaxIterations == 0 the loop would spin forever, and with it set the
		// loop would only count down through empty passes. Return at once in
		// both cases, as adk-python does.
		subAgents := ctx.Agent().SubAgents()
		if len(subAgents) == 0 {
			return
		}

		indexes := utils.GraphResumeIndexes(ctx)
		resumingPass := len(indexes) > 0
		var completedPasses uint
		startIdx := 0
		if resumingPass {
			startIdx = indexes[0]
			completedPasses = a.loadPausedPasses(ctx)
		}
		if a.maxIterations > 0 && completedPasses >= a.maxIterations {
			a.clearPausedPasses(ctx)
			return
		}
		var remaining uint
		if a.maxIterations > 0 {
			remaining = a.maxIterations - completedPasses
		}
		passCtx := ctx

		for {
			// A pass can yield no events (for example when every sub-agent is
			// empty), so the consumer never gets a chance to stop the loop.
			// Stop on cancellation here and yield the context's error.
			if err := ctx.Err(); err != nil {
				a.clearPausedPasses(ctx)
				yield(nil, err)
				return
			}

			shouldEscalate := false
			shouldPause := false
			for i := startIdx; i < len(subAgents); i++ {
				subAgent := subAgents[i]
				subCtx := passCtx
				if resumingPass {
					if slices.Contains(indexes, i) {
						subAgent = utils.ResumeSubAgent(passCtx, subAgent)
						subCtx = utils.WithWorkflowResuming(passCtx)
					} else {
						subCtx = utils.WithGraphResumeDone(passCtx)
					}
				}
				for event, err := range subAgent.Run(subCtx) {
					// A failed sub-agent ends the loop, as in adk-python. Retrying
					// it would repeat the same failure, forever when there is no
					// MaxIterations.
					if !yield(event, err) || err != nil {
						a.clearPausedPasses(ctx)
						return
					}
					if event != nil {
						if event.Actions.Escalate {
							shouldEscalate = true
						}
						if utils.ShouldPauseInvocation(event) {
							shouldPause = true
						}
					}
				}
				if shouldPause || utils.HasUnresolvedLongRunningCalls(ctx) {
					a.savePausedPasses(ctx, completedPasses)
					ctx.EndInvocation()
					return
				}
				if shouldEscalate {
					a.clearPausedPasses(ctx)
					return
				}
			}

			completedPasses++
			startIdx = 0
			resumingPass = false
			passCtx = utils.WithGraphResumeDone(ctx)

			if a.maxIterations > 0 {
				remaining--
				if remaining == 0 {
					a.clearPausedPasses(ctx)
					return
				}
			}
		}
	}
}
