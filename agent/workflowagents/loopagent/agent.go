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

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agentstate"
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

type loopAgent struct {
	maxIterations uint
}

// The agent state keys of a loop, as in adk-python's LoopAgentState.
const (
	currentSubAgent = "current_sub_agent"
	timesLooped     = "times_looped"
)

func (a *loopAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	if states := agentstate.FromContext(ctx); states != nil {
		return a.runResumable(ctx, states)
	}
	count := a.maxIterations

	return func(yield func(*session.Event, error) bool) {
		// Without sub-agents there is nothing to run, escalate or fail. With
		// MaxIterations == 0 the loop would spin forever, and with it set the
		// loop would only count down through empty passes. Return at once in
		// both cases, as adk-python does.
		if len(ctx.Agent().SubAgents()) == 0 {
			return
		}
		subAgents := ctx.Agent().SubAgents()
		if indexes := utils.GraphResumeIndexes(ctx); len(indexes) > 0 {
			subAgents = subAgents[indexes[0]:]
		}
		passCtx := ctx
		for {
			// A pass can yield no events (for example when every sub-agent is
			// empty), so the consumer never gets a chance to stop the loop.
			// Stop on cancellation here and yield the context's error.
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}

			shouldExit := false
			for _, subAgent := range subAgents {
				for event, err := range subAgent.Run(passCtx) {
					// A failed sub-agent ends the loop, as in adk-python. Retrying
					// it would repeat the same failure, forever when there is no
					// MaxIterations.
					if !yield(event, err) || err != nil {
						return
					}

					if event != nil && event.Actions.Escalate {
						shouldExit = true
					}
				}
				if shouldExit {
					return
				}
			}
			subAgents = ctx.Agent().SubAgents()
			passCtx = utils.WithGraphResumeDone(ctx)

			if count > 0 {
				count--
				if count == 0 {
					return
				}
			}
		}
	}
}

// runResumable runs the loop in a resumable invocation, as adk-python's
// LoopAgent does: it saves the sub-agent it is on and how many passes it has
// finished, stops when the invocation pauses on a long-running call, and on the
// next turn continues from there, so MaxIterations counts passes across turns.
func (a *loopAgent) runResumable(ctx agent.InvocationContext, states *agentstate.Store) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		subAgents := ctx.Agent().SubAgents()
		if len(subAgents) == 0 {
			return
		}
		looped, start := 0, 0
		state, resuming := states.Load(ctx.Agent().Name())
		if resuming {
			looped = agentstate.Int(state, timesLooped)
			start = agentstate.Index(subAgents, state, currentSubAgent)
		} else if indexes := utils.GraphResumeIndexes(ctx); len(indexes) > 0 {
			start = indexes[0]
		}
		passCtx := ctx
		for a.maxIterations == 0 || uint(looped) < a.maxIterations {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}
			for _, subAgent := range subAgents[start:] {
				// The sub-agent being resumed was saved on the turn it started.
				if !resuming && !yield(states.Checkpoint(ctx, map[string]any{currentSubAgent: subAgent.Name(), timesLooped: looped}), nil) {
					return
				}
				resuming = false
				subCtx, scope := agentstate.NewScope(passCtx)
				escalated := false
				for event, err := range subAgent.Run(subCtx) {
					// A failed sub-agent ends the loop, as in Run.
					if !yield(event, err) || err != nil {
						return
					}
					if agentstate.Pauses(event) {
						agentstate.MarkPaused(subCtx)
					}
					escalated = escalated || event != nil && event.Actions.Escalate
				}
				// The loop continues from this sub-agent once its call is
				// answered on a later turn.
				if scope.Paused() {
					return
				}
				if escalated {
					yield(states.End(ctx, ctx.Agent().Name()), nil)
					return
				}
			}
			start = 0
			looped++
			passCtx = utils.WithGraphResumeDone(ctx)
		}
		yield(states.End(ctx, ctx.Agent().Name()), nil)
	}
}
