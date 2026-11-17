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

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	"google.golang.org/adk/v2/internal/converters"
	"google.golang.org/adk/v2/internal/llminternal"
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
	// In a resumable run (runner.Config.Resumable), the agent stops after
	// the sub-agent whose event pauses the invocation, once that sub-agent's
	// events end. A resumed run counts the passes recorded in its checkpoint
	// (see New) against MaxIterations.
	MaxIterations uint
}

// New creates a LoopAgent.
//
// LoopAgent repeatedly runs its sub-agents in sequence for a specified number
// of iterations or until a termination condition is met.
//
// Use the LoopAgent when your workflow involves repetition or iterative
// refinement, such as like revising code.
//
// A resumable run (runner.Config.Resumable) records its progress in the
// session. Before it starts each sub-agent, the agent yields an event without
// content whose Actions.AgentState is the checkpoint
// {"current_sub_agent": <sub-agent name>, "times_looped": <completed passes>},
// and when the loop ends, after its last pass or an escalation, it yields an
// event with Actions.EndOfAgent set. It yields no end event when a sub-agent
// pauses the invocation or fails, or when the caller stops early. After each
// pass that does not pause, the stored states and end marks of every agent
// below the loop are cleared, so that each one starts the next pass afresh;
// a nested ParallelAgent, for example, runs its sub-agents again. Resuming the
// invocation (runner.WithInvocationID, or a function response to a paused
// call) keeps the checkpoint's pass count and restarts at the sub-agent it
// names, without writing that checkpoint again; a name that is no longer a
// sub-agent restarts the pass at its first sub-agent. A checkpoint with other
// keys or values of the wrong type, such as a negative times_looped, makes the
// run fail with an error. A run that is not resumable yields neither kind of
// event.
//
// The resume behavior described above holds only when this agent is the run's
// root or the root reaches it only through ParallelAgent, SequentialAgent or
// LoopAgent agents; in every other tree, a resumed turn is handled by the
// runner's rule for the agent that made the answered call (see
// runner.WithInvocationID), which may leave this agent's remaining work undone.
// Resuming an invocation (runner.WithInvocationID, or a function response to a
// paused call or an input request) is not supported when a graph workflow
// agent, made by workflowagent.New, runs below a ParallelAgent,
// SequentialAgent or LoopAgent, or when a graph workflow agent, ParallelAgent,
// SequentialAgent or LoopAgent runs as an agent node (workflow.AgentNode) of a
// graph workflow agent: in those trees the resume behavior described above
// does not apply, and a resumed turn may, for example, fail, run nodes or
// sub-agents again, or end before every branch has finished.
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

// loopState is the checkpoint a LoopAgent writes in a resumable run: the
// sub-agent it is about to start and the number of passes it has completed.
// It is adk-python's LoopAgentState (agents/loop_agent.py).
type loopState struct {
	CurrentSubAgent string `json:"current_sub_agent"`
	TimesLooped     uint   `json:"times_looped"`
}

// restartState returns the number of completed passes and the sub-agent index
// Run starts at, and whether that sub-agent's checkpoint is already in the
// session. Without a stored state it starts the first pass at 0. A stored
// state keeps its pass count and restarts at the sub-agent it names, or at 0
// when it names none or a name that is no longer a sub-agent
// (agents/loop_agent.py, _get_start_state; adk-python also logs a warning for
// an unknown name). Graph narrowing (utils.GraphResumeIndexes) overrides the
// stored index. A stored state of the wrong shape is an error.
func (a *loopAgent) restartState(ctx agent.InvocationContext, rc *runconfig.RunConfig, subAgents []agent.Agent) (timesLooped uint, start int, resuming bool, err error) {
	name := ctx.Agent().Name()
	if raw, ok := rc.AgentState(name); ok {
		st, err := converters.FromMapStructureStrict[loopState](raw)
		if err != nil {
			return 0, 0, false, fmt.Errorf("loop agent %q: stored agent state: %w", name, err)
		}
		resuming, timesLooped = true, st.TimesLooped
		if i := slices.IndexFunc(subAgents, func(a agent.Agent) bool { return a.Name() == st.CurrentSubAgent }); st.CurrentSubAgent != "" && i >= 0 {
			start = i
		}
	}
	if indexes := utils.GraphResumeIndexes(ctx); len(indexes) > 0 {
		start, resuming = indexes[0], false
	}
	return timesLooped, start, resuming, nil
}

// descendantNames returns the names of every agent below a, depth first, as
// adk-python's InvocationContext.reset_sub_agent_states recurses
// (agents/invocation_context.py).
func descendantNames(a agent.Agent) []string {
	var names []string
	for _, sub := range a.SubAgents() {
		names = append(names, sub.Name())
		names = append(names, descendantNames(sub)...)
	}
	return names
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
		name := ctx.Agent().Name()
		rc := runconfig.FromContext(ctx)
		resumable := rc != nil && rc.Resumable
		timesLooped, start, resuming, err := a.restartState(ctx, rc, subAgents)
		if err != nil {
			yield(nil, err)
			return
		}
		var descendants []string
		if resumable {
			descendants = descendantNames(ctx.Agent())
		}
		passCtx := ctx
		for a.maxIterations == 0 || timesLooped < a.maxIterations {
			// A pass can yield no events (for example when every sub-agent is
			// empty), so the consumer never gets a chance to stop the loop.
			// Stop on cancellation here and yield the context's error.
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}

			shouldExit, paused := false, false
			for i := start; i < len(subAgents); i++ {
				subAgent := subAgents[i]
				// On a resume the checkpoint naming the first sub-agent is
				// already in the session (agents/loop_agent.py).
				if resumable && !resuming {
					state := map[string]any{"current_sub_agent": subAgent.Name(), "times_looped": int(timesLooped)}
					if !yield(llminternal.NewAgentStateEvent(ctx, name, state), nil) {
						return
					}
				}
				resuming = false
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
					if resumable && event != nil && llminternal.ShouldPauseInvocation(ctx, event) {
						paused = true
					}
				}
				if shouldExit || paused {
					break
				}
			}
			// The paused sub-agent's remaining events have reached the caller;
			// no later sub-agent or pass runs, the descendants keep their
			// states and end marks, and the loop writes no end event, as in
			// adk-python (loop_agent.py, _run_async_impl). Only a resumable run
			// pauses, as in adk-java (InvocationContext.shouldPauseInvocation).
			if paused {
				return
			}
			start = 0
			timesLooped++
			// Every descendant starts the next pass afresh, also after the pass
			// that escalated (loop_agent.py, ctx.reset_sub_agent_states).
			if resumable {
				rc.ResetAgentRuns(descendants)
			}
			passCtx = utils.WithGraphResumeDone(ctx)
			if shouldExit {
				break
			}
		}
		if resumable {
			yield(llminternal.NewEndOfAgentEvent(ctx, name), nil)
		}
	}
}
