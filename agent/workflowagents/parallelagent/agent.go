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

// Package parallelagent provides an agent that runs its sub-agents in parallel.
package parallelagent

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"sync"

	"golang.org/x/sync/errgroup"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/converters"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// Config defines the configuration for a ParallelAgent.
type Config struct {
	// Basic agent setup.
	AgentConfig agent.Config
}

// New creates a ParallelAgent.
//
// Parallel agent runs its sub-agents in parallel in isolated manner.
//
// This approach is beneficial for scenarios requiring multiple perspectives or
// attempts on a single task, such as:
// - Running different algorithms simultaneously.
// - Generating multiple responses for review by a subsequent evaluation agent.
//
// When a direct sub-agent yields an event with Actions.Escalate set, the
// ParallelAgent yields that event, asks that sub-agent for no further event,
// cancels the other sub-agents and waits for them to return. An escalation
// that a deeper agent yields, for example a LoopAgent's sub-agent inside a
// branch, does not stop the siblings. A sub-agent must return once its
// context is canceled, or the ParallelAgent waits for it forever. Once the
// caller stops reading, the ParallelAgent releases no sub-agent: it cancels
// them and waits for them to return, so no sub-agent runs past the event the
// caller did not take.
//
// In a resumable run (runner.Config.Resumable), the ParallelAgent first
// yields an agent-state event with an empty state, authored by itself, before
// any sub-agent starts. It does not start a sub-agent that has already ended
// in the invocation. It yields an end-of-agent event once every sub-agent has
// ended, or after an escalation by a direct sub-agent. Agents built by
// llmagent, sequentialagent, loopagent, parallelagent and workflowagent record
// their end themselves, a graph workflow agent only when it finishes with no
// node waiting for input; a sub-agent built with [agent.New] has to call
// [agent.RecordEndOfAgent] when its Run is finished, or the ParallelAgent
// yields no end-of-agent event. No end-of-agent event follows a sub-agent's
// error, a stop by the caller, or an event that pauses the invocation (a
// long-running function call); after a pause the other sub-agents still run to
// completion. Resuming the invocation runs only the sub-agents whose
// end-of-agent event is not in the session, and yields no second agent-state
// event; a record made by agent.RecordEndOfAgent is not stored, so such a
// sub-agent runs again.
//
// The resume behavior described above holds only when this agent is the run's
// root or the root reaches it only through ParallelAgent, SequentialAgent or
// LoopAgent agents; in every other tree, the runner decides which agent runs a
// resumed turn (see runner.WithInvocationID and Runner.Run), which may leave
// this agent's remaining work undone.
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
		return nil, fmt.Errorf("ParallelAgent doesn't allow custom Run implementations")
	}

	cfg.AgentConfig.Run = run

	parallelAgent, err := agent.New(cfg.AgentConfig)
	if err != nil {
		return nil, err
	}

	internalAgent, ok := parallelAgent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	state := agentinternal.Reveal(internalAgent)
	state.AgentType = agentinternal.TypeParallelAgent
	state.Config = cfg

	return parallelAgent, nil
}

// parallelState is the stored state of a ParallelAgent. It is always empty, as
// adk-python's BaseAgentState is; decoding into it rejects any key.
type parallelState struct{}

// asksParentToExit reports whether ev is an escalation by one of the direct
// sub-agents named in direct. An escalation that a nested agent re-yields
// keeps its deeper author and does not count (agents/parallel_agent.py).
func asksParentToExit(ev *session.Event, direct map[string]bool) bool {
	return ev.Actions.Escalate && direct[ev.Author]
}

// allEnded reports whether every agent in subAgents has ended in the run of
// rc.
func allEnded(rc *runconfig.RunConfig, subAgents []agent.Agent) bool {
	return !slices.ContainsFunc(subAgents, func(a agent.Agent) bool { return !rc.AgentEnded(a.Name()) })
}

func run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		curAgent := ctx.Agent()
		all := curAgent.SubAgents()
		if len(all) == 0 {
			return
		}
		rc := runconfig.FromContext(ctx)
		resumable := rc != nil && rc.Resumable
		if resumable {
			if raw, ok := rc.AgentState(curAgent.Name()); ok {
				if _, err := converters.FromMapStructureStrict[parallelState](raw); err != nil {
					yield(nil, fmt.Errorf("parallel agent %q: stored agent state: %w", curAgent.Name(), err))
					return
				}
			} else if !yield(llminternal.NewAgentStateEvent(ctx, curAgent.Name(), map[string]any{}), nil) {
				return
			}
		}

		// Cancelable so an early consumer stop aborts each sub-agent's in-flight Run
		// promptly. Teardown still runs to completion and the iterator waits for it
		// (see stop), so a sub-agent that blocks in teardown blocks run.
		subAgentsCtx, cancelSubAgents := context.WithCancel(ctx)

		var (
			errGroup, errGroupCtx = errgroup.WithContext(subAgentsCtx)
			doneChan              = make(chan bool)
			resultsChan           = make(chan result)
		)

		subAgents := all
		if indexes := utils.GraphResumeIndexes(ctx); len(indexes) > 0 {
			selected := make([]agent.Agent, 0, len(indexes))
			for _, i := range indexes {
				selected = append(selected, all[i])
			}
			subAgents = selected
		}
		subAgents = slices.DeleteFunc(slices.Clone(subAgents), func(a agent.Agent) bool { return rc.AgentEnded(a.Name()) })
		for _, sa := range subAgents {
			branch := fmt.Sprintf("%s.%s", curAgent.Name(), sa.Name())
			if ctx.Branch() != "" {
				branch = fmt.Sprintf("%s.%s", ctx.Branch(), branch)
			}
			subAgent := sa
			errGroup.Go(func() error {
				// Keep the parent's isolation scope: a workflow node stamps it on
				// every event the branches emit, and the LLM history filter matches
				// it exactly, so an unscoped branch would lose its own turns.
				subCtx := icontext.NewInvocationContext(errGroupCtx, icontext.InvocationContextParams{
					Artifacts:      ctx.Artifacts(),
					Memory:         ctx.Memory(),
					Session:        ctx.Session(),
					Branch:         branch,
					IsolationScope: ctx.IsolationScope(),
					Agent:          subAgent,
					UserContent:    ctx.UserContent(),
					RunConfig:      ctx.RunConfig(),
					InvocationID:   ctx.InvocationID(),
				})

				if err := runSubAgent(subCtx, subAgent, resultsChan, doneChan); err != nil {
					return fmt.Errorf("failed to run sub-agent %q: %w", subAgent.Name(), err)
				}

				return nil
			})
		}

		go func() {
			if err := errGroup.Wait(); err != nil {
				select {
				case resultsChan <- result{err: err}:
				case <-doneChan:
				}
			}
			close(resultsChan)
		}()

		// Await sub-agent goroutines (incl. their deferred teardown) before
		// returning, even on early stop: cancel them, then drain resultsChan, which
		// the funnel closes only after errGroup.Wait(). Otherwise teardown can
		// outlive the run and touch an already-ended context.
		stop := sync.OnceFunc(func() {
			cancelSubAgents()
			close(doneChan)
			for range resultsChan { // drain until the funnel closes resultsChan
			}
		})
		defer stop()

		direct := make(map[string]bool, len(all))
		for _, sa := range all {
			direct[sa.Name()] = true
		}
		var escalated, paused, failed, stopped bool
		for res := range resultsChan {
			if res.err != nil {
				failed = true
			}
			// Recorded before the escalation check: once yield has returned
			// false, nothing may be yielded again, the end event included.
			if !yield(res.event, res.err) {
				stopped = true
			}
			if resumable && res.event != nil && llminternal.ShouldPauseInvocation(ctx, res.event) {
				paused = true
			}
			if res.event != nil && asksParentToExit(res.event, direct) {
				// The escalating branch's ack stays open, so it never asks its
				// sub-agent for another event; stop cancels it.
				escalated = true
				break
			}
			// Checked before the release: once the caller stopped, no branch may
			// run past the yield of an event the caller did not take; stop
			// cancels the branch instead.
			if stopped {
				break
			}
			// Signal sub-agent that event processing (including session append) is complete
			if res.ackChan != nil {
				close(res.ackChan)
			}
		}
		// The end event is decided only after every branch has returned, so
		// the ends they recorded are all visible.
		stop()
		if stopped || failed || paused || !resumable {
			return
		}
		if escalated || allEnded(rc, all) {
			yield(llminternal.NewEndOfAgentEvent(ctx, curAgent.Name()), nil)
		}
	}
}

func runSubAgent(ctx agent.InvocationContext, agent agent.Agent, results chan<- result, done <-chan bool) error {
	for event, err := range agent.Run(ctx) {
		if err != nil {
			return err
		}

		ackChan := make(chan struct{})

		select {
		case <-done:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case results <- result{
			event:   event,
			ackChan: ackChan,
		}:
			// Wait for runner to finish processing before continuing to next iteration
			select {
			case <-ackChan:
			case <-done:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return nil
}

type result struct {
	event   *session.Event
	err     error
	ackChan chan struct{}
}
