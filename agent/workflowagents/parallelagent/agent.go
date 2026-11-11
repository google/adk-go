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

	"golang.org/x/sync/errgroup"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agentstate"
	icontext "google.golang.org/adk/v2/internal/context"
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

func run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	subAgents := ctx.Agent().SubAgents()
	if indexes := utils.GraphResumeIndexes(ctx); len(indexes) > 0 {
		selected := make([]agent.Agent, 0, len(indexes))
		for _, i := range indexes {
			selected = append(selected, subAgents[i])
		}
		subAgents = selected
	}
	if states := agentstate.FromContext(ctx); states != nil {
		return runResumable(ctx, states, subAgents)
	}
	return runSubAgents(ctx, subAgents, nil)
}

// runResumable runs the sub-agents in a resumable invocation, as adk-python's
// ParallelAgent does: on the turn after a pause it runs only the sub-agents
// that have not finished, and it finishes once all of them have. A sub-agent
// that pauses does not stop its siblings.
func runResumable(ctx agent.InvocationContext, states *agentstate.Store, subAgents []agent.Agent) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if len(ctx.Agent().SubAgents()) == 0 {
			return
		}
		if _, resuming := states.Load(ctx.Agent().Name()); resuming {
			subAgents = slices.DeleteFunc(slices.Clone(ctx.Agent().SubAgents()), func(sub agent.Agent) bool {
				return states.Ended(sub.Name())
			})
		} else if !yield(states.Checkpoint(ctx, map[string]any{}), nil) {
			return
		}
		scopes := make([]*agentstate.Scope, len(subAgents))
		for event, err := range runSubAgents(ctx, subAgents, scopes) {
			if !yield(event, err) || err != nil {
				return
			}
		}
		paused := false
		for i, sub := range subAgents {
			if scopes[i] != nil && scopes[i].Paused() {
				paused = true
				continue
			}
			// The sub-agent finished, so a later turn does not run it again.
			// Workflow agents record their own end.
			if !states.Ended(sub.Name()) && !yield(states.End(ctx, sub.Name()), nil) {
				return
			}
		}
		if !paused {
			yield(states.End(ctx, ctx.Agent().Name()), nil)
		}
	}
}

// runSubAgents runs subAgents concurrently, each on its own branch. When
// scopes is not nil, sub-agent i runs in a new agentstate.Scope stored at
// scopes[i], which is safe to read once the returned sequence is done.
func runSubAgents(ctx agent.InvocationContext, subAgents []agent.Agent, scopes []*agentstate.Scope) iter.Seq2[*session.Event, error] {
	curAgent := ctx.Agent()

	// Cancelable so an early consumer stop aborts each sub-agent's in-flight Run
	// promptly. Teardown still runs to completion and the iterator waits for it
	// (see the deferred cleanup), so a sub-agent that blocks in teardown blocks run.
	subAgentsCtx, cancelSubAgents := context.WithCancel(ctx)

	var (
		errGroup, errGroupCtx = errgroup.WithContext(subAgentsCtx)
		doneChan              = make(chan bool)
		resultsChan           = make(chan result)
	)

	for i, sa := range subAgents {
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
			if scopes != nil {
				subCtx, scopes[i] = agentstate.NewScope(subCtx)
			}

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

	return func(yield func(*session.Event, error) bool) {
		// Await sub-agent goroutines (incl. their deferred teardown) before
		// returning, even on early stop: cancel them, then drain resultsChan, which
		// the funnel closes only after errGroup.Wait(). Otherwise teardown can
		// outlive the run and touch an already-ended context.
		defer func() {
			cancelSubAgents()
			close(doneChan)
			for range resultsChan { // drain until the funnel closes resultsChan
			}
		}()

		for res := range resultsChan {
			shouldContinue := yield(res.event, res.err)

			// Signal sub-agent that event processing (including session append) is complete
			if res.ackChan != nil {
				close(res.ackChan)
			}

			if !shouldContinue {
				break
			}
		}
	}
}

func runSubAgent(ctx agent.InvocationContext, agent agent.Agent, results chan<- result, done <-chan bool) error {
	for event, err := range agent.Run(ctx) {
		if err != nil {
			return err
		}
		if agentstate.Pauses(event) {
			agentstate.MarkPaused(ctx)
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
