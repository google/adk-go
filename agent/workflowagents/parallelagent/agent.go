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
	"errors"
	"fmt"
	"iter"
	"slices"
	"sync/atomic"

	"golang.org/x/sync/errgroup"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	icontext "google.golang.org/adk/v2/internal/context"
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
// If a direct sub-agent yields an event with Actions.Escalate, the agent yields
// that event, cancels the remaining branches and ends. If an error that would
// reach this agent has already been yielded by then, that error is returned
// instead of the event. That covers an error a direct sub-agent yields,
// including one a SequentialAgent or LoopAgent sub-agent passes up from its own
// sub-agents, and an error from inside a nested ParallelAgent reached through
// ParallelAgents, SequentialAgents and LoopAgents only, even while the agents
// in between are still cleaning up. Other agents, such as a workflow agent or a custom agent,
// can catch or retry an error, so an error from inside one counts only once
// that agent has yielded it. An error from a branch that a nested ParallelAgent
// canceled after its own escalation never counts. Escalations from deeper
// agents are passed through and do not stop this agent.
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
	curAgent := ctx.Agent()

	// Cancelable so an early consumer stop aborts each sub-agent's in-flight Run
	// promptly. Teardown still runs to completion and the iterator waits for it
	// (see the deferred cleanup), so a sub-agent that blocks in teardown blocks run.
	subAgentsCtx, cancelSubAgents := context.WithCancel(ctx)

	var (
		errGroup, errGroupCtx = errgroup.WithContext(subAgentsCtx)
		doneChan              = make(chan bool)
		resultsChan           = make(chan result)
		// The first error a sub-agent yielded, recorded as soon as it is
		// yielded, before that sub-agent's Run unwinds. The escalation path
		// seals it with errSealed, so errors from the branches it cancels are
		// neither recorded nor reported.
		firstErr atomic.Pointer[error]
	)

	subAgentNames := make(map[string]bool, len(ctx.Agent().SubAgents()))
	for _, sa := range ctx.Agent().SubAgents() {
		subAgentNames[sa.Name()] = true
		branch := fmt.Sprintf("%s.%s", curAgent.Name(), sa.Name())
		if ctx.Branch() != "" {
			branch = fmt.Sprintf("%s.%s", ctx.Branch(), branch)
		}
		subAgent := sa
		errGroup.Go(func() error {
			wrap := func(err error) error {
				return fmt.Errorf("failed to run sub-agent %q: %w", subAgent.Name(), err)
			}
			onErr := func(err error) {
				err = wrap(err)
				// Once recorded, this agent returns the error: in place of an
				// escalation, or from errGroup.Wait otherwise. An enclosing
				// ParallelAgent would get it only after that Wait, so report
				// it there now as well.
				if firstErr.CompareAndSwap(nil, &err) {
					reportErr(ctx, err)
				}
			}
			// A ParallelAgent nested below subAgent reports its error here as
			// soon as it is recorded, if the agents in between pass errors
			// straight up.
			reporterCtx := context.WithValue(errGroupCtx, errReporterKey{}, errReporter{
				invocationID: ctx.InvocationID(),
				root:         subAgent,
				report:       onErr,
			})
			// Keep the parent's isolation scope: a workflow node stamps it on
			// every event the branches emit, and the LLM history filter matches
			// it exactly, so an unscoped branch would lose its own turns.
			subCtx := icontext.NewInvocationContext(reporterCtx, icontext.InvocationContextParams{
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

			if err := runSubAgent(subCtx, subAgent, resultsChan, doneChan, onErr); err != nil {
				return wrap(err)
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
			// An escalation from a direct sub-agent ends this agent, as in
			// adk-python (_asks_this_agent_to_exit). Escalations from deeper agents
			// are passed through and do not end this one. The deferred cleanup
			// cancels the remaining branches.
			if res.event != nil && res.event.Actions.Escalate && subAgentNames[res.event.Author] {
				// An error a sub-agent yielded before this point must still reach
				// the caller, even if that sub-agent is still unwinding or the
				// caller has canceled since. Sealing the record keeps errors from
				// the branches canceled below out of it.
				if !firstErr.CompareAndSwap(nil, &errSealed) {
					yield(nil, *firstErr.Load())
					return
				}
				yield(res.event, nil)
				return
			}

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

// runSubAgent forwards the events of agent to results. onErr is called with an
// error the agent yields before runSubAgent returns it, since returning waits
// for the agent's Run to unwind.
func runSubAgent(ctx agent.InvocationContext, agent agent.Agent, results chan<- result, done <-chan bool, onErr func(error)) error {
	for event, err := range agent.Run(ctx) {
		if err != nil {
			onErr(err)
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

// errReporterKey is the context key of the errReporter of the closest
// enclosing ParallelAgent.
type errReporterKey struct{}

// errReporter records an error in an enclosing ParallelAgent. It is scoped to
// one invocation, so an agent run in a separate invocation (for example
// through an agent tool) does not report into it.
type errReporter struct {
	invocationID string
	// root is the enclosing ParallelAgent's direct sub-agent this reporter
	// was made for.
	root   agent.Agent
	report func(error)
}

// errSealed marks the error record of a ParallelAgent that has ended on an
// escalation.
var errSealed = errors.New("parallel agent ended on an escalation")

// reportErr records err in the closest enclosing ParallelAgent of ctx, if the
// agents between the two hand an error straight up.
func reportErr(ctx agent.InvocationContext, err error) {
	if r, ok := ctx.Value(errReporterKey{}).(errReporter); ok && r.invocationID == ctx.InvocationID() && passesErrorsUp(r.root, ctx.Agent()) {
		r.report(err)
	}
}

// passesErrorsUp reports whether target is root, or is reached from root only
// through SequentialAgents and LoopAgents. Those yield a sub-agent's error to
// their caller as soon as it is yielded. Any other agent may catch or retry it,
// so its error counts only once it yields it.
func passesErrorsUp(root, target agent.Agent) bool {
	if root == target {
		return true
	}
	internalAgent, ok := root.(agentinternal.Agent)
	if !ok {
		return false
	}
	switch agentinternal.Reveal(internalAgent).AgentType {
	case agentinternal.TypeSequentialAgent, agentinternal.TypeLoopAgent:
		return slices.ContainsFunc(root.SubAgents(), func(sub agent.Agent) bool { return passesErrorsUp(sub, target) })
	}
	return false
}

type result struct {
	event   *session.Event
	err     error
	ackChan chan struct{}
}
