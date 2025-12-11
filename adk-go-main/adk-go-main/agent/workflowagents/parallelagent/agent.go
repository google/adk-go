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
	"fmt"
	"iter"

	"golang.org/x/sync/errgroup"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
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
	curAgent := ctx.Agent()

	var (
		errGroup, errGroupCtx = errgroup.WithContext(ctx)
		doneChan              = make(chan bool)
		resultsChan           = make(chan result)
	)

	for _, sa := range ctx.Agent().SubAgents() {
		branch := fmt.Sprintf("%s.%s", curAgent.Name(), sa.Name())
		if ctx.Branch() != "" {
			branch = fmt.Sprintf("%s.%s", ctx.Branch(), branch)
		}
		subAgent := sa
		errGroup.Go(func() error {
			subCtx := icontext.NewInvocationContext(errGroupCtx, icontext.InvocationContextParams{
				Artifacts:   ctx.Artifacts(),
				Memory:      ctx.Memory(),
				Session:     ctx.Session(),
				Branch:      branch,
				Agent:       subAgent,
				UserContent: ctx.UserContent(),
				RunConfig:   ctx.RunConfig(),
			})

			if err := runSubAgent(subCtx, subAgent, resultsChan, doneChan); err != nil {
				return fmt.Errorf("failed to run sub-agent %q: %w", subAgent.Name(), err)
			}

			return nil
		})
	}

	go func() {
		_ = errGroup.Wait() // this error is already sent to the user via iterator
		close(resultsChan)
	}()

	return func(yield func(*session.Event, error) bool) {
		defer close(doneChan)

		for res := range resultsChan {
			if !yield(res.event, res.err) {
				break
			}
		}
	}
}

func runSubAgent(ctx agent.InvocationContext, agent agent.Agent, results chan<- result, done <-chan bool) error {
	for event, err := range agent.Run(ctx) {
		select {
		case <-done:
			return nil
		case <-ctx.Done():
			select {
			case <-done:
			case results <- result{
				err: ctx.Err(),
			}:
			}
			return ctx.Err()
		case results <- result{
			event: event,
			err:   err,
		}:
			if err != nil {
				return err
			}
		}
	}
	return nil
}

type result struct {
	event *session.Event
	err   error
}
