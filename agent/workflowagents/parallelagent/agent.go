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

package parallelagent

import (
	"fmt"
	"iter"
	"sync"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

type Config struct {
	// Basic agent setup.
	AgentConfig agent.Config
}

// New creates a ParallelAgent.
//
// It is a shell agent that run its sub-agents in parallel in isolated manner.
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

	return agent.New(cfg.AgentConfig)
}

func run(ctx agent.Context) iter.Seq2[*session.Event, error] {
	curAgent := ctx.Agent()

	var wg sync.WaitGroup
	results := make(chan result)

	for _, subAgent := range ctx.Agent().SubAgents() {
		branch := fmt.Sprintf("%s.%s", curAgent.Name(), subAgent.Name())
		if ctx.Branch() != "" {
			branch = fmt.Sprintf("%s.%s", ctx.Branch(), branch)
		}

		wg.Add(1)
		go func() {
			defer wg.Done()

			ctx := agent.NewContext(ctx, subAgent, ctx.UserContent(), ctx.Session(), branch)
			runSubAgent(ctx, subAgent, results)
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	return func(yield func(*session.Event, error) bool) {
		for res := range results {
			if !yield(res.event, res.err) {
				break
			}
		}

		ctx.End()
	}
}

func runSubAgent(ctx agent.Context, agent agent.Agent, results chan<- result) {
	for event, err := range agent.Run(ctx) {
		select {
		case <-ctx.Done():
			return
		case results <- result{
			event: event,
			err:   err,
		}:
		}
	}
}

type result struct {
	event *session.Event
	err   error
}
