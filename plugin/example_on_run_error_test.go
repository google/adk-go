// Copyright 2026 Google LLC
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

package plugin_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// invocationSpans holds one open record per invocation, the names of the
// agents that failed in it. OnAgentErrorCallback and OnRunErrorCallback both
// write to it, and the runner does not serialize the two hooks (parallel
// sub-agents call OnAgentErrorCallback concurrently), so a mutex guards it.
type invocationSpans struct {
	mu     sync.Mutex
	failed map[string][]string // by invocation ID
}

// A plugin that keeps per-invocation state closes it in both
// AfterRunCallback and OnRunErrorCallback: when a run ends because of a run
// error, only OnRunErrorCallback is called. Here the agent fails and the
// caller stops ranging at the error, which ends the run. The caller receives
// the error first; OnRunErrorCallback runs once the loop body has returned,
// before the range statement completes, with the same error value. A caller
// that kept ranging past an error where the run goes on would not end the run
// with it: the run would continue and, if nothing else failed, end with
// AfterRunCallback.
func ExampleConfig_onRunError() {
	spans := &invocationSpans{failed: map[string][]string{}}
	closer, err := plugin.New(plugin.Config{
		Name: "span_closer",
		BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			spans.mu.Lock()
			defer spans.mu.Unlock()
			spans.failed[ctx.InvocationID()] = []string{}
			return nil, nil
		},
		OnAgentErrorCallback: func(ctx agent.Context, a agent.Agent, _ error) {
			spans.mu.Lock()
			defer spans.mu.Unlock()
			spans.failed[ctx.InvocationID()] = append(spans.failed[ctx.InvocationID()], a.Name())
		},
		OnRunErrorCallback: func(ctx agent.InvocationContext, err error) {
			spans.mu.Lock()
			defer spans.mu.Unlock()
			fmt.Printf("span closed: run error %q, failed agents %v\n", err, spans.failed[ctx.InvocationID()])
			delete(spans.failed, ctx.InvocationID())
		},
		AfterRunCallback: func(ctx agent.InvocationContext) {
			spans.mu.Lock()
			defer spans.mu.Unlock()
			fmt.Println("span closed: completed")
			delete(spans.failed, ctx.InvocationID())
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	worker, err := agent.New(agent.Config{
		Name: "worker",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("worker failed"))
			}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             worker,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{closer}},
	})
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("hello", genai.RoleUser)
	for _, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			fmt.Println("caller received:", err)
			break
		}
	}
	fmt.Println("open spans:", len(spans.failed))

	// Output:
	// caller received: worker failed
	// span closed: run error "worker failed", failed agents [worker]
	// open spans: 0
}
