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

package agent_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// A custom agent inside a resumable ParallelAgent calls RecordEndOfAgent when
// its Run is finished. Once every sub-agent has ended, the ParallelAgent
// writes its end-of-agent event; the call itself yields no event.
func ExampleRecordEndOfAgent() {
	worker, err := agent.New(agent.Config{
		Name: "worker",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = genai.NewContentFromText("done", genai.RoleModel)
				if !yield(ev, nil) {
					return
				}
				agent.RecordEndOfAgent(ctx)
			}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	par, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{worker}},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             par,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		Resumable:         true,
	})
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("go", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		switch {
		case ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent:
			fmt.Println(ev.Author, "ended")
		case ev.Actions.AgentState != nil:
			fmt.Println(ev.Author, "state", ev.Actions.AgentState)
		default:
			fmt.Println(ev.Author, ev.Content.Parts[0].Text)
		}
	}
	// Output:
	// par state map[]
	// worker done
	// par ended
}
