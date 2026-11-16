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

package sequentialagent_test

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// In a resumable run, a sequential agent writes a checkpoint naming each
// sub-agent before it starts it. Resuming the invocation restarts at the
// sub-agent the last checkpoint names, so the sub-agents before it do not run
// again. Here the session already holds the checkpoint
// {"current_sub_agent": "review"} that an earlier, interrupted run wrote.
func ExampleNew_resumeFromCheckpoint() {
	draft, err := llmagent.New(llmagent.Config{
		Name:  "draft",
		Model: oneReplyModel{reply: genai.NewContentFromText("A draft.", genai.RoleModel)},
	})
	if err != nil {
		log.Fatal(err)
	}
	review, err := llmagent.New(llmagent.Config{
		Name:  "review",
		Model: oneReplyModel{reply: genai.NewContentFromText("Looks good.", genai.RoleModel)},
	})
	if err != nil {
		log.Fatal(err)
	}
	pipeline, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "pipeline", SubAgents: []agent.Agent{draft, review}},
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "docs", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	userMsg := session.NewEvent(ctx, "inv-1")
	userMsg.Author = "user"
	userMsg.Content = genai.NewContentFromText("Write the release notes", genai.RoleUser)
	checkpoint := session.NewEvent(ctx, "inv-1")
	checkpoint.Author = "pipeline"
	checkpoint.Actions.AgentState = map[string]any{"current_sub_agent": "review"}
	for _, ev := range []*session.Event{userMsg, checkpoint} {
		if err := sessions.AppendEvent(ctx, created.Session, ev); err != nil {
			log.Fatal(err)
		}
	}

	r, err := runner.New(runner.Config{AppName: "docs", Agent: pipeline, SessionService: sessions, Resumable: true})
	if err != nil {
		log.Fatal(err)
	}
	for ev, err := range r.Run(ctx, "user", "s1", nil, agent.RunConfig{}, runner.WithInvocationID("inv-1")) {
		if err != nil {
			log.Fatal(err)
		}
		switch {
		case ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent:
			fmt.Printf("%s: ended\n", ev.Author)
		case ev.Actions.AgentState != nil:
			fmt.Printf("%s: checkpoint %v\n", ev.Author, ev.Actions.AgentState)
		default:
			fmt.Printf("%s: %s\n", ev.Author, ev.Content.Parts[0].Text)
		}
	}

	// Output:
	// review: Looks good.
	// review: ended
	// pipeline: ended
}
