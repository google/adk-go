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

package savefilesasartifacts_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/savefilesasartifacts"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// The plugin saves the PDF a user attaches as an artifact of the session, and
// the agent receives a placeholder in its place. The in-memory artifact
// service gives its artifacts memory:// URIs, which a model cannot read, so no
// file reference follows the placeholder.
func ExampleNew() {
	ctx := context.Background()

	saveFiles, err := savefilesasartifacts.New(savefilesasartifacts.Config{})
	if err != nil {
		log.Fatal(err)
	}

	// A stand-in for an LLM agent that prints the user message it receives.
	root, err := agent.New(agent.Config{
		Name: "root_agent",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {
				for _, part := range ic.UserContent().Parts {
					fmt.Printf("agent sees: %s\n", part.Text)
				}
			}
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	sessions := session.InMemoryService()
	artifacts := artifact.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:         "app",
		Agent:           root,
		SessionService:  sessions,
		ArtifactService: artifacts,
		PluginConfig:    runner.PluginConfig{Plugins: []*plugin.Plugin{saveFiles}},
	})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"}); err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromParts([]*genai.Part{
		genai.NewPartFromText("Summarize this report."),
		{InlineData: &genai.Blob{DisplayName: "report.pdf", Data: []byte("%PDF-1.7"), MIMEType: "application/pdf"}},
	}, genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", "s", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("event from %s, artifact delta %v\n", ev.Author, ev.Actions.ArtifactDelta)
	}

	list, err := artifacts.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("saved artifacts: %v\n", list.FileNames)

	// Output:
	// event from root_agent, artifact delta map[report.pdf:1]
	// agent sees: Summarize this report.
	// agent sees: [Uploaded Artifact: "report.pdf"]
	// saved artifacts: [report.pdf]
}
