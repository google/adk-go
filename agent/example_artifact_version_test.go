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
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// A callback or tool saves artifacts through its context and looks up the
// record of a version with ArtifactVersion. A nil version selects the latest
// version, and a version that does not exist is an error wrapping
// fs.ErrNotExist. The runner's ArtifactService stores the artifacts.
func ExampleArtifactVersion() {
	reporter, err := agent.New(agent.Config{
		Name: "reporter",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
			for _, text := range []string{"draft", "final"} {
				if _, err := ctx.Artifacts().Save(ctx, "report.txt", genai.NewPartFromText(text)); err != nil {
					return nil, err
				}
			}

			latest, err := agent.ArtifactVersion(ctx, "report.txt", nil)
			if err != nil {
				return nil, err
			}
			fmt.Println("latest version:", latest.Version, latest.MimeType)

			first := int64(1)
			v1, err := agent.ArtifactVersion(ctx, "report.txt", &first)
			if err != nil {
				return nil, err
			}
			fmt.Println("version 1 URI:", v1.CanonicalURI)

			missing := int64(5)
			_, err = agent.ArtifactVersion(ctx, "report.txt", &missing)
			fmt.Println("version 5 not found:", errors.Is(err, fs.ErrNotExist))
			return nil, nil
		}},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "alice", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:         "app",
		Agent:           reporter,
		SessionService:  sessions,
		ArtifactService: artifact.InMemoryService(),
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("write the report", genai.RoleUser)
	for _, err := range r.Run(ctx, "alice", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}
	// Output:
	// latest version: 2 text/plain
	// version 1 URI: memory://apps/app/users/alice/sessions/s1/artifacts/report.txt/versions/1
	// version 5 not found: true
}
