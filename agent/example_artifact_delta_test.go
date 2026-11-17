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
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// A callback that saved an artifact through the artifact service itself, not
// through ctx.Artifacts(), reports the saved version with MergeArtifactDelta.
// The version goes into the artifact delta of the callback's event, which the
// agent then emits. A context that carries no event actions, such as one
// built by NewContext, cannot record and returns an error wrapping
// errors.ErrUnsupported.
func ExampleMergeArtifactDelta() {
	artifacts := artifact.InMemoryService()
	importer, err := agent.New(agent.Config{
		Name: "importer",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
			saved, err := artifacts.Save(ctx, &artifact.SaveRequest{
				AppName: ctx.AppName(), UserID: ctx.UserID(), SessionID: ctx.SessionID(),
				FileName: "upload.txt", Part: genai.NewPartFromText("uploaded text"),
			})
			if err != nil {
				return nil, err
			}
			return nil, agent.MergeArtifactDelta(ctx, map[string]int64{"upload.txt": saved.Version})
		}},
		Run: func(ictx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {
				err := agent.MergeArtifactDelta(agent.NewContext(ictx), map[string]int64{"upload.txt": 1})
				fmt.Println("from NewContext:", errors.Is(err, errors.ErrUnsupported))
			}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             importer,
		SessionService:    session.InMemoryService(),
		ArtifactService:   artifacts,
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("import", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(ev.Author, ev.Actions.ArtifactDelta)
	}

	// Output:
	// importer map[upload.txt:1]
	// from NewContext: true
}
