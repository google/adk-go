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

package agentengine_test

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/client/agentengine"
)

// ExampleClient_StreamQuery shows one turn against a deployed agent.
func ExampleClient_StreamQuery() {
	ctx := context.Background() // an entrypoint: examples and main may use it
	client, err := agentengine.New(ctx, agentengine.Config{
		Project: "my-project", Location: "us-central1", ReasoningEngine: "1234567890",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = client.Close() }()

	events := client.StreamQuery(ctx, agentengine.StreamQueryRequest{
		UserID:    "user-1",
		SessionID: "thread-42",
		Message:   genai.NewContentFromText("What is the status of ticket 17?", genai.RoleUser),
	})
	for ev, err := range events {
		if err != nil {
			log.Print(err)
			return
		}
		// A failed model call arrives as an event, not as an error.
		if ev.ErrorMessage != "" {
			log.Printf("%s: %s", ev.ErrorCode, ev.ErrorMessage)
			return
		}
		// The final non-partial event repeats the text of the partial chunks.
		if ev.Content == nil || ev.Partial {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil || p.Text == "" {
				continue
			}
			fmt.Print(p.Text)
		}
	}
}
