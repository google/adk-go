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

package examples

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type RunConfig struct {
	SessionService  session.Service
	ArtifactService artifact.Service
	StreamingMode   agent.StreamingMode
}

func Run(ctx context.Context, rootAgent agent.Agent, runConfig *RunConfig) {
	userID, appName := "test_user", "test_app"

	if runConfig == nil {
		runConfig = &RunConfig{}
	}

	sessionService := runConfig.SessionService
	if sessionService == nil {
		sessionService = session.InMemoryService()
	}

	resp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		log.Fatalf("Failed to create the session service: %v", err)
	}

	session := resp.Session

	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           rootAgent,
		SessionService:  sessionService,
		ArtifactService: runConfig.ArtifactService,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nUser -> ")

		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}

		userMsg := genai.NewContentFromText(userInput, genai.RoleUser)

		streamingMode := runConfig.StreamingMode
		if streamingMode == "" {
			streamingMode = agent.StreamingModeSSE
		}
		fmt.Print("\nAgent -> ")
		for event, err := range r.Run(ctx, userID, session.ID(), userMsg, &agent.RunConfig{
			StreamingMode: streamingMode,
		}) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
			} else {
				for _, p := range event.LLMResponse.Content.Parts {
					// if its running in streaming mode, don't print the non partial llmResponses
					if streamingMode != agent.StreamingModeSSE || event.LLMResponse.Partial {
						fmt.Print(p.Text)
					}
				}
			}
		}
	}
}
