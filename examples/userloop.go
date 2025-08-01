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

	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

func Run(ctx context.Context, rootAgent types.Agent) {
	userID, appName := "test_user", "test_app"

	sessionService := &session.InMemorySessionService{}

	session, err := sessionService.Create(ctx, &types.SessionCreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		panic(err)
	}

	runner := runner.Runner{
		AppName:        appName,
		RootAgent:      rootAgent,
		SessionService: sessionService,
	}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nUser -> ")

		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}

		userMsg := genai.NewContentFromText(userInput, genai.RoleUser)

		fmt.Print("\nAgent -> ")
		for event, err := range runner.Run(ctx, userID, session.ID, userMsg, &types.AgentRunConfig{
			StreamingMode: types.StreamingModeSSE,
		}) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
			} else {
				for _, p := range event.LLMResponse.Content.Parts {
					fmt.Print(p.Text)
				}
			}
		}
	}
}
