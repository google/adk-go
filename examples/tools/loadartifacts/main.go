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

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/loadartifactstool"
)

// Note: you need to run the program from the loadartifacts directory
// to fetch the image successfully.
func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	llmagent, err := llmagent.New(llmagent.Config{
		Name:        "artifact_describer",
		Model:       model,
		Description: "Agent to answer questions about artifacts.",
		Instruction: "When user asks about the artifact, load them and describe them.",
		Tools: []tool.Tool{
			loadartifactstool.New(),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	userID, appName := "test_user", "test_app"
	sessionService := session.InMemoryService()
	// Create session.
	resp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		log.Fatalf("Failed to create the session service: %v", err)
	}

	session := resp.Session
	artifactService := artifact.InMemoryService()
	// Populate artifacts that can be described later.
	imageBytes, err := os.ReadFile("animal_picture.png")
	if err != nil {
		log.Fatalf("Failed to read image file: %v", err)
	}
	genai.NewPartFromBytes(imageBytes, "image/png")

	_, err = artifactService.Save(ctx, &artifact.SaveRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: session.ID(),
		FileName:  "animal_picture.png",
		Part:      genai.NewPartFromBytes(imageBytes, "image/png"),
	})
	if err != nil {
		log.Fatalf("Failed to save artifact: %v", err)
	}

	_, err = artifactService.Save(ctx, &artifact.SaveRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: session.ID(),
		FileName:  "haiku.txt",
		Part: genai.NewPartFromText(
			"An old silent pond..." +
				"A frog jumps into the pond," +
				"splash! Silence again."),
	})
	if err != nil {
		log.Fatalf("Failed to save artifact: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           llmagent,
		SessionService:  sessionService,
		ArtifactService: artifactService,
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

		fmt.Print("\nAgent -> ")
		streamingMode := agent.StreamingModeSSE
		for event, err := range r.Run(ctx, userID, session.ID(), userMsg, agent.RunConfig{
			StreamingMode: streamingMode,
		}) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
			} else {
				if event.LLMResponse.Content == nil {
					continue
				}
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
