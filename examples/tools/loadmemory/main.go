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

// Package main provides an example ADK agent that uses the load_memory and
// preload_memory tools to retrieve memories from previous conversations.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/memory"
	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/model/gemini"
	"github.com/sjzsdu/adk-go/runner"
	"github.com/sjzsdu/adk-go/session"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/loadmemorytool"
	"github.com/sjzsdu/adk-go/tool/preloadmemorytool"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	llmAgent, err := llmagent.New(llmagent.Config{
		Name:        "memory_assistant",
		Model:       model,
		Description: "Agent that can recall information from memory.",
		Instruction: "You are a helpful assistant with access to memory. " +
			"Relevant memory may be preloaded automatically for each request. " +
			"If the preloaded context is not enough, use the load_memory tool " +
			"to search for additional relevant information. " +
			"If you find relevant memories, use them to provide informed responses.",
		Tools: []tool.Tool{
			preloadmemorytool.New(),
			loadmemorytool.New(),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	userID, appName := "test_user", "memory_app"
	sessionService := session.InMemoryService()
	memoryService := memory.InMemoryService()

	// Create a previous session with some conversation history to populate memory.
	previousSession, err := createPreviousSessionWithHistory(ctx, sessionService, appName, userID)
	if err != nil {
		log.Fatalf("Failed to create previous session: %v", err)
	}

	// Add the previous session to memory so it can be searched.
	if err := memoryService.AddSession(ctx, previousSession); err != nil {
		log.Fatalf("Failed to add session to memory: %v", err)
	}

	fmt.Println("Memory populated with previous conversation about a trip to Tokyo.")
	fmt.Println("Memories will be preloaded automatically for each request.")
	fmt.Println("Try asking: 'What do you remember about my trip?'")

	// Create a new session for the current conversation.
	resp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	currentSession := resp.Session

	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          llmAgent,
		SessionService: sessionService,
		MemoryService:  memoryService,
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
		for event, err := range r.Run(ctx, userID, currentSession.ID(), userMsg, agent.RunConfig{
			StreamingMode: streamingMode,
		}) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
			} else {
				if event.LLMResponse.Content == nil {
					continue
				}
				for _, p := range event.LLMResponse.Content.Parts {
					// If running in streaming mode, only print partial responses.
					if streamingMode != agent.StreamingModeSSE || event.LLMResponse.Partial {
						fmt.Print(p.Text)
					}
				}
			}
		}
	}
}

// createPreviousSessionWithHistory creates a session with pre-populated
// conversation history that will be added to memory.
func createPreviousSessionWithHistory(
	ctx context.Context,
	sessionService session.Service,
	appName, userID string,
) (session.Session, error) {
	resp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	s := resp.Session

	// Add some conversation history about a trip to Tokyo.
	events := []struct {
		author  string
		content string
	}{
		{"user", "I just got back from an amazing trip to Tokyo!"},
		{"model", "That sounds wonderful! Tokyo is such a vibrant city. What were the highlights of your trip?"},
		{"user", "I visited the Senso-ji temple in Asakusa, it was beautiful. I also tried authentic ramen at a small shop in Shinjuku."},
		{"model", "Senso-ji is one of Tokyo's oldest and most significant temples. The ramen in Shinjuku is legendary! Did you try any other Japanese cuisine?"},
		{"user", "Yes! I had sushi at Tsukiji outer market and tried takoyaki in Shibuya. The food was incredible."},
		{"model", "You really experienced the best of Tokyo's food scene! Tsukiji is famous for its fresh seafood. Did you get to see any other sights?"},
		{"user", "I went to the top of Tokyo Skytree and saw Mount Fuji in the distance. The view was breathtaking."},
		{"model", "Seeing Mount Fuji from Tokyo Skytree is a special experience, especially on a clear day. It sounds like you had an unforgettable trip!"},
	}

	for _, e := range events {
		event := session.NewEvent("previous-session")
		event.Author = e.author
		event.LLMResponse = model.LLMResponse{
			Content: genai.NewContentFromText(e.content, genai.Role(e.author)),
		}
		if err := sessionService.AppendEvent(ctx, s, event); err != nil {
			return nil, fmt.Errorf("failed to append event: %w", err)
		}
	}

	return s, nil
}
