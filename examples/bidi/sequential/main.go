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

// Package main provides an example of using sequential agents with real-time bidirectional streaming.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

func main() {
	log.SetOutput(os.Stdout)
	ctx := context.Background()

	// gemini-3.1-flash-live-preview
	// gemini-2.5-flash-native-audio-preview-12-2025
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-native-audio-preview-12-2025", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	ideaGenerator, err := llmagent.New(llmagent.Config{
		Name:        "idea_generator",
		Model:       model,
		Description: "Brainstorms creative story ideas with the user.",
		Instruction: "You are the Idea Generator. Ask the user for a topic they are interested in, and brainstorm 3 creative story ideas for them. Discuss the options and help them choose their favorite idea. Once the user confirms their choice, call the task_completed function so the Story Teller agent can take over to narrate the story.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create idea generator agent: %v", err)
	}

	storyTeller, err := llmagent.New(llmagent.Config{
		Name:        "story_teller",
		Model:       model,
		Description: "Narrates an engaging story based on the chosen idea.",
		// Note: Sending content history is currently not implemented for the Live API.
		// Therefore, the agent asks the user to remind them of the chosen idea instead of reviewing history.
		Instruction: "You are the Story Teller. The previous agent has just finalized a story idea with the user. Greet the user, ask them to remind you of the chosen idea, and then narrate an exciting, highly engaging short story about it using your voice.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create story teller agent: %v", err)
	}

	seqAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "bidi-demo",
			SubAgents: []agent.Agent{ideaGenerator, storyTeller},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create sequential agent: %v", err)
	}

	uiMode := true

	if uiMode {
		ss := session.InMemoryService()

		fs := http.FileServer(http.Dir("examples/bidi/static"))
		http.Handle("/", fs)
		http.Handle("/static/", http.StripPrefix("/static/", fs))

		controller := controllers.NewRuntimeAPIController(ss, nil, agent.NewSingleLoader(seqAgent), nil, 0, runner.PluginConfig{}, true)

		http.HandleFunc("/run_live", func(w http.ResponseWriter, req *http.Request) {
			err := controller.RunLiveHandler(w, req)
			if err != nil {
				log.Printf("RunLiveHandler failed: %v", err)
			}
		})

		fmt.Println("Serving UI on http://localhost:8081")
		log.Fatal(http.ListenAndServe(":8081", nil))
	}
}
