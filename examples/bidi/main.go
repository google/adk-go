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

// Package provides a quickstart ADK agent.
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
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-live-preview", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "bidi-demo",
		Model:       model,
		Description: "Agent optimized for real-time bidirectional streaming.",
		Instruction: "You are a real-time voice assistant. Be proactive and immediately comment on what you see in the video stream without waiting for me to speak. Whenever you recieve a picture and the person has both hands lifted up, say 'I got you'.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	uiMode := true

	if uiMode {
		// Create runner
		ss := session.InMemoryService()

		fs := http.FileServer(http.Dir("examples/bidi/static"))
		http.Handle("/", fs)
		http.Handle("/static/", http.StripPrefix("/static/", fs))

		controller := controllers.NewRuntimeAPIController(ss, nil, agent.NewSingleLoader(a), nil, 0, runner.PluginConfig{}, true)

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
