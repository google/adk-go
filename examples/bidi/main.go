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
	"path/filepath"
	"runtime"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/geminitool"
)

func main() {
	log.SetOutput(os.Stdout)
	ctx := context.Background()

	// gemini-3.1-flash-live-preview
	// gemini-2.5-flash-native-audio-preview-12-2025
	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-live-preview", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	type EmptyArgs struct{}
	type MessageResult struct {
		Message string `json:"message"`
	}

	cameraTool, err := functiontool.New(functiontool.Config{
		Name:        "camera_toggle",
		Description: "Turns the camera on or off.",
	}, func(ctx tool.Context, args EmptyArgs) (MessageResult, error) {
		fmt.Println("Camera tool was called!")
		return MessageResult{Message: "Camera tool called successfully!"}, nil
	})
	if err != nil {
		log.Fatalf("Failed to create camera tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "bidi-demo",
		Model:       model,
		Description: "Agent optimized for real-time bidirectional streaming.",
		Instruction: "You are a real-time voice assistant.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
			cameraTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	uiMode := true

	if uiMode {
		// Create runner
		ss := session.InMemoryService()

		_, filename, _, ok := runtime.Caller(0)
		if !ok {
			log.Fatal("No caller information")
		}
		staticDir := filepath.Join(filepath.Dir(filename), "static")
		fs := http.FileServer(http.Dir(staticDir))
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
