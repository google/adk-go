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

// Package provides an example ADK REST API server with an ADK agent.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/cmd/launcher"
	"github.com/sjzsdu/adk-go/server/adkrest"
	"github.com/sjzsdu/adk-go/session"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/geminitool"
	"github.com/sjzsdu/adk-go/util/modelfactory"
)

func main() {
	ctx := context.Background()

	flag.Parse()

	// 从命令行参数创建模型配置
	modelConfig := modelfactory.NewFromFlags()
	model := modelfactory.MustCreateModel(ctx, modelConfig)

	// Create an agent
	a, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Configure the ADK REST API
	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: session.InMemoryService(),
	}

	// Create the REST API handler - this returns a standard http.Handler
	apiHandler := adkrest.NewHandler(config, 120*time.Second)

	// Create a standard net/http ServeMux
	mux := http.NewServeMux()

	// Register the API handler at the /api/ path
	// You can use any HTTP server or router here - not tied to gorilla/mux
	mux.Handle("/api/", http.StripPrefix("/api", apiHandler))

	// Add a simple health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte("OK")); err != nil {
			log.Printf("Failed to write response: %v", err)
		}
	})

	// Start the server
	log.Println("Starting server on :8080")
	log.Println("API available at http://localhost:8080/api/")
	log.Println("Health check at http://localhost:8080/health")

	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
