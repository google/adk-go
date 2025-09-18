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

// package main is the entry point for the REST API server.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strconv"

	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifactservice"
	"google.golang.org/adk/cmd/restapi/config"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
	"google.golang.org/adk/cmd/restapi/services"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/llm/gemini"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/genai"
)

func corsWithArgs(serverConfig *config.ADKAPIServerConfigs) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return serverConfig.Cors.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		}))
	}
}

func saveReportfunc(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error) {
	if llmResponse == nil || llmResponse.Content == nil || llmResponseError != nil {
		return llmResponse, llmResponseError
	}
	for _, part := range llmResponse.Content.Parts {
		err := ctx.Artifacts().Save(uuid.NewString(), *part)
		if err != nil {
			return nil, err
		}
	}
	return llmResponse, llmResponseError
}

func agentLoader(apiKey string) services.AgentLoader {
	ctx := context.Background()
	model, err := gemini.NewModel(ctx, "gemini-2.0-flash", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		panic(err)
	}

	agent1, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		AfterModel: []llmagent.AfterModelCallback{
			saveReportfunc,
		},
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		panic(err)
	}

	agent2, err := llmagent.New(llmagent.Config{
		Name:        "foobar",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Agents created: %v, %v", agent1, agent2)

	return services.NewStaticAgentLoader(
		map[string]agent.Agent{
			"weather_time_agent": agent1,
			"foobar":             agent2,
		},
	)
}

func main() {

	serverConfig, err := config.LoadConfig()
	agentLoader := agentLoader(serverConfig.GeminiAPIKey)
	if err != nil {
		panic(err)
	}
	log.Printf("Starting server on port %d", serverConfig.Port)
	sessionService := sessionservice.Mem()
	artifactService := artifactservice.Mem()

	router := routers.NewRouter(
		routers.NewSessionsAPIRouter(handlers.NewSessionsAPIController(sessionService)),
		routers.NewRuntimeAPIRouter(handlers.NewRuntimeAPIRouter(sessionService, agentLoader, artifactService)),
		routers.NewAppsAPIRouter(handlers.NewAppsAPIController(agentLoader)),
		routers.NewDebugAPIRouter(&handlers.DebugAPIController{}),
		routers.NewArtifactsAPIRouter(handlers.NewArtifactsAPIController(artifactService)),
	)
	router.Use(corsWithArgs(serverConfig))

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(serverConfig.Port), router))
}
