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
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
	"google.golang.org/adk/cmd/restapi/services"
	"google.golang.org/adk/cmd/restapi/utils"
	"google.golang.org/adk/llm/gemini"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/genai"
)

func corsWithArgs(serverArgs utils.AdkAPIArgs) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "http://"+serverArgs.FrontAddress)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func agentLoader() services.AgentLoader {
	ctx := context.Background()
	model, err := gemini.NewModel(ctx, "gemini-2.0-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GEMINI_API_KEY"),
	})
	if err != nil {
		panic(err)
	}

	agent1, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
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

	serverArgs := utils.ParseArgs()
	agentLoader := agentLoader()

	log.Printf("Starting server on port %d with front address %s", serverArgs.Port, serverArgs.FrontAddress)
	sessionService := sessionservice.Mem()

	router := routers.NewRouter(
		routers.NewSessionsApiRouter(handlers.NewSessionsApiController(sessionService)),
		routers.NewRuntimeApiRouter(handlers.NewRuntimeApiRouter(sessionService, agentLoader)),
		routers.NewAppsApiRouter(handlers.NewAppsApiController(agentLoader)),
		routers.NewDebugApiRouter(&handlers.DebugApiController{}),
		routers.NewArtifactsApiRouter(&handlers.ArtifactsApiController{}),
	)
	router.Use(corsWithArgs(serverArgs))

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(serverArgs.Port), router))
}
