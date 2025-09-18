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
	"log"
	"net/http"
	"strconv"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/restapi/config"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
	"google.golang.org/adk/cmd/restapi/services"
	"google.golang.org/adk/sessionservice"
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

func agentLoader(apiKey string) services.AgentLoader {
	ctx := context.Background()
	return services.NewStaticAgentLoader(
		map[string]agent.Agent{
			"llm_auditor": getAgent(ctx, apiKey),
		},
	)
}

func main() {

	serverConfig, err := config.LoadConfig()
	agentLoader := agentLoader(serverConfig.GeminiAPIKey)
	if err != nil {
		panic(err)
	}
	log.Printf("Starting server on port %d\n", serverConfig.Port)
	sessionService := sessionservice.Mem()

	router := routers.NewRouter(
		routers.NewSessionsAPIRouter(handlers.NewSessionsAPIController(sessionService)),
		routers.NewRuntimeAPIRouter(handlers.NewRuntimeAPIRouter(sessionService, agentLoader)),
		routers.NewAppsAPIRouter(handlers.NewAppsAPIController(agentLoader)),
		routers.NewDebugAPIRouter(handlers.NewDebugAPIRouter(sessionService, agentLoader)),
		routers.NewArtifactsAPIRouter(&handlers.ArtifactsAPIController{}),
	)
	router.Use(corsWithArgs(serverConfig))

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(serverConfig.Port), router))
}
