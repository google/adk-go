// Package main is the entry point for the REST API server.
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
	"google.golang.org/adk/llm/gemini"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/adk/web/handlers"
	"google.golang.org/adk/web/routers"
	"google.golang.org/adk/web/services"
	"google.golang.org/adk/web/utils"
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

	router := routers.NewRouter(
		routers.NewSessionsApiRouter(handlers.NewSessionsApiController(sessionservice.Mem())),
		routers.NewRuntimeApiRouter(handlers.NewRuntimeApiRouter(sessionservice.Mem(), agentLoader)),
		routers.NewAppsApiRouter(handlers.NewAppsApiController(agentLoader)),
		routers.NewDebugApiRouter(&handlers.DebugApiController{}),
		routers.NewArtifactsApiRouter(&handlers.ArtifactsApiController{}),
	)
	router.Use(corsWithArgs(serverArgs))

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(serverArgs.Port), router))
}
