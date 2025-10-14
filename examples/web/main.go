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
	"os"

	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/restapi/services"
	"google.golang.org/adk/cmd/web"
	"google.golang.org/adk/examples/web/agents"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/genai"
)

func saveReportfunc(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
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

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("GOOGLE_API_KEY")

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	sessionService := session.InMemoryService()
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
		AfterModel: []llmagent.AfterModelCallback{saveReportfunc},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	llmAuditor := agents.GetLLmAuditorAgent(ctx, apiKey)

	agentLoader := services.NewStaticAgentLoader(
		map[string]agent.Agent{
			"weather_time_agent": rootAgent,
			"llm_auditor":        llmAuditor,
		},
	)
	artifactservice := artifact.InMemoryService()

	config := web.ParseArgs()
	fmt.Printf("%+v", config)
	web.Serve(config, &web.ServeConfig{
		SessionService:  sessionService,
		AgentLoader:     agentLoader,
		ArtifactService: artifactservice,
	})

}
