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
	"log"
	"os"

	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/run"
	"google.golang.org/adk/cmd/restapi/services"
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
		_, err := ctx.Artifacts().Save(ctx, uuid.NewString(), part)
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
	imageGeneratorAgent := GetImageGeneratorAgent(ctx, apiKey)

	agentLoader := services.NewStaticAgentLoader(
		map[string]agent.Agent{
			"weather_time_agent": rootAgent,
			"llm_auditor":        llmAuditor,
			"image_generator":    imageGeneratorAgent,
		},
	)
	artifactservice := artifact.InMemoryService()

	config := &adk.Config{
		ArtifactService: artifactservice,
		SessionService:  sessionService,
		AgentLoader:     agentLoader,
	}

	run.Run(ctx, config)

}

func generateImage(ctx tool.Context, input generateImageInput) generateImageResult {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  os.Getenv("GOOGLE_CLOUD_PROJECT"),
		Location: os.Getenv("GOOGLE_CLOUD_LOCATION"),
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return generateImageResult{
			Status: "fail",
		}
	}

	response, err := client.Models.GenerateImages(
		ctx,
		"imagen-3.0-generate-002",
		input.Prompt,
		&genai.GenerateImagesConfig{NumberOfImages: 1})
	if err != nil {
		return generateImageResult{
			Status: "fail",
		}
	}

	_, err = ctx.Artifacts().Save(ctx, input.Filename, genai.NewPartFromBytes(response.GeneratedImages[0].Image.ImageBytes, "image/png"))
	if err != nil {
		return generateImageResult{
			Status: "fail",
		}
	}

	return generateImageResult{
		Status:   "success",
		Filename: input.Filename,
	}
}

type generateImageInput struct {
	Prompt   string `json:"prompt"`
	Filename string `json:"filename"`
}

type generateImageResult struct {
	Filename string `json:"filename"`
	Status   string `json:"Status"`
}

func GetImageGeneratorAgent(ctx context.Context, apiKey string) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-2.0-flash-001", nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	generateImageTool, err := tool.NewFunctionTool(
		tool.FunctionToolConfig{
			Name:        "generate_image",
			Description: "Generates image and saves in artifact service.",
		},
		generateImage)
	if err != nil {
		log.Fatalf("Failed to create generate image tool: %v", err)
	}
	imageGeneratorAgent, err := llmagent.New(llmagent.Config{
		Name:        "image_generator",
		Model:       model,
		Description: "Agent to generate pictures, answers questions about it and saves it locally if asked.",
		Instruction: "You are an agent whose job is to generate or edit an image based on the user's prompt.",
		Tools: []tool.Tool{
			generateImageTool, tool.NewLoadArtifactsTool(),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	return imageGeneratorAgent
}
