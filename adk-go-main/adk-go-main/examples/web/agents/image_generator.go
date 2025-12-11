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

// Package agents contains sample agents to demonstate ADK Web Capabilities.
package agents

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/loadartifactstool"
)

func generateImage(ctx tool.Context, input generateImageInput) (generateImageResult, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  os.Getenv("GOOGLE_CLOUD_PROJECT"),
		Location: os.Getenv("GOOGLE_CLOUD_LOCATION"),
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return generateImageResult{
			Status: "fail",
		}, nil
	}

	response, err := client.Models.GenerateImages(
		ctx,
		"imagen-3.0-generate-002",
		input.Prompt,
		&genai.GenerateImagesConfig{NumberOfImages: 1})
	if err != nil {
		return generateImageResult{
			Status: "fail",
		}, nil
	}

	_, err = ctx.Artifacts().Save(ctx, input.Filename, genai.NewPartFromBytes(response.GeneratedImages[0].Image.ImageBytes, "image/png"))
	if err != nil {
		return generateImageResult{
			Status: "fail",
		}, nil
	}

	return generateImageResult{
		Status:   "success",
		Filename: input.Filename,
	}, nil
}

type generateImageInput struct {
	Prompt   string `json:"prompt"`
	Filename string `json:"filename"`
}

type generateImageResult struct {
	Filename string `json:"filename"`
	Status   string `json:"Status"`
}

func GetImageGeneratorAgent(ctx context.Context, model model.LLM) agent.Agent {
	generateImageTool, err := functiontool.New(
		functiontool.Config{
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
			generateImageTool, loadartifactstool.New(),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	return imageGeneratorAgent
}
