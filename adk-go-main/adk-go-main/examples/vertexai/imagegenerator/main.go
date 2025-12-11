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

// Package main demonstrates how to create an agent that can generate images
// using Vertex AI's Imagen model, save them as artifacts, and then save them
// to the local filesystem.
package main

import (
	"context"
	"log"
	"os"
	"path/filepath"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/loadartifactstool"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.0-flash-001", nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	generateImageTool, err := functiontool.New(
		functiontool.Config{
			Name:        "generate_image",
			Description: "Generates image and saves in artifact service.",
		},
		generateImage)
	if err != nil {
		log.Fatalf("Failed to create generate image tool: %v", err)
	}

	saveImageTool, err := functiontool.New(
		functiontool.Config{
			Name:        "save_image_locally",
			Description: "Saves images locally based on the filename.",
		},
		saveImage)
	if err != nil {
		log.Fatalf("Failed to create generate image tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "image_generator",
		Model:       model,
		Description: "Agent to generate pictures, answers questions about it and saves it locally if asked.",
		Instruction: "You are an agent whose job is to generate an image, describe it and save it locally if asked." +
			" Also user will provide the filename and you should save it in the artifacts with that filename." +
			" When user ask to save image locally you can call save_image_locally to do it.",
		Tools: []tool.Tool{
			loadartifactstool.New(), generateImageTool, saveImageTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		ArtifactService: artifact.InMemoryService(),
		AgentLoader:     agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

// This is a function tool to generate images using Vertex AI's Imagen model.
func generateImage(ctx tool.Context, input generateImageInput) (generateImageResult, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  os.Getenv("GOOGLE_CLOUD_PROJECT"),
		Location: os.Getenv("GOOGLE_CLOUD_LOCATION"),
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return generateImageResult{}, err
	}

	response, err := client.Models.GenerateImages(
		ctx,
		"imagen-3.0-generate-002",
		input.Prompt,
		&genai.GenerateImagesConfig{NumberOfImages: 1})
	if err != nil {
		return generateImageResult{}, err
	}

	_, err = ctx.Artifacts().Save(ctx, input.Filename, genai.NewPartFromBytes(response.GeneratedImages[0].Image.ImageBytes, "image/png"))
	if err != nil {
		return generateImageResult{}, err
	}
	return generateImageResult{Filename: input.Filename, Status: "success"}, nil
}

type generateImageInput struct {
	Prompt   string `json:"prompt"`
	Filename string `json:"filename"`
}

type generateImageResult struct {
	Filename string `json:"filename"`
	Status   string `json:"Status"`
}

// This is function tool that loads image from the artifacts service and
// saves is to the local filesystem.
func saveImage(ctx tool.Context, input saveImageInput) (saveImageResult, error) {
	filename := input.Filename
	resp, err := ctx.Artifacts().Load(ctx, filename)
	if err != nil {
		log.Printf("Failed to load artifact '%s': %v", filename, err)
		return saveImageResult{}, err
	}

	if resp.Part.InlineData == nil || len(resp.Part.InlineData.Data) == 0 {
		log.Printf("Artifact '%s' has no inline data", filename)
		return saveImageResult{}, err
	}

	// Ensure the filename has a .png extension for the local file.
	localFilename := filename
	if filepath.Ext(localFilename) != ".png" {
		localFilename += ".png"
	}

	// Create an "output" directory in the current working directory if it doesn't exist.
	outputDir := "output"
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		log.Printf("Failed to create output directory '%s': %v", outputDir, err)
		return saveImageResult{}, err
	}

	localPath := filepath.Join(outputDir, localFilename)
	err = os.WriteFile(localPath, resp.Part.InlineData.Data, 0o644)
	if err != nil {
		log.Printf("Failed to write image to local file '%s': %v", localPath, err)
		return saveImageResult{}, err
	}

	log.Printf("Successfully saved image to %s", localPath)
	return saveImageResult{Status: "success"}, nil
}

type saveImageInput struct {
	Filename string `json:"filename"`
}

type saveImageResult struct {
	Status string `json:"Status"`
}
