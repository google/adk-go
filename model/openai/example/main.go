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

// Command openai-example demonstrates using the OpenAI-compatible model adapter.
//
// Before running this example, set the following environment variables:
//
//	export OPENAI_API_KEY="your-api-key"                   # API key for authentication
//	export OPENAI_BASE_URL="https://api.example.com/v1"  # API endpoint URL (e.g., openai, local LLM server)
//
// Usage:
//
//	go run main.go
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"

	"google.golang.org/adk/model"
	"google.golang.org/adk/model/openai"
	"google.golang.org/genai"
)

// ModelName is the model identifier. Modify this for your setup.
const ModelName = "x-ai/grok-4.1-fast:free"

// Google Cloud Storage public sample resources.
const (
	// SampleImageURL is a sample image from Google Cloud public datasets.
	SampleImageURL = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/scones.jpg"

	// SampleAudioURL is a sample audio file from Google Cloud.
	SampleAudioURL = "https://storage.googleapis.com/cloud-samples-data/generative-ai/audio/pixel.mp3"

	// SampleVideoURL is a sample video file from Google Cloud.
	SampleVideoURL = "https://storage.googleapis.com/cloud-samples-data/generative-ai/video/pixel8.mp4"

	// SamplePDFURL is a sample PDF document from Google Cloud.
	SamplePDFURL = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf"
)

func main() {
	ctx := context.Background()

	// Create OpenAI-compatible model
	// APIKey and BaseURL are read from environment variables:
	//   OPENAI_API_KEY - API key for authentication
	//   OPENAI_BASE_URL - API endpoint URL
	llm, err := openai.NewModel(ctx, ModelName, nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	fmt.Printf("Using model: %s\n\n", ModelName)

	// Example 1: Simple text generation
	runTextExample(ctx, llm)

	// Example 2: With system instruction
	runSystemInstructionExample(ctx, llm)

	// Example 3: Streaming
	runStreamingExample(ctx, llm)

	// Example 4: Function calling
	runFunctionCallingExample(ctx, llm)

	// Example 5: Image analysis
	runImageExample(ctx, llm)

	// Example 6: Audio analysis
	runAudioExample(ctx, llm)

	// Example 7: Video analysis
	runVideoExample(ctx, llm)

	// Example 8: PDF analysis
	runPDFExample(ctx, llm)
}

// float32Ptr returns a pointer to the given float32 value.
func float32Ptr(f float32) *float32 {
	return &f
}

// fetchURL fetches data from the given URL and returns the response body.
func fetchURL(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch URL: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error: %s", resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return data, nil
}

// runTextExample demonstrates simple text generation.
func runTextExample(ctx context.Context, llm model.LLM) {
	fmt.Println("=== Example 1: Simple Text Generation ===")
	req := &model.LLMRequest{
		Contents: genai.Text("What is 2+2? Answer with just the number."),
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0),
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		if len(resp.Content.Parts) > 0 {
			fmt.Printf("Response: %s\n", resp.Content.Parts[0].Text)
		}
	}
}

// runSystemInstructionExample demonstrates using system instructions.
func runSystemInstructionExample(ctx context.Context, llm model.LLM) {
	fmt.Println("\n=== Example 2: With System Instruction ===")
	req := &model.LLMRequest{
		Contents: genai.Text("Tell me a joke"),
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are a helpful assistant who speaks like a pirate.", "system"),
			Temperature:       float32Ptr(0.7),
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		if len(resp.Content.Parts) > 0 {
			fmt.Printf("Response: %s\n", resp.Content.Parts[0].Text)
		}
	}
}

// runStreamingExample demonstrates streaming responses.
func runStreamingExample(ctx context.Context, llm model.LLM) {
	fmt.Println("\n=== Example 3: Streaming Response ===")
	req := &model.LLMRequest{
		Contents: genai.Text("Count from 1 to 5"),
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0),
		},
	}

	fmt.Print("Response: ")
	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		if resp.Partial && len(resp.Content.Parts) > 0 {
			fmt.Print(resp.Content.Parts[0].Text)
		}
	}
	fmt.Println()
}

// runFunctionCallingExample demonstrates function calling capabilities.
func runFunctionCallingExample(ctx context.Context, llm model.LLM) {
	fmt.Println("\n=== Example 4: Function Calling ===")
	req := &model.LLMRequest{
		Contents: genai.Text("What's the weather in Paris?"),
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0),
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "get_weather",
							Description: "Get the current weather for a location",
							Parameters: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"location": {
										Type:        genai.TypeString,
										Description: "The city name",
									},
								},
								Required: []string{"location"},
							},
						},
					},
				},
			},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				fmt.Printf("Function called: %s\n", part.FunctionCall.Name)
				fmt.Printf("Arguments: %v\n", part.FunctionCall.Args)
			} else if part.Text != "" {
				fmt.Printf("Response: %s\n", part.Text)
			}
		}
	}
}

// runImageExample demonstrates image analysis.
func runImageExample(ctx context.Context, llm model.LLM) {
	fmt.Println("\n=== Example 5: Image Analysis ===")
	fmt.Printf("Fetching image from: %s\n", SampleImageURL)

	imageData, err := fetchURL(SampleImageURL)
	if err != nil {
		log.Printf("Failed to fetch image: %v", err)
		return
	}
	fmt.Printf("Image size: %d bytes\n", len(imageData))

	// Create content with image and text
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "image/jpeg",
							Data:     imageData,
						},
					},
					{Text: "What do you see in this image? Describe it briefly."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		if len(resp.Content.Parts) > 0 {
			fmt.Printf("Response: %s\n", resp.Content.Parts[0].Text)
		}
	}
}

// runAudioExample demonstrates audio analysis.
func runAudioExample(ctx context.Context, llm model.LLM) {
	fmt.Println("\n=== Example 6: Audio Analysis ===")
	fmt.Printf("Fetching audio from: %s\n", SampleAudioURL)

	audioData, err := fetchURL(SampleAudioURL)
	if err != nil {
		log.Printf("Failed to fetch audio: %v", err)
		return
	}
	fmt.Printf("Audio size: %d bytes\n", len(audioData))

	// Create content with audio and text
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "audio/mpeg",
							Data:     audioData,
						},
					},
					{Text: "What is being said in this audio? Please transcribe or summarize it."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		if len(resp.Content.Parts) > 0 {
			fmt.Printf("Response: %s\n", resp.Content.Parts[0].Text)
		}
	}
}

// runVideoExample demonstrates video analysis.
func runVideoExample(ctx context.Context, llm model.LLM) {
	fmt.Println("\n=== Example 7: Video Analysis ===")
	fmt.Printf("Fetching video from: %s\n", SampleVideoURL)

	videoData, err := fetchURL(SampleVideoURL)
	if err != nil {
		log.Printf("Failed to fetch video: %v", err)
		return
	}
	fmt.Printf("Video size: %d bytes\n", len(videoData))

	// Create content with video and text
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "video/mp4",
							Data:     videoData,
						},
					},
					{Text: "What is happening in this video? Describe the main content briefly."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		if len(resp.Content.Parts) > 0 {
			fmt.Printf("Response: %s\n", resp.Content.Parts[0].Text)
		}
	}
}

// runPDFExample demonstrates PDF document analysis.
func runPDFExample(ctx context.Context, llm model.LLM) {
	fmt.Println("\n=== Example 8: PDF Analysis ===")
	fmt.Printf("Fetching PDF from: %s\n", SamplePDFURL)

	pdfData, err := fetchURL(SamplePDFURL)
	if err != nil {
		log.Printf("Failed to fetch PDF: %v", err)
		return
	}
	fmt.Printf("PDF size: %d bytes\n", len(pdfData))

	// Create content with PDF and text
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "application/pdf",
							Data:     pdfData,
						},
					},
					{Text: "What is this PDF document about? Summarize the main topic and key points."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("GenerateContent error: %v", err)
			return
		}
		if len(resp.Content.Parts) > 0 {
			fmt.Printf("Response: %s\n", resp.Content.Parts[0].Text)
		}
	}
}
