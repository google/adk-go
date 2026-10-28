// Copyright 2026 Google LLC
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

// Package main runs a non-interactive ADK agent backed by Anthropic Claude on
// either Vertex AI or the Anthropic API, sending an inline PNG image and
// invoking a function tool during the turn.
package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// defaultModel is a versioned Claude model ID served on both Vertex AI and the
// first-party Anthropic API. Vertex AI does not serve -latest aliases.
const defaultModel = "claude-sonnet-4-5-20250929"

// sampleBluePNG is a 1x1 solid blue PNG image embedded directly so the sample
// runs without external asset files.
var sampleBluePNG = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
	0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
	0xde, 0x00, 0x00, 0x00, 0x10, 0x49, 0x44, 0x41,
	0x54, 0x78, 0x9c, 0x62, 0x62, 0x60, 0xf8, 0x0f,
	0x08, 0x00, 0x00, 0xff, 0xff, 0x01, 0x0b, 0x01,
	0x02, 0xd6, 0x95, 0x43, 0xa8, 0x00, 0x00, 0x00,
	0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60,
	0x82,
}

type paletteInput struct {
	Color string `json:"color" jsonschema:"The dominant color name in lowercase, such as blue, red, or green."`
}

type paletteOutput struct {
	Color   string `json:"color"`
	HexCode string `json:"hex_code"`
	Family  string `json:"family"`
}

func getPaletteInfo(_ agent.Context, in paletteInput) (paletteOutput, error) {
	color := strings.ToLower(strings.TrimSpace(in.Color))
	switch color {
	case "blue":
		return paletteOutput{Color: "blue", HexCode: "#0055FF", Family: "cool primary"}, nil
	case "red":
		return paletteOutput{Color: "red", HexCode: "#FF2200", Family: "warm primary"}, nil
	default:
		return paletteOutput{Color: color, HexCode: "#336699", Family: "standard palette"}, nil
	}
}

func main() {
	log.SetFlags(0)
	if err := run(context.Background()); err != nil {
		log.Fatal(explain(err))
	}
}

func run(ctx context.Context) error {
	cfg, err := resolveClientConfig()
	if err != nil {
		return err
	}

	modelName := cmp.Or(strings.TrimSpace(os.Getenv("ANTHROPIC_MODEL")), defaultModel)
	llm, err := anthropicmodel.NewModel(ctx, modelName, cfg)
	if err != nil {
		return fmt.Errorf("create Claude model: %w", err)
	}

	paletteTool, err := functiontool.New(functiontool.Config{
		Name:        "get_palette_info",
		Description: "Looks up the canonical design-system hex code and color family for a color name.",
	}, getPaletteInfo)
	if err != nil {
		return fmt.Errorf("create tool: %w", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "claude_vision_agent",
		Model:       llm,
		Description: "Inspects an inline image and looks up its design-system palette metadata.",
		Instruction: "Identify the dominant color in the attached image, call get_palette_info with that color name, and report the hex code and color family.",
		Tools:       []tool.Tool{paletteTool},
	})
	if err != nil {
		return fmt.Errorf("create agent: %w", err)
	}

	sessionService := session.InMemoryService()
	const (
		appName   = "anthropic_example"
		userID    = "example_user"
		sessionID = "session_1"
	)
	if _, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          a,
		SessionService: sessionService,
	})
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	prompt := "What color is the attached image? Use get_palette_info to report its hex code and color family."
	userMsg := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			genai.NewPartFromBytes(sampleBluePNG, "image/png"),
			genai.NewPartFromText(prompt),
		},
	}

	fmt.Printf("Backend: %s (%s)\n", cfg.Backend, modelName)
	fmt.Printf("User -> [inline image/png, %d bytes] %s\n", len(sampleBluePNG), prompt)

	for event, err := range r.Run(ctx, userID, sessionID, userMsg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
		if event == nil || event.LLMResponse.Content == nil {
			continue
		}
		for _, part := range event.LLMResponse.Content.Parts {
			switch {
			case part.FunctionCall != nil:
				fmt.Printf("Tool Call -> %s(%v)\n", part.FunctionCall.Name, part.FunctionCall.Args)
			case part.FunctionResponse != nil:
				fmt.Printf("Tool Result -> %s: %v\n", part.FunctionResponse.Name, part.FunctionResponse.Response)
			case part.Text != "" && !part.Thought:
				fmt.Printf("Agent -> %s\n", strings.TrimSpace(part.Text))
			}
		}
	}
	return nil
}

func resolveClientConfig() (anthropicmodel.ClientConfig, error) {
	backendEnv := strings.ToLower(strings.TrimSpace(os.Getenv("ANTHROPIC_BACKEND")))
	project := strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_PROJECT"))
	location := strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_LOCATION"))
	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))

	switch backendEnv {
	case "vertex":
		if project == "" {
			return anthropicmodel.ClientConfig{}, errors.New("GOOGLE_CLOUD_PROJECT must be set when ANTHROPIC_BACKEND=vertex")
		}
		if location == "" {
			return anthropicmodel.ClientConfig{}, errors.New("GOOGLE_CLOUD_LOCATION must be set when ANTHROPIC_BACKEND=vertex")
		}
		return anthropicmodel.ClientConfig{
			Backend:  anthropicmodel.BackendVertex,
			Project:  project,
			Location: location,
		}, nil

	case "anthropic":
		if apiKey == "" {
			return anthropicmodel.ClientConfig{}, errors.New("ANTHROPIC_API_KEY must be set when ANTHROPIC_BACKEND=anthropic")
		}
		return anthropicmodel.ClientConfig{
			Backend: anthropicmodel.BackendAnthropic,
			APIKey:  apiKey,
		}, nil

	case "":
		if project != "" || location != "" {
			if project == "" {
				return anthropicmodel.ClientConfig{}, errors.New("GOOGLE_CLOUD_PROJECT must be set alongside GOOGLE_CLOUD_LOCATION for Vertex AI")
			}
			if location == "" {
				return anthropicmodel.ClientConfig{}, errors.New("GOOGLE_CLOUD_LOCATION must be set alongside GOOGLE_CLOUD_PROJECT for Vertex AI")
			}
			return anthropicmodel.ClientConfig{
				Backend:  anthropicmodel.BackendVertex,
				Project:  project,
				Location: location,
			}, nil
		}
		if apiKey != "" {
			return anthropicmodel.ClientConfig{
				Backend: anthropicmodel.BackendAnthropic,
				APIKey:  apiKey,
			}, nil
		}
		return anthropicmodel.ClientConfig{}, errors.New("set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (for Vertex AI) or ANTHROPIC_API_KEY (for the Anthropic API)")

	default:
		return anthropicmodel.ClientConfig{}, fmt.Errorf("unsupported ANTHROPIC_BACKEND %q; valid values are \"vertex\" or \"anthropic\"", backendEnv)
	}
}

func explain(err error) string {
	var apiErr *anthropicmodel.APIError
	if errors.As(err, &apiErr) {
		return fmt.Sprintf("Claude API error (status %d, type %s): %s", apiErr.StatusCode, apiErr.Type, apiErr.Message)
	}
	return err.Error()
}
