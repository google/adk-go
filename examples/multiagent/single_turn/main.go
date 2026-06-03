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

// Package main demonstrates how a "single_turn" mode agent can act as an
// autonomous sub-agent to an LLM agent, utilizing schemas and tools without
// ever interacting with the user.
//
// The phone_recommender sub-agent receives structured input
// (UserPreferences), uses a mocked tool (check_phone_price), and returns
// structured output (PhoneRecommendation). The root_agent translates the
// user's natural language request into the structured input and delegates
// to the sub-agent.
package main

import (
	"context"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// CheckPhonePriceInput is the input for the check_phone_price tool.
type CheckPhonePriceInput struct {
	ModelName string `json:"model_name" jsonschema:"the Pixel phone model to look up"`
}

// CheckPhonePriceOutput is the output of the check_phone_price tool.
type CheckPhonePriceOutput struct {
	Price float64 `json:"price"`
}

// checkPhonePrice is a mock tool to check the current price of a Pixel phone
// model.
func checkPhonePrice(_ agent.ToolContext, in CheckPhonePriceInput) (CheckPhonePriceOutput, error) {
	prices := map[string]float64{
		"Pixel 10a":         499.0,
		"Pixel 10":          799.0,
		"Pixel 10 Pro":      999.0,
		"Pixel 10 Pro XL":   1199.0,
		"Pixel 10 Pro Fold": 1799.0,
	}
	// Simple mock logic, defaulting to 799 if not found exactly.
	for key, value := range prices {
		if strings.Contains(strings.ToLower(in.ModelName), strings.ToLower(key)) {
			return CheckPhonePriceOutput{Price: value}, nil
		}
	}
	return CheckPhonePriceOutput{Price: 799.0}, nil
}

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-lite", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	checkPhonePriceTool, err := functiontool.New(functiontool.Config{
		Name:        "check_phone_price",
		Description: "Mock tool to check the current price of a Pixel phone model.",
	}, checkPhonePrice)
	if err != nil {
		log.Fatalf("Failed to create check_phone_price tool: %v", err)
	}

	// UserPreferences input schema, mirroring the Python pydantic model.
	userPreferencesSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"budget": {
				Type:        genai.TypeInteger,
				Description: "The user's maximum budget in USD",
			},
			"primary_use": {
				Type: genai.TypeString,
				Description: "What the user primarily uses their phone for" +
					" (e.g., photography, gaming, basics)",
			},
			"preferred_size": {
				Type:        genai.TypeString,
				Description: "Preferred phone size (e.g., small, large, any)",
			},
		},
		Required: []string{"budget", "primary_use", "preferred_size"},
	}

	// PhoneRecommendation output schema for the structured recommendation.
	phoneRecommendationSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"model_name": {Type: genai.TypeString},
			"price":      {Type: genai.TypeNumber},
			"reason":     {Type: genai.TypeString},
		},
		Required: []string{"model_name", "price", "reason"},
	}

	phoneRecommender, err := llmagent.New(llmagent.Config{
		Name:         "phone_recommender",
		Model:        model,
		Description:  "Recommends a Pixel phone based on preferences.",
		Mode:         llmagent.ModeSingleTurn,
		InputSchema:  userPreferencesSchema,
		OutputSchema: phoneRecommendationSchema,
		Tools:        []tool.Tool{checkPhonePriceTool},
		Instruction: `You are an expert Google Pixel hardware recommender.
Based on the provided UserPreferences, recommend exactly one Pixel phone model.
You must use the ` + "`check_phone_price`" + ` tool to find the exact current price of the model you are recommending before you finish your task.
Only recommend these phones: Pixel 10a, Pixel 10, Pixel 10 Pro, Pixel 10 Pro XL, Pixel 10 Pro Fold.
`,
	})
	if err != nil {
		log.Fatalf("Failed to create phone_recommender: %v", err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:      "root_agent",
		Model:     model,
		SubAgents: []agent.Agent{phoneRecommender},
		Instruction: `You are a helpful phone sales associate.
If the user is asking for a phone recommendation, use the ` + "`phone_recommender`" + ` to get a structured recommendation.
Once the recommender finishes, present the model, price, and reason to the user in a friendly way.
`,
	})
	if err != nil {
		log.Fatalf("Failed to create root_agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
