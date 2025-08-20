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
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/examples"
	"google.golang.org/adk/llm/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GEMINI_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	weatherAgent, err := llmagent.New(llmagent.Config{
		Name:        "weather_agent",
		Model:       model,
		Description: "Agent to answer questions about the current weather.",
		Instruction: `
		I can answer your questions about the weather in a city.`,
		Tools: []tool.Tool{
			tool.MustNewFunctionTool(
				tool.FunctionToolConfig{
					Name:        "get_weather_report",
					Description: "Retrieves the current weather report for a specified city.",
				},
				weatherReport,
			),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create weather agent: %v", err)
	}

	timeAgent, err := llmagent.New(llmagent.Config{
		Name:        "time_agent",
		Model:       model,
		Description: "Agent to answer questions about the current time.",
		Instruction: "I can answer your questions about the current time.",
		Tools: []tool.Tool{
			tool.MustNewGenaiTool(&genai.Tool{
				GoogleSearch: &genai.GoogleSearch{},
			}),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create time agent: %v", err)
	}

	agent, err := llmagent.New(llmagent.Config{
		Name:        "dispatcher_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: `
			Delegate weather requests to weather_agent and time requests to time_agent.
			If weather_agent or time_agent don't have required information, notify the user.
			For any other queries reply with: "I cannot answer."
		`,
		SubAgents: []agent.Agent{
			weatherAgent, timeAgent,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	examples.Run(ctx, agent)
}

type Args struct {
	City string `json:"city"`
}
type Result struct {
	Report string `json:"report"`
	Status string `json:"status"`
}

var resultSet = map[string]Result{
	"london": {
		Status: "success",
		Report: "The weather in London was cloudy with a temperature of 18 degrees Celsius.",
	},
	"paris": {
		Status: "success",
		Report: "The weather in Paris was sunny with a temperature of 25 derees Celsius.",
	},
}

func weatherReport(ctx context.Context, input Args) Result {
	city := strings.ToLower(input.City)
	if ret, ok := resultSet[city]; ok {
		return ret
	}
	return Result{
		Status: "error",
		Report: fmt.Sprintf("Weather information for %q is not available.", city),
	}
}
