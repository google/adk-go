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

// Package provides an example of hiding tools behind search_tools so the model
// loads a tool's declaration only when it needs that tool.
package main

import (
	"context"
	"errors"
	"log"
	"os"
	"slices"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolsearch"
)

/*
	  Example user queries:

		What time is it?
		Multiply 6 by 7.
		Convert 21 degrees Celsius to Fahrenheit.
*/
func main() {
	ctx := context.Background()

	m, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	a, err := newAgent(m)
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

// newAgent builds an agent that sees get_current_time and search_tools up
// front. Every other tool in the catalog is declared to the model only after
// search_tools has returned it.
func newAgent(m model.LLM) (agent.Agent, error) {
	tools, err := catalogTools()
	if err != nil {
		return nil, err
	}
	var gatedNames []string
	for _, t := range tools {
		if t.Name() != "get_current_time" {
			gatedNames = append(gatedNames, t.Name())
		}
	}

	gated, err := toolsearch.New(catalog(tools), toolsearch.Config{
		CoreToolNames: []string{"get_current_time"},
		// Optional. Listing the gated names in the search_tools description
		// lets the model select one by exact name instead of guessing keywords.
		GatedToolNames: gatedNames,
	})
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:        "toolsearch_agent",
		Model:       m,
		Description: "Agent that discovers its tools on demand.",
		Instruction: "You help with time, arithmetic, unit conversion and text questions. Use a tool for every calculation.",
		Toolsets:    []tool.Toolset{gated},
	})
}

// catalog is the base toolset that toolsearch wraps. In a real agent it would
// typically be one or more MCP toolsets with many tools.
type catalog []tool.Tool

func (catalog) Name() string                                       { return "catalog" }
func (c catalog) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return c, nil }

type numbers struct {
	A float64 `json:"a" jsonschema:"The first number."`
	B float64 `json:"b" jsonschema:"The second number."`
}

type numberResult struct {
	Value float64 `json:"value"`
}

type temperature struct {
	Value float64 `json:"value" jsonschema:"The temperature to convert."`
	From  string  `json:"from" jsonschema:"The unit to convert from: celsius or fahrenheit."`
}

type temperatureResult struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit"`
}

type text struct {
	Text string `json:"text" jsonschema:"The text to process."`
}

type textResult struct {
	Text  string `json:"text,omitempty"`
	Count int    `json:"count,omitempty"`
}

type empty struct{}

type timeResult struct {
	Time string `json:"time"`
}

func catalogTools() ([]tool.Tool, error) {
	var tools []tool.Tool
	var errs []error
	add := func(t tool.Tool, err error) {
		tools = append(tools, t)
		errs = append(errs, err)
	}

	add(functiontool.New(functiontool.Config{Name: "get_current_time", Description: "Return the current time in RFC 3339 format."},
		func(agent.Context, empty) (timeResult, error) {
			return timeResult{Time: time.Now().Format(time.RFC3339)}, nil
		}))
	add(functiontool.New(functiontool.Config{Name: "add_numbers", Description: "Add two numbers and return their sum."},
		func(_ agent.Context, n numbers) (numberResult, error) {
			return numberResult{Value: n.A + n.B}, nil
		}))
	add(functiontool.New(functiontool.Config{Name: "multiply_numbers", Description: "Multiply two numbers and return their product."},
		func(_ agent.Context, n numbers) (numberResult, error) {
			return numberResult{Value: n.A * n.B}, nil
		}))
	add(functiontool.New(functiontool.Config{Name: "convert_temperature", Description: "Convert a temperature between Celsius and Fahrenheit."},
		func(_ agent.Context, t temperature) (temperatureResult, error) {
			if strings.EqualFold(t.From, "fahrenheit") {
				return temperatureResult{Value: (t.Value - 32) * 5 / 9, Unit: "celsius"}, nil
			}
			return temperatureResult{Value: t.Value*9/5 + 32, Unit: "fahrenheit"}, nil
		}))
	add(functiontool.New(functiontool.Config{Name: "count_words", Description: "Count the words in a text."},
		func(_ agent.Context, t text) (textResult, error) {
			return textResult{Count: len(strings.Fields(t.Text))}, nil
		}))
	add(functiontool.New(functiontool.Config{Name: "reverse_text", Description: "Reverse the characters of a text."},
		func(_ agent.Context, t text) (textResult, error) {
			r := []rune(t.Text)
			slices.Reverse(r)
			return textResult{Text: string(r)}, nil
		}))

	return tools, errors.Join(errs...)
}
