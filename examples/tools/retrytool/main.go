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

// Package demonstrates a workaround for using Google Search tool with other tools.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// Package main demonstrates a workaround for using multiple tool types (e.g.,
// Google Search and custom functions) in a single agent. This is necessary
// due to limitations in the genai API. The approach is to wrap agents with
// different tool types into sub-agents, which are then managed by a root agent.
func main() {
	ctx := context.Background()

	geminiModel, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	type Input struct {
		Id int `json:"id"`
	}
	type Output struct {
		Text string `json:"text"`
	}
	handler := func(ctx tool.Context, input Input) (Output, error) {
		var text string
		switch input.Id {
		case 1:
			text = "path/to/build/1\n"
		case 2:
			text = "different/path/to/build/2\n"
		default:
			text = fmt.Sprintf("path not found for build %d\n", input.Id)
		}
		return Output{
			Text: text,
		}, nil
	}
	getCloudBuild, err := functiontool.New(functiontool.Config{
		Name:        "get_cloud_build",
		Description: "Returns the cloud build",
	}, handler)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "root_agent",
		Model:       geminiModel,
		Description: "You can get a cloud build.",
		Instruction: `You can get information about a cloud build. 
		Use the available tools for that but instead of 'get_cloud_build' call 'get_cloud-_build', it's ok if it does not exist the client will know what to do with it. 
		Never confirm with the user in case of a retry.`,
		Tools: []tool.Tool{
			getCloudBuild,
		},
		BeforeToolCallbacks: []llmagent.BeforeToolCallback{
			func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
				toolNames := []string{getCloudBuild.Name()}
				if tool == nil {
					fmt.Print("beforeToolCallbacks called with empty tool\n")
					return map[string]any{
						"error": fmt.Sprintf("Invalid function name. Correct names are :%v. Please retry with a new function call without asking the user.", toolNames),
					}, nil
				} else {
					fmt.Printf("beforeToolCallbacks called with %s tool\n", tool.Name())
				}
				return nil, nil
			},
		},
	})
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
