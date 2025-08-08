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
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/llm/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GEMINI_API_KEY"),
	})
	if err != nil {
		panic(err)
	}

	agent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
	})
	if err != nil {
		panic(err)
	}

	runAgent(agent)
}

func runAgent(agent agent.Agent) {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nUser -> ")

		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}

		ctx := &agentContext{
			Context:     context.Background(),
			userContent: genai.NewContentFromText(userInput, genai.RoleUser),
		}

		fmt.Print("\nAgent -> ")
		for event, err := range agent.Run(ctx) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
			} else {
				for _, p := range event.LLMResponse.Content.Parts {
					fmt.Print(p.Text)
				}
			}
		}
	}
}

// TODO: move to runner
type agentContext struct {
	context.Context
	ended        bool
	userContent  *genai.Content
	invocationID string
	branch       string
	agentName    string
	artifacts    agent.Artifacts
	session      session.Session
}

func (c *agentContext) UserContent() *genai.Content {
	return c.userContent
}
func (c *agentContext) InvocationID() string {
	return c.invocationID
}
func (c *agentContext) Branch() string {
	return c.branch
}
func (c *agentContext) AgentName() string {
	return c.agentName
}
func (c *agentContext) Session() session.Session {
	return c.session
}
func (c *agentContext) Artifacts() agent.Artifacts {
	return c.artifacts
}
func (c *agentContext) End() {
	c.ended = true
}
func (c *agentContext) Ended() bool {
	return c.ended
}
func (c *agentContext) Report(*session.Event) {}
