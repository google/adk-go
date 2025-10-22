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

// package console provides a simple way to test an agent from console application
package console

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// ConsoleConfig contains command-line params for console launcher
type ConsoleConfig struct {
	streamingMode agent.StreamingMode
	rootAgentName string
}

// ConsoleLauncher allows to interact with an agent in console
type ConsoleLauncher struct {
	Config *ConsoleConfig
}

// Run starts console loop. User-provided text is fed to the chosen agent (the only one if there's only one, specified by name otherwise)
func (l ConsoleLauncher) Run(ctx context.Context, config *adk.Config) error {
	userID, appName := "test_user", "test_app"

	sessionService := config.SessionService
	if sessionService == nil {
		sessionService = session.InMemoryService()
	}

	resp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return fmt.Errorf("failed to create the session service: %v", err)
	}

	matchingAgent, err := config.AgentLoader.LoadAgent(l.Config.rootAgentName)
	if err != nil {
		return fmt.Errorf("failed to find the matching agent: %v", err)
	}

	session := resp.Session

	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           matchingAgent,
		SessionService:  sessionService,
		ArtifactService: config.ArtifactService,
	})
	if err != nil {
		return fmt.Errorf("failed to create runner: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nUser -> ")

		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}

		userMsg := genai.NewContentFromText(userInput, genai.RoleUser)

		streamingMode := l.Config.streamingMode
		if streamingMode == "" {
			streamingMode = agent.StreamingModeSSE
		}
		fmt.Print("\nAgent -> ")
		for event, err := range r.Run(ctx, userID, session.ID(), userMsg, agent.RunConfig{
			StreamingMode: streamingMode,
		}) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
			} else {
				for _, p := range event.LLMResponse.Content.Parts {
					// if its running in streaming mode, don't print the non partial llmResponses
					if streamingMode != agent.StreamingModeSSE || event.LLMResponse.Partial {
						fmt.Print(p.Text)
					}
				}
			}
		}
	}
}

// BuildLauncher parses command line args and returns ready-to-run console launcher.
func BuildLauncher(args []string) (launcher.Launcher, []string, error) {
	consoleConfig, argsLeft, err := ParseArgs(args)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot parse arguments for console: %v: %w", args, err)
	}
	return &ConsoleLauncher{Config: consoleConfig}, argsLeft, nil
}

func ParseArgs(args []string) (*ConsoleConfig, []string, error) {
	fs := flag.NewFlagSet("console", flag.ContinueOnError)

	var streaming = ""
	var rootAgentName = ""
	fs.StringVar(&streaming, "streaming_mode", string(agent.StreamingModeSSE), fmt.Sprintf("defines streaming mode (%s|%s|%s)", agent.StreamingModeNone, agent.StreamingModeSSE, agent.StreamingModeBidi))
	fs.StringVar(&rootAgentName, "root_agent_name", "", "If you have multiple agents you should specify which one should be user for interactions. You can leave if empty if you have only one agent - it will be used by default")

	err := fs.Parse(args)
	if err != nil || !fs.Parsed() {
		return &(ConsoleConfig{}), nil, fmt.Errorf("failed to parse flags: %v", err)
	}
	if streaming != string(agent.StreamingModeNone) && streaming != string(agent.StreamingModeSSE) && streaming != string(agent.StreamingModeBidi) {
		return &(ConsoleConfig{}), nil, fmt.Errorf("invalid streaming_mode: %v. Should be (%s|%s|%s)", streaming, agent.StreamingModeNone, agent.StreamingModeSSE, agent.StreamingModeBidi)
	}
	res := ConsoleConfig{
		streamingMode: agent.StreamingMode(streaming),
		rootAgentName: rootAgentName,
	}
	return &res, fs.Args(), nil
}
