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

// Package console provides a simple way to interact with an agent from console application
package console

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/universal"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// consoleConfig contains command-line params for console launcher
type consoleConfig struct {
	streamingMode       agent.StreamingMode
	streamingModeString string // command-line param to be converted to agent.StreamingMode
}

// Launcher allows to interact with an agent in console
type Launcher struct {
	flags  *flag.FlagSet
	config *consoleConfig
}

// NewLauncher creates new console launcher
func NewLauncher() *Launcher {
	config := &consoleConfig{}

	fs := flag.NewFlagSet("console", flag.ContinueOnError)
	fs.StringVar(&config.streamingModeString, "streaming_mode", string(agent.StreamingModeSSE),
		fmt.Sprintf("defines streaming mode (%s|%s|%s)", agent.StreamingModeNone, agent.StreamingModeSSE, agent.StreamingModeBidi))

	return &Launcher{config: config, flags: fs}
}

// Run implements launcher.SubLauncher. It starts the console interaction loop.
func (l *Launcher) Run(ctx context.Context, config *adk.Config) error {
	userID, appName := "console_user", "console_app"

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

	rootAgent := config.AgentLoader.RootAgent()

	session := resp.Session

	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           rootAgent,
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

		streamingMode := l.config.streamingMode
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

// Parse implements launcher.SubLauncher. After parsing console-specific
// arguments returns remaining un-parsed arguments
func (l *Launcher) Parse(args []string) ([]string, error) {
	err := l.flags.Parse(args)
	if err != nil || !l.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse flags: %v", err)
	}
	if l.config.streamingModeString != string(agent.StreamingModeNone) &&
		l.config.streamingModeString != string(agent.StreamingModeSSE) &&
		l.config.streamingModeString != string(agent.StreamingModeBidi) {
		return nil, fmt.Errorf("invalid streaming_mode: %v. Should be (%s|%s|%s)", l.config.streamingModeString,
			agent.StreamingModeNone, agent.StreamingModeSSE, agent.StreamingModeBidi)
	}
	l.config.streamingMode = agent.StreamingMode(l.config.streamingModeString)
	return l.flags.Args(), nil
}

// Keyword implements launcher.SubLauncher.
func (l *Launcher) Keyword() string {
	return "console"
}

// CommandLineSyntax implements launcher.SubLauncher.
func (l *Launcher) CommandLineSyntax() string {
	return util.FormatFlagUsage(l.flags)
}

// SimpleDescription implements launcher.SubLauncher.
func (l *Launcher) SimpleDescription() string {
	return "runs an agent in console mode."
}

// Execute implements launcher.Launcher. It parses arguments and runs the launcher.
func (l *Launcher) Execute(ctx context.Context, config *adk.Config, args []string) error {
	remainingArgs, err := l.Parse(args)
	if err != nil {
		return fmt.Errorf("cannot parse args: %w", err)
	}
	// do not accept additional arguments
	err = universal.ErrorOnUnparsedArgs(remainingArgs)
	if err != nil {
		return fmt.Errorf("cannot parse all the arguments: %w", err)
	}
	return l.Run(ctx, config)
}
