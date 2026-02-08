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

// Package console provides a simple way to interact with an agent from console application.
package console

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/internal/telemetry"
	"google.golang.org/adk/cmd/launcher/universal"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// consoleConfig contains command-line params for console launcher
type consoleConfig struct {
	streamingMode       agent.StreamingMode
	streamingModeString string // command-line param to be converted to agent.StreamingMode
	otelToCloud         bool
	shutdownTimeout     time.Duration
}

// consoleLauncher allows to interact with an agent in console
type consoleLauncher struct {
	flags  *flag.FlagSet  // flags are used to parse command-line arguments
	config *consoleConfig // config contains parsed command-line parameters
}

// NewLauncher creates new console launcher
func NewLauncher() launcher.SubLauncher {
	config := &consoleConfig{}

	fs := flag.NewFlagSet("console", flag.ContinueOnError)
	fs.StringVar(&config.streamingModeString, "streaming_mode", string(agent.StreamingModeSSE),
		fmt.Sprintf("defines streaming mode (%s|%s)", agent.StreamingModeNone, agent.StreamingModeSSE))
	fs.DurationVar(&config.shutdownTimeout, "shutdown-timeout", 2*time.Second, "Console shutdown timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for waiting for active requests to finish during shutdown")
	fs.BoolVar(&config.otelToCloud, "otel_to_cloud", false, "Enables/disables OpenTelemetry export to GCP: telemetry.googleapis.com. See adk-go/telemetry package for details about supported options, credentials and environment variables.")
	return &consoleLauncher{config: config, flags: fs}
}

// Run implements launcher.SubLauncher. It starts the console interaction loop.
func (l *consoleLauncher) Run(ctx context.Context, config *launcher.Config) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	telemetry, err := telemetry.InitAndSetGlobalOtelProviders(ctx, config, l.config.otelToCloud)
	if err != nil {
		return fmt.Errorf("telemetry initialization failed: %v", err)
	}
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), l.config.shutdownTimeout)
		defer cancel()
		if err := telemetry.Shutdown(shutdownCtx); err != nil {
			log.Printf("telemetry shutdown failed: %v", err)
		}
	}()

	// userID and appName are not important at this moment, we can just use any
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
		PluginConfig:    config.PluginConfig,
	})
	if err != nil {
		return fmt.Errorf("failed to create runner: %v", err)
	}

	inputChan := make(chan string)
	readErrChan := make(chan error, 1)

	go func() {
		reader := bufio.NewReader(os.Stdin)
		for {
			userInput, err := reader.ReadString('\n')
			if err != nil {
				readErrChan <- err
				return
			}
			inputChan <- userInput
		}
	}()

	fmt.Print("\nUser -> ")

	for {
		select {
		case <-ctx.Done():
			return nil
		case err := <-readErrChan:
			if errors.Is(err, io.EOF) {
				fmt.Println("\nEOF detected, exiting...")
				return nil
			}
			log.Fatal(err)
		case userInput := <-inputChan:
			userMsg := genai.NewContentFromText(userInput, genai.RoleUser)

			streamingMode := l.config.streamingMode
			if streamingMode == "" {
				streamingMode = agent.StreamingModeSSE
			}
			fmt.Print("\nAgent -> ")
			prevText := ""
			for event, err := range r.Run(ctx, userID, session.ID(), userMsg, agent.RunConfig{
				StreamingMode: streamingMode,
			}) {
				if err != nil {
					fmt.Printf("\nAGENT_ERROR: %v\n", err)
				} else {
					if event.LLMResponse.Content == nil {
						continue
					}

					text := ""
					for _, p := range event.LLMResponse.Content.Parts {
						text += p.Text
					}

					if streamingMode != agent.StreamingModeSSE {
						fmt.Print(text)
						continue
					}

					// In SSE mode, always print partial responses and capture them.
					if !event.IsFinalResponse() {
						fmt.Print(text)
						prevText += text
						continue
					}

					// Only print final response if it doesn't match previously captured text.
					if text != prevText {
						fmt.Print(text)
					}

					prevText = ""
				}
			}
			fmt.Print("\nUser -> ")
		}
	}
}

// Parse implements launcher.SubLauncher. After parsing console-specific
// arguments returns remaining un-parsed arguments
func (l *consoleLauncher) Parse(args []string) ([]string, error) {
	err := l.flags.Parse(args)
	if err != nil || !l.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse flags: %v", err)
	}
	if l.config.streamingModeString != string(agent.StreamingModeNone) &&
		l.config.streamingModeString != string(agent.StreamingModeSSE) {
		return nil, fmt.Errorf("invalid streaming_mode: %v. Should be (%s|%s)", l.config.streamingModeString,
			agent.StreamingModeNone, agent.StreamingModeSSE)
	}
	l.config.streamingMode = agent.StreamingMode(l.config.streamingModeString)
	return l.flags.Args(), nil
}

// Keyword implements launcher.SubLauncher. Returns the command-line keyword for this launcher.
func (l *consoleLauncher) Keyword() string {
	return "console"
}

// CommandLineSyntax implements launcher.SubLauncher. Returns the command-line syntax for the console launcher.
func (l *consoleLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(l.flags)
}

// SimpleDescription implements launcher.SubLauncher. Returns a simple description of the console launcher.
func (l *consoleLauncher) SimpleDescription() string {
	return "runs an agent in console mode."
}

// Execute implements launcher.Launcher. It parses arguments and runs the launcher.
func (l *consoleLauncher) Execute(ctx context.Context, config *launcher.Config, args []string) error {
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
